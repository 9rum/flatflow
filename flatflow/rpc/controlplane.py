# Copyright 2024 The FlatFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from typing import Optional

import flatbuffers
import grpc
import torch
import torch.fx
from numpy.typing import ArrayLike
from torch._ops import OpOverload

from flatflow.ops import *  # noqa: F403
from flatflow.rpc.controlplane_generated import (
    BroadcastRequestAddEpoch,
    BroadcastRequestAddIndices,
    BroadcastRequestAddRank,
    BroadcastRequestEnd,
    BroadcastRequestStart,
    BroadcastRequestStartIndicesVector,
    BroadcastResponse,
    InitRequestAddGlobalBatchSize,
    InitRequestAddGraph,
    InitRequestAddMicroBatchSize,
    InitRequestAddSizes,
    InitRequestEnd,
    InitRequestStart,
    InitRequestStartSizesVector,
)
from flatflow.rpc.controlplane_grpc_fb import ControlPlaneStub
from flatflow.rpc.empty_generated import EmptyEnd, EmptyStart

__all__ = ["ControlPlaneClient"]


class ControlPlaneClient(object):
    """A client class that simplifies communication with the control plane.

    Args:
        rank (int): Rank of the current process within the data-parallel group.
        channel (grpc.Channel): A channel object.
    """

    rank: int
    stub: ControlPlaneStub

    def __init__(self, rank: int, channel: grpc.Channel) -> None:
        self.rank = rank
        # Block until the control plane is ready.
        grpc.channel_ready_future(channel).result()
        self.stub = ControlPlaneStub(channel)

    def Init(
        self,
        global_batch_size: int,
        micro_batch_size: int,
        graph: torch.fx.Graph,
        sizes: Sequence[int],
    ) -> None:
        """Initializes the training environment.

        Args:
            global_batch_size (int): The global batch size.
            micro_batch_size (int): The micro-batch size.
            graph (torch.fx.Graph): A computational graph traced from the given model.
            sizes (Sequence[int]): A vector representing the mapping from an index to
                the user-defined size of the corresponding data sample.
        """
        assert self.rank == 0

        builder = flatbuffers.Builder()

        nodes = []

        for node in graph.nodes:
            if isinstance(node.target, OpOverload):
                opcode = node.target.name()
                assert opcode in _OPCODES, (
                    f"{opcode} is not a supported operator.\n"
                    "Please make sure you are using the latest version of FlatFlow\n"
                    "or file an issue to https://github.com/9rum/flatflow/issues.\n"
                    "The latest release can be found at https://github.com/9rum/flatflow/releases."
                )
                target = _OPCODES[opcode]

                args = []

                for arg in node.args:
                    if isinstance(arg, torch.fx.Node) and "tensor_meta" in arg.meta:
                        shape = []

                        for maybe_sym_int in arg.meta["tensor_meta"].shape:
                            if isinstance(maybe_sym_int, torch.SymInt):
                                expr = maybe_sym_int.node.expr
                                symbol = next(iter(expr.free_symbols))
                                shape.append(
                                    [expr.coeff(symbol, 0), expr.coeff(symbol, 1)]
                                )
                            else:
                                shape.append([maybe_sym_int, 0])

                        TensorMetadataStartShapeVector(builder, len(shape))
                        for sym_int in reversed(shape):
                            CreateSymInt(builder, sym_int)
                        _shape = builder.EndVector()

                        TensorMetadataStart(builder)
                        TensorMetadataAddShape(builder, _shape)
                        _arg = TensorMetadataEnd(builder)
                        args.append(_arg)

                NodeStartArgsVector(builder, len(args))
                for arg in reversed(args):
                    builder.PrependUOffsetTRelative(arg)
                _args = builder.EndVector()

                shape = []

                if "tensor_meta" in node.meta:
                    for maybe_sym_int in node.meta["tensor_meta"].shape:
                        if isinstance(maybe_sym_int, torch.SymInt):
                            expr = maybe_sym_int.node.expr
                            symbol = next(iter(expr.free_symbols))
                            shape.append([expr.coeff(symbol, 0), expr.coeff(symbol, 1)])
                        else:
                            shape.append([maybe_sym_int, 0])

                TensorMetadataStartShapeVector(builder, len(shape))
                for sym_int in reversed(shape):
                    CreateSymInt(builder, sym_int)
                _shape = builder.EndVector()

                TensorMetadataStart(builder)
                TensorMetadataAddShape(builder, _shape)
                _meta = TensorMetadataEnd(builder)

                NodeStart(builder)
                NodeAddTarget(builder, target)
                NodeAddArgs(builder, _args)
                NodeAddMeta(builder, _meta)
                _node = NodeEnd(builder)
                nodes.append(_node)

        GraphStartNodesVector(builder, len(nodes))
        for node in reversed(nodes):
            builder.PrependUOffsetTRelative(node)
        _nodes = builder.EndVector()

        GraphStart(builder)
        GraphAddNodes(builder, _nodes)
        _graph = GraphEnd(builder)

        InitRequestStartSizesVector(builder, len(sizes))
        for size in reversed(sizes):
            builder.PrependUint32(size)
        _sizes = builder.EndVector()

        InitRequestStart(builder)
        InitRequestAddGlobalBatchSize(builder, global_batch_size)
        InitRequestAddMicroBatchSize(builder, micro_batch_size)
        InitRequestAddGraph(builder, _graph)
        InitRequestAddSizes(builder, _sizes)
        request = InitRequestEnd(builder)
        builder.Finish(request)

        self.stub.Init(bytes(builder.Output()))

    def Broadcast(
        self, epoch: int, indices: Optional[Sequence[int]] = None
    ) -> ArrayLike:
        """Returns the reordered computation schedule for the next training epoch.

        Args:
            epoch (int): The epoch number.
            indices (Sequence[int], optional): The original computation schedule.

        Returns:
            ArrayLike: The reordered computation schedule.
        """
        builder = flatbuffers.Builder()

        if self.rank == 0:
            assert indices is not None
            BroadcastRequestStartIndicesVector(builder, len(indices))
            for index in reversed(indices):
                builder.PrependUint64(index)
            _indices = builder.EndVector()

        BroadcastRequestStart(builder)
        BroadcastRequestAddEpoch(builder, epoch)
        BroadcastRequestAddRank(builder, self.rank)
        if self.rank == 0:
            BroadcastRequestAddIndices(builder, _indices)
        request = BroadcastRequestEnd(builder)
        builder.Finish(request)

        response = self.stub.Broadcast(bytes(builder.Output()))
        return BroadcastResponse.GetRootAs(response).IndicesAsNumpy()  # type: ignore[call-arg]

    def Finalize(self) -> None:
        """Terminates the training environment."""
        assert self.rank == 0

        builder = flatbuffers.Builder()

        EmptyStart(builder)
        empty = EmptyEnd(builder)
        builder.Finish(empty)

        self.stub.Finalize(bytes(builder.Output()))
