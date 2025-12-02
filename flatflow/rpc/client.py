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

from collections.abc import Generator, Sequence

import flatbuffers
import grpc
import numpy
import torch.fx

from flatflow.ops import serialize
from flatflow.rpc.controlplane_generated import (
    InitRequestAddBody,
    InitRequestAddTrailer,
    InitRequestBodyAddSizes,
    InitRequestBodyAddTotalSize,
    InitRequestBodyEnd,
    InitRequestBodyStart,
    InitRequestBodyStartSizesVector,
    InitRequestEnd,
    InitRequestStart,
    InitRequestTrailerAddDataParallelRank,
    InitRequestTrailerAddDataParallelWorldSize,
    InitRequestTrailerAddGlobalBatchSize,
    InitRequestTrailerAddGraph,
    InitRequestTrailerAddMicroBatchSize,
    InitRequestTrailerEnd,
    InitRequestTrailerStart,
    ScatterRequestAddEpoch,
    ScatterRequestAddIndices,
    ScatterRequestEnd,
    ScatterRequestStart,
    ScatterRequestStartIndicesVector,
    ScatterResponse,
)
from flatflow.rpc.controlplane_grpc_fb import ControlPlaneStub
from flatflow.rpc.empty_generated import EmptyEnd, EmptyStart

__all__ = ["ControlPlaneClient"]


class ControlPlaneClient(object):
    """A client class that simplifies communication with the control plane.

    Args:
        port (int): The port number on which the control plane runs.
    """

    stub: ControlPlaneStub

    def __init__(self, port: int) -> None:
        # Block until the control plane is ready.
        channel = grpc.insecure_channel(f"[::1]:{port}")
        grpc.channel_ready_future(channel).result()
        self.stub = ControlPlaneStub(channel)

    def Init(
        self,
        data_parallel_rank: int,
        data_parallel_world_size: int,
        global_batch_size: int,
        micro_batch_size: int,
        graph: torch.fx.Graph,
        sizes: Sequence[int],
    ) -> None:
        """Initializes the training environment.

        Args:
            data_parallel_rank (int): Rank of the current process within the
                data-parallel group.
            data_parallel_world_size (int): Number of processes within the data-parallel
                group.
            global_batch_size (int): The global batch size.
            micro_batch_size (int): The micro-batch size.
            graph (torch.fx.Graph): A computational graph traced from the given model.
            sizes (Sequence[int]): A vector representing the mapping from an index to
                the user-defined size of the corresponding data sample.
        """
        builder = flatbuffers.Builder()

        def impl() -> Generator[bytes, None, None]:
            stride = 1 << 19
            for offset in range(0, len(sizes), stride):
                view = sizes[offset : offset + stride][::-1]
                InitRequestBodyStartSizesVector(builder, len(view))
                for size in view:
                    builder.PrependUint32(size)
                _sizes = builder.EndVector()

                InitRequestBodyStart(builder)
                InitRequestBodyAddTotalSize(builder, len(sizes))
                InitRequestBodyAddSizes(builder, _sizes)
                body = InitRequestBodyEnd(builder)

                InitRequestStart(builder)
                InitRequestAddBody(builder, body)
                request = InitRequestEnd(builder)
                builder.Finish(request)
                yield bytes(builder.Output())
                builder.Clear()

            _graph = serialize(builder, graph)

            InitRequestTrailerStart(builder)
            InitRequestTrailerAddDataParallelRank(builder, data_parallel_rank)
            InitRequestTrailerAddDataParallelWorldSize(
                builder, data_parallel_world_size
            )
            InitRequestTrailerAddGlobalBatchSize(builder, global_batch_size)
            InitRequestTrailerAddMicroBatchSize(builder, micro_batch_size)
            InitRequestTrailerAddGraph(builder, _graph)
            trailer = InitRequestTrailerEnd(builder)

            InitRequestStart(builder)
            InitRequestAddTrailer(builder, trailer)
            request = InitRequestEnd(builder)
            builder.Finish(request)
            yield bytes(builder.Output())

        self.stub.Init(impl())

    def Scatter(self, epoch: int, indices: Sequence[int]) -> numpy.ndarray:
        """Returns the reordered computation schedule for the next training epoch.

        Args:
            epoch (int): The epoch number.
            indices (Sequence[int]): The original computation schedule.

        Returns:
            numpy.ndarray: The reordered computation schedule.
        """
        builder = flatbuffers.Builder()

        def impl() -> Generator[bytes, None, None]:
            stride = 1 << 18
            for offset in range(0, len(indices), stride):
                view = indices[offset : offset + stride][::-1]
                ScatterRequestStartIndicesVector(builder, len(view))
                for index in view:
                    builder.PrependUint64(index)
                _indices = builder.EndVector()

                ScatterRequestStart(builder)
                ScatterRequestAddEpoch(builder, epoch)
                ScatterRequestAddIndices(builder, _indices)
                request = ScatterRequestEnd(builder)
                builder.Finish(request)
                yield bytes(builder.Output())
                builder.Clear()

        responses = self.stub.Scatter(impl())
        for response in responses:
            yield from ScatterResponse.GetRootAs(response).IndicesAsNumpy()  # type: ignore[call-arg]

    def Finalize(self) -> None:
        """Terminates the training environment."""
        builder = flatbuffers.Builder()

        EmptyStart(builder)
        empty = EmptyEnd(builder)
        builder.Finish(empty)

        self.stub.Finalize(bytes(builder.Output()))
