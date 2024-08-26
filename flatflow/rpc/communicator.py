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

from flatflow.rpc.BroadcastRequest import (
    BroadcastRequestAddCosts,
    BroadcastRequestAddEpoch,
    BroadcastRequestAddRank,
    BroadcastRequestEnd,
    BroadcastRequestStart,
    BroadcastRequestStartCostsVector,
)
from flatflow.rpc.BroadcastResponse import BroadcastResponse
from flatflow.rpc.communicator_grpc_fb import CommunicatorStub
from flatflow.rpc.Empty import Empty, EmptyEnd, EmptyStart
from flatflow.rpc.InitRequest import (
    InitRequestAddGlobalBatchSize,
    InitRequestAddHeterogeneous,
    InitRequestAddHiddenSize,
    InitRequestAddMicroBatchSize,
    InitRequestAddOrder,
    InitRequestAddRank,
    InitRequestAddSeed,
    InitRequestAddSizes,
    InitRequestAddUseFlatShuffle,
    InitRequestEnd,
    InitRequestStart,
    InitRequestStartSizesVector,
)

__all__ = ["CommunicatorClient"]


class CommunicatorClient(object):
    """A client class that simplifies communication with the communicator runtime.

    Args:
        channel (grpc.Channel): A Channel object.
    """

    rank: int
    stub: CommunicatorStub

    def __init__(self, channel: grpc.Channel) -> None:
        # Block until the communicator runtime is ready.
        grpc.channel_ready_future(channel).result()
        self.stub = CommunicatorStub(channel)

    def Init(
        self,
        global_batch_size: int,
        micro_batch_size: int,
        order: int,
        rank: int,
        seed: int,
        heterogeneous: bool,
        use_flat_shuffle: bool,
        hidden_size: Optional[int] = None,
        sizes: Optional[Sequence[int]] = None,
    ) -> Empty:
        """Initializes the training environment.

        Args:
            global_batch_size (int): The global batch size.
            micro_batch_size (int): The micro-batch size.
            order (int): The order of complexity for a given size; e.g., Transformers
                have quadratic complexity for the context length, so its order is two.
            rank (int): Rank of the current process within the data parallel size.
            seed (int): Random seed used to shuffle the samples.
            heterogeneous (bool): If ``True``, the scheduler generates a computation
                schedule considering the heterogeneity of cluster.
            use_flat_shuffle (bool): If ``False``, the scheduler shuffles only between
                micro-batches with the same cost.
            hidden_size (int, optional): The hidden dimension size.
            sizes (Sequence[int], optional): A vector representing the mapping from an
                index to the user-defined size of the corresponding data sample.

        Returns:
            An :class:`~flatflow.rpc.Empty` object.
        """
        self.rank = rank

        if hidden_size is None:
            hidden_size = 0

        builder = flatbuffers.Builder()

        if rank == 0:
            assert sizes is not None
            InitRequestStartSizesVector(builder, len(sizes))
            # Since we prepend the sizes, this loop iterates in reverse order.
            for size in reversed(sizes):
                builder.PrependUint16(size)
            _sizes = builder.EndVector()

        InitRequestStart(builder)
        InitRequestAddGlobalBatchSize(builder, global_batch_size)
        InitRequestAddHiddenSize(builder, hidden_size)
        InitRequestAddMicroBatchSize(builder, micro_batch_size)
        InitRequestAddOrder(builder, order)
        InitRequestAddRank(builder, rank)
        InitRequestAddSeed(builder, seed)
        if rank == 0:
            InitRequestAddSizes(builder, _sizes)
        InitRequestAddHeterogeneous(builder, heterogeneous)
        InitRequestAddUseFlatShuffle(builder, use_flat_shuffle)
        request = InitRequestEnd(builder)
        builder.Finish(request)
        buffer = bytes(builder.Output())

        empty = self.stub.Init(buffer)
        return Empty.GetRootAs(empty)

    def Broadcast(self, epoch: int, costs: Optional[Sequence[float]] = None) -> BroadcastResponse:
        """Gets computation schedule from the scheduler.

        If the scheduler provides profile-guided optimization (PGO), the given cost is
        used to estimate the complexity.

        Args:
            epoch (int): The epoch number.
            costs (Sequence[float], optional): The evaluated costs used in PGO.

        Returns:
            A :class:`~flatflow.rpc.BroadcastResponse` object containing the computation
            schedule. If the scheduler does not support PGO, :meth:`Converged` always
            returns ``True``.
        """
        builder = flatbuffers.Builder()

        if costs is not None:
            BroadcastRequestStartCostsVector(builder, len(costs))
            # Since we prepend the costs, this loop iterates in reverse order.
            for cost in reversed(costs):
                builder.PrependFloat64(cost)
            _costs = builder.EndVector()

        BroadcastRequestStart(builder)
        BroadcastRequestAddEpoch(builder, epoch)
        BroadcastRequestAddRank(builder, self.rank)
        if costs is not None:
            BroadcastRequestAddCosts(builder, _costs)
        request = BroadcastRequestEnd(builder)
        builder.Finish(request)
        buffer = bytes(builder.Output())

        response = self.stub.Broadcast(buffer)
        return BroadcastResponse.GetRootAs(response)

    def Finalize(self) -> Empty:
        """Terminates the training environment.

        Returns:
            An :class:`~flatflow.rpc.Empty` object.
        """
        assert self.rank == 0

        builder = flatbuffers.Builder()

        EmptyStart(builder)
        empty = EmptyEnd(builder)
        builder.Finish(empty)
        buffer = bytes(builder.Output())

        empty = self.stub.Finalize(buffer)
        return Empty.GetRootAs(empty)
