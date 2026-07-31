# Adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/data/distributed.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from collections.abc import Iterator
from functools import partial

import numpy
import torch

from flatflow.ffi import sched, sched_unstable
from flatflow.ops import serialize
from flatflow.sys import getsizeof
from flatflow.torch.utils.data.dataset import Dataset

__all__ = ["DistributedSampler"]


class DistributedSampler(torch.utils.data.DistributedSampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is a drop-in replacement for :class:`torch.utils.data.DistributedSampler` that
    reorders the computation schedule. Once a computation schedule identical to that of
    the PyTorch counterpart is made, it passes the indices to the scheduler to reorder
    them according to scheduling objectives. Please refer to the below for details on
    scheduling policies and the corresponding behavior of the scheduler.

    Args:
        dataset (Dataset): Dataset used for sampling.
        tensor_parallel_world_size (int): Number of processes participating in the
            tensor-parallel process groups.
        context_parallel_world_size (int): Number of processes participating in the
            context-parallel process groups.
        data_parallel_world_size (int): Number of processes participating in the
            data-parallel process groups.
        data_parallel_rank (int): Rank of the current process within the data-parallel
            process groups.
        global_batch_size (int): The global batch size.
        micro_batch_size (int): The micro-batch size.
        graph (torch.fx.Graph): Graph representation of a PyTorch program.
        shuffle (bool, optional): If ``True``, sampler will shuffle the indices.
            Default: ``True``.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all processes
            in the distributed group. Default: ``0``.
        drop_last (bool, optional): If ``True``, then the sampler will drop the tail of
            the data to make it evenly divisible across the number of replicas. If
            ``False``, the sampler will add extra indices to make the data evenly
            divisible across the replicas. Default: ``False``.
        unstable (bool, optional): How aggressively the scheduler is allowed to move
            samples around. If ``False``, reordering is confined to within each step so
            that every batch keeps its original composition and the resulting checkpoint
            is left unchanged, which we call iterative reordering. If ``True``, the
            scheduler may move samples freely across step boundaries. This will change
            the computation schedule and therefore yield a different checkpoint than a
            stable run. Default: ``True``.
        policy (str, optional): Scheduling policy to select the scheduling objectives.
            ``"fast"`` minimizes computation stalls, ``"mem"`` prioritizes memory
            balance and ``"joint"`` strikes a balance between the two.
            Default: ``"joint"``.
    """

    def __init__(
        self,
        dataset: Dataset,
        tensor_parallel_world_size: int,
        context_parallel_world_size: int,
        data_parallel_world_size: int,
        data_parallel_rank: int,
        global_batch_size: int,
        micro_batch_size: int,
        graph: torch.fx.Graph,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        unstable: bool = True,
        policy: str = "joint",
    ) -> None:
        super().__init__(
            dataset,
            num_replicas=data_parallel_world_size,
            rank=data_parallel_rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        self.tensor_parallel_world_size = tensor_parallel_world_size
        self.context_parallel_world_size = context_parallel_world_size
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.buf = serialize(graph)
        self.unstable = unstable
        self.policy = policy

        sizes = map(partial(getsizeof, dataset), range(len(dataset)))  # type: ignore[arg-type]
        self.sizes = numpy.fromiter(sizes, dtype=numpy.int64, count=len(dataset))  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).numpy()  # type: ignore[arg-type]
            indices = indices.view(numpy.uintp)
        else:
            indices = numpy.arange(len(self.dataset), dtype=numpy.uintp)  # type: ignore[arg-type]

        if indices.size < self.total_size:
            indices = numpy.resize(indices, self.total_size)
        else:
            indices = indices[: self.total_size]
        assert indices.size == self.total_size

        if self.unstable:
            indices = sched_unstable(
                indices,  # type: ignore[arg-type]
                self.sizes,  # type: ignore[arg-type]
                self.buf,
                self.tensor_parallel_world_size,
                self.context_parallel_world_size,
                self.num_replicas,
                self.rank,
                self.global_batch_size,
                self.micro_batch_size,
                self.policy,
            )
        else:
            indices = sched(
                indices,  # type: ignore[arg-type]
                self.sizes,  # type: ignore[arg-type]
                self.buf,
                self.tensor_parallel_world_size,
                self.context_parallel_world_size,
                self.num_replicas,
                self.rank,
                self.global_batch_size,
                self.micro_batch_size,
                self.policy,
            )
        assert indices.size == self.num_samples

        return map(operator.index, indices)
