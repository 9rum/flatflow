import os
import subprocess
from typing import TypeVar, Optional, Iterable, Iterator

import grpc
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from chronica import sys
from chronica.torch.utils.data import Dataset
from chronica.rpc import Arguments, Feedback, Schedule, STATIC, DYNAMIC, SchedulerStub

__all__ = ["DistributedSampler"]

T_co = TypeVar('T_co', covariant=True)


class DistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~chronica.torch.utils.data.DistributedSampler` instance
    as a :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): Not used but for PyTorch compatibility.
        seed (int, optional): Random seed used to shuffle the sampler.
            This number should be identical across all processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): If ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        batch_size (int, optional): How many samples per batch to load. Default: ``1``.
        master_addr (str, optional): Address of the master node (rank 0).
            If rendezvous protocol is enabled using ``torchrun``, the sampler automatically gets the address
            from the environment variable.
        master_port (int, optional): Port on the master node (rank 0) to be used for initializing
            the scheduler server. Default: ``50051``.
        schedule (str, optional): Schedule type (must be either ``"static"`` or ``"dynamic"``).
            By default, ``"static"`` is set for static scheduling that reduces the workload imbalance between workers.
            If ``"dynamic"``, the scheduler provides a feedback-directed optimization that adaptively adjusts
            the workload on each worker.
        interval (int, optional): Interval, in # of steps, to report the performance indicators for dynamic scheduling.
        partition (bool, optional): If ``True``, then the sampler will restrict remote data fetching.
            It is especially useful when the data is distributed among devices and machines. In such a case,
            ``groups`` should tell the mapping about which workers are on which nodes. Default: ``False``.
        groups (Iterable, optional): Mapping from worker rank to node rank. For instance, if the cluster is homogeneous
            of two nodes with four GPUs each, ``groups`` would be ``[0, 0, 0, 0, 1, 1, 1, 1]``. On the other hand,
            if the cluster is heterogeneous with four GPUs on node #0 and two GPUs on node #1,
            then ``groups`` would be ``[0, 0, 0, 0, 1, 1]``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make scheduling work properly across multiple epochs.
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, batch_size: int = 1,
                 master_addr: str = None, master_port: int = 50051,
                 schedule: str = None, interval: int = 1,
                 partition: bool = False, groups: Iterable = None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if num_replicas <= rank or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval [0, {}]".format(rank, num_replicas - 1))
        if master_addr is None:
            master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        if schedule is None or schedule == "static":
            self.schedule = STATIC
        elif schedule == "dynamic":
            self.schedule = DYNAMIC
        else:
            raise ValueError("Invalid schedule type {}, schedule should be either static or dynamic".format(schedule))
        self.rank = rank
        self.interval = interval

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        self.num_samples = len(dataset) // num_replicas  # type: ignore[arg-type]
        if not drop_last and len(dataset) % num_replicas != 0:  # type: ignore[arg-type]
            self.num_samples += 1
        total_size = self.num_samples * num_replicas

        self.indices = list(range(len(dataset)))  # type: ignore[arg-type]
        if drop_last:
            # remove tail of data to make it evenly divisible.
            self.indices = self.indices[:total_size]
        else:
            # add extra samples to make it evenly divisible.
            padding_size = total_size - len(self.indices)
            # deterministically shuffle based on seed.
            g = torch.Generator()
            g.manual_seed(seed)
            if partition:
                assert groups is not None
                last = max(groups)
                base = len(list(filter(lambda rank: rank < last, groups))) * self.num_samples
                perm = torch.randperm(len(self.indices[base:]), generator=g).add(base).tolist()
                if len(self.indices[base:]) < padding_size:
                    self.indices += self.indices[base:] * (padding_size // len(self.indices[base:])) + perm[:padding_size % len(self.indices[base:])]
                else:
                    self.indices += perm[:padding_size]
            else:
                perm = torch.randperm(len(self.indices), generator=g).tolist()
                if len(self.indices) < padding_size:
                    self.indices += self.indices * (padding_size // len(self.indices)) + perm[:padding_size % len(self.indices)]
                else:
                    self.indices += perm[:padding_size]
        assert len(self.indices) == total_size

        channel = grpc.insecure_channel("{}:{}".format(master_addr, master_port))
        # block until the scheduler server is initialized.
        grpc.channel_ready_future(channel).result()
        self.stub = SchedulerStub(channel)

        if self.rank == 0:
            sizes = list(map(lambda index: sys.getsizeof(dataset, index), self.indices))
            self.stub.Init(Arguments(num_replicas, batch_size, sizes, groups, partition, self.schedule))

    def __iter__(self) -> Iterator[T_co]:
        return self

    def __next__(self) -> T_co:
        pass

    def __len__(self) -> int:
        return self.num_samples

    def __del__(self) -> None:
        if self.rank == 0:
            self.stub.Finalize()

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. This ensures scheduling work properly across multiple epochs.

        Args:
            epoch (int): Epoch number.
        """
        if self.rank == 0 and 0 < epoch:
            self.stub.Reset()
