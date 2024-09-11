# Adapted from https://github.com/pytorch/pytorch/blob/v2.4.0/torch/utils/data/distributed.py
import math
import os
from typing import Iterator, Optional, TypeVar

import grpc
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from collections.abc import Sequence

import flatflow
from flatflow.rpc import CommunicatorClient, run
from flatflow.torch.utils.data.dataset import Dataset

__all__ = ["DistributedSampler"]

T_co = TypeVar('T_co', covariant=True)

class DistributedSampler(Sampler[T_co]):
  r"""Sampler that restricts data loading to a subset of the dataset.

  It is especially useful in conjunction with
  :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
  process can pass a :class:`~flatflow.torch.utils.data.DistributedSampler` instance
  as a :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
  original dataset that is exclusive to it.

  .. note::
      Dataset is assumed to be of constant size and that any instance of it always
      returns the same elements in the same order.

  Args:
      dataset (Dataset): Dataset used for sampling.
      global_batch_size (int): Global batch size integer value.
         Calculated as data_parallel_size * per_replica_batch_size.
      micro_batch_size (int): Micro batch size integer value.
      num_replicas (int, optional): Number of processes participating in
          distributed training. By default, :attr:`world_size` is retrieved from the
          current distributed group.
      rank (int, optional): Rank of the current process within :attr:`num_replicas`.
          By default, :attr:`rank` is retrieved from the current distributed
          group.
      shuffle (bool, optional): Not used but for PyTorch compatibility.
      seed (int, optional): Random seed used to shuffle the sampler.
          This number should be identical across all processes in the distributed group. (default: ``0``)
      drop_last (bool, optional): If ``True``, then the sampler will drop the
          tail of the data to make it evenly divisible across the number of
          replicas. If ``False``, the sampler will add extra indices to make
          the data evenly divisible across the replicas. (default: ``False``)
      order (int, optional): Indicates model complexity. (CNN: 1, transformer: 2)
      use_flat_shuffle (bool, optional): If ``True``, then the sampler will shuffle in inter-range.
          This will enable faster training.
      master_addr (str, optional): Address of the master node (rank 0).
          If rendezvous protocol is enabled using ``torchrun``, the sampler automatically gets the address
          from the environment variable.
      master_port (int, optional): Port on the master node (rank 0) to be used for initializing
          the communicator server. (default: ``50051``)
      heterogeneous (str, optional): Indicates whether cluster is heterogeneous or not.
          Currently, only False is supported.
      hidden_size (bool, optional): Indicates the hidden dimension size of model.
          This is given only when order is 2.
          
  .. warning::
      In distributed mode, calling the :meth:`set_epoch` method at
      the beginning of each epoch **before** creating the :class:`DataLoader` iterator
      is necessary to make scheduling work properly across multiple epochs.
  """
  def __init__(self, dataset: Dataset, global_batch_size: int, micro_batch_size: int, 
               num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True, 
               seed: int = 0, drop_last: bool = False, order: Optional[int] = 2, use_flat_shuffle: bool = True,
               master_addr: Optional[str] = None, master_port: int = 50051, heterogeneous: bool = False,
               hidden_size: Optional[int] = False) -> None:
    
    if len(dataset) <= 0:
      raise ValueError(f"Dataset must be greater than 0, but {len(dataset)}")
    if micro_batch_size <= 0:
        raise RuntimeError(f"micro_batch_size size must be greater than 0, but {micro_batch_size}")
    if data_parallel_size <= 0:
        raise RuntimeError(f"data parallel size must be greater than 0, but {data_parallel_size}")
    if num_replicas is None:
      if not dist.is_available():
        raise RuntimeError("Requires distributed package to be available")
      num_replicas = dist.get_world_size()
    if rank is None:
      if not dist.is_available():
        raise RuntimeError("Requires distributed package to be available")
      rank = dist.get_rank()
    if master_addr is None:
      master_addr = os.getenv("MASTER_ADDR")
      if master_addr is None:
        raise ValueError("Invalid master address {}, either master address or MASTER_ADDR should be given".format(master_addr))
    if order == 2:
      if hidden_size == None:
        raise ValueError("hidden_size should be assigned")

    self.num_replicas = num_replicas
    self.dataset = dataset
    self.rank = rank
    self.map = list(range(len(self.dataset)))  # type: ignore[arg-type]
    self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
    
    if self.rank == 0:
      data_parallel_size = torch.distributed.get_world_size()
      run(master_port, data_parallel_size)
    
    addr = os.getenv("MASTER_ADDR", master_addr)
    channel = grpc.insecure_channel(f"{addr}:{master_port}")
    self.stub = CommunicatorClient(channel)

    if self.rank == 0:
      sizes = list(map(lambda index: flatflow.sys.getsizeof(self.dataset, index), list(range(len(self.dataset)))))
      self.stub.Init(global_batch_size, micro_batch_size, order, rank, seed, heterogeneous, use_flat_shuffle, hidden_size, sizes)
    else:
      self.stub.Init(global_batch_size, micro_batch_size, order, rank, seed, heterogeneous, use_flat_shuffle, hidden_size)
    

  def __iter__(self) -> Iterator[T_co]:
    broadcast = self.stub.Broadcast(epoch=self.epoch)
    self.indices = broadcast.IndicesAsNumpy()  # type: ignore[arg-type]
    converged = broadcast.converged()
    assert converged, "Converge option is not supported yet : internal Failed"
    self._num_yielded = 0
    return self
  
  def __next__(self) -> T_co:
    if self.num_samples <= self._num_yielded:
      raise StopIteration
    index = self.map[self.indices[self._num_yielded]]

    self._num_yielded += 1
    return index
  
  def __len__(self) -> int:
    num_available_samples: int = self.num_samples - self._num_yielded
    return (num_available_samples + self.global_batch_size - 1) // self.global_batch_size

  
  def __del__(self) -> None:
    if self.rank == 0:
      self.client.Finalize()

  def set_epoch(self, epoch: int) -> None:
    r"""Sets the epoch for this sampler. This ensures all replicas use a different random ordering for each epoch.
    Otherwise, the next iteration of this sampler will yield the same ordering.

    Args:
        epoch (int): Epoch number.
    """
    self.epoch = epoch