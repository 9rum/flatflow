# Adapted from https://github.com/pytorch/pytorch/blob/v2.4.0/torch/utils/data/distributed.py
import math
import os
import shutil
import subprocess
import time
from typing import Iterable, Iterator, Optional, TypeVar

import grpc
import numpy as np
import torch
import torch.distributed as dist
from google.protobuf.empty_pb2 import Empty
from sklearn.linear_model import LinearRegression
from torch.utils.data import Sampler
from collections.abc import Sequence

from flatflow.rpc import CommunicatorClient, run
from flatflow.torch.utils.data.dataset import Dataset

__all__ = ["DistributedSampler"]

T_co = TypeVar('T_co', covariant=True)

class DistributedSampler(Sampler[T_co]):
  def __init__(self, dataset: Dataset, global_batch_size: int, micro_batch_size: int, order: Optional[int] = 2,
               rank: Optional[int] = None, use_flat_shuffle: bool = True,
               seed: int = 0, master_addr: Optional[str] = None,
               master_port: int = 50051, heterogeneous: bool = None,
               hidden_size: int = False, sizes: Optional[Sequence[int]] = None) -> None:
    
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
    self.stub.Init(global_batch_size, micro_batch_size, order, rank, seed, heterogeneous, use_flat_shuffle, hidden_size, sizes)
    

  def __iter__(self) -> Iterator[T_co]:
    broadcast = self.stub.Broadcast(epoch=self.epoch)  # type: ignore[arg-type]
    self.indices = [broadcast.Indices(i) for i in range(broadcast.IndicesLength())]
    self._num_yielded = 0
    return self
  
  def __next__(self) -> T_co:
    if self.num_samples <= self._num_yielded:
      raise StopIteration
    index = self.map[self.indices[self._num_yielded]]

    self._num_yielded += 1
    return index
  
  def __len__(self) -> int:
    """Length of Batch Sampler.

    ..note::
        When `rampup_batch_size` is enabled, the return value can be not exactly precise.

    """
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