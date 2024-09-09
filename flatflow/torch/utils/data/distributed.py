# Adapted from https://github.com/pytorch/pytorch/blob/v2.4.0/torch/utils/data/distributed.py
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
  def __init__(self, dataset: Dataset, global_batch_size: int, micro_batch_size: int,
               rank: Optional[int] = None, use_flat_shuffle: bool = True,
               seed: int = 0, master_addr: Optional[str] = None,
               master_port: int = 50051, heterogeneous: bool = None,
               hidden_size: int = False, sizes: Optional[Sequence[int]] = None) -> None:
    addr = os.getenv("MASTER_ADDR", master_addr)
    channel = grpc.insecure_channel(f"{addr}:{master_port}")
    self.stub = CommunicatorClient(channel)
    self.stub.Init(global_batch_size, micro_batch_size, rank, seed, heterogeneous, use_flat_shuffle, hidden_size, sizes)
    
    if rank == 0:
      data_parallel_size = torch.distributed.get_world_size()
      run(master_port, data_parallel_size)

  def __iter__(self) -> Iterator[T_co]:
      broadcast = self.stub.Broadcast(epoch=self.epoch)  # type: ignore[arg-type]
      self.indices = [broadcast.Indices(i) for i in range(broadcast.IndicesLength())]

      self._num_yielded = 0
      return self
  
  def __next__(self) -> T_co:
    if self.num_samples <= self._num_yielded:
        raise StopIteration
    index = self.map[self.indices[self._num_yielded]] #todo

    self._num_yielded += 1
    return index

  def __del__(self) -> None:
    if self.rank == 0:
        self.client.Finalize()


if __name__ == "__main__":
    dataset = torch.utils.data.TensorDataset(torch.rand(100, 2), torch.rand(100, 1))
    sampler = DistributedSampler(dataset, global_batch_size=32, micro_batch_size=16, rank=0)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None)
    for data in dataloader:
        print(data)
        break
    del sampler
    del dataloader
    del dataset
    print("Done")