# Adapted from https://github.com/pytorch/pytorch/blob/v2.4.0/torch/utils/data/distributed.py
# Copyright 2024 The FlatFlow Authors.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
from typing import Optional, TypeVar

import grpc
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_sampler import BaseMegatronBatchSampler

import flatflow
from flatflow.rpc import CommunicatorClient, run
from flatflow.torch.utils.data.dataset import Dataset

T_co = TypeVar('T_co', covariant=True)

class DistributedSampler(BaseMegatronBatchSampler):
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
      port (int, optional): Port on the master node (rank 0) to be used for initializing
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
  def __init__(self,
        dataset: Dataset,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        global_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool,
        pad_samples_to_global_batch_size=False,
        order: Optional[int] = 2,
        use_flat_shuffle: bool = True,
        seed: Optional[int] = 0,
        heterogeneous: bool = False,
        hidden_size: Optional[int] = False,
        port: int = 50051,
        ) -> None:

    super().__init__(
        total_samples=total_samples,
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
        drop_last=drop_last,
        global_batch_size=global_batch_size,
        pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
    )
    self.order = order
    self.use_flat_shuffle = use_flat_shuffle
    self.dataset = dataset
    self.rank = data_parallel_rank
    self.epoch = 0
    self.indices = []
    self.last_batch_size = self.total_samples % self._global_batch_size
    addr = os.getenv("MASTER_ADDR")
    channel = grpc.insecure_channel(f"{addr}:{port}")

    if self.rank == 0:
        sizes = [flatflow.sys.getsizeof(item) for item in self.dataset]
        run(port, data_parallel_size)

    self.client = CommunicatorClient(channel)
    if self.rank == 0:
        self.client.Init(global_batch_size, micro_batch_size, order, self.rank, seed, heterogeneous, use_flat_shuffle, hidden_size, sizes) #noqa E501
    else:
        self.client.Init(global_batch_size, micro_batch_size, order, self.rank, seed, heterogeneous, use_flat_shuffle, hidden_size) #noqa E501

  def __iter__(self):
    broadcast = self.client.Broadcast(epoch=self.epoch)
    self.indices = list(broadcast.IndicesAsNumpy())
    batch = []
    for idx in range(len(self.indices)):
        batch.append(self.indices[idx])
        if len(batch) == self._global_batch_size_on_this_data_parallel_rank:
            self.consumed_samples += self._global_batch_size
            yield batch
            batch = []
    if len(batch) > 0 and not self.drop_last:
        yield batch


  def __len__(self) -> int:
    """Length of Random Batch Sampler.

    ..note::
        When `rampup_batch_size` is enabled, the return value can be not exactly precise.

    """
    active_total_samples = self.total_samples - (self.last_batch_size if self.drop_last else 0)
    num_available_samples = active_total_samples - self.consumed_samples % active_total_samples
    if self.drop_last:
        return num_available_samples // self.global_batch_size
    else:
        return (num_available_samples + self.global_batch_size - 1) // self.global_batch_size

  def set_epoch(self, epoch: int) -> None:
    r"""Sets the epoch for this sampler. This ensures all replicas use a different random ordering for each epoch.
    Otherwise, the next iteration of this sampler will yield the same ordering.

    Args:
        epoch (int): Epoch number.
    """
    self.epoch = epoch

  def __del__(self) -> None:
    if self.rank == 0:
      self.client.Finalize()
