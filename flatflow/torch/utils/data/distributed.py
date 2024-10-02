# Adapted from https://github.com/pytorch/pytorch/blob/v2.4.0/torch/utils/data/distributed.py
# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0rc0/nemo/collections/nlp/data/language_modeling/megatron/megatron_batch_samplers.py
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
import warnings
from typing import Optional, TypeVar

import grpc
from megatron.core import parallel_state

import flatflow
from flatflow.rpc import CommunicatorClient, run
from flatflow.torch.utils.data.dataset import Dataset

T_co = TypeVar('T_co', covariant=True)

__all__ = ["DistributedSampler"]

class DistributedSampler:
  r"""Sampler that restricts data loading to a subset of the dataset.
    This Sampler is adopted from nemo's megatron_batch_samplers.
    Planning to apply Torch based support later if necessary.

  .. note::
      Dataset is assumed to be of constant size and that any instance of it always
      returns the same elements in the same order.

  Args:
      dataset (Dataset): Dataset used for sampling.
      total_samples (int): Total number of samples in the dataset.
      consumed_samples (int): Number of samples consumed by the model.
      micro_batch_size (int): Micro batch size integer value.
      global_batch_size (int): Global batch size integer value.
         Calculated as data_parallel_size * per_replica_batch_size.
      data_parallel_rank (int): Data parallel rank integer value.
      data_parallel_size (int): Data parallel size integer value.
      drop_last (bool, optional): If ``True``, then the sampler will drop the
          tail of the data to make it evenly divisible across the number of
          replicas. If ``False``, the sampler will add extra indices to make
          the data evenly divisible across the replicas. (default: ``False``)
      order (int, optional): Indicates model complexity. (CNN: 1, transformer: 2)
      use_flat_shuffle (bool, optional): If ``True``, then the sampler will shuffle in inter-range.
          This will enable faster training.
      seed (int, optional): Random seed used to shuffle the sampler.
          This number should be identical across all processes in the distributed group. (default: ``0``)
      heterogeneous (str, optional): Indicates whether cluster is heterogeneous or not.
          Currently, only False is supported.
      hidden_size (bool, optional): Indicates the hidden dimension size of model.
          This is given only when order is 2.
      port (int, optional): Port on the master node (rank 0) to be used for initializing
          the communicator server. (default: ``50051``)
  .. warning::
      In distributed mode, calling the :meth:`set_epoch` method at
      the beginning of each epoch **before** creating the :class:`DataLoader` iterator
      is necessary to make scheduling work properly across multiple epochs.
  """
  _global_batch_size: int
  _num_micro_batches: int
  _global_batch_size_on_this_data_parallel_rank: int

  def __init__(self,
        dataset: Dataset,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        global_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool,
        order: Optional[int] = 2,
        use_flat_shuffle: bool = True,
        seed: Optional[int] = 0,
        heterogeneous: bool = False,
        hidden_size: Optional[int] = False,
        port: int = 50051,
        ) -> None:
    """Constructor of Megatron-LM style Batch Sampler."""
    if total_samples <= 0:
        raise RuntimeError("no sample to consume: {}".format(total_samples))
    if micro_batch_size <= 0:
        raise RuntimeError(f"micro_batch_size size must be greater than 0, but {micro_batch_size}")
    if data_parallel_size <= 0:
        raise RuntimeError(f"data parallel size must be greater than 0, but {data_parallel_size}")
    if data_parallel_rank >= data_parallel_size:
        raise RuntimeError(
            "data_parallel_rank should be smaller than data size, but {} >= {}".format(
                data_parallel_rank, data_parallel_size
            )
        )

    self.order = order
    self.use_flat_shuffle = use_flat_shuffle
    self.total_samples: int = total_samples
    self.consumed_samples: int = consumed_samples
    self.micro_batch_size: int = micro_batch_size
    self.data_parallel_rank: int = data_parallel_rank
    self.data_parallel_size: int = data_parallel_size
    self.drop_last: bool = drop_last
    self.tensor_parallel_world_size: int = parallel_state.get_tensor_model_parallel_world_size()
    self.tensor_parallel_rank: int = parallel_state.get_tensor_model_parallel_rank()
    self.pipeline_parallel_rank: int = parallel_state.get_pipeline_model_parallel_rank()
    self.pipeline_parallel_world_size: int = parallel_state.get_pipeline_model_parallel_world_size()
    self.micro_batch_times_data_parallel_size: int = self.micro_batch_size * self.data_parallel_size
    self.dataset = dataset

    self.update_global_batch_size(global_batch_size)

    self.global_rank = self.tensor_parallel_world_size * self.pipeline_parallel_world_size * self.data_parallel_rank \
                + self.pipeline_parallel_rank * self.tensor_parallel_world_size \
                + self.tensor_parallel_rank
    self.rank = data_parallel_rank
    self.epoch = 0
    self.indices = []
    self.last_batch_size = self.total_samples % self._global_batch_size
    addr = os.getenv("MASTER_ADDR")
    channel = grpc.insecure_channel(f"{addr}:{port}")
    if self.global_rank==0:
        run(port, data_parallel_size)
    
    self.client = CommunicatorClient(channel)

    if self.global_rank == 0:
        sizes = [flatflow.sys.getsizeof(item) for item in self.dataset]
        self.client.Init(global_batch_size, micro_batch_size, order, self.rank, seed, heterogeneous, use_flat_shuffle, hidden_size, sizes) #noqa E501
    elif self.global_rank % (self.tensor_parallel_world_size * self.pipeline_parallel_world_size) == 0:
        self.client.Init(global_batch_size, micro_batch_size, order, self.rank, seed, heterogeneous, use_flat_shuffle, hidden_size) #noqa E501
    
  def update_global_batch_size(self, new_global_batch_size: int) -> None:
        """Update the global batch size."""
        self._global_batch_size = new_global_batch_size
        if self._global_batch_size % self.micro_batch_times_data_parallel_size != 0:
            raise RuntimeError(
                f"`global_batch_size` ({self._global_batch_size}) is not divisible by "
                f"`micro_batch_size ({self.micro_batch_size}) x data_parallel_size "
                f"({self.data_parallel_size})`"
            )
        self._num_micro_batches = self._global_batch_size // self.micro_batch_times_data_parallel_size
        self._global_batch_size_on_this_data_parallel_rank = self._num_micro_batches * self.micro_batch_size

  @property
  def global_batch_size(self) -> int:
        return self._global_batch_size

  @global_batch_size.setter
  def global_batch_size(self, new_global_batch_size: int) -> None:
    warnings.warn("`self.update_global_batch_size(new_global_batch_size)` is recommended.")
    self.update_global_batch_size(new_global_batch_size=new_global_batch_size)

  def __iter__(self):
    if(self.global_rank % (self.tensor_parallel_world_size * self.pipeline_parallel_world_size) == 0):
        broadcast = self.client.Broadcast(epoch=self.epoch)
        self.indices = list(broadcast.IndicesAsNumpy())
    batch = []
    for idx in range(self.consumed_samples, self.total_samples):
        batch.append(idx)
        if len(batch) == self._global_batch_size:
            indices = [
                batch[i] for i in range(self.data_parallel_rank, self._global_batch_size, self.data_parallel_size,)
            ]
            assert len(indices) == self._global_batch_size_on_this_data_parallel_rank
            yield indices
            batch = []

    if len(batch) > 0 and not self.drop_last:
        indices = [batch[i] for i in range(self.data_parallel_rank, len(batch), self.data_parallel_size)]
        yield indices


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
