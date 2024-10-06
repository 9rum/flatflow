# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0rc0/nemo/collections/nlp/data/language_modeling/megatron/megatron_batch_samplers.py
# Copyright (c) 2024, The FlatFlow Authors.
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

import abc
import os
import warnings
from typing import Optional

import grpc
import torch
from megatron.core import parallel_state
from nemo.utils.decorators import experimental

import flatflow
from flatflow.rpc import CommunicatorClient, run
from flatflow.torch.utils.data.dataset import Dataset

__all__ = [
    "MegatronPretrainingBatchSampler",
    "MegatronPretrainingRandomBatchSampler",
]


class BaseMegatronBatchSampler:
    """Megatron style BatchSampler.

    Let mbs, gbs, tp, pp, and dp stand for "micro batch size", "global batch size",
    "tensor model parallel world size", "pipeline model parallel world size", and
    "data parallel world size", the number of micro batches (hereafter, nmb) is defined as
    :math:`nmb = gbs \\div (mbs \\times dp)`.

    See `apex/transformer/microbatches.py#L91-L98 <https://github.com/NVIDIA/apex/blob/
    44c3043685b6115e7b81b3458a6c76601b1e55b4/apex/transformer/microbatches.py#L91-L98>`_
    for the initial settings of the number of micro batches and
    `apex/transformer/microbatches.py#L160-L177 <https://github.com/NVIDIA/apex/blob/
    44c3043685b6115e7b81b3458a6c76601b1e55b4/apex/transformer/microbatches.py#L160-L177>_`.
    for warming up of global batch size.

    e.g.) `(mbs, gbs, tp, pp, dp) = (1, 16, 1, 1, 2)`, then the number of micro batches is
    :math:`gbs \\div (mbs \\times dp) = 16 \\div (1 \\times 2) = 8`.
    In this case, an instance of Megatron Batch Sampler on each data parallel rank is expected
    returns :math:`nmb \\times mbs = 8` indices.
    """

    _global_batch_size: int
    _num_micro_batches: int
    _global_batch_size_on_this_data_parallel_rank: int

    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        global_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool,
        pad_samples_to_global_batch_size=False,
    ) -> None:
        """Constructor of Megatron-LM style Batch Sampler.

        Args:
            total_samples: The size of dataset.
            consumed_samples: The number of samples that have been used.
            micro_batch_size: The size of each micro batch.
            global_batch_size: The size of global batch.
            data_parallel_rank: The value you can obtain via
                `parallel_state.get_data_parallel_rank()` of megatron.core.
            data_parallel_size: The value you can obtain via
                `parallel_state.get_data_parallel_world_size()` of megatron.core.
        """
        # Sanity checks.
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
        # Keep a copy of input params for later use.
        self.total_samples: int = total_samples
        self.consumed_samples: int = consumed_samples
        self.micro_batch_size: int = micro_batch_size
        self.data_parallel_rank: int = data_parallel_rank
        self.data_parallel_size: int = data_parallel_size
        self.drop_last: bool = drop_last
        self.pad_samples_to_global_batch_size = pad_samples_to_global_batch_size
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * self.data_parallel_size

        self.update_global_batch_size(global_batch_size)

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

    def __len__(self) -> int:
        """Length of Batch Sampler.

        ..note::
            When `rampup_batch_size` is enabled, the return value can be not exactly precise.

        """
        num_available_samples: int = self.total_samples - self.consumed_samples
        if self.drop_last:
            return num_available_samples // self.global_batch_size
        else:
            return (num_available_samples + self.global_batch_size - 1) // self.global_batch_size

    @abc.abstractmethod
    def __iter__(self):
        ...


class MegatronPretrainingBatchSampler(BaseMegatronBatchSampler):
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
        pad_samples_to_global_batch_size (bool, optional): If ``True``, then the sampler will pad (default: ``False``)
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
    def __init__(self,
          dataset: Dataset,
          total_samples: int,
          consumed_samples: int,
          micro_batch_size: int,
          global_batch_size: int,
          data_parallel_rank: int,
          data_parallel_size: int,
          drop_last: bool,
          pad_samples_to_global_batch_size = False,
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
          global_batch_size=global_batch_size,
          data_parallel_rank=data_parallel_rank,
          data_parallel_size=data_parallel_size,
          drop_last=drop_last,
          pad_samples_to_global_batch_size=pad_samples_to_global_batch_size
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
          self.client.Init(global_batch_size, micro_batch_size, order, self.rank, seed, heterogeneous, use_flat_shuffle, hidden_size, sizes)
      elif self.global_rank % (self.tensor_parallel_world_size * self.pipeline_parallel_world_size) == 0:
          self.client.Init(global_batch_size, micro_batch_size, order, self.rank, seed, heterogeneous, use_flat_shuffle, hidden_size)

    def set_epoch(self, epoch: int) -> None:
      r"""Sets the epoch for this sampler. This ensures all replicas use a different random ordering for each epoch.
      Otherwise, the next iteration of this sampler will yield the same ordering.

      Args:
          epoch (int): Epoch number.
      """
      self.epoch = epoch

    def __iter__(self):
      if(self.global_rank % (self.tensor_parallel_world_size * self.pipeline_parallel_world_size) == 0):
          broadcast = self.client.Broadcast(epoch=self.epoch)
          self.indices = list(broadcast.IndicesAsNumpy())
      batch = []
      # Last batch will be dropped if drop_last is not set False
      for idx in range(self.consumed_samples, self.total_samples):
          batch.append(idx)
          if len(batch) == self._global_batch_size:
              indices = [
                  batch[i] for i in range(self.data_parallel_rank, self._global_batch_size, self.data_parallel_size,)
              ]
              assert len(indices) == self._global_batch_size_on_this_data_parallel_rank
              yield indices
              batch = []

      # Check the last partial batch and see drop_last is set
      if 0 < len(batch) and not self.drop_last:
          indices = [batch[i] for i in range(self.data_parallel_rank, len(batch), self.data_parallel_size)]
          if self.pad_samples_to_global_batch_size:
              num_pad = self._global_batch_size // self.data_parallel_size - len(indices)
              indices = indices + [-1] * num_pad
          yield indices

    def __del__(self) -> None:
      if hasattr(self.client, 'rank') and self.client.rank == 0:
        self.client.Finalize()

@experimental
class MegatronPretrainingRandomBatchSampler(BaseMegatronBatchSampler):

    # NOTE (mkozuki): [[Argument of `dataset` and `data_sharding`]]
    # From the commit below, it seems like `dataset` argument and `data_sharding` argument
    # are necessary for ViT training. However, to keep this simple,
    # I omit those two arguments.
    # commit: https://github.com/NVIDIA/Megatron-LM/commit/7a77abd9b6267dc0020a60b424b4748fc22790bb
    #
    # NOTE (degert): I have re-written this class somewhat to give the length correctly when consumed_samples
    # are larger than total_samples, which happens with 1 < epochs training when using this Sampler
    # I have also added an explicit seed which allows us to remove Dataset-side shuffling in Nemo-Aligner
    #
    # This class does not currently work with pad_samples_to_global_batch_size=True
    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        global_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool,
        pad_samples_to_global_batch_size: bool = False,
        seed: int = 0,
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
        assert (
            not pad_samples_to_global_batch_size
        ), "`MegatronPretrainingRandomBatchSampler` does not support sample padding"
        if (not drop_last) and 1 < self.micro_batch_times_data_parallel_size:
            raise RuntimeError(
                "`MegatronPretrainingRandomBatchSampler` does not support drop_last=False when 1 < micro_batch_size * data_parallel_size. \
                  please reduce your MBS and data parallelism to 1 if you want to use drop_last=False, or switch to drop_last=True to avoid this error"
            )
        self.last_batch_size = self.total_samples % self._global_batch_size
        self.seed = seed

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

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self._global_batch_size_on_this_data_parallel_rank:
                self.consumed_samples += self._global_batch_size
                yield batch
                batch = []
        # Check the last partial batch and see drop_last is set
        if 0 < len(batch) and not self.drop_last:
            yield batch
