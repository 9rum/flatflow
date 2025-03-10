# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0/nemo/collections/nlp/data/language_modeling/megatron/megatron_batch_samplers.py
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

import os
from typing import Optional
import math

import grpc
import torch.distributed
from megatron.core import parallel_state
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import BaseMegatronBatchSampler

from flatflow import sys
from flatflow.rpc import ControlPlaneClient, run
from flatflow.torch.utils.data.dataset import Dataset

__all__ = ["MegatronPretrainingBatchSampler"]


class MegatronPretrainingBatchSampler(BaseMegatronBatchSampler):
    """Sampler that restricts data loading to a subset of the dataset.
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
        seed (int, optional): Random seed used to shuffle the sampler.
            This number should be identical across all processes in the distributed group. (default: ``0``)
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the indices.
        port (int, optional): Port on the master node (rank 0) to be used for initializing
            the communicator server. (default: ``50051``)
    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make scheduling work properly across multiple epochs.
    """

    def __init__(
        self,
        dataset: Dataset,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        global_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool,
        graph: torch.fx.graph,
        pad_samples_to_global_batch_size=False,
        seed: Optional[int] = 0,
        shuffle: bool = True,
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
            pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
        )
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
        self.global_rank = torch.distributed.get_rank()
        self.epoch = 0
        self.indices = []
        self.last_batch_size = self.total_samples % self._global_batch_size
        self.world_size = torch.distributed.get_world_size()
        self.num_data_parallel_group = self.world_size // (
            self.tensor_parallel_world_size * self.pipeline_parallel_world_size
        )
        self.seed = seed
        self.shuffle = shuffle
        self.total_size = total_samples 
        sizes = [sys.getsizeof(self.dataset, index) for index in range(len(self.dataset))]
        self.total_length = len(sizes)

        addr = os.getenv("MASTER_ADDR")
        channel = grpc.insecure_channel(f"{addr}:{port}")

        if self.data_parallel_rank == 0:
            run(port, data_parallel_size)

        self.client = ControlPlaneClient(self.global_rank, channel)
        if self.pipeline_parallel_rank == 0 and self.tensor_parallel_rank == 0:
            self.client.Init(
                global_batch_size,
                micro_batch_size,
                graph,
                sizes if self.global_rank == 0 else None,
            )

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler. This ensures all replicas use a different random ordering for each epoch.
        Otherwise, the next iteration of this sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def __iter__(self):
        indices = []
        model_parallel_group = parallel_state.get_model_parallel_group() 
        model_parallel_src_rank = torch.distributed.get_process_group_ranks(model_parallel_group)[0]
        is_model_parallel_src = (self.global_rank == model_parallel_src_rank)

        while True:
            if self.consumed_samples > self.total_length // self.num_data_parallel_group:
                break
            schedule_size = [0]
            if self.shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))

            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            else:
                # remove tail of data to make it evenly divisible.
                indices = indices[: self.total_size]
            assert len(indices) == self.total_size

            # receive the reordered computation schedule from the control plane
            if is_model_parallel_src:
                schedule = self.client.Broadcast(self.epoch, indices)
                schedule_size = [len(schedule)]

            torch.distributed.broadcast_object_list(schedule_size, src=model_parallel_src_rank, group=model_parallel_group)
            if not is_model_parallel_src:
                schedule = [0] * schedule_size[0]
            torch.distributed.broadcast_object_list(schedule, src=model_parallel_src_rank, group=model_parallel_group)

            self.consumed_samples += schedule_size[0]

            batch = []
            for idx in range(schedule_size[0]):
                batch.append(schedule[idx])
                if len(batch) == self._global_batch_size_on_this_data_parallel_rank:
                    yield batch
                    batch = []

            if len(batch) > 0 and not self.drop_last and self.pad_samples_to_global_batch_size:
                num_pad = self._global_batch_size_on_this_data_parallel_rank - len(batch)
                batch = batch + [-1] * num_pad
                yield batch

    def __del__(self) -> None:
        if hasattr(self.client, "rank") and self.client.rank == 0:
            self.client.Finalize()
