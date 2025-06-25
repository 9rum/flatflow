# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0/nemo/collections/nlp/data/language_modeling/megatron/megatron_batch_samplers.py
# Copyright (c) 2024, The FlatFlow Authors.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import grpc
import torch.distributed
import torch.fx
from megatron.core import parallel_state
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import BaseMegatronBatchSampler

from flatflow import sys
from flatflow.rpc import ControlPlaneClient, run  # type: ignore[attr-defined]
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
        graph (torch.fx.Graph): The exported computational graph.
        pad_samples_to_global_batch_size (bool, optional): If ``True``, then the sampler will pad (default: ``False``).

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
        graph: torch.fx.Graph,
        pad_samples_to_global_batch_size=False,
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
        self.world_size = torch.distributed.get_world_size()
        self.num_data_parallel_group = self.world_size // (
            self.tensor_parallel_world_size * self.pipeline_parallel_world_size
        )
        self.schedule = []
        self.schedule_size = [0]

        if self.pipeline_parallel_rank == 0 and self.tensor_parallel_rank == 0:
            # Launch a control plane for this worker. Network communication is avoided
            # so port conflicts are handled internally by the extension.
            port = run()

            # The control plane runs locally on every data parallel worker and
            # communicates through the IPv6 loopback interface only.
            channel = grpc.insecure_channel(f"[::1]:{port}")
            self.client = ControlPlaneClient(channel)
            sizes = [sys.getsizeof(self.dataset, index) for index in range(len(self.dataset))]
            self.client.Init(
                data_parallel_size,
                global_batch_size,
                micro_batch_size,
                self.data_parallel_rank,
                graph,
                sizes,
            )

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler. This ensures all replicas use a different random ordering for each epoch.
        Otherwise, the next iteration of this sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def __iter__(self):
        model_parallel_group = parallel_state.get_model_parallel_group()
        model_parallel_src_rank = torch.distributed.get_process_group_ranks(model_parallel_group)[0]
        is_model_parallel_src = (self.global_rank == model_parallel_src_rank)

        # receive the reordered computation schedule from the control plane
        if is_model_parallel_src:
            self.schedule = self.client.Scatter(self.epoch, list(range(len(self.dataset))))
            self.schedule_size = [len(self.schedule)]
            self.epoch += 1
        else:
            self.schedule_size = [0]

        torch.distributed.broadcast_object_list(self.schedule_size, src=model_parallel_src_rank, group=model_parallel_group)
        if not is_model_parallel_src:
            self.schedule = [0] * self.schedule_size[0]
        torch.distributed.broadcast_object_list(self.schedule, src=model_parallel_src_rank, group=model_parallel_group)

        micro_batch = []
        for idx_from_schedule in self.schedule:
            micro_batch.append(idx_from_schedule)
            if len(micro_batch) == self._global_batch_size_on_this_data_parallel_rank:
                self.consumed_samples += len(micro_batch)
                yield micro_batch
                micro_batch = []

        if 0 < len(micro_batch) and not self.drop_last:
            if self.pad_samples_to_global_batch_size:
                num_pad = self._global_batch_size_on_this_data_parallel_rank - len(micro_batch)
                micro_batch = micro_batch + [-1] * num_pad
            yield micro_batch
            self.consumed_samples += len(micro_batch)
        self.consumed_samples = 0

    def __del__(self) -> None:
        if hasattr(self, "client"):
            self.client.Finalize()
