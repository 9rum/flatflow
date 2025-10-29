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

import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import grpc
import torch.distributed
import torch.fx
from megatron.core import parallel_state
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import BaseMegatronBatchSampler

from flatflow import sys
from flatflow.rpc import ControlPlaneClient, run  # type: ignore[attr-defined]
from flatflow.torch.utils.data import Dataset

__all__ = ["MegatronPretrainingBatchSampler"]


class MegatronPretrainingBatchSampler(BaseMegatronBatchSampler):
    """Megatron-LM style pre-training batch sampler.

    Args:
        dataset (Dataset): Dataset used for sampling.
        total_samples (int): Total number of samples in the dataset.
        consumed_samples (int): Number of samples consumed by the model.
        micro_batch_size (int): Micro batch size integer value.
        global_batch_size (int): Global batch size integer value.
          Calculated as data_parallel_size * per_replica_batch_size.
        data_parallel_rank (int): Data parallel rank integer value.
        data_parallel_size (int): Data parallel size integer value.
        drop_last (bool): If ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. (default: ``False``)
        graph (torch.fx.Graph): The exported computational graph.
        pad_samples_to_global_batch_size (bool, optional): If ``True``, then the sampler will pad (default: ``False``).
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
        self.dataset = dataset
        self.epoch = 0

        if drop_last:
            self.total_size = len(dataset) // global_batch_size * global_batch_size
        else:
            assert pad_samples_to_global_batch_size
            self.total_size = ((len(dataset) - 1) // global_batch_size + 1) * global_batch_size
        num_samples = self.total_size // data_parallel_size
        self.indices = [0] * num_samples

        pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
        tensor_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
        if pipeline_parallel_rank == 0 and tensor_parallel_rank == 0:
            # Launch a control plane for this worker. Network communication is avoided
            # so port conflicts are handled internally by the extension.
            port = run()

            # The control plane runs locally on every data parallel worker and
            # communicates through the IPv6 loopback interface only.
            channel = grpc.insecure_channel(f"[::1]:{port}")
            self.client = ControlPlaneClient(channel)

            func = partial(sys.getsizeof, dataset)
            max_workers = len(os.sched_getaffinity(os.getpid()))
            with ProcessPoolExecutor(max_workers) as executor:
                sizes = list(executor.map(func, range(len(dataset))))

            if drop_last:
                sizes = sizes[: self.total_size]
            else:
                sizes.extend([sizes[-1]] * (self.total_size - len(dataset)))

            self.client.Init(
                data_parallel_rank,
                data_parallel_size,
                global_batch_size,
                micro_batch_size,
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
        rank = torch.distributed.get_rank()
        is_model_parallel_src = rank == model_parallel_src_rank

        # Receive the reordered computation schedule from the control plane.
        if is_model_parallel_src:
            self.indices = self.client.Scatter(self.epoch, list(range(self.total_size)))
            self.epoch += 1
        torch.distributed.broadcast_object_list(self.indices, model_parallel_src_rank, model_parallel_group)

        batch = []
        for idx in self.indices:  # type: ignore[attr-defined]
            batch.append(idx if idx < len(self.dataset) else -1)
            if len(batch) == self._global_batch_size_on_this_data_parallel_rank:
                yield batch
                batch = []

    def __del__(self) -> None:
        if hasattr(self, "client"):
            self.client.Finalize()
