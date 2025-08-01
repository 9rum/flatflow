# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Dataloaders."""

import abc
from itertools import chain
from typing import Optional

import torch
import nemo.collections.nlp.data.language_modeling.megatron.data_samplers

from nemo.utils import logging
import grpc
import torch.distributed
from megatron.core import parallel_state
from flatflow import sys
from flatflow.rpc import ControlPlaneClient, run
from flatflow.torch.utils.data import Dataset

class _FlatflowMixin:
    """Mixin that injects RPC + scheduling functionality."""
    def _init_rpc(
        self,
        dataset: Dataset,
        graph: torch.fx.Graph,
    ):
        self.dataset = dataset
        self.epoch: int = 0

        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        is_src = pp_rank == 0 and tp_rank == 0

        if is_src:
            port = run()
            ch   = grpc.insecure_channel(f"[::1]:{port}")
            self._client = ControlPlaneClient(ch)

            sizes = [
                sys.getsizeof(dataset, i if i < len(dataset) else len(dataset) - 1)
                for i in range(self.total_samples)
            ]
            self._client.Init(
                self.data_parallel_rank,
                self.data_parallel_size,
                self.global_batch_size,
                self.micro_batch_size,
                graph,
                sizes,
            )

    def _broadcast_schedule(self):
        mp_group = parallel_state.get_model_parallel_group()
        src_rank = torch.distributed.get_process_group_ranks(mp_group)[0]
        rank = torch.distributed.get_rank()

        if rank == src_rank:
            # indices → 0,1,2,...,total_samples-1
            indices = list(range(self.total_samples))
            self._indices = self._client.Scatter(self.epoch, indices)
            self.epoch += 1
        torch.distributed.broadcast_object_list(self._indices, src_rank, mp_group)


class MegatronPretrainingSampler(
    _FlatflowMixin, nemo.collections.nlp.data.language_modeling.megatron.data_samplers.MegatronPretrainingSampler
):
    def __init__(
        self,
        dataset: Dataset,
        graph: torch.fx.Graph,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_rpc(dataset, graph)

    def __iter__(self):
        self._broadcast_schedule()
        
        batch = []
        for idx in self._indices: # type: ignore[attr-defined]
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start, end = self.get_start_end_idx()
                yield batch[start:end]   # yield per micro_batch_size
                batch = []

        if len(batch) and not self.drop_last:
            start, end = self.get_start_end_idx()
            yield batch[start:end]

    def __del__(self):
        if hasattr(self, "_client"):
            self._client.Finalize()

class MegatronCorePretrainingSampler(MegatronPretrainingSampler):
    """_get_padding_indices is different"""

    def _get_padding_indices(self, pad_samples_num: int):
        return [None] * pad_samples_num
