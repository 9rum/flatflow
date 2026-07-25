# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0/nemo/collections/nlp/data/language_modeling/megatron/data_samplers.py
# Copyright (c) 2025, The FlatFlow Authors.
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

import operator

import numpy as np
import torch.fx
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingSampler as BaseMegatronPretrainingSampler

from flatflow.ffi import sched, sched_unstable
from flatflow.nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from flatflow.nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import GPTDataset
from flatflow.ops import serialize

__all__ = ["MegatronCorePretrainingSampler", "MegatronPretrainingSampler"]


class MegatronPretrainingSampler(BaseMegatronPretrainingSampler):
    """Megatron-LM style pre-training sampler.

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
    """

    def __init__(
        self,
        dataset: BlendableDataset | GPTDataset,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        global_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        tensor_parallel_size: int,
        context_parallel_size: int,
        drop_last: bool,
        graph: torch.fx.Graph,
        unstable: bool = True,
        policy: str = "joint",
        pad_samples_to_global_batch_size: bool = False,
        *args,
        **kwargs,
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
            *args,
            **kwargs,
        )
        self.dataset = dataset
        self.tensor_parallel_size = tensor_parallel_size
        self.context_parallel_size = context_parallel_size
        self.unstable = unstable
        self.policy = policy
        self.epoch = 0

        if drop_last:
            self.total_size = len(dataset) // global_batch_size * global_batch_size
            sizes = dataset._sizes[: self.total_size]
        else:
            assert pad_samples_to_global_batch_size
            self.total_size = ((len(dataset) - 1) // global_batch_size + 1) * global_batch_size
            sizes = np.append(dataset._sizes, np.repeat(dataset._sizes[-1], self.total_size - len(dataset)))  # noqa: E501

        self.sizes = np.ascontiguousarray(sizes, dtype=np.int64)
        self.buf = serialize(graph)

        del dataset._sizes

    def __iter__(self):
        self.epoch += 1

        indices = np.ascontiguousarray(np.arange(self.total_size, dtype=np.uintp))
        if self.unstable:
            batches = sched_unstable(
                indices,
                self.sizes,
                self.buf,
                self.tensor_parallel_size,
                self.context_parallel_size,
                self.data_parallel_size,
                self.data_parallel_rank,
                self.global_batch_size,
                self.micro_batch_size,
                self.policy,
            )
        else:
            batches = sched(
                indices,
                self.sizes,
                self.buf,
                self.tensor_parallel_size,
                self.context_parallel_size,
                self.data_parallel_size,
                self.data_parallel_rank,
                self.global_batch_size,
                self.micro_batch_size,
                self.policy,
            )
        batches = map(operator.index, batches)

        batch = []
        for idx in batches:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            assert not self.pad_samples_to_global_batch_size
            yield batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class MegatronCorePretrainingSampler(MegatronPretrainingSampler):
    def _get_padding_indices(self, pad_samples_num: int):
        return [None] * pad_samples_num
