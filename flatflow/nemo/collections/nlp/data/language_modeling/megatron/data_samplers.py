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

import numpy
import torch.fx
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingSampler as BaseMegatronPretrainingSampler

from flatflow.nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from flatflow.nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import GPTDataset
from flatflow.rpc import ControlPlaneClient, run  # type: ignore[attr-defined]

__all__ = ["MegatronPretrainingSampler", "MegatronCorePretrainingSampler"]


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
        drop_last: bool,
        graph: torch.fx.Graph,
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
        self.epoch = 0

        if drop_last:
            self.total_size = len(dataset) // global_batch_size * global_batch_size
        else:
            assert pad_samples_to_global_batch_size
            self.total_size = ((len(dataset) - 1) // global_batch_size + 1) * global_batch_size

        port = run()
        self.client = ControlPlaneClient(port)

        if drop_last:
            sizes = dataset._sizes[: self.total_size]
        else:
            sizes = numpy.append(dataset._sizes, numpy.repeat(dataset._sizes[-1], self.total_size - len(dataset)))

        self.client.Init(
            data_parallel_rank,
            data_parallel_size,
            global_batch_size,
            micro_batch_size,
            graph,
            sizes,
        )

        del dataset._sizes

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler. This ensures all replicas use a different random ordering for each epoch.
        Otherwise, the next iteration of this sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def __iter__(self):
        indices = self.client.Scatter(self.epoch, numpy.arange(self.total_size, dtype=numpy.uint64))
        self.epoch += 1

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if batch and not self.drop_last:
            assert (
                not self.pad_samples_to_global_batch_size
            ), "with pad_samples_to_global_batch_size all batches should be complete"
            yield batch

    def __del__(self) -> None:
        if hasattr(self, "client"):
            self.client.Finalize()


class MegatronCorePretrainingSampler(MegatronPretrainingSampler):
    def _get_padding_indices(self, pad_samples_num: int):
        return [None] * pad_samples_num
