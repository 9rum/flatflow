# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0/nemo/collections/nlp/data/language_modeling/megatron/blendable_dataset.py
# Copyright (c) 2024, The FlatFlow Authors.
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import bisect
from collections.abc import Iterable

import numpy

from flatflow.torch.utils.data import Dataset
from flatflow.nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import GPTDataset

__all__ = ["BlendableDataset"]


class BlendableDataset(Dataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            s += len(e)
            r.append(s)
        return r

    def __init__(self, datasets: Iterable[GPTDataset]):
        self.datasets = list(datasets)
        assert 0 < len(self.datasets), "datasets should not be an empty iterable"
        self.cumulative_sizes = self.cumsum(self.datasets)
        self._sizes = numpy.concatenate([dataset._sizes for dataset in self.datasets])

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if len(self) + idx < 0:
                raise ValueError("absolute value of index should not exceed dataset length")
            idx += len(self)
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def __sizeof__(self, idx):
        return self._sizes[idx]

    def create_data_mmap(self):
        for dataset in self.datasets:
            dataset.create_data_mmap()
