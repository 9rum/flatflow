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

import nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset

from flatflow import sys
from flatflow.torch.utils.data import Dataset

__all__ = ["BlendableDataset"]


class BlendableDataset(
    Dataset, nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset.BlendableDataset
):
    def __sizeof__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return sys.getsizeof(self.datasets[dataset_idx], sample_idx)
