# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0/nemo/core/classes/dataset.py
# Copyright (c) 2024, The FlatFlow Authors.
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import nemo.core.classes

import flatflow.torch.utils.data

__all__ = [
    "Dataset",
    "IterableDataset",
]


class Dataset(flatflow.torch.utils.data.Dataset, nemo.core.classes.Dataset):
    """Data set with output ports.

    .. note::
        Subclasses of :class:`~flatflow.nemo.core.classes.Dataset` should *not* implement :attr:`input_types`.
    """

    def _collate_fn(self, batch):
        return flatflow.torch.utils.data.default_collate(batch)


class IterableDataset(flatflow.torch.utils.data.IterableDataset, nemo.core.classes.IterableDataset):
    """Iterable data set with output ports.

    .. note::
        Subclasses of :class:`~flatflow.nemo.core.classes.IterableDataset` should *not* implement :attr:`input_types`.
    """

    def _collate_fn(self, batch):
        return flatflow.torch.utils.data.default_collate(batch)
