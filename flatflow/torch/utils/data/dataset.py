# Adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/data/dataset.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import bisect

import torch

from flatflow.sys import getsizeof

__all__ = [
    "ChainDataset",
    "ConcatDataset",
    "Dataset",
    "IterableDataset",
]


class Dataset(torch.utils.data.Dataset):
    """An abstract class representing a :class:`Dataset`.

    This class extends :class:`torch.utils.data.Dataset` for use in conjunction with
    :class:`flatflow.torch.utils.data.DistributedSampler`.  In addition to the methods
    provided in the PyTorch counterpart, subclasses should overwrite :meth:`__sizeof__`,
    which is expected to return the user-defined size of data sample at position
    :param:`index`.
    """

    def __add__(self, other: "Dataset") -> "ConcatDataset":
        return ConcatDataset([self, other])

    def __sizeof__(self, index: int) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __sizeof__.")


class IterableDataset(Dataset, torch.utils.data.IterableDataset):
    """An iterable dataset."""

    def __add__(self, other: Dataset):
        return ChainDataset([self, other])


class ConcatDataset(Dataset, torch.utils.data.ConcatDataset):
    """Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (Iterable[Dataset]): List of datasets to be concatenated.
    """

    def __sizeof__(self, index: int) -> int:
        if index < 0:
            if len(self) + index < 0:
                raise ValueError(
                    "Absolute value of index should not exceed dataset length."
                )
            index += len(self)
        row = bisect.bisect(self.cumulative_sizes, index)
        if row == 0:
            col = index
        else:
            col = index - self.cumulative_sizes[row - 1]
        return getsizeof(self.datasets[row], col)


class ChainDataset(IterableDataset, torch.utils.data.ChainDataset):
    """Dataset for chaining multiple :class:`IterableDataset`s.

    This class is useful to assemble different existing dataset streams. The chaining
    operation is done on-the-fly, so concatenating large-scale datasets with this class
    will be efficient.

    Args:
        datasets (Iterable[Dataset]): Datasets to be chained together.
    """
