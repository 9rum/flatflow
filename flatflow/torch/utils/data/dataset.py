# Adapted from https://github.com/pytorch/pytorch/blob/v2.4.0/torch/utils/data/dataset.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import bisect
from collections.abc import Iterable, Sequence
from typing import TypeVar

import torch
from typing_extensions import deprecated

from flatflow import sys

__all__ = [
    "ChainDataset",
    "ConcatDataset",
    "Dataset",
    "IterableDataset",
]

T_co = TypeVar("T_co", covariant=True)


class Dataset(torch.utils.data.Dataset[T_co]):
    """An abstract class representing a data set.

    This is an extension of :class:`torch.utils.data.Dataset` for use with
    :class:`~flatflow.torch.utils.data.DistributedSampler`.  In addition to the
    methods supported in :class:`torch.utils.data.Dataset`, subclasses could
    also optionally overwrite :meth:`__sizeof__`, which is expected to return
    the user-defined size of the data sample at position :param:`index`.
    """

    def __add__(self, other: torch.utils.data.Dataset[T_co]) -> "ConcatDataset[T_co]":
        return ConcatDataset([self, other])

    def __sizeof__(self, index: int) -> int:
        return 1


class IterableDataset(Dataset[T_co], torch.utils.data.IterableDataset[T_co]):
    """An iterable data set.

    All data sets that represent an iterable of data samples should subclass it.
    Such form of data sets is particularly useful when data come from a stream.

    All subclasses should overwrite :meth:`__iter__`, which would return an
    iterator of samples in this data set.

    When a subclass is used with :class:`~flatflow.torch.utils.data.DataLoader`,
    each item in the data set will be yielded from the data loader iterator.
    When :attr:`0 < num_workers`, each worker process will have a different copy
    of the data set object, so it is often desired to configure each copy
    independently to avoid having duplicate data returned from the workers.
    :func:`torch.utils.data.get_worker_info`, when called in a worker process,
    returns information about the worker. It can be used in either the data set's
    :meth:`__iter__` method or the data loader's :attr:`worker_init_fn` option
    to modify each copy's behavior.
    """

    def __add__(self, other: torch.utils.data.Dataset[T_co]):
        return ChainDataset([self, other])


class ConcatDataset(Dataset[T_co]):
    """Data set as a concatenation of multiple data sets.

    This class is useful to assemble different existing data sets.

    Args:
        datasets (Iterable[Dataset]): List of data sets to be concatenated.
    """

    datasets: Sequence[torch.utils.data.Dataset[T_co]]
    cumulative_sizes: Sequence[int]
    _cumulative_sizes: Sequence[int]

    @staticmethod
    def cumsum(sequence: Sequence[torch.utils.data.Dataset]) -> Sequence[int]:
        r = [0] * len(sequence)
        for i, e in enumerate(sequence):
            r[i] = r[i - 1] + len(e)  # type: ignore[arg-type]
        return r

    @staticmethod
    def _cumsum(sequence: Sequence[torch.utils.data.Dataset]) -> Sequence[int]:
        r = [0] * (len(sequence) + 1)
        for i, e in enumerate(sequence):
            r[i + 1] = r[i] + len(e)  # type: ignore[arg-type]
        return r

    def __init__(self, datasets: Iterable[torch.utils.data.Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert 0 < len(self.datasets), "datasets should not be an empty iterable"
        for d in self.datasets:
            assert not isinstance(d, torch.utils.data.IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)
        self._cumulative_sizes = self._cumsum(self.datasets)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> T_co:
        if idx < 0:
            if len(self) + idx < 0:
                raise ValueError("absolute value of index should not exceed dataset length")
            idx += len(self)
        dataset_idx = bisect.bisect(self.cumulative_sizes, idx)
        sample_idx = idx - self._cumulative_sizes[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]

    def __sizeof__(self, idx: int) -> int:
        if idx < 0:
            if len(self) + idx < 0:
                raise ValueError("absolute value of index should not exceed dataset length")
            idx += len(self)
        dataset_idx = bisect.bisect(self.cumulative_sizes, idx)
        sample_idx = idx - self._cumulative_sizes[dataset_idx]
        return sys.getsizeof(self.datasets[dataset_idx], sample_idx)

    @property
    @deprecated("`cummulative_sizes` attribute is renamed to `cumulative_sizes`", category=FutureWarning)
    def cummulative_sizes(self) -> Sequence[int]:
        return self.cumulative_sizes


class ChainDataset(IterableDataset):
    """Data set for chaining multiple iterable data sets.

    This class is useful to assemble different existing data set streams.
    The chaining operation is done on-the-fly, so concatenating large-scale
    data sets with this class will be efficient.

    Args:
        datasets (Iterable[Dataset]): data sets to be chained together
    """

    def __init__(self, datasets: Iterable[torch.utils.data.Dataset]) -> None:
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            assert isinstance(d, torch.utils.data.IterableDataset), "ChainDataset only supports IterableDataset"
            yield from d

    def __len__(self) -> int:
        total = 0
        for d in self.datasets:
            assert isinstance(d, torch.utils.data.IterableDataset), "ChainDataset only supports IterableDataset"
            total += len(d)  # type: ignore[arg-type]
        return total
