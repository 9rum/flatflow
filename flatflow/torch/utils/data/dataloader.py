# Adapted from https://github.com/pytorch/pytorch/blob/v2.4.0/torch/utils/data/dataloader.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional, TypeVar, Union

import torch
import torch.utils.data.graph_settings
from torch.utils.data import _DatasetKind, default_convert, get_worker_info
from torch.utils.data.dataloader import _InfiniteConstantSampler
from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper, _MapDataPipeSerializationWrapper

from flatflow.torch.utils.data._utils import default_collate

__all__ = [
    "DataLoader",
    "_DatasetKind",
    "default_collate",
    "default_convert",
    "get_worker_info",
]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

_collate_fn_t = Callable[[Sequence[T]], Any]
_worker_init_fn_t = Callable[[int], Any]


class DataLoader(torch.utils.data.DataLoader[T_co]):
    """Data loader combines a data set and a sampler, and provides an iterable over the given data set.

    This is a port of :class:`torch.utils.data.DataLoader`.

    One notable API difference is that if :attr:`collate_fn` is not provided,
    i.e., when automatic batching (collation), concatenation-based batching is applied.
    That is, the resulting collated tensor does not have a batch size dimension;
    instead, it is yielded with id offsets to denote sequence boundaries.

    Args:
        dataset (Dataset): data set from which to load the data
        batch_size (int, optional): how many samples per batch to load (default: ``1``)
        shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``)
        sampler (Sampler or Iterable, optional): Defines the strategy to draw samples from the data set.
            Can be any :class:`Iterable` with :meth:`__len__` implemented.
            If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): Like :attr:`sampler`, but returns a batch of indices at a time.
            Mutually exclusive with :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
        num_workers (int, optional): How many subprocesses to use for data loading.
            ``0`` means that the data will be loaded in the main process (default: ``0``).
        collate_fn (Callable, optional): Merges a list of samples to form a mini-batch of tensors.
            Used when using batched loading from a map-style data set.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors into device/CUDA pinned memory
            before returning them.
        drop_last (bool, optional): Set to ``True`` to drop the last incomplete batch, if the data set size is not
            divisible by the batch size. If ``False`` and the size of data set is not divisible by the batch size, then
            the last batch will be smaller (default: ``False``).
        timeout (float, optional): If positive, the timeout value for collecting a batch from workers.
            Should always be non-negative (default: ``0``).
        worker_init_fn (Callable, optional): If not ``None``, this will be called on each worker subprocess
            with the worker id (an int in ``[0, num_workers)``) as input,
            after seeding and before data loading (default: ``None``).
        multiprocessing_context (str or multiprocessing.context.BaseContext, optional): If ``None``, the default
            multiprocessing context of your operating system will be used (default: ``None``).
        generator (torch.Generator, optional): If not ``None``, this RNG will be used by
            :class:`torch.utils.data.RandomSampler` to generate random indices and
            multiprocessing to generate ``base_seed`` for workers (default: ``None``).
        prefetch_factor (int, optional): Number of batches loaded in advance by each worker. ``2`` means there will be
            a total of ``2 * num_workers`` batches prefetched across all workers (default value depends on the set value
            for :attr:`num_workers`. If value of ``num_workers=0`` default is ``None``. Otherwise, if value of
            ``0 < num_workers`` default is ``2``).
        persistent_workers (bool, optional): If ``True``, the data loader will not shut down the worker processes
            after a data set has been consumed once. This allows to maintain the worker data set instances alive
            (default: ``False``).
        pin_memory_device (str, optional): The device to :attr:`pin_memory` to if :attr:`pin_memory` is ``True``.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        sampler: Optional[Union[torch.utils.data.Sampler, Iterable]] = None,
        batch_sampler: Optional[Union[torch.utils.data.Sampler[Sequence], Iterable[Sequence]]] = None,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ) -> None:
        if num_workers < 0:
            raise ValueError("num_workers option should be non-negative; use num_workers=0 to disable multiprocessing.")

        if timeout < 0:
            raise ValueError("timeout option should be non-negative.")

        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError(
                "prefetch_factor option could only be specified in multiprocessing. "
                "Let 0 < num_workers to enable multiprocessing, otherwise set prefetch_factor to None."
            )
        elif 0 < num_workers and prefetch_factor is None:
            prefetch_factor = 2
        elif prefetch_factor is not None and prefetch_factor < 0:
            raise ValueError("prefetch_factor option should be non-negative.")

        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers option needs 0 < num_workers.")

        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context

        if isinstance(self.dataset, torch.utils.data.IterDataPipe):
            self.dataset = _IterDataPipeSerializationWrapper(self.dataset)
        elif isinstance(self.dataset, torch.utils.data.MapDataPipe):
            self.dataset = _MapDataPipeSerializationWrapper(self.dataset)

        if isinstance(dataset, torch.utils.data.IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            if isinstance(dataset, torch.utils.data.IterDataPipe):
                if shuffle is not None:
                    dataset = torch.utils.data.graph_settings.apply_shuffle_settings(dataset, shuffle=shuffle)
            elif shuffle:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    f"shuffle option, but got shuffle={shuffle}."
                )

            if sampler is not None:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    f"sampler option, but got sampler={sampler}."
                )
            elif batch_sampler is not None:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    f"batch_sampler option, but got batch_sampler={batch_sampler}."
                )
        else:
            shuffle = bool(shuffle)
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle.")

        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last."
                )
            batch_size = None
            drop_last = False
        elif batch_size is None:
            if drop_last:
                raise ValueError(
                    "batch_size=None option disables auto-batching and is mutually exclusive with drop_last."
                )

        if sampler is None:
            if self._dataset_kind == _DatasetKind.Iterable:
                sampler = _InfiniteConstantSampler()
            else:
                if shuffle:
                    sampler = torch.utils.data.RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
                else:
                    sampler = torch.utils.data.SequentialSampler(dataset)  # type: ignore[arg-type]

        if batch_size is not None and batch_sampler is None:
            batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = default_collate
            else:
                collate_fn = default_convert

        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers

        self.__initialized = True
        self._IterableDataset_len_called = None

        self._iterator = None

        self.check_worker_number_rationality()
