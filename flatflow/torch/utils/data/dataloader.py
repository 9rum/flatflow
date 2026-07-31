# Adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/data/dataloader.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable

import torch
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t

from flatflow.torch.utils.data._utils.collate import default_collate
from flatflow.torch.utils.data.dataset import Dataset

__all__ = ["DataLoader", "default_collate"]


class DataLoader(torch.utils.data.DataLoader):
    """Data loader combines a dataset and a sampler, and provides an iterable over the
    given dataset.

    It is a drop-in replacement for :class:`torch.utils.data.DataLoader`. The only
    difference is that if :attr:`collate_fn` is not provided, i.e., when automatic
    batching (collation), concatenation-based batching is applied. That is, the collated
    tensor does not have an additional outer dimension.

    Args:
        dataset (Dataset): Dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load (default: ``1``).
        shuffle (bool, optional): Set to ``True`` to have the data reshuffled at every
            epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): Defines the strategy to draw samples
            from the dataset. Can be any :class:`Iterable` with :meth:`__len__`
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): Like :attr:`sampler`, but returns
            a batch of indices at a time. Mutually exclusive with :attr:`batch_size`,
            :attr:`shuffle`, :attr:`sampler` and :attr:`drop_last`.
        num_workers (int, optional): How many subprocesses to use for data loading.
            ``0`` means that the data will be loaded in the main process
            (default: ``0``).
        collate_fn (Callable, optional): Merges a list of samples to form a mini-batch
            of tensor(s). Used when using batched loading from a map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors into
            device pinned memory before returning them.
        drop_last (bool, optional): Set to ``True`` to drop the last incomplete batch if
            the dataset size is not divisible by the batch size. If ``False`` and the
            size of dataset is not divisible by the batch size, then the last batch will
            be smaller (default: ``False``).
        timeout (float, optional): If positive, the timeout value for collecting a batch
            from workers. Should always be non-negative (default: ``0``).
        worker_init_fn (Callable, optional): If not ``None``, this will be called on
            each worker subprocess with the worker id (an int in ``[0, num_workers)``)
            as input, after seeding and before data loading (default: ``None``).
        multiprocessing_context (str or multiprocessing.context.BaseContext, optional):
            If ``None``, the default multiprocessing context of your operating system
            will be used (default: ``None``).
        generator (torch.Generator, optional): If not ``None``, this RNG will be used by
            :class:`torch.utils.data.RandomSampler` to generate random indices and
            multiprocessing to generate ``base_seed`` for workers (default: ``None``).
        prefetch_factor (int, optional): Number of batches loaded in advance by each
            worker. ``2`` means there will be a total of ``2 * num_workers`` batches
            prefetched across all workers (default value depends on the set value for
            :attr:`num_workers`. If value of ``num_workers=0`` default is ``None``.
            Otherwise, if value of ``0 < num_workers`` default is ``2``).
        persistent_workers (bool, optional): If ``True``, the data loader will not shut
            down the worker processes after a dataset has been consumed once. This
            allows to maintain the workers dataset instances alive (default: ``False``).
        pin_memory_device (str, optional): The device to :attr:`pin_memory` on if
            :attr:`pin_memory` is ``True``.
        in_order (bool, optional): If ``False``, the data loader will not enforce that
            batches are returned in a first-in, first-out order. Only applies when
            ``0 < num_workers`` (default: ``True``).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: torch.utils.data.Sampler | Iterable | None = None,
        batch_sampler: torch.utils.data.Sampler | Iterable | None = None,
        num_workers: int = 0,
        collate_fn: _collate_fn_t | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: _worker_init_fn_t | None = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
    ) -> None:
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
            in_order=in_order,
        )

        if collate_fn is None and self._auto_collation:
            self.collate_fn = default_collate
