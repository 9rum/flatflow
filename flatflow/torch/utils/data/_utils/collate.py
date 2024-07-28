# Adapted from https://github.com/pytorch/pytorch/blob/v2.4.0/torch/utils/data/_utils/collate.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Mapping
from typing import Optional, Union

import torch
from torch.utils.data._utils.collate import collate

__all__ = ["default_collate"]


def collate_tensor_fn(batch, *, collate_fn_map: Optional[Mapping[Union[type, tuple[type, ...]], Callable]] = None):
    elem = batch[0]
    out = None
    if elem.is_nested:
        raise RuntimeError(
            "Batches of nested tensors are not currently supported by the default collate_fn; "
            "please provide a custom collate_fn to handle them appropriately."
        )
    if elem.layout in {torch.sparse_coo, torch.sparse_csr, torch.sparse_bsr, torch.sparse_csc, torch.sparse_bsc}:
        raise RuntimeError(
            "Batches of sparse tensors are not currently supported by the default collate_fn; "
            "please provide a custom collate_fn to handle them appropriately."
        )
    offsets = [0] * (len(batch) + 1)
    for i, e in enumerate(batch):
        offsets[i + 1] = offsets[i] + e.size(0)

    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy.
        numel = sum(x.numel() for x in batch)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    return torch.cat(batch, 0, out=out), offsets


default_collate_fn_map: Mapping[Union[type, tuple[type, ...]], Callable] = {torch.Tensor: collate_tensor_fn}


def default_collate(batch):
    """Take in a batch of data and concatenate the elements within the batch.

    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~flatflow.torch.utils.data.DataLoader`.

    For layers that require the notion of data samples, id offsets are provided
    along with the concatenated tensor to specify the sequence boundaries.

    Args:
        batch: a single batch to be collated
    """
    return collate(batch, collate_fn_map=default_collate_fn_map)
