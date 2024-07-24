# Adapted from https://github.com/pytorch/pytorch/blob/v2.3.1/torch/utils/data/_utils/collate.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Mapping
from typing import Optional, Union

import torch


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
