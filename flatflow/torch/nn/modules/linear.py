# Adapted from https://github.com/pytorch/pytorch/blob/v2.4.0/torch/nn/modules/linear.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch

__all__ = ["Linear"]


class Linear(torch.nn.Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module is a port of :class:`torch.nn.Linear` for use in conjunction with the concatenation-based batching.

    One API difference is that :param:`offsets` is given as an additional argument to :meth:`forward`.
    Note that :param:`offsets` is not used directly in this module since linear transformations do not
    require any notion of data samples.

    Another API difference is that :meth:`forward` returns :param:`offsets` as an additional return value.
    This is to propagate :param:`offsets` in case the downstream layers require the notion of data samples.

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool): If ``False``, the layer will not learn an additive bias (default: ``True``).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input: torch.Tensor, offsets: Sequence[int]) -> tuple[torch.Tensor, Sequence[int]]:
        return super().forward(input), offsets
