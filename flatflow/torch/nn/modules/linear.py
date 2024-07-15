from typing import Dict

import torch
from torch import Tensor, nn


class Linear(nn.Linear):
    r"""Custom Linear layer for flatflow.
    Offsets are required to handle variable length inputs.
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> offsets = {'cu_seqlens_q': [0, 2, 3, 6], 'cu_seqlens_k': [0, 2, 3, 6], 'max_seqlen_q': 3, 'max_seqlen_k': 3}
        >>> output = m(input, offsets)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input_tensor: Tensor, offsets: Dict) -> Tensor:
        """
        Args:
        input_tensor : torch.Tensor
            Input tensor to be passed to the forward method of the Linear layer
        offsets : Dict[str, Union[List[Any], int]]
            Offset values of cu_seqlens and max_seqlen
            Consist of four keys : cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        """
        return super().forward(input_tensor), offsets
