import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
from torch import nn


class Attention(nn.Module):
    def __init__(self, offsets):
        """Attention module.
        Assigns the input offsets to the class variables.
        Args:
            offsets : Dict[str, Union[List[Any], int]]
        """
        super().__init__()
        self.cu_seqlens_q = offsets["cu_seqlens_q"]
        self.cu_seqlens_k = offsets["cu_seqlens_k"]
        self.max_seqlen_q = offsets["max_seqlen_q"]
        self.max_seqlen_k = offsets["max_seqlen_k"]

    def _flash_attention_forward(
        self,
        query,
        key,
        value,
        dropout=0.0,
        softmax_scale=None,
        use_causal=True,
    ):
        """
        Calls the forward method of Flash Attention.
        Based on flatflow programming model tensors are not in stacked manner,
        therefore unpad_input & pad_input is unnecessary.

        Args:
            query : torch.Tensor
                Input query to be passed to Flash Attention API
            key : torch.Tensor
                Input key to be passed to Flash Attention API
            value : torch.Tensor
                Input value to be passed to Flash Attention API
            cu_seqlens_q : List[int]
                Index of different query data sample's offset
                (batch_size + 1,), dtype torch.int32.
                The cumulative sequence lengths of the sequences in the batch, used to index into q
            cu_seqlens_k : List[int]
                Index of different kv data sample's offset
                (batch_size + 1,), dtype torch.int32.
                The cumulative sequence lengths of the sequences in the batch, used to index into kv
            max_seqlen_q : int
                Maximum query sequence length in the batch
            max_seqlen_k : int
                Maximum key sequence length in the batch
            dropout : optional[float]
                Dropout probability
            softmax_scale : optional[float]
                Scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_causal : optional[bool]
                Boolean value whether to use causal mask or not
        """
        return flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k=self.cu_seqlens_k,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=use_causal,
        )
