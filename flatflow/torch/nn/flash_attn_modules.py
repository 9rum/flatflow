import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
from torch import nn


class Attention(nn.Module):
    def __init__(self, config):
        """Attention module.
        Assigns the input offsets to the class variables.
        Args:
            config : Dict[str, Union[float, bool, int]]
            dropout : optional[float]
                Dropout probability
            softmax_scale : optional[float]
                Scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_causal : optional[bool]
                Boolean value whether to use causal mask or not
            window_size : optional[Tuple[int, int]]
                Window size for local attention
            alibi_slopes : optional[Union[torch.Tensor, Tuple[torch.Tensor]]]
                alibli slops tensor of shape (num_heads,) or (batch_size, num_heads)
            return_attn_probs : optional[bool]
                Whether to return the attention probabilities. This option is for testing only.
        """
        super().__init__()
        self.dropout = getattr(config, "dropout", 0.0)
        self.softmax_scale = getattr(config, "softmax_scale", None)
        self.use_causal = getattr(config, "use_causal", True)
        self.window_size = getattr(config, "window_size", None)
        self.alibi_slopes = getattr(config, "alibi_slopes", None)
        self.deterministic = getattr(config, "deterministic", False)
        self.return_attn_probs = getattr(config, "return_attn_probs", False)

    def flash_attn_unpadded_func(
        self,
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
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
        """
        return flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=self.dropout,
            softmax_scale=self.softmax_scale,
            causal=self.use_causal,
        )
