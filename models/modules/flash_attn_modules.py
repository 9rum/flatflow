try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    from pip._internal import main as pip 
    pip(['install', 'flash-attn' , '--no-build-isolation'])

def _flash_attention_forward(
    query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k , dropout=0.0, softmax_scale=None, use_causal=True,
):
    """
    Calls the forward method of Flash Attention.
    Based on flatflow programming model tensors are not in stacked manner,
    therefore unpad_input & pad_input is unnecessary.

    Args:
        query (`torch.Tensor`):
            Input query to be passed to Flash Attention API
        key (`torch.Tensor`):
            Input key to be passed to Flash Attention API
        value (`torch.Tensor`):
            Input value to be passed to Flash Attention API
        cu_seqlens_q (`list`):
            Index of different query data sample's offset
            (batch_size + 1,), dtype torch.int32.
            The cumulative sequence lengths of the sequences in the batch, used to index into q
        cu_seqlens_k (`list`):
            Index of different kv data sample's offset
            (batch_size + 1,), dtype torch.int32. 
            The cumulative sequence lengths of the sequences in the batch, used to index into kv
        max_seqlen_q (`int`):
            Maximum query sequence length in the batch
        max_seqlen_k (`int`):
            Maximum key sequence length in the batch
        dropout (`float`, *optional*):
            Dropout probability
        softmax_scale (`float`, *optional*):
            Scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_causal (`boolean`, *optional*):
            Boolean value whether to use causal mask or not
    """
    attn_output_unpad = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_query_seqlen,
        max_seqlen_k=max_key_seqlen,
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=use_causal,
    )

    return attn_output_unpad