import os
import sys

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input
from torch.profiler import ProfilerActivity, profile, record_function

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from flatflow.torch.nn.flash_attn_modules import Attention

is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)


def _format_time(time_us):
    """Define how to format time in FunctionEvent."""
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return f"{time_us / US_IN_SECOND:.3f}s"
    if time_us >= US_IN_MS:
        return f"{time_us / US_IN_MS:.3f}ms"
    return f"{time_us:.3f}us"


def _format_memory(nbytes):
    """Return a formatted memory size string."""
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if abs(nbytes) >= GB:
        return f"{nbytes * 1.0 / GB:.2f} Gb"
    elif abs(nbytes) >= MB:
        return f"{nbytes * 1.0 / MB:.2f} Mb"
    elif abs(nbytes) >= KB:
        return f"{nbytes * 1.0 / KB:.2f} Kb"
    else:
        return str(nbytes) + " b"


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full(
            (batch_size, 1), max_seqlen, device=device, dtype=torch.int32
        )
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(
            max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device
        )
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size)
        < lengths
    )
    return padding_mask


def attn_bias_from_alibi_slopes(
    slopes,
    seqlen_q,
    seqlen_k,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
):
    batch, nheads = slopes.shape
    device = slopes.device
    slopes = rearrange(slopes, "b h -> b h 1 1")
    if causal:
        return (
            torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32) * slopes
        )
    else:
        row_idx = rearrange(
            torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
        )
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        relative_pos = torch.abs(row_idx + sk - sq - col_idx)
        return -slopes * relative_pos.to(dtype=slopes.dtype)


def generate_qkv(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    kvpacked=False,
    qkvpacked=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
            q, query_padding_mask
        )
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * seqlen_q,
            step=seqlen_q,
            dtype=torch.int32,
            device=q_unpad.device,
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(
            k, key_padding_mask
        )
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0,
            (batch_size + 1) * seqlen_k,
            step=seqlen_k,
            dtype=torch.int32,
            device=k_unpad.device,
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(
                dqkv_unpad, indices_q, batch_size, seqlen_q
            )
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(
                dkv_unpad, indices_k, batch_size, seqlen_k
            )
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(
                dk_unpad, indices_k, batch_size, seqlen_k
            )
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(
                dk_unpad, "(b s) h d -> b s h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


@pytest.mark.parametrize("kvpacked", [False])
@pytest.mark.parametrize(
    "dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16])
)
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("alibi", [True])
@pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (108, 216),
        (256, 512),
    ],
)
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
def test_flash_attn_varlen_output(
    seqlen_q,
    seqlen_k,
    d,
    dropout_p,
    causal,
    local,
    alibi,
    deterministic,
    mha_type,
    dtype,
    kvpacked,
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip("Not enough GPU memory")

    device = "cuda"
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    query_padding_mask = generate_random_padding_mask(
        seqlen_q, batch_size, device, mode="random"
    )
    key_padding_mask = generate_random_padding_mask(
        seqlen_k, batch_size, device, mode="random"
    )

    if alibi:
        alibi_slopes = (
            torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        )
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
        )
    else:
        alibi_slopes, attn_bias = None, None

    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_k,
        cu_seqlens_q,
        max_seqlen_k,
        max_seqlen_q,
    ) = (None, None, None, None, None, None, None)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("flash_attention_forward"):
            (
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                q,
                k,
                v,
                output_pad_fn,
                dq_pad_fn,
                dk_pad_fn,
            ) = generate_qkv(
                q, k, v, query_padding_mask, key_padding_mask, kvpacked=False
            )
            out_unpad, sm_lse, S_dmask = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
            out = output_pad_fn(out_unpad)
    events = prof.key_averages()
    flash_attn_cuda_time = sum(
        getattr(event, "device_time_total", 0) for event in events
    )
    flash_attn_cpu_time = sum(getattr(event, "cpu_time_total", 0) for event in events)
    flash_attn_cuda_memory = sum(
        getattr(event, "device_memory_usage", 0) for event in events
    )
    flash_attn_cpu_memory = sum(
        getattr(event, "cpu_memory_usage", 0) for event in events
    )
    offsets = dict()
    offsets["cu_seqlens_q"] = cu_seqlens_q
    offsets["cu_seqlens_k"] = cu_seqlens_k
    offsets["max_seqlen_q"] = max_seqlen_q
    offsets["max_seqlen_k"] = max_seqlen_k
    flatflow_attention = Attention(offsets)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("flash_attention_forward"):
            unpad_output = flatflow_attention._flash_attention_forward(
                q_unpad,
                k_unpad,
                v_unpad,
                dropout_p,
                use_causal=causal,
            )
    events = prof.key_averages()
    flatflow_cuda_time = sum(getattr(event, "device_time_total", 0) for event in events)
    flatflow_cpu_time = sum(getattr(event, "cpu_time_total", 0) for event in events)
    flatflow_cuda_memory = sum(
        getattr(event, "device_memory_usage", 0) for event in events
    )
    flatflow_cpu_memory = sum(getattr(event, "cpu_memory_usage", 0) for event in events)

    assert (
        flatflow_cuda_time <= flash_attn_cuda_time
    ), "FlatFlow CUDA time should be less"
    assert (
        flatflow_cpu_time <= flash_attn_cpu_time
    ), f"FlatFlow CPU time should be less // flatflow_cpu_time"
    assert (
        flatflow_cuda_memory <= flash_attn_cuda_memory
    ), "FlatFlow CUDA memory usage should be less"
    assert (
        flatflow_cpu_memory <= flash_attn_cpu_memory
    ), "FlatFlow CPU memory usage should be less"
