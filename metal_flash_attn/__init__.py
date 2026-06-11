"""Metal flash-attention for PyTorch MPS — CUDA flash-attn API surface.

Inference-only (no backward). Unsupported features raise NotImplementedError so
callers can fall back, rather than silently approximating.
"""
from __future__ import annotations

import math

import torch

from ._kernel import MAX_HEAD_DIM, flash_attn_forward
from ._version import __version__

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "__version__",
]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _check_supported(
    q, k, v, dropout_p, window_size, softcap, alibi_slopes,
    deterministic, return_attn_probs,
):
    if dropout_p:
        raise NotImplementedError("metal_flash_attn: dropout_p > 0 not supported (inference-only)")
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("metal_flash_attn: sliding window_size not supported yet")
    if softcap:
        raise NotImplementedError("metal_flash_attn: softcap not supported")
    if alibi_slopes is not None:
        raise NotImplementedError("metal_flash_attn: alibi_slopes not supported")
    if deterministic:
        raise NotImplementedError("metal_flash_attn: deterministic not supported")
    if return_attn_probs:
        raise NotImplementedError("metal_flash_attn: return_attn_probs not supported (probs never materialized)")
    if q.device.type != "mps":
        raise NotImplementedError(f"metal_flash_attn: device {q.device.type!r} not supported (MPS only)")
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise NotImplementedError(f"metal_flash_attn: dtype {q.dtype} (q/k/v must match, fp16/bf16/fp32)")
    D = q.shape[-1]
    if D > MAX_HEAD_DIM:
        raise NotImplementedError(f"metal_flash_attn: head_dim {D} > {MAX_HEAD_DIM}")
    if k.shape[-1] != D or v.shape[-1] != D:
        raise ValueError("metal_flash_attn: q/k/v head_dim mismatch")


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Drop-in for flash_attn.flash_attn_func (forward only).

    q: [B, Lq, Hq, D]; k, v: [B, Lk, Hkv, D] (heads-third, CUDA flash-attn layout).
    Hkv must divide Hq (GQA/MQA). Returns [B, Lq, Hq, D] in q.dtype.
    """
    _check_supported(
        q, k, v, dropout_p, window_size, softcap, alibi_slopes,
        deterministic, return_attn_probs,
    )
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"metal_flash_attn: expected 4-D [B,S,H,D] tensors, got q.dim()={q.dim()}")
    Hq, Hkv = q.shape[2], k.shape[2]
    if Hkv == 0 or Hq % Hkv != 0:
        raise ValueError(f"metal_flash_attn: num_heads_kv ({Hkv}) must divide num_heads_q ({Hq})")
    if k.shape[1] != v.shape[1]:
        raise ValueError("metal_flash_attn: k/v seqlen mismatch")

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    out = flash_attn_forward(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        scale=scale, causal=causal,
    )
    return out.transpose(1, 2).contiguous()


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    """Drop-in for flash_attn.flash_attn_varlen_func (forward only).

    q: [total_q, Hq, D]; k, v: [total_k, Hkv, D] packed without padding;
    cu_seqlens_*: [B+1] int32 cumulative sequence lengths.
    Returns [total_q, Hq, D] in q.dtype.
    """
    _check_supported(
        q, k, v, dropout_p, window_size, softcap, alibi_slopes,
        deterministic, return_attn_probs,
    )
    if block_table is not None:
        raise NotImplementedError("metal_flash_attn: block_table (paged KV cache) not supported")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(f"metal_flash_attn: expected 3-D [total,H,D] tensors, got q.dim()={q.dim()}")
    if cu_seqlens_q.shape[0] != cu_seqlens_k.shape[0]:
        raise ValueError(
            f"metal_flash_attn: cu_seqlens_q ({cu_seqlens_q.shape[0]}) and "
            f"cu_seqlens_k ({cu_seqlens_k.shape[0]}) imply different batch sizes"
        )
    Hq, Hkv = q.shape[1], k.shape[1]
    if Hkv == 0 or Hq % Hkv != 0:
        raise ValueError(f"metal_flash_attn: num_heads_kv ({Hkv}) must divide num_heads_q ({Hq})")
    if k.shape[0] != v.shape[0]:
        raise ValueError("metal_flash_attn: k/v total token count mismatch")

    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    if cu_q[-1] != q.shape[0]:
        raise ValueError(
            f"metal_flash_attn: cu_seqlens_q[-1]={cu_q[-1]} != total_q={q.shape[0]}"
        )
    if cu_k[-1] != k.shape[0]:
        raise ValueError(
            f"metal_flash_attn: cu_seqlens_k[-1]={cu_k[-1]} != total_k={k.shape[0]}"
        )

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    out = torch.empty_like(q)
    for i in range(len(cu_q) - 1):
        qs = q[cu_q[i]:cu_q[i + 1]]
        ks = k[cu_k[i]:cu_k[i + 1]]
        vs = v[cu_k[i]:cu_k[i + 1]]
        if qs.shape[0] == 0:
            continue
        o = flash_attn_forward(
            qs.permute(1, 0, 2)[None], ks.permute(1, 0, 2)[None],
            vs.permute(1, 0, 2)[None], scale=scale, causal=causal,
        )  # [1, Hq, Lq_i, D]
        out[cu_q[i]:cu_q[i + 1]] = o[0].permute(1, 0, 2)
    return out


def flash_attn_qkvpacked_func(qkv, *args, **kwargs):
    """Drop-in for flash_attn.flash_attn_qkvpacked_func. qkv: [B, S, 3, H, D]."""
    if qkv.dim() != 5 or qkv.shape[2] != 3:
        raise ValueError(
            f"metal_flash_attn: expected qkv [B,S,3,H,D], got {tuple(qkv.shape)}"
        )
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(q, k, v, *args, **kwargs)


def flash_attn_kvpacked_func(q, kv, *args, **kwargs):
    """Drop-in for flash_attn.flash_attn_kvpacked_func. kv: [B, Sk, 2, Hkv, D]."""
    if kv.dim() != 5 or kv.shape[2] != 2:
        raise ValueError(
            f"metal_flash_attn: expected kv [B,S,2,H,D], got {tuple(kv.shape)}"
        )
    k, v = kv.unbind(dim=2)
    return flash_attn_func(q, k, v, *args, **kwargs)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, *args, **kwargs):
    """Drop-in for flash_attn.flash_attn_varlen_qkvpacked_func. qkv: [total, 3, H, D]."""
    if qkv.dim() != 4 or qkv.shape[1] != 3:
        raise ValueError(
            f"metal_flash_attn: expected qkv [total,3,H,D], got {tuple(qkv.shape)}"
        )
    q, k, v = qkv.unbind(dim=1)
    return flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, *args, **kwargs
    )


def flash_attn_varlen_kvpacked_func(
    q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *args, **kwargs
):
    """Drop-in for flash_attn.flash_attn_varlen_kvpacked_func. kv: [total_k, 2, Hkv, D]."""
    if kv.dim() != 4 or kv.shape[1] != 2:
        raise ValueError(
            f"metal_flash_attn: expected kv [total,2,H,D], got {tuple(kv.shape)}"
        )
    k, v = kv.unbind(dim=1)
    return flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *args, **kwargs
    )
