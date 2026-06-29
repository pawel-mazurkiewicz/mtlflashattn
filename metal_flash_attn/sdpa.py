"""Gated F.scaled_dot_product_attention patch for MPS.

The v0 kernel is memory-bounded but slower than stock fused SDPA at sizes that
fit, so the patch only reroutes when materializing the Lq x Lk score matrix
would genuinely threaten unified memory (score bytes >= min_score_gb,
default 12 GB / MTLFLASHATTN_SDPA_MIN_GB). Everything else — and any kernel
error — falls through to the original op. Never crashes the caller.

Kill switch: MTLFLASHATTN_SDPA=off.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from ._kernel import MAX_HEAD_DIM, _select_tier, flash_attn_forward

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# TensorOps tiers that beat stock fused SDPA by 3-4x (or more, causal); worth
# routing even when the score matrix would fit in memory.
_FAST_TIERS = frozenset({"v2", "v2_fp32", "v2_bf16"})

_orig = None
_min_score_bytes = None
_min_seq = None       # correctness gate: stock MPS fused SDPA is wrong past ~4k
_fast_min_seq = None  # speed gate floor: tiny attention stays on stock


def _eligibility(q, k, v, attn_mask, dropout_p, is_causal):
    """Return (eligible, reason). reason names the disqualifying gate."""
    if q.device.type != "mps":
        return False, "not-mps"
    if attn_mask is not None:
        return False, "attn_mask"
    if dropout_p:
        return False, "dropout"
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False, f"ndim({q.dim()})"
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        return False, f"dtype({q.dtype})"
    D = q.shape[-1]
    if D > MAX_HEAD_DIM or k.shape[-1] != D or v.shape[-1] != D:
        return False, f"head_dim({D})"
    Hq, Hkv = q.shape[1], k.shape[1]
    if Hkv == 0 or Hq % Hkv != 0:
        return False, f"heads({Hq}/{Hkv})"
    if k.shape[2] != v.shape[2]:
        return False, "kv-len-mismatch"
    Lq, Lk = q.shape[2], k.shape[2]
    # torch sdpa's is_causal is TOP-LEFT aligned; the kernel is bottom-right
    # (CUDA flash-attn convention). Identical only when Lq == Lk.
    if is_causal and Lq != Lk:
        return False, "causal-cross-length"
    maxlen = max(Lq, Lk)
    # 1. Correctness: stock MPS fused SDPA is silently numerically wrong past
    #    ~4k tokens (per-element errors up to ~28 on real DiT q/k/v). Route to
    #    our (exact, or fp32-accumulating) kernel regardless of tier or memory.
    if maxlen >= _min_seq:
        return True, "correctness-large-seq"
    # 2. Speed: a fast TensorOps tier is 3-4x faster than stock even when the
    #    score matrix fits; fire above a modest floor (tiny attention is cheap
    #    and stock is fine there).
    if maxlen >= _fast_min_seq and _select_tier(q, k, v) in _FAST_TIERS:
        return True, "fast-tier"
    # 3. Memory: OOM rescue for slow tiers / many heads at moderate seq.
    score_bytes = q.shape[0] * Hq * Lq * Lk * 2
    if score_bytes >= _min_score_bytes:
        return True, "oom-rescue"
    return False, f"fits({score_bytes / 1024**3:.2f}GB)"


def _uneven_v_attention(query, key, value, attn_mask, is_causal, scale, enable_gqa=False,
                        q_chunk=4096):
    """Defensive exact path for value_head_dim != query/key_head_dim.

    Some torch/macOS versions have been observed to mishandle the wide-value case
    in stock MPS SDPA (wrong-shaped or numerically wrong output) — notably with
    Hunyuan3D's PBR reference attention, where per-material value projections are
    concatenated into one tensor. Current torch (2.12) / macOS 27 handles it
    correctly, so this is insurance rather than a fix for a reproducing bug: we
    compute it directly, fp32-accumulated and chunked over the query length so the
    Lq x Lk score matrix never has to be fully materialized.
    """
    Hq, Hkv = query.shape[1], key.shape[1]
    # Expand kv heads only when the caller opted into GQA — mirroring stock SDPA,
    # which rejects unequal head counts unless enable_gqa=True. We raise a clean
    # error here rather than defer to stock: on MPS the unequal-heads case
    # hard-aborts the process (LLVM ERROR), which would crash the caller.
    if Hq != Hkv:
        if not (enable_gqa and Hkv != 0 and Hq % Hkv == 0):
            raise ValueError(
                f"metal_flash_attn: SDPA query/kv head mismatch (Hq={Hq}, Hkv={Hkv}); "
                "pass enable_gqa=True for GQA/MQA"
            )
        rep = Hq // Hkv
        key = key.repeat_interleave(rep, dim=1)
        value = value.repeat_interleave(rep, dim=1)

    s = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    Lq, Lk = query.shape[-2], key.shape[-2]

    bias = None
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            bias = torch.zeros(attn_mask.shape, dtype=torch.float32, device=query.device)
            bias = bias.masked_fill(~attn_mask, float("-inf"))
        else:
            bias = attn_mask.float()

    key_t = key.float().transpose(-2, -1)
    value_f = value.float()
    outs = []
    for i in range(0, Lq, q_chunk):
        qi = query[..., i:i + q_chunk, :].float()
        scores = torch.matmul(qi, key_t) * s
        if bias is not None:
            scores = scores + (bias[..., i:i + qi.shape[-2], :] if bias.shape[-2] == Lq else bias)
        if is_causal:  # top-left aligned, matching torch SDPA semantics
            qpos = torch.arange(i, i + qi.shape[-2], device=query.device).unsqueeze(-1)
            kpos = torch.arange(Lk, device=query.device).unsqueeze(0)
            scores = scores.masked_fill(kpos > qpos, float("-inf"))
        # fully masked rows (bool mask or additive -inf) make softmax emit NaN;
        # zero them like ref_attention does, rather than propagating NaN.
        probs = torch.nan_to_num(scores.softmax(dim=-1), nan=0.0)
        outs.append(torch.matmul(probs, value_f))
    return torch.cat(outs, dim=-2).to(value.dtype)


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
          is_causal=False, scale=None, **kwargs):
    eligible, _ = _eligibility(query, key, value, attn_mask, dropout_p, is_causal)
    if eligible:
        try:
            s = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
            return flash_attn_forward(query, key, value, scale=s, causal=is_causal)
        except Exception as e:  # never crash — fall back to stock SDPA
            print(f"[metal_flash_attn/sdpa] kernel fell back ({e}); using stock SDPA")
    # Defensive shield for value_head_dim != query_head_dim: some torch/macOS
    # versions have mishandled the wide-value case (current torch handles it). Cheap
    # insurance — compute it exactly ourselves rather than trust the stock path.
    if (query.device.type == "mps" and not dropout_p
            and query.dim() == 4 and value.dim() == 4
            and value.shape[-1] != query.shape[-1]):
        return _uneven_v_attention(query, key, value, attn_mask, is_causal, scale,
                                   enable_gqa=kwargs.get("enable_gqa", False))
    return _orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                 is_causal=is_causal, scale=scale, **kwargs)


def install(min_score_gb=None, min_seq=None, fast_min_seq=None):
    """Patch F.scaled_dot_product_attention. Returns True if newly installed.

    Gates (any one fires the kernel): correctness (max seq >= min_seq, default
    MTLFLASHATTN_SDPA_MIN_SEQ=4096), speed (fast TensorOps tier and max seq >=
    fast_min_seq, default MTLFLASHATTN_SDPA_FAST_MIN_SEQ=1024), and memory
    (score bytes >= min_score_gb, default MTLFLASHATTN_SDPA_MIN_GB=12).
    """
    global _orig, _min_score_bytes, _min_seq, _fast_min_seq
    if _orig is not None:
        return False
    if os.environ.get("MTLFLASHATTN_SDPA", "auto").lower() in ("off", "0", "false"):
        return False
    if min_score_gb is None:
        min_score_gb = float(os.environ.get("MTLFLASHATTN_SDPA_MIN_GB", "12"))
    if min_seq is None:
        min_seq = int(os.environ.get("MTLFLASHATTN_SDPA_MIN_SEQ", "4096"))
    if fast_min_seq is None:
        fast_min_seq = int(os.environ.get("MTLFLASHATTN_SDPA_FAST_MIN_SEQ", "1024"))
    _min_score_bytes = int(min_score_gb * (1024 ** 3))
    _min_seq = min_seq
    _fast_min_seq = fast_min_seq
    _orig = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _sdpa
    torch.nn.functional.scaled_dot_product_attention = _sdpa
    return True


def uninstall():
    """Restore the original op. Returns True if a patch was removed."""
    global _orig, _min_score_bytes, _min_seq, _fast_min_seq
    if _orig is None:
        return False
    F.scaled_dot_product_attention = _orig
    torch.nn.functional.scaled_dot_product_attention = _orig
    _orig = None
    _min_score_bytes = None
    _min_seq = None
    _fast_min_seq = None
    return True
