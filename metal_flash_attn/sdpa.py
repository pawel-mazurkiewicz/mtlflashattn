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

from ._kernel import MAX_HEAD_DIM, flash_attn_forward

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_orig = None
_min_score_bytes = None


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
    score_bytes = q.shape[0] * Hq * Lq * Lk * 2
    if score_bytes < _min_score_bytes:
        return False, f"fits({score_bytes / 1024**3:.2f}GB)"
    return True, "flash"


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
          is_causal=False, scale=None, **kwargs):
    eligible, _ = _eligibility(query, key, value, attn_mask, dropout_p, is_causal)
    if eligible:
        try:
            s = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
            return flash_attn_forward(query, key, value, scale=s, causal=is_causal)
        except Exception as e:  # never crash — fall back to stock SDPA
            print(f"[metal_flash_attn/sdpa] kernel fell back ({e}); using stock SDPA")
    return _orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                 is_causal=is_causal, scale=scale, **kwargs)


def install(min_score_gb=None):
    """Patch F.scaled_dot_product_attention. Returns True if newly installed."""
    global _orig, _min_score_bytes
    if _orig is not None:
        return False
    if os.environ.get("MTLFLASHATTN_SDPA", "auto").lower() in ("off", "0", "false"):
        return False
    if min_score_gb is None:
        min_score_gb = float(os.environ.get("MTLFLASHATTN_SDPA_MIN_GB", "12"))
    _min_score_bytes = int(min_score_gb * (1024 ** 3))
    _orig = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _sdpa
    torch.nn.functional.scaled_dot_product_attention = _sdpa
    return True


def uninstall():
    """Restore the original op. Returns True if a patch was removed."""
    global _orig, _min_score_bytes
    if _orig is None:
        return False
    F.scaled_dot_product_attention = _orig
    torch.nn.functional.scaled_dot_product_attention = _orig
    _orig = None
    _min_score_bytes = None
    return True
