"""Metal flash-attention forward kernel (v0: one thread per query row).

JIT-compiled via torch.mps.compile_shader — pure Python, no .metallib, no xcrun.
Online softmax over all keys: the Lq x Lk score matrix is never materialized,
so peak memory is O(B*H*Lq*D). fp32 accumulation; inputs are upcast.

Internal layout is heads-second [B, H, N, D]. Callers hand in any strided view
(e.g. a transpose of heads-third [B, N, H, D]); the fp32 upcast + .contiguous()
performs the layout conversion in a single copy.
"""
from __future__ import annotations

import torch

MAX_HEAD_DIM = 128  # kernel uses a thread-local acc[128]

_MSL = r"""
#include <metal_stdlib>
using namespace metal;

kernel void flash_attn_fwd(
    device const float* Q   [[buffer(0)]],   // [B,Hq,Lq,D]
    device const float* K   [[buffer(1)]],   // [B,Hkv,Lk,D]
    device const float* V   [[buffer(2)]],   // [B,Hkv,Lk,D]
    device float*       O   [[buffer(3)]],   // [B,Hq,Lq,D]
    device const int*   SH  [[buffer(4)]],   // [B,Hq,Hkv,Lq,Lk,D,causal]
    device const float* PR  [[buffer(5)]],   // [scale]
    uint3 gid [[thread_position_in_grid]])
{
    const int B=SH[0], Hq=SH[1], Hkv=SH[2], Lq=SH[3], Lk=SH[4], D=SH[5], causal=SH[6];
    const float scale = PR[0];

    const int qi = int(gid.x);
    const int bh = int(gid.y);
    if (qi >= Lq || bh >= B*Hq) return;
    const int b   = bh / Hq;
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);            // grouped-query mapping

    const int q_base = ((b*Hq + hq)*Lq + qi)*D;
    const int kv_bh  = (b*Hkv + hkv);

    // causal aligns bottom-right (key j attends iff j <= qi + (Lk-Lq))
    const int kmax = causal ? (qi + (Lk - Lq) + 1) : Lk;

    float m = -INFINITY;
    float l = 0.0f;
    float acc[128];
    for (int d=0; d<D; ++d) acc[d]=0.0f;

    for (int kj=0; kj<kmax; ++kj) {
        const int k_base = (kv_bh*Lk + kj)*D;
        float s = 0.0f;
        for (int d=0; d<D; ++d) s += Q[q_base+d]*K[k_base+d];
        s *= scale;
        float m_new = max(m, s);
        float corr  = exp(m - m_new);
        float p     = exp(s - m_new);
        l = l*corr + p;
        for (int d=0; d<D; ++d) acc[d] = acc[d]*corr + p*V[k_base+d];
        m = m_new;
    }
    float inv = (l > 0.0f) ? (1.0f/l) : 0.0f;
    for (int d=0; d<D; ++d) O[q_base+d] = acc[d]*inv;
}
"""

_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        if not torch.backends.mps.is_available():
            raise RuntimeError("metal_flash_attn requires PyTorch MPS")
        if not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError(
                "metal_flash_attn requires torch.mps.compile_shader (PyTorch >= 2.5)"
            )
        _lib = torch.mps.compile_shader(_MSL)
    return _lib


def flash_attn_forward(q, k, v, scale, causal):
    """q: [B,Hq,Lq,D], k/v: [B,Hkv,Lk,D] (heads-second, any strides).

    Returns [B,Hq,Lq,D] contiguous, in q.dtype.
    """
    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[2]
    qf = q.float().contiguous()
    kf = k.float().contiguous()
    vf = v.float().contiguous()
    out = torch.empty(B, Hq, Lq, D, device=q.device, dtype=torch.float32)
    sh = torch.tensor(
        [B, Hq, Hkv, Lq, Lk, D, 1 if causal else 0],
        dtype=torch.int32, device=q.device,
    )
    pr = torch.tensor([float(scale)], dtype=torch.float32, device=q.device)
    _get_lib().flash_attn_fwd(
        qf, kf, vf, out, sh, pr,
        threads=(Lq, B * Hq, 1), group_size=(64, 1, 1),
    )
    return out.to(q.dtype)
