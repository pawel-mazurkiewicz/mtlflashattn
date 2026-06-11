"""Metal flash-attention forward kernel (v0: one thread per query row).

JIT-compiled via torch.mps.compile_shader — pure Python, no .metallib, no xcrun.
Online softmax over all keys: the Lq x Lk score matrix is never materialized,
so peak memory is O(B*H*Lq*D). fp32 accumulation; inputs are upcast.

Internal layout is heads-second [B, H, N, D]. Callers hand in any strided view
(e.g. a transpose of heads-third [B, N, H, D]); the fp32 upcast + .contiguous()
performs the layout conversion in a single copy.
"""
from __future__ import annotations

import os

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

// ---------------------------------------------------------------------------
// v1: FA-2 with simdgroup_matrix (M1+). One simdgroup per 8 query rows,
// 4 simdgroups per threadgroup. QK^T and PV as half 8x8 simdgroup matmuls with
// DIRECT device-memory K/V reads (no threadgroup staging — unified-memory
// anti-pattern per MPP guide / cider). The S tile round-trips through
// threadgroup memory for the online softmax (unavoidable pre-TensorOps);
// softmax state (m, l) and the O accumulator are fp32, striped across lanes
// (lane owns key-column `lane` of S, and output columns c with c % 32 == lane),
// so per-row rescaling never needs the opaque simdgroup_matrix layout.
// Q is padded to a multiple of 8 rows, K/V to a multiple of 32; padded keys are
// masked with -inf in the softmax phase, padded query rows are never stored.

constant constexpr int BC     = 32;  // keys per block
constant constexpr int NS     = 4;   // simdgroups per threadgroup
constant constexpr int MAXDT  = 16;  // max D/8 (D <= 128)

kernel void flash_attn_fwd_v1(
    device const half*  Q   [[buffer(0)]],   // [B,Hq,Lqp,D] padded
    device const half*  K   [[buffer(1)]],   // [B,Hkv,Lkp,D] padded
    device const half*  V   [[buffer(2)]],   // [B,Hkv,Lkp,D] padded
    device half*        O   [[buffer(3)]],   // [B,Hq,Lq,D] unpadded
    device const int*   SH  [[buffer(4)]],   // [B,Hq,Hkv,Lq,Lk,Lqp,Lkp,D,causal]
    device const float* PR  [[buffer(5)]],   // [scale]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[1], Hkv=SH[2], Lq=SH[3], Lk=SH[4],
              Lqp=SH[5], Lkp=SH[6], D=SH[7], causal=SH[8];
    const float scale = PR[0];
    const int DT = D / 8;

    threadgroup half Stile [NS][8*BC];
    threadgroup half Ptile [NS][8*BC];
    threadgroup half PVtile[NS][8*8*MAXDT];
    // Q staged in threadgroup memory (NOT registers): Q is reused across every
    // K block, and freeing D/8 simdgroup matrices per thread is what keeps
    // occupancy alive at D=128. This is reuse-staging, not the K/V streaming
    // anti-pattern.
    threadgroup half Qs[NS][8*8*MAXDT];

    const int q_row0 = int(tgid.x)*(8*NS) + int(sgid)*8;
    if (q_row0 >= Lqp) return;            // uniform per simdgroup

    const int bh  = int(tgid.y);
    const int b   = bh / Hq;
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);

    device const half* Qb = Q + ulong(bh)*ulong(Lqp)*ulong(D);
    device const half* Kb = K + ulong(b*Hkv+hkv)*ulong(Lkp)*ulong(D);
    device const half* Vb = V + ulong(b*Hkv+hkv)*ulong(Lkp)*ulong(D);

    for (int i=int(lane); i<8*D; i+=32)
        Qs[sgid][(i/D)*(8*MAXDT) + (i%D)] = Qb[q_row0*D + i];
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float m[8], l[8], acc[8][4];
    for (int r=0; r<8; ++r) {
        m[r] = -INFINITY; l[r] = 0.0f;
        for (int t=0; t<4; ++t) acc[r][t] = 0.0f;
    }

    const int shift = Lk - Lq;            // bottom-right causal alignment
    for (int j0=0; j0<Lkp; j0+=BC) {
        if (causal && j0 > q_row0 + 7 + shift) break;

        // S = Q K^T  (4 key sub-tiles of 8)
        for (int ct=0; ct<4; ++ct) {
            simdgroup_half8x8 S = make_filled_simdgroup_matrix<half,8,8>(0.0h);
            for (int dt=0; dt<DT; ++dt) {
                simdgroup_half8x8 Qt, Kt;
                simdgroup_load(Qt, &Qs[sgid][dt*8], ulong(8*MAXDT));
                simdgroup_load(Kt, Kb + (j0+ct*8)*D + dt*8, ulong(D), ulong2(0,0), true);
                simdgroup_multiply_accumulate(S, Qt, Kt, S);
            }
            simdgroup_store(S, &Stile[sgid][ct*8], ulong(BC));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax — lane owns key column j0+lane
        const int j = j0 + int(lane);
        float corr[8];
        for (int r=0; r<8; ++r) {
            const int qrow = q_row0 + r;
            float s = float(Stile[sgid][r*BC + lane]) * scale;
            const bool valid = (j < Lk) && (!causal || j <= qrow + shift);
            s = valid ? s : -INFINITY;
            const float mb = simd_max(s);
            const float mn = max(m[r], mb);
            float p, c;
            if (mn == -INFINITY) { p = 0.0f; c = 1.0f; }   // row fully masked so far
            else { p = exp(s - mn); c = exp(m[r] - mn); }
            l[r] = l[r]*c + simd_sum(p);
            m[r] = mn;
            corr[r] = c;
            Ptile[sgid][r*BC + lane] = half(p);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // PV: [8 x BC] @ [BC x D], V read directly from device memory
        simdgroup_half8x8 Pt[4];
        for (int ct=0; ct<4; ++ct)
            simdgroup_load(Pt[ct], &Ptile[sgid][ct*8], ulong(BC));
        for (int dt=0; dt<DT; ++dt) {
            simdgroup_half8x8 PV = make_filled_simdgroup_matrix<half,8,8>(0.0h);
            for (int ct=0; ct<4; ++ct) {
                simdgroup_half8x8 Vt;
                simdgroup_load(Vt, Vb + (j0+ct*8)*D + dt*8, ulong(D));
                simdgroup_multiply_accumulate(PV, Pt[ct], Vt, PV);
            }
            simdgroup_store(PV, &PVtile[sgid][dt*8], ulong(8*MAXDT));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // rescale + accumulate in fp32 lanes (lane owns cols c % 32 == lane)
        for (int r=0; r<8; ++r) {
            int t = 0;
            for (int c=int(lane); c<D; c+=32, ++t)
                acc[r][t] = acc[r][t]*corr[r] + float(PVtile[sgid][r*(8*MAXDT) + c]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);  // before next S overwrite
    }

    device half* Ob = O + ulong(bh)*ulong(Lq)*ulong(D);
    for (int r=0; r<8; ++r) {
        const int qrow = q_row0 + r;
        if (qrow >= Lq) break;
        const float inv = (l[r] > 0.0f) ? (1.0f/l[r]) : 0.0f;
        int t = 0;
        for (int c=int(lane); c<D; c+=32, ++t)
            Ob[qrow*D + c] = half(acc[r][t] * inv);
    }
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


def _v1_eligible(q, k, v):
    D = q.shape[-1]
    return (
        q.dtype == torch.float16
        and k.dtype == torch.float16
        and v.dtype == torch.float16
        and D % 8 == 0
        and D <= MAX_HEAD_DIM
    )


def _select_tier(q, k, v):
    return "v1" if _v1_eligible(q, k, v) else "v0"


def flash_attn_forward(q, k, v, scale, causal):
    """q: [B,Hq,Lq,D], k/v: [B,Hkv,Lk,D] (heads-second, any strides).

    Returns [B,Hq,Lq,D] contiguous, in q.dtype.

    Tier selection: MTLFLASHATTN_KERNEL=auto|v0|v1 (auto picks v1 for fp16
    with D % 8 == 0, else the memory-safe scalar v0).
    """
    mode = os.environ.get("MTLFLASHATTN_KERNEL", "auto").lower()
    if mode == "v1":
        if not _v1_eligible(q, k, v):
            raise RuntimeError(
                f"metal_flash_attn: v1 kernel forced but ineligible "
                f"(dtype={q.dtype}, D={q.shape[-1]}; needs fp16, D%8==0, D<={MAX_HEAD_DIM})"
            )
        return _flash_v1(q, k, v, scale, causal)
    if mode == "auto" and _select_tier(q, k, v) == "v1":
        return _flash_v1(q, k, v, scale, causal)
    return _flash_v0(q, k, v, scale, causal)


def _flash_v1(q, k, v, scale, causal):
    """simdgroup_matrix FA-2 kernel. fp16 in/out, fp32 softmax state."""
    import torch.nn.functional as F

    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[2]
    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    Lqp = -(-Lq // 8) * 8
    Lkp = -(-Lk // 32) * 32
    if Lqp != Lq:
        qc = F.pad(qc, (0, 0, 0, Lqp - Lq))
    if Lkp != Lk:
        kc = F.pad(kc, (0, 0, 0, Lkp - Lk))
        vc = F.pad(vc, (0, 0, 0, Lkp - Lk))
    out = torch.empty(B, Hq, Lq, D, device=q.device, dtype=torch.float16)
    sh = torch.tensor(
        [B, Hq, Hkv, Lq, Lk, Lqp, Lkp, D, 1 if causal else 0],
        dtype=torch.int32, device=q.device,
    )
    pr = torch.tensor([float(scale)], dtype=torch.float32, device=q.device)
    ntg_x = -(-Lqp // 32)  # 32 query rows per threadgroup (4 simdgroups x 8)
    _get_lib().flash_attn_fwd_v1(
        qc, kc, vc, out, sh, pr,
        threads=(ntg_x * 128, B * Hq, 1), group_size=(128, 1, 1),
    )
    return out


def _flash_v0(q, k, v, scale, causal):
    """One thread per query row, fp32 scalar online softmax. Memory-safe baseline."""
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
