"""Metal flash-attention forward kernel (v0: one thread per query row).

JIT-compiled via torch.mps.compile_shader — pure Python, no .metallib, no xcrun.
Online softmax over all keys: the Lq x Lk score matrix is never materialized,
so peak memory is O(B*H*Lq*D). fp32 accumulation; inputs are upcast.

Internal layout is heads-second [B, H, N, D]. Callers hand in any strided view
(e.g. a transpose of heads-third [B, N, H, D]); the fp32 upcast + .contiguous()
performs the layout conversion in a single copy.
"""
from __future__ import annotations

import atexit
import os
import platform

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
    device const float* PR  [[buffer(5)]],   // [scale, softcap]
    uint3 gid [[thread_position_in_grid]])
{
    const int B=SH[0], Hq=SH[1], Hkv=SH[2], Lq=SH[3], Lk=SH[4], D=SH[5], causal=SH[6];
    const float scale = PR[0];
    const float softcap = PR[1];
    const int wl = int(PR[2]);   // sliding-window left bound, -1 = open
    const int wr = int(PR[3]);   // sliding-window right bound, -1 = open

    const int qi = int(gid.x);
    const int bh = int(gid.y);
    if (qi >= Lq || bh >= B*Hq) return;
    const int b   = bh / Hq;
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);            // grouped-query mapping

    const int q_base = ((b*Hq + hq)*Lq + qi)*D;
    const int kv_bh  = (b*Hkv + hkv);

    // causal aligns bottom-right (key j attends iff j <= qi + (Lk-Lq)); sliding
    // window further bounds keys to [center - wl, center + wr] (wl/wr < 0 = open)
    const int center = qi + (Lk - Lq);
    int kmax = causal ? (center + 1) : Lk;
    if (wr >= 0) kmax = min(kmax, center + wr + 1);
    const int kmin = (wl >= 0) ? max(0, center - wl) : 0;

    float m = -INFINITY;
    float l = 0.0f;
    float acc[128];
    for (int d=0; d<D; ++d) acc[d]=0.0f;

    for (int kj=kmin; kj<kmax; ++kj) {
        const int k_base = (kv_bh*Lk + kj)*D;
        float s = 0.0f;
        for (int d=0; d<D; ++d) s += Q[q_base+d]*K[k_base+d];
        s *= scale;
        if (softcap > 0.0f) s = softcap * tanh(s / softcap);
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
    device const float* PR  [[buffer(5)]],   // [scale, softcap]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[1], Hkv=SH[2], Lq=SH[3], Lk=SH[4],
              Lqp=SH[5], Lkp=SH[6], D=SH[7], causal=SH[8];
    const float scale = PR[0];
    const float softcap = PR[1];
    const int wl = int(PR[2]);   // sliding-window left bound, -1 = open
    const int wr = int(PR[3]);   // sliding-window right bound, -1 = open
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
            if (softcap > 0.0f) s = softcap * tanh(s / softcap);
            const bool valid = (j < Lk) && (!causal || j <= qrow + shift)
                && (wl < 0 || j >= qrow + shift - wl) && (wr < 0 || j <= qrow + shift + wr);
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

# ---------------------------------------------------------------------------
# v2: Metal 4 TensorOps (MPP) — macOS 26+/Xcode 27 SDK. Neural Accelerator on
# M5+, simdgroup path on M1-M4. One threadgroup = BR query rows, 4 simdgroups
# run matmul2d cooperatively. QK^T lands in a fp32 cooperative tensor; scale +
# (causal/bounds) masking, the cross-block online-softmax state (m, l, corr)
# lives in threadgroup arrays updated via cooperative-tensor COORDINATES
# (get_multidimensional_index) — no reliance on opaque layouts. P round-trips
# threadgroup memory into the PV matmul (multiply_accumulate into a fp32
# cooperative O across K blocks); register-resident P reuse is a later,
# measured optimization. Head dim is baked per-compile (@D@), libs cached.
_V2_MSL = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BR = 32;     // query rows per threadgroup
constant constexpr int BC = 32;     // keys per block
constant constexpr int DD = @D@;    // head dim (specialized per compile)
constant constexpr int NSG = 4;     // simdgroups cooperating per matmul

kernel void flash_attn_fwd_v2(
    device half*  Q  [[buffer(0)]],   // [B,Hq,Lq,DD]  (non-const: MPP static_assert)
    device half*  K  [[buffer(1)]],   // [B,Hkv,Lk,DD]
    device half*  V  [[buffer(2)]],   // [B,Hkv,Lk,DD]
    device half*  O  [[buffer(3)]],   // [B,Hq,Lq,DD]
    device int*   SH [[buffer(4)]],   // [B,Hq,Hkv,Lq,Lk,causal]
    device float* PR [[buffer(5)]],   // [scale, softcap]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[1], Hkv=SH[2], Lq=SH[3], Lk=SH[4], causal=SH[5];
    const float scale = PR[0];
    const float softcap = PR[1];
    const int wl = int(PR[2]);   // sliding-window left bound, -1 = open
    const int wr = int(PR[3]);   // sliding-window right bound, -1 = open

    const int bh  = int(tgid.y);
    const int b   = bh / Hq;
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);
    const int q0  = int(tgid.x) * BR;

    // NOTE: tensor::slice() as a matmul2d operand reads WRONG DATA (verified
    // empirically on macOS 27.0 beta 1, M5 Max — see dev_min_repro3.py).
    // Operands are constructed as pointer-offset tensors instead; the
    // remaining-rows extents preserve the op's bounds-checked edge handling.
    device half* Qp = Q + ulong(bh)*ulong(Lq)*DD;
    device half* Kp = K + ulong(b*Hkv+hkv)*ulong(Lk)*DD;
    device half* Vp = V + ulong(b*Hkv+hkv)*ulong(Lk)*DD;

    // QK^T: S[BR,BC] = Q[BR,DD] @ K[BC,DD]^T (NT), written to a THREADGROUP
    // tensor destination — reduce_rows is single-simdgroup-only, so the online
    // softmax runs as scalar lane-per-column code on the TG tile instead
    // (v1's engine). Only ctO stays cooperative (in-register across K blocks).
    constexpr auto qk_desc = matmul2d_descriptor(
        BR, BC, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc, execution_simdgroups<NSG>> qk_op;

    // PV: O[BR,DD] += P[BR,BC] @ V[BC,DD]  (NN, accumulating across K blocks)
    constexpr auto pv_desc = matmul2d_descriptor(
        BR, DD, static_cast<int>(dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroups<NSG>> pv_op;

    threadgroup float tgS[BR*BC];
    threadgroup half  tgP[BR*BC];
    threadgroup float tgCorr[BR];
    threadgroup float tgInvL[BR];

    // Operand extents must be the CLAMPED TILE size (min(tile, remaining)) —
    // oversized extents make the op infer k/n from them and read beyond the tile.
    auto mQ = tensor(Qp + q0*DD, dextents<int,2>{DD, min(BR, Lq - q0)}, array<int,2>{1, DD});
    auto tS = tensor(&tgS[0], dextents<int,2>{BC, BR}, array<int,2>{1, BC});
    auto tP = tensor(&tgP[0], dextents<int,2>{BC, BR}, array<int,2>{1, BC});

    using PvA = __tensor_ops_detail::__remove_addrspace_t<decltype(tP)>;
    using PvB = __tensor_ops_detail::__remove_addrspace_t<decltype(mQ)>;

    auto ctO = pv_op.get_destination_cooperative_tensor<PvA, PvB, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i)
        if (ctO.is_valid_element(i)) ctO[i] = 0.0f;

    // each simdgroup owns 8 of the BR=32 rows for the scalar softmax phase
    const int r0 = int(sgid) * (BR / NSG);
    float m[BR / NSG], l[BR / NSG];
    #pragma clang loop unroll(full)
    for (int r = 0; r < BR/NSG; ++r) { m[r] = -INFINITY; l[r] = 0.0f; }

    const int shift = Lk - Lq;   // bottom-right causal alignment
    for (int j0 = 0; j0 < Lk; j0 += BC) {
        if (causal && j0 > q0 + BR - 1 + shift) break;

        // tensor destinations ACCUMULATE (C = A*B + C) — zero S every block
        for (int i = int(tid); i < BR*BC; i += NSG*32) tgS[i] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mK = tensor(Kp + j0*DD, dextents<int,2>{DD, min(BC, Lk - j0)}, array<int,2>{1, DD});
        qk_op.run(mQ, mK, tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax on the TG tile — lane owns key column j0+lane
        const int j = j0 + int(lane);
        #pragma clang loop unroll(full)
        for (int r = 0; r < BR/NSG; ++r) {
            const int row  = r0 + r;
            const int qrow = q0 + row;
            float s = tgS[row*BC + lane] * scale;
            if (softcap > 0.0f) s = softcap * tanh(s / softcap);
            const bool ok = (j < Lk) && (qrow < Lq) && (!causal || j <= qrow + shift)
                && (wl < 0 || j >= qrow + shift - wl) && (wr < 0 || j <= qrow + shift + wr);
            s = ok ? s : -INFINITY;
            const float mb = simd_max(s);
            const float mn = max(m[r], mb);
            float p, c;
            if (mn == -INFINITY) { p = 0.0f; c = 1.0f; }   // row fully masked so far
            else { p = exp(s - mn); c = exp(m[r] - mn); }
            l[r] = l[r]*c + simd_sum(p);
            m[r] = mn;
            tgCorr[row] = c;
            tgP[row*BC + lane] = half(p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // rescale running O by corr (coordinates), accumulate this block's PV
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
            if (!ctO.is_valid_element(i)) continue;
            auto idx = ctO.get_multidimensional_index(i);
            ctO[i] *= tgCorr[int(idx[1])];
        }
        auto mV = tensor(Vp + j0*DD, dextents<int,2>{DD, min(BC, Lk - j0)}, array<int,2>{1, DD});
        pv_op.run(tP, mV, ctO);
        threadgroup_barrier(mem_flags::mem_threadgroup);  // tgS/tgP reused next block
    }

    if (lane == 0) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < BR/NSG; ++r)
            tgInvL[r0 + r] = (l[r] > 0.0f) ? (1.0f / l[r]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // normalize and store (manual, bounds-checked, fp32 -> fp16)
    device half* Ob = O + ulong(bh)*ulong(Lq)*DD;
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
        if (!ctO.is_valid_element(i)) continue;
        auto idx = ctO.get_multidimensional_index(i);
        const int row = int(idx[1]);
        const int r   = q0 + row;
        const int c   = int(idx[0]);
        if (r >= Lq || c >= DD) continue;
        Ob[r*DD + c] = half(ctO[i] * tgInvL[row]);
    }
}

// ---------------------------------------------------------------------------
// v2r: register-resident P variant — the WWDC recipe with the REAL constraints
// applied (all discovered empirically, none documented):
//   * input cooperative tensors require SINGLE-simdgroup ops
//     ("Input cooperative tensors require a single SIMD group"),
//   * the source cooperative tensor's element type must equal the left-input
//     element type => S must accumulate in HALF (v1-like precision tradeoff),
//   * the PV descriptor needs a STATIC k ("Inner dimension cannot be dynamic").
// Each simdgroup independently owns SR=16 query rows; reduce_rows works at this
// scope; P feeds PV via get_left_input_cooperative_tensor — zero threadgroup
// round-trips for S or P. K/V are padded to BC multiples by the dispatcher
// (static k means the tail block reads a full BC rows).

constant constexpr int SR = @SR@;   // query rows per simdgroup (v2r); chosen by D

kernel void flash_attn_fwd_v2r(
    device half*  Q  [[buffer(0)]],   // [B,Hq,Lq,DD]
    device half*  K  [[buffer(1)]],   // [B,Hkv,Lkp,DD] padded to BC multiple
    device half*  V  [[buffer(2)]],   // [B,Hkv,Lkp,DD] padded to BC multiple
    device half*  O  [[buffer(3)]],   // [B,Hq,Lq,DD]
    device int*   SH [[buffer(4)]],   // [B,Hq,Hkv,Lq,Lk,causal,Lkp]
    device float* PR [[buffer(5)]],   // [scale, softcap]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[1], Hkv=SH[2], Lq=SH[3], Lk=SH[4], causal=SH[5], Lkp=SH[6];
    const float scale = PR[0];
    const float softcap = PR[1];
    const int wl = int(PR[2]);   // sliding-window left bound, -1 = open
    const int wr = int(PR[3]);   // sliding-window right bound, -1 = open

    const int bh  = int(tgid.y);
    const int b   = bh / Hq;
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);
    const int q0  = int(tgid.x)*(SR*NSG) + int(sgid)*SR;

    threadgroup float tgM[NSG][SR], tgL[NSG][SR], tgCorr[NSG][SR], tgBlk[NSG][SR];

    if (q0 >= Lq) return;             // uniform per simdgroup

    device half* Qp = Q + ulong(bh)*ulong(Lq)*DD;
    device half* Kp = K + ulong(b*Hkv+hkv)*ulong(Lkp)*DD;
    device half* Vp = V + ulong(b*Hkv+hkv)*ulong(Lkp)*DD;

    constexpr auto qk_desc = matmul2d_descriptor(
        SR, BC, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc, execution_simdgroup> qk_op;
    constexpr auto pv_desc = matmul2d_descriptor(
        SR, DD, BC, false, false, false,   // static k: left-input reuse requires it
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroup> pv_op;

    if (lane < SR) { tgM[sgid][lane] = -INFINITY; tgL[sgid][lane] = 0.0f; }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    auto mQ = tensor(Qp + q0*DD, dextents<int,2>{DD, min(SR, Lq - q0)}, array<int,2>{1, DD});
    using TT = __tensor_ops_detail::__remove_addrspace_t<decltype(mQ)>;

    auto ctS = qk_op.get_destination_cooperative_tensor<TT, TT, half>();
    using CtP = decltype(pv_op.get_left_input_cooperative_tensor<half, half, float>());
    auto ctO = pv_op.get_destination_cooperative_tensor<CtP, TT, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i) ctO[i] = 0.0f;

    const int shift = Lk - Lq;        // bottom-right causal alignment
    for (int j0 = 0; j0 < Lk; j0 += BC) {
        if (causal && j0 > q0 + SR - 1 + shift) break;

        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctS.get_capacity(); ++i) ctS[i] = half(0.0h);
        auto mK = tensor(Kp + j0*DD, dextents<int,2>{DD, BC}, array<int,2>{1, DD});
        qk_op.run(mQ, mK, ctS);

        // scale + bounds/causal mask (half storage, fp32 math)
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctS.get_capacity(); ++i) {
            if (!ctS.is_valid_element(i)) continue;
            auto idx = ctS.get_multidimensional_index(i);
            const int jj = j0 + int(idx[0]);
            const int ii = q0 + int(idx[1]);
            const bool ok = (jj < Lk) && (ii < Lq) && (!causal || jj <= ii + shift)
                && (wl < 0 || jj >= ii + shift - wl) && (wr < 0 || jj <= ii + shift + wr);
            float sv = float(ctS[i]) * scale;
            if (softcap > 0.0f) sv = softcap * tanh(sv / softcap);
            ctS[i] = ok ? half(sv) : half(-INFINITY);
        }

        // block row-max via reduce_rows (legal at single-simdgroup scope)
        auto ctMb = qk_op.get_row_reduction_destination_cooperative_tensor<TT, TT, half>();
        reduce_rows(ctS, ctMb, reduction_operation::max, half(-INFINITY));
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctMb.get_capacity(); ++i) {
            if (!ctMb.is_valid_element(i)) continue;
            auto idx = ctMb.get_multidimensional_index(i);
            tgBlk[sgid][int(idx[0])] = float(ctMb[i]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < SR) {
            const float mo = tgM[sgid][lane];
            const float mn = max(mo, tgBlk[sgid][lane]);
            tgCorr[sgid][lane] = (mn == -INFINITY) ? 1.0f : exp(mo - mn);
            tgM[sgid][lane] = mn;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // P = exp(S - m) in place (half)
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctS.get_capacity(); ++i) {
            if (!ctS.is_valid_element(i)) continue;
            auto idx = ctS.get_multidimensional_index(i);
            const float mn = tgM[sgid][int(idx[1])];
            ctS[i] = (mn == -INFINITY) ? half(0.0h) : half(exp(float(ctS[i]) - mn));
        }

        // block row-sum via reduce_rows, merge running l
        auto ctLb = qk_op.get_row_reduction_destination_cooperative_tensor<TT, TT, half>();
        reduce_rows(ctS, ctLb, reduction_operation::sum, half(0.0h));
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctLb.get_capacity(); ++i) {
            if (!ctLb.is_valid_element(i)) continue;
            auto idx = ctLb.get_multidimensional_index(i);
            tgBlk[sgid][int(idx[0])] = float(ctLb[i]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < SR)
            tgL[sgid][lane] = tgL[sgid][lane]*tgCorr[sgid][lane] + tgBlk[sgid][lane];
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // rescale running O, accumulate PV with register-resident P
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
            if (!ctO.is_valid_element(i)) continue;
            auto idx = ctO.get_multidimensional_index(i);
            ctO[i] *= tgCorr[sgid][int(idx[1])];
        }
        auto ctP = pv_op.get_left_input_cooperative_tensor<half, half, float>(ctS);
        auto mV = tensor(Vp + j0*DD, dextents<int,2>{DD, BC}, array<int,2>{1, DD});
        pv_op.run(ctP, mV, ctO);
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
    device half* Ob = O + ulong(bh)*ulong(Lq)*DD;
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
        if (!ctO.is_valid_element(i)) continue;
        auto idx = ctO.get_multidimensional_index(i);
        const int row = int(idx[1]);
        const int r   = q0 + row;
        const int c   = int(idx[0]);
        if (r >= Lq || c >= DD) continue;
        const float l = tgL[sgid][row];
        Ob[r*DD + c] = half((l > 0.0f) ? (ctO[i] / l) : 0.0f);
    }
}

// probe: can the (single-simdgroup, half) QK destination feed PV as left input?
kernel void v2r_compat(
    device half* DUMMY [[buffer(0)]],
    device int*  OUT   [[buffer(1)]])
{
    constexpr auto qk_desc = matmul2d_descriptor(
        SR, BC, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc, execution_simdgroup> qk_op;
    constexpr auto pv_desc = matmul2d_descriptor(
        SR, DD, BC, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroup> pv_op;

    auto t = tensor(DUMMY, dextents<int,2>{DD, SR}, array<int,2>{1, DD});
    using TT = __tensor_ops_detail::__remove_addrspace_t<decltype(t)>;
    auto ctS = qk_op.get_destination_cooperative_tensor<TT, TT, half>();
    OUT[0] = pv_op.is_compatible_as_left_input<half, half, float>(ctS) ? 1 : 0;
}
"""

_V2_DTYPE_MSL = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BR = 32;
constant constexpr int BC = 32;
constant constexpr int DD = @D@;
constant constexpr int NSG = 4;

kernel void flash_attn_fwd_v2_dtype(
    device @MSL_T@* Q  [[buffer(0)]],
    device @MSL_T@* K  [[buffer(1)]],
    device @MSL_T@* V  [[buffer(2)]],
    device @MSL_T@* O  [[buffer(3)]],
    device int*      SH [[buffer(4)]],
    device float*    PR [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int Hq=SH[1], Hkv=SH[2], Lq=SH[3], Lk=SH[4], causal=SH[5];
    const float scale = PR[0];
    const float softcap = PR[1];
    const int wl = int(PR[2]);   // sliding-window left bound, -1 = open
    const int wr = int(PR[3]);   // sliding-window right bound, -1 = open

    const int bh  = int(tgid.y);
    const int b   = bh / Hq;
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);
    const int q0  = int(tgid.x) * BR;

    device @MSL_T@* Qp = Q + ulong(bh)*ulong(Lq)*DD;
    device @MSL_T@* Kp = K + ulong(b*Hkv+hkv)*ulong(Lk)*DD;
    device @MSL_T@* Vp = V + ulong(b*Hkv+hkv)*ulong(Lk)*DD;

    constexpr auto qk_desc = matmul2d_descriptor(
        BR, BC, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<qk_desc, execution_simdgroups<NSG>> qk_op;

    constexpr auto pv_desc = matmul2d_descriptor(
        BR, DD, static_cast<int>(dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroups<NSG>> pv_op;

    threadgroup float tgS[BR*BC];
    threadgroup @P_T@ tgP[BR*BC];
    threadgroup float tgCorr[BR];
    threadgroup float tgInvL[BR];

    auto mQ = tensor(Qp + q0*DD, dextents<int,2>{DD, min(BR, Lq - q0)}, array<int,2>{1, DD});
    auto tS = tensor(&tgS[0], dextents<int,2>{BC, BR}, array<int,2>{1, BC});
    auto tP = tensor(&tgP[0], dextents<int,2>{BC, BR}, array<int,2>{1, BC});

    using PvA = __tensor_ops_detail::__remove_addrspace_t<decltype(tP)>;
    using PvB = __tensor_ops_detail::__remove_addrspace_t<decltype(mQ)>;

    auto ctO = pv_op.get_destination_cooperative_tensor<PvA, PvB, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i)
        if (ctO.is_valid_element(i)) ctO[i] = 0.0f;

    const int r0 = int(sgid) * (BR / NSG);
    float m[BR / NSG], l[BR / NSG];
    #pragma clang loop unroll(full)
    for (int r = 0; r < BR/NSG; ++r) { m[r] = -INFINITY; l[r] = 0.0f; }

    const int shift = Lk - Lq;
    for (int j0 = 0; j0 < Lk; j0 += BC) {
        if (causal && j0 > q0 + BR - 1 + shift) break;

        for (int i = int(tid); i < BR*BC; i += NSG*32) tgS[i] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mK = tensor(Kp + j0*DD, dextents<int,2>{DD, min(BC, Lk - j0)}, array<int,2>{1, DD});
        qk_op.run(mQ, mK, tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int j = j0 + int(lane);
        #pragma clang loop unroll(full)
        for (int r = 0; r < BR/NSG; ++r) {
            const int row  = r0 + r;
            const int qrow = q0 + row;
            float s = tgS[row*BC + lane] * scale;
            if (softcap > 0.0f) s = softcap * tanh(s / softcap);
            const bool ok = (j < Lk) && (qrow < Lq) && (!causal || j <= qrow + shift)
                && (wl < 0 || j >= qrow + shift - wl) && (wr < 0 || j <= qrow + shift + wr);
            s = ok ? s : -INFINITY;
            const float mb = simd_max(s);
            const float mn = max(m[r], mb);
            float p, c;
            if (mn == -INFINITY) { p = 0.0f; c = 1.0f; }
            else { p = exp(s - mn); c = exp(m[r] - mn); }
            l[r] = l[r]*c + simd_sum(p);
            m[r] = mn;
            tgCorr[row] = c;
            tgP[row*BC + lane] = @P_T@(p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
            if (!ctO.is_valid_element(i)) continue;
            auto idx = ctO.get_multidimensional_index(i);
            ctO[i] *= tgCorr[int(idx[1])];
        }
        auto mV = tensor(Vp + j0*DD, dextents<int,2>{DD, min(BC, Lk - j0)}, array<int,2>{1, DD});
        pv_op.run(tP, mV, ctO);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < BR/NSG; ++r)
            tgInvL[r0 + r] = (l[r] > 0.0f) ? (1.0f / l[r]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device @MSL_T@* Ob = O + ulong(bh)*ulong(Lq)*DD;
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
        if (!ctO.is_valid_element(i)) continue;
        auto idx = ctO.get_multidimensional_index(i);
        const int row = int(idx[1]);
        const int r   = q0 + row;
        const int c   = int(idx[0]);
        if (r >= Lq || c >= DD) continue;
        Ob[r*DD + c] = @MSL_T@(ctO[i] * tgInvL[row]);
    }
}
"""

_lib = None
_v2_libs = {}
_v2_dtype_libs = {}
_v2_support = None
_v2_reuse = {}


def _v2_reuse_ok(D):
    """True if the register-resident-P (v2r) kernel should be used for this head dim.

    Measured on M5 Max: v2r ~2x v2 at D=64 (25 vs 12 TF/s; causal 23 TF/s = 20x
    stock) but loses at D=128 (16.6 vs 18.5 TF/s — SR x 128 fp32 O registers per
    simdgroup collapse occupancy). auto => v2r only for D <= 64.
    Tradeoff: v2r accumulates S in half (API forces source element type == left
    input type), so its precision is v1-like rather than v2's fp32.
    MTLFLASHATTN_V2_PREUSE=1 forces it for all D, =0 disables.
    """
    mode = os.environ.get("MTLFLASHATTN_V2_PREUSE", "auto").lower()
    if mode in ("0", "off", "false"):
        return False
    if mode not in ("1", "on", "true") and D > 64:
        return False
    return _v2r_compat_probe(D)


def _v2r_compat_probe(D):
    """Cached on-device check that the single-simdgroup QK destination can feed
    PV as a left input at this head dim (the v2r register-resident-P recipe)."""
    ok = _v2_reuse.get(D)
    if ok is None:
        try:
            lib = _get_v2_lib(D)
            dummy = torch.zeros(32 * D, dtype=torch.float16, device="mps")
            out = torch.zeros(1, dtype=torch.int32, device="mps")
            lib.v2r_compat(dummy, out, threads=(32, 1, 1), group_size=(32, 1, 1))
            torch.mps.synchronize()
            ok = bool(out.item())
        except Exception:
            ok = False
        _v2_reuse[D] = ok
    return ok


_V2R_MIN_LK = 256  # below this, v2r is slower AND less precise than the TG kernel


def _v2r_dtype_ok(D, Lk, mode):
    """Register-resident-P gate for the dtype kernels, by dtype, head dim, and
    KV length. Measured M5 Max (v2r SR=16 vs the threadgroup-round-trip kernel):
      bf16: 2.5x at D<=64, 1.23x at D=128 -> all eligible D.
      fp32: 1.45x at D<=64, ~1.0x/loss at D=128 (fp32 S/O register pressure)
            -> D<=64 only.
    But v2r only amortizes past Lk~256 (it does the softmax reduction in the
    element type, so few keys cost both speed and precision -- e.g. Lk=5 image-
    conditioning cross-attention is 0.64x and 3x the error). Gate on Lk.
    MTLFLASHATTN_V2_PREUSE=1 forces D ceiling (not the Lk floor), =0 disables."""
    if D % 8 != 0 or D > MAX_HEAD_DIM:
        return False
    if Lk < _V2R_MIN_LK:
        return False
    env = os.environ.get("MTLFLASHATTN_V2_PREUSE", "auto").lower()
    if env in ("0", "off", "false"):
        return False
    ceiling = MAX_HEAD_DIM if mode == "v2_bf16" else 64
    if env not in ("1", "on", "true") and D > ceiling:
        return False
    return _v2r_compat_probe(D)


def _v2r_sr(D):
    # SR=16 everywhere. Measured (M5 Max): SR=8 and SR=32 both collapse (D=128
    # bf16: SR=8 -> 7.6, SR=16 -> 23.8, SR=32 -> 5.9 TF/s) -- SR=16 matches the
    # native 16-row matmul granularity; fewer rows waste it, more rows spill.
    return 16


def _get_v2_lib(D):
    lib = _v2_libs.get(D)
    if lib is None:
        src = _V2_MSL.replace("@D@", str(D)).replace("@SR@", str(_v2r_sr(D)))
        lib = torch.mps.compile_shader(src)
        _v2_libs[D] = lib
    return lib


def _build_v2r_dtype_msl():
    """Derive a dtype-parameterized register-resident-P v2r kernel from the
    proven fp16 v2r source — single source of truth, so any fix to the fp16
    kernel carries over. `@ET@` is the element type (e.g. bfloat). Confirmed on
    M5 Max: bfloat cooperative-tensor reuse via get_left_input_cooperative_tensor
    compiles and is_compatible_as_left_input(bfloat)==1; 2.5x over the
    threadgroup-round-trip baseline at D<=64 (dev/spike_v2r_bf16.py)."""
    src = _V2_MSL
    i = src.index("constant constexpr int SR")
    body = src[i:]  # SR const + comments + flash_attn_fwd_v2r + v2r_compat
    header = (
        "#include <metal_stdlib>\n"
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "using namespace metal;\n"
        "using namespace mpp::tensor_ops;\n"
        "constant constexpr int BR = 32;\n"
        "constant constexpr int BC = 32;\n"
        "constant constexpr int DD = @D@;\n"
        "constant constexpr int NSG = 4;\n"
    )
    s = header + body
    s = s.replace("flash_attn_fwd_v2r", "flash_attn_fwd_v2r_dtype")
    s = s.replace("v2r_compat", "v2r_dtype_compat")
    return s.replace("half", "@ET@")


_V2R_DTYPE_MSL = _build_v2r_dtype_msl()
_v2r_dtype_libs = {}


def _get_v2r_dtype_lib(D, et):
    key = (D, et)
    lib = _v2r_dtype_libs.get(key)
    if lib is None:
        src = (
            _V2R_DTYPE_MSL
            .replace("@D@", str(D))
            .replace("@SR@", str(_v2r_sr(D)))
            .replace("@ET@", et)
        )
        lib = torch.mps.compile_shader(src)
        _v2r_dtype_libs[key] = lib
    return lib


_V2_DTYPE_SPECS = {
    "v2_fp32": (torch.float32, "float", "float"),
    "v2_bf16": (torch.bfloat16, "bfloat", "bfloat"),
}


def _v2_dtype_spec(mode):
    return _V2_DTYPE_SPECS.get(mode)


def _v2_mode_for_runtime_dtype(dtype):
    if dtype == torch.float16:
        return "v2"
    if dtype == torch.float32:
        return "v2_fp32"
    if dtype == torch.bfloat16:
        return "v2_bf16"
    return None


def _v2_dtype_eligible(q, k, v, dtype):
    D = q.shape[-1]
    return (
        q.dtype == dtype
        and k.dtype == dtype
        and v.dtype == dtype
        and D % 8 == 0
        and D <= MAX_HEAD_DIM
    )


def _get_v2_dtype_lib(D, mode):
    spec = _v2_dtype_spec(mode)
    if spec is None:
        raise ValueError(f"unknown v2 dtype mode: {mode}")
    _, msl_t, p_t = spec
    key = (D, mode)
    lib = _v2_dtype_libs.get(key)
    if lib is None:
        src = (
            _V2_DTYPE_MSL
            .replace("@D@", str(D))
            .replace("@MSL_T@", msl_t)
            .replace("@P_T@", p_t)
        )
        lib = torch.mps.compile_shader(src)
        _v2_dtype_libs[key] = lib
    return lib


def _v2_supported():
    """TensorOps v2 needs macOS 26+ (MPP headers) and a working compile."""
    global _v2_support
    if _v2_support is None:
        try:
            major = int(platform.mac_ver()[0].split(".")[0])
        except (ValueError, IndexError):
            major = 0
        if major < 26 or not torch.backends.mps.is_available():
            _v2_support = False
        else:
            try:
                _get_v2_lib(64)
                _v2_support = True
            except Exception:
                _v2_support = False
    return _v2_support


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


def _torch_fallback_eligible(q, k, v):
    return (
        q.dtype == torch.float32
        and k.dtype == torch.float32
        and v.dtype == torch.float32
    )


def _v2_fp32_min_seq():
    raw = os.environ.get("MTLFLASHATTN_V2_FP32_MIN_SEQ", "2048")
    try:
        return max(1, int(raw))
    except ValueError:
        return 2048


def _v2_fp32_auto_eligible(q, k, v):
    """fp32 auto promotion: TensorOps v2_fp32 wins over the chunked PyTorch
    fallback only on long sequences (measured crossover ~1k-4k tokens); below
    the gate the fallback is faster. Gate on max(Lq, Lk)."""
    if not _v2_dtype_eligible(q, k, v, torch.float32):
        return False
    if max(q.shape[2], k.shape[2]) < _v2_fp32_min_seq():
        return False
    return _v2_supported()


def _v2_bf16_auto_eligible(q, k, v):
    """bf16 auto promotion: bf16 is the native fast TensorOps dtype, and v2_bf16
    beats the chunked PyTorch fallback at every measured length — so promote
    whenever D is eligible and TensorOps is available, with no length gate.
    The alternative under auto was scalar v0 (10x slower)."""
    return _v2_dtype_eligible(q, k, v, torch.bfloat16) and _v2_supported()


def _select_tier(q, k, v):
    if _torch_fallback_eligible(q, k, v):
        if _v2_fp32_auto_eligible(q, k, v):
            return "v2_fp32"
        return "torch"
    if q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16:
        if _v2_bf16_auto_eligible(q, k, v):
            return "v2_bf16"
        return "torch"  # fp32 chunked fallback beats scalar v0 for bf16 too
    if not _v1_eligible(q, k, v):
        return "v0"
    return "v2" if _v2_supported() else "v1"


_trace = {}  # (dtype, D, Lq, Lk, causal, label) -> call count


def _trace_enabled():
    return os.environ.get("MTLFLASHATTN_TRACE", "").lower() in ("1", "on", "true", "yes")


def _effective_kernel_label(q, k, v):
    """Resolve the kernel that flash_attn_forward will actually run, for tracing.
    Mirrors the dispatch in flash_attn_forward / _flash_v2 / _flash_v2_dtype."""
    D = q.shape[-1]
    mode = os.environ.get("MTLFLASHATTN_KERNEL", "auto").lower()
    if mode in ("torch", "pytorch"):
        return "torch"
    if mode == "v2_dtype" or mode == "v2_typed":
        mode = _v2_mode_for_runtime_dtype(q.dtype) or "v0"
    Lk = k.shape[2]
    if mode in ("v2_bf16", "v2_fp32"):
        return f"v2r({_v2_dtype_spec(mode)[1]})" if _v2r_dtype_ok(D, Lk, mode) else f"{mode}(TG)"
    if mode == "v2":
        return "v2r" if _v2_reuse_ok(D) else "v2"
    if mode == "v1":
        return "v1"
    if mode == "v0":
        return "v0"
    # auto
    tier = _select_tier(q, k, v)
    if tier == "v2":
        return "v2r" if _v2_reuse_ok(D) else "v2"
    if tier in ("v2_fp32", "v2_bf16"):
        return f"v2r({_v2_dtype_spec(tier)[1]})" if _v2r_dtype_ok(D, Lk, tier) else f"{tier}(TG)"
    return tier  # v1, v0, torch


def _record_trace(q, k, v, causal):
    try:
        label = _effective_kernel_label(q, k, v)
        key = (str(q.dtype).replace("torch.", ""), int(q.shape[-1]),
               int(q.shape[2]), int(k.shape[2]), bool(causal), label)
        _trace[key] = _trace.get(key, 0) + 1
    except Exception:
        pass  # tracing must never break a real run


def _trace_summary_lines():
    if not _trace:
        return []
    total = sum(_trace.values())
    lines = [f"[MTLFLASHATTN_TRACE] {total} attention call(s), "
             f"{len(_trace)} distinct shape/kernel combos:"]
    for (dt, D, Lq, Lk, causal, label), n in sorted(
            _trace.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  calls={n:<6} {dt:<9} D={D:<4} Lq={Lq:<6} Lk={Lk:<6} "
                     f"causal={'T' if causal else 'F'}  -> {label}")
    return lines


@atexit.register
def _trace_atexit():
    if _trace_enabled() and _trace:
        import sys
        print("\n".join(_trace_summary_lines()), file=sys.stderr, flush=True)


def flash_attn_forward(q, k, v, scale, causal, softcap=0.0, window_left=-1, window_right=-1):
    """q: [B,Hq,Lq,D], k/v: [B,Hkv,Lk,D] (heads-second, any strides).

    Returns [B,Hq,Lq,D] contiguous, in q.dtype.

    Tier selection: MTLFLASHATTN_KERNEL=auto|torch|v0|v1|v2|v2_fp32|v2_bf16|v2_dtype.
    Auto picks the fastest fp16 Metal tier, and for fp32 routes to the TensorOps
    v2_fp32 kernel on long sequences (>= MTLFLASHATTN_V2_FP32_MIN_SEQ, default
    2048) where it beats the chunked PyTorch fallback, falling back to the
    PyTorch path on short sequences. bf16 routes to the TensorOps v2_bf16 kernel
    whenever D is eligible (no length gate — it beats the fallback everywhere),
    else the chunked PyTorch path. v0 stays the explicit scalar debug baseline
    for unsupported dtypes/shapes. v2_dtype is an explicit mixed-dtype TensorOps
    mode for real pipelines that emit fp16, fp32, and bf16 attention.
    """
    if _trace_enabled():
        _record_trace(q, k, v, causal)
    mode = os.environ.get("MTLFLASHATTN_KERNEL", "auto").lower()
    if mode in ("torch", "pytorch"):
        return _flash_torch(q, k, v, scale, causal, softcap, window_left, window_right)
    if mode in ("v2_dtype", "v2_typed"):
        runtime_mode = _v2_mode_for_runtime_dtype(q.dtype)
        if runtime_mode is None or k.dtype != q.dtype or v.dtype != q.dtype:
            raise RuntimeError(
                "metal_flash_attn: v2_dtype kernel forced but ineligible "
                f"(q={q.dtype}, k={k.dtype}, v={v.dtype}; needs matching "
                "fp16, fp32, or bf16 tensors)"
            )
        if runtime_mode == "v2":
            if not _v1_eligible(q, k, v):
                raise RuntimeError(
                    f"metal_flash_attn: v2_dtype fp16 kernel forced but ineligible "
                    f"(D={q.shape[-1]}; needs D%8==0, D<={MAX_HEAD_DIM})"
                )
        elif not _v2_dtype_eligible(q, k, v, q.dtype):
            raise RuntimeError(
                f"metal_flash_attn: v2_dtype {q.dtype} kernel forced but ineligible "
                f"(D={q.shape[-1]}; needs D%8==0, D<={MAX_HEAD_DIM})"
            )
        if not _v2_supported():
            raise RuntimeError(
                "metal_flash_attn: v2_dtype kernel forced but TensorOps unavailable "
                "(needs macOS 26+ with MetalPerformancePrimitives)"
            )
        if runtime_mode == "v2":
            return _flash_v2(q, k, v, scale, causal, softcap, window_left, window_right)
        return _flash_v2_dtype(q, k, v, scale, causal, softcap, window_left, window_right, runtime_mode)
    dtype_spec = _v2_dtype_spec(mode)
    if dtype_spec is not None:
        dtype = dtype_spec[0]
        if not _v2_dtype_eligible(q, k, v, dtype):
            raise RuntimeError(
                f"metal_flash_attn: {mode} kernel forced but ineligible "
                f"(dtype={q.dtype}, D={q.shape[-1]}; needs {dtype}, "
                f"D%8==0, D<={MAX_HEAD_DIM})"
            )
        if not _v2_supported():
            raise RuntimeError(
                f"metal_flash_attn: {mode} kernel forced but TensorOps unavailable "
                "(needs macOS 26+ with MetalPerformancePrimitives)"
            )
        return _flash_v2_dtype(q, k, v, scale, causal, softcap, window_left, window_right, mode)
    if mode == "v2":
        if not _v1_eligible(q, k, v):  # same shape/dtype constraints as v1
            raise RuntimeError(
                f"metal_flash_attn: v2 kernel forced but ineligible "
                f"(dtype={q.dtype}, D={q.shape[-1]}; needs fp16, D%8==0, D<={MAX_HEAD_DIM})"
            )
        if not _v2_supported():
            raise RuntimeError(
                "metal_flash_attn: v2 kernel forced but TensorOps unavailable "
                "(needs macOS 26+ with MetalPerformancePrimitives)"
            )
        return _flash_v2(q, k, v, scale, causal, softcap, window_left, window_right)
    if mode == "v1":
        if not _v1_eligible(q, k, v):
            raise RuntimeError(
                f"metal_flash_attn: v1 kernel forced but ineligible "
                f"(dtype={q.dtype}, D={q.shape[-1]}; needs fp16, D%8==0, D<={MAX_HEAD_DIM})"
            )
        return _flash_v1(q, k, v, scale, causal, softcap, window_left, window_right)
    if mode == "auto":
        tier = _select_tier(q, k, v)
        if tier == "v2":
            return _flash_v2(q, k, v, scale, causal, softcap, window_left, window_right)
        if tier == "v2_fp32":
            return _flash_v2_dtype(q, k, v, scale, causal, softcap, window_left, window_right, "v2_fp32")
        if tier == "v2_bf16":
            return _flash_v2_dtype(q, k, v, scale, causal, softcap, window_left, window_right, "v2_bf16")
        if tier == "v1":
            return _flash_v1(q, k, v, scale, causal, softcap, window_left, window_right)
        if tier == "torch":
            return _flash_torch(q, k, v, scale, causal, softcap, window_left, window_right)
    return _flash_v0(q, k, v, scale, causal, softcap, window_left, window_right)


def _torch_chunk_size():
    raw = os.environ.get("MTLFLASHATTN_TORCH_CHUNK", "2048")
    try:
        chunk = int(raw)
    except ValueError:
        chunk = 2048
    return max(1, chunk)


def _flash_torch(q, k, v, scale, causal, softcap, window_left, window_right):
    """Chunked PyTorch matmul-softmax-matmul fallback. Computes in fp32."""
    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[2]
    if Hkv == 0 or Hq % Hkv != 0:
        raise ValueError(f"metal_flash_attn: num_heads_kv ({Hkv}) must divide num_heads_q ({Hq})")

    qf = q.float()
    kf = k.float()
    vf = v.float()
    if Hkv != Hq:
        rep = Hq // Hkv
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)

    kt = kf.transpose(-1, -2)
    out = torch.empty(B, Hq, Lq, D, device=q.device, dtype=torch.float32)
    chunk = _torch_chunk_size()
    windowed = window_left >= 0 or window_right >= 0
    key_pos = None
    if causal or windowed:
        key_pos = torch.arange(Lk, device=q.device)[None, :]
    for start in range(0, Lq, chunk):
        end = min(start + chunk, Lq)
        scores = (qf[:, :, start:end] @ kt) * scale
        if softcap:
            scores = softcap * torch.tanh(scores / softcap)
        if causal or windowed:
            center = torch.arange(start, end, device=q.device)[:, None] + (Lk - Lq)
            if causal:
                scores = scores.masked_fill(key_pos > center, float("-inf"))
            if window_left >= 0:
                scores = scores.masked_fill(key_pos < center - window_left, float("-inf"))
            if window_right >= 0:
                scores = scores.masked_fill(key_pos > center + window_right, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        out[:, :, start:end] = probs @ vf
    return out.to(q.dtype)


# Per-call dispatch tensors (shape `sh`, scale `pr`) are read-only kernel inputs.
# Building them via torch.tensor([...], device='mps') costs ~270us each (alloc +
# host->device copy + sync) — ~90% of a tiny Lk=5 cross-attention call. They are
# value-stable for a given key, so cache and reuse them: sharing one tensor across
# async dispatches is safe because the kernel only reads them and the contents
# never change for a given key.
_sh_cache = {}
_pr_cache = {}


def _sh_tensor(values, device):
    """Cached read-only int32 shape tensor for kernel dispatch (see note above)."""
    key = (tuple(int(x) for x in values), str(device))
    t = _sh_cache.get(key)
    if t is None:
        t = torch.tensor(list(values), dtype=torch.int32, device=device)
        _sh_cache[key] = t
    return t


def _pr_tensor(scale, softcap, window_left, window_right, device):
    """Cached read-only fp32 [scale, softcap, window_left, window_right] tensor.

    softcap=0 disables logit soft-capping; kernels apply
    s = softcap * tanh(s / softcap) to the scaled scores when softcap > 0.
    window_left/window_right are sliding-window key bounds relative to the
    bottom-right-aligned diagonal (center = qpos + (Lk - Lq)); -1 means
    unbounded on that side. Stored as fp32 and read back with int() in-shader
    (exact for window sizes < 2**24).
    """
    key = (float(scale), float(softcap), int(window_left), int(window_right), str(device))
    t = _pr_cache.get(key)
    if t is None:
        t = torch.tensor(
            [float(scale), float(softcap), float(window_left), float(window_right)],
            dtype=torch.float32, device=device,
        )
        _pr_cache[key] = t
    return t


def _flash_v2r_dtype(q, k, v, scale, causal, softcap, window_left, window_right, et):
    """Register-resident-P v2r kernel in element type `et` (bf16). Mirrors the
    fp16 v2r dispatch: static-k PV reads full BC-row tiles, so pad K/V to a BC
    multiple. ~2.5x over the threadgroup-round-trip dtype kernel at D<=64."""
    import torch.nn.functional as F

    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[2]
    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    out = torch.empty(B, Hq, Lq, D, device=q.device, dtype=q.dtype)
    pr = _pr_tensor(scale, softcap, window_left, window_right, q.device)
    Lkp = -(-Lk // 32) * 32
    if Lkp != Lk:
        kc = F.pad(kc, (0, 0, 0, Lkp - Lk))
        vc = F.pad(vc, (0, 0, 0, Lkp - Lk))
    sh = _sh_tensor([B, Hq, Hkv, Lq, Lk, 1 if causal else 0, Lkp], q.device)
    rows_per_tg = _v2r_sr(D) * 4
    ntg_x = -(-Lq // rows_per_tg)
    _get_v2r_dtype_lib(D, et).flash_attn_fwd_v2r_dtype(
        qc, kc, vc, out, sh, pr,
        threads=(ntg_x * 128, B * Hq, 1), group_size=(128, 1, 1),
    )
    return out


def _flash_v2_dtype(q, k, v, scale, causal, softcap, window_left, window_right, mode):
    """Experimental TensorOps v2 dtype specialization. No auto promotion yet."""
    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[2]
    # Register-resident P (v2r) beats the threadgroup round-trip: bf16 ~2.5x at
    # D<=64 and ~1.23x at D=128; fp32 ~1.45x at D<=64 (bit-exact) but loses at
    # D=128. _v2r_dtype_ok encodes the per-dtype D ceiling (bf16: all eligible D,
    # fp32: D<=64).
    if mode in ("v2_bf16", "v2_fp32") and _v2r_dtype_ok(D, Lk, mode):
        return _flash_v2r_dtype(q, k, v, scale, causal, softcap, window_left, window_right, _v2_dtype_spec(mode)[1])
    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    out = torch.empty(B, Hq, Lq, D, device=q.device, dtype=q.dtype)
    sh = _sh_tensor([B, Hq, Hkv, Lq, Lk, 1 if causal else 0], q.device)
    pr = _pr_tensor(scale, softcap, window_left, window_right, q.device)
    ntg_x = -(-Lq // 32)
    _get_v2_dtype_lib(D, mode).flash_attn_fwd_v2_dtype(
        qc, kc, vc, out, sh, pr,
        threads=(ntg_x * 128, B * Hq, 1), group_size=(128, 1, 1),
    )
    return out


def _flash_v2(q, k, v, scale, causal, softcap, window_left, window_right):
    """TensorOps matmul2d FA kernel. fp16 in/out, fp32 cooperative accumulation."""
    import torch.nn.functional as F

    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[2]
    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    out = torch.empty(B, Hq, Lq, D, device=q.device, dtype=torch.float16)
    pr = _pr_tensor(scale, softcap, window_left, window_right, q.device)
    lib = _get_v2_lib(D)
    if _v2_reuse_ok(D):
        # v2r: static-k PV reads full BC-row tiles — pad K/V to a BC multiple
        Lkp = -(-Lk // 32) * 32
        if Lkp != Lk:
            kc = F.pad(kc, (0, 0, 0, Lkp - Lk))
            vc = F.pad(vc, (0, 0, 0, Lkp - Lk))
        sh = _sh_tensor([B, Hq, Hkv, Lq, Lk, 1 if causal else 0, Lkp], q.device)
        rows_per_tg = _v2r_sr(D) * 4  # 4 simdgroups x SR rows
        ntg_x = -(-Lq // rows_per_tg)
        lib.flash_attn_fwd_v2r(
            qc, kc, vc, out, sh, pr,
            threads=(ntg_x * 128, B * Hq, 1), group_size=(128, 1, 1),
        )
        return out
    sh = _sh_tensor([B, Hq, Hkv, Lq, Lk, 1 if causal else 0], q.device)
    ntg_x = -(-Lq // 32)  # BR=32 query rows per threadgroup
    lib.flash_attn_fwd_v2(
        qc, kc, vc, out, sh, pr,
        threads=(ntg_x * 128, B * Hq, 1), group_size=(128, 1, 1),
    )
    return out


def _flash_v1(q, k, v, scale, causal, softcap, window_left, window_right):
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
    sh = _sh_tensor([B, Hq, Hkv, Lq, Lk, Lqp, Lkp, D, 1 if causal else 0], q.device)
    pr = _pr_tensor(scale, softcap, window_left, window_right, q.device)
    ntg_x = -(-Lqp // 32)  # 32 query rows per threadgroup (4 simdgroups x 8)
    _get_lib().flash_attn_fwd_v1(
        qc, kc, vc, out, sh, pr,
        threads=(ntg_x * 128, B * Hq, 1), group_size=(128, 1, 1),
    )
    return out


def _flash_v0(q, k, v, scale, causal, softcap, window_left, window_right):
    """One thread per query row, fp32 scalar online softmax. Memory-safe baseline."""
    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[2]
    qf = q.float().contiguous()
    kf = k.float().contiguous()
    vf = v.float().contiguous()
    out = torch.empty(B, Hq, Lq, D, device=q.device, dtype=torch.float32)
    sh = _sh_tensor([B, Hq, Hkv, Lq, Lk, D, 1 if causal else 0], q.device)
    pr = _pr_tensor(scale, softcap, window_left, window_right, q.device)
    _get_lib().flash_attn_fwd(
        qf, kf, vf, out, sh, pr,
        threads=(Lq, B * Hq, 1), group_size=(64, 1, 1),
    )
    return out.to(q.dtype)
