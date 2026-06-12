# mtlflashattn

**Memory-efficient, fast flash-attention for PyTorch on Apple Silicon (MPS)** — with a
guarded `flash_attn` drop-in shim so existing code uses it automatically.

`mtlflashattn` never materializes the `Lq×Lk` score matrix (fixing OOM/SIGKILL on long
attention), and on M5 / macOS 27 it drives Apple's **TensorOps / Neural Accelerators** via
`matmul2d` to run **3–11× faster than the stock fused MPS SDPA** — which, separately, is
*silently numerically wrong* past ~4k tokens. Pure-Python `torch.mps.compile_shader`; no C++/
ObjC/Swift extension, no `.metallib`, no `xcrun` at runtime.

> Likely the first working `matmul2d`/TensorOps flash-attention outside Apple.

---

## Why

- **OOM rescue.** Stock attention allocates `Lq×Lk` scores; at long sequence it blows past
  unified memory and the process is SIGKILL'd. This kernel is O(L·D) memory — it never forms
  the score matrix.
- **Correctness.** The MPS *fused* `scaled_dot_product_attention` diverges past ~4k tokens
  (per-element errors up to ~28 on real DiT q/k/v → grid artifacts / variance collapse). This
  kernel accumulates softmax/output state in fp32 and matches a chunked-fp32 reference to
  ~1e-4–6e-4 at 1k–32k tokens.
- **Speed.** On M5 it uses the Neural Accelerators (TensorOps `matmul2d`) with a
  register-resident online-softmax pipeline.
- **Zero-friction adoption.** A guarded `flash_attn` import shim means libraries that
  `from flash_attn import flash_attn_func` get the Metal version on Apple Silicon with no code
  change — while a real CUDA `flash_attn` install always wins.

## Requirements

- Apple Silicon Mac, `torch >= 2.5` (`torch.mps.compile_shader`).
- **v2 / v2r (TensorOps)**: M5+ and macOS 26+ (MetalPerformancePrimitives). Auto-detected.
- **v1 (`simdgroup_matrix`)**: any Apple Silicon (M1+), used when TensorOps is unavailable.
- **v0 / torch fallback**: anywhere MPS runs.

## Install

```bash
pip install mtlflashattn        # or: uv add mtlflashattn
```

The guarded `flash_attn` shim auto-activates at interpreter start via a `.pth` file. Kill it
with `MTLFLASHATTN_SHIM=off`.

## Usage

### 1. As the `flash_attn` drop-in (most code needs nothing)

```python
import torch
from flash_attn import flash_attn_func        # served by mtlflashattn on MPS

q = torch.randn(1, 8192, 16, 64, device="mps", dtype=torch.bfloat16)  # [B, S, H, D]
k = torch.randn_like(q); v = torch.randn_like(q)
out = flash_attn_func(q, k, v, causal=True)
```

Exposes the CUDA flash-attn surface: `flash_attn_func`, `flash_attn_varlen_func`,
`flash_attn_qkvpacked_func`, `flash_attn_kvpacked_func`, `flash_attn_varlen_qkvpacked_func`,
`flash_attn_varlen_kvpacked_func`, and the `flash_attn.flash_attn_interface` submodule.
Unsupported features (dropout, alibi, softcap, `return_attn_probs`, sliding window, D>128,
backward) **raise `NotImplementedError`** so callers fall back rather than get wrong results.

### 2. Direct API

```python
import metal_flash_attn as mfa
out = mfa.flash_attn_func(q, k, v, causal=True, softmax_scale=0.125)
```

### 3. As an `F.scaled_dot_product_attention` patch

```python
import metal_flash_attn.sdpa as sdpa
sdpa.install()    # reroutes F.scaled_dot_product_attention on MPS, gated; opt-in
...
sdpa.uninstall()
```

Fires on three gates: **correctness** (`max(Lq,Lk) ≥ MTLFLASHATTN_SDPA_MIN_SEQ`, default 4096),
**speed** (a fast TensorOps tier above `MTLFLASHATTN_SDPA_FAST_MIN_SEQ`, default 1024), and
**memory** (`MTLFLASHATTN_SDPA_MIN_GB`, default 12). Tiny attention stays on stock. Never
crashes the caller — any kernel error falls through to the original op.

## Kernel tiers

Selected automatically by dtype, head dim, and sequence length (`MTLFLASHATTN_KERNEL=auto`):

| Tier | Hardware | Notes |
|---|---|---|
| **v2r** | M5+ / macOS 27+ | TensorOps `matmul2d`, register-resident P (no threadgroup round-trip). Fastest. bf16 all D; fp32 D≤64; fp16 D≤64. Gated to Lk≥256. |
| **v2** | M5+ / macOS 26+ | TensorOps `matmul2d`, threadgroup-staged P. fp16/bf16/fp32 (`v2_fp32`/`v2_bf16`). |
| **v1** | M1+ | `simdgroup_matrix` 8×8 FA-2. fp16 fallback when TensorOps is unavailable. |
| **torch** | any MPS | Chunked fp32 matmul-softmax-matmul. The safe fp32 short-sequence path. |
| **v0** | any MPS | Scalar one-thread-per-row. Exact, memory-safe debug baseline. |

Force a tier with `MTLFLASHATTN_KERNEL=v0|v1|v2|v2_fp32|v2_bf16|v2_dtype|torch`.

## Benchmarks

M5 Max, macOS 27.0, torch 2.12, B=1 H=16, vs stock fused SDPA (**ratio = flash/stock, <1 is
faster**; effective TF/s in parens).

**fp16, non-causal:**

| shape | stock | v1 | v2 / v2r |
|---|---|---|---|
| D=64 L=8k | 92 ms (3.0) | 0.55× (5.4) | **0.09× (30.8 TF/s, v2r)** |
| D=128 L=8k | 101 ms (5.4) | 0.97× (5.6) | **0.29× (18.9 TF/s, v2)** |

**dtype-specialized v2r (register-resident P), effective TF/s:**

| shape | bf16 | fp32 | precision |
|---|---|---|---|
| D=64 L=8k | **30.6** (2.5× over TG round-trip) | **12.5** (1.45×) | fp32 bit-exact; bf16 ~bf16 noise |
| D=128 L=26k | **22.5** (1.23×) | 8.8 (TG) | bf16 ~bf16 noise |

**Causal** adds block-skipping the MPS path lacks: fp16 D=64 L=8k v2r ≈ **23 TF/s (~20× stock)**.

Reproduce: `python bench/bench_attn.py` (fp16) and `python dev/bench_dtype_kernels.py` (all dtypes).

## Diagnostics

`MTLFLASHATTN_TRACE=1` makes every `flash_attn_forward` accumulate per
`(dtype, D, Lq, Lk, causal, resolved-kernel)` call counts and print a summary to stderr at exit
— so you can see exactly which tier each stage of a real workload hits. Zero overhead when off.

```
[MTLFLASHATTN_TRACE] 4560 attention call(s), 6 distinct shape/kernel combos:
  calls=990    bfloat16  D=128  Lq=26136 Lk=26136 causal=F  -> v2r(bfloat)
  calls=990    bfloat16  D=128  Lq=26136 Lk=5     causal=F  -> v2_bf16(TG)
  ...
```

## Environment knobs

| Variable | Default | Effect |
|---|---|---|
| `MTLFLASHATTN_KERNEL` | `auto` | Force a tier. |
| `MTLFLASHATTN_SHIM` | `auto` | `off` disables the `flash_attn` import shim. |
| `MTLFLASHATTN_TRACE` | off | `1` prints a per-shape/tier call summary at exit. |
| `MTLFLASHATTN_V2_PREUSE` | `auto` | Force/disable the register-resident-P (v2r) path. |
| `MTLFLASHATTN_V2_FP32_MIN_SEQ` | `2048` | fp32 length gate: above → v2_fp32, below → torch fallback. |
| `MTLFLASHATTN_TORCH_CHUNK` | `2048` | Query-chunk size for the torch fallback. |
| `MTLFLASHATTN_SDPA` / `_MIN_GB` / `_MIN_SEQ` / `_FAST_MIN_SEQ` | — | SDPA-patch gating. |

## Scope

Inference forward pass only (no backward). Supports `softmax_scale`, bottom-right `causal`,
GQA/MQA, fp16/bf16/fp32, D≤128. Sliding window, dropout, alibi, softcap, KV-cache decode, and
FlexAttention are not implemented (the shim raises so callers fall back).

## How it works / engineering notes

The full build log — the TensorOps/MPP API gotchas discovered empirically on macOS 27 beta
(`tensor::slice()` reads wrong data as a `matmul2d` operand; operand extents must be clamped;
tensor destinations always accumulate; `reduce_rows` requires single-simdgroup scope and its
row-reduction output is per-thread *co-located* with the source — the reduced value for row `r`
sits on every lane that owns row `r`'s columns, with the reduction's row index at `idx[0]` vs the
source's at `idx[1]`; the register-resident P recipe forces the S element type to match the left
input), the per-tier design, and the speed/precision analysis are documented inline in the kernel
source (`metal_flash_attn/_kernel.py`).

## License

MIT.
