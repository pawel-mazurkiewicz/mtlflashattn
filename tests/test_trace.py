"""MTLFLASHATTN_TRACE: aggregate per-shape/kernel call accounting for diagnosing
which tier each stage of a real workload actually hits."""
import math

import pytest
import torch

mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires MPS"
)


def require_v2():
    from metal_flash_attn import _kernel

    if not _kernel._v2_supported():
        pytest.skip("v2 TensorOps kernel not supported on this machine")


@mps_only
def test_trace_disabled_by_default_records_nothing(monkeypatch):
    from metal_flash_attn import _kernel

    monkeypatch.delenv("MTLFLASHATTN_TRACE", raising=False)
    _kernel._trace.clear()
    q = torch.randn(1, 2, 64, 64, device="mps", dtype=torch.float16)
    _kernel.flash_attn_forward(q, q, q, scale=0.125, causal=False)
    assert _kernel._trace == {}


@mps_only
def test_trace_records_dtype_shape_and_tier(monkeypatch):
    from metal_flash_attn import _kernel

    monkeypatch.setenv("MTLFLASHATTN_TRACE", "1")
    _kernel._trace.clear()

    q = torch.randn(1, 4, 128, 64, device="mps", dtype=torch.float16)
    _kernel.flash_attn_forward(q, q, q, scale=0.125, causal=False)
    _kernel.flash_attn_forward(q, q, q, scale=0.125, causal=False)

    keys = list(_kernel._trace)
    assert len(keys) == 1
    dt, D, Lq, Lk, causal, label = keys[0]
    assert dt == "float16" and D == 64 and Lq == 128 and Lk == 128
    assert _kernel._trace[keys[0]] == 2  # accumulates
    assert _trace_summary_has(_kernel, "float16")


@mps_only
def test_trace_label_matches_dispatch_for_dtypes(monkeypatch):
    from metal_flash_attn import _kernel

    require_v2()
    monkeypatch.delenv("MTLFLASHATTN_KERNEL", raising=False)
    monkeypatch.setenv("MTLFLASHATTN_V2_FP32_MIN_SEQ", "2048")

    # bf16 D<=64 -> register-resident v2r
    qb = torch.randn(1, 2, 256, 64, device="mps", dtype=torch.bfloat16)
    assert _kernel._effective_kernel_label(qb, qb, qb) == "v2r(bfloat)"

    # fp32 long seq D<=64 -> v2r(float); short seq -> torch
    qf_long = torch.randn(1, 2, 4096, 64, device="mps", dtype=torch.float32)
    qf_short = torch.randn(1, 2, 256, 64, device="mps", dtype=torch.float32)
    assert _kernel._effective_kernel_label(qf_long, qf_long, qf_long) == "v2r(float)"
    assert _kernel._effective_kernel_label(qf_short, qf_short, qf_short) == "torch"

    # bf16 D=128 -> v2r too (1.23x over TG round-trip)
    qb128 = torch.randn(1, 2, 256, 128, device="mps", dtype=torch.bfloat16)
    assert _kernel._effective_kernel_label(qb128, qb128, qb128) == "v2r(bfloat)"

    # fp32 D=128 -> threadgroup round-trip (v2r loses on fp32 register pressure)
    qf128 = torch.randn(1, 2, 4096, 128, device="mps", dtype=torch.float32)
    assert _kernel._effective_kernel_label(qf128, qf128, qf128) == "v2_fp32(TG)"


def _trace_summary_has(_kernel, needle):
    return any(needle in line for line in _kernel._trace_summary_lines())
