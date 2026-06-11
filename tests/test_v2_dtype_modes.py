"""Explicit fp32/bf16 TensorOps v2 modes.

These modes are deliberately opt-in while benchmarking decides whether either
dtype should be promoted to auto dispatch.
"""
import math

import pytest
import torch

from test_flash_attn_func import ref_attention

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires MPS"
)


def require_v2():
    from metal_flash_attn import _kernel

    if not _kernel._v2_supported():
        pytest.skip("v2 TensorOps kernel not supported on this machine")


def run_forced_dtype_mode(monkeypatch, mode, dtype, tol, seed=0, causal=False):
    from metal_flash_attn import _kernel

    require_v2()
    monkeypatch.setenv("MTLFLASHATTN_KERNEL", mode)

    def fail_v0(*args, **kwargs):
        raise AssertionError(f"{mode} must not fall back to scalar v0")

    def fail_torch(*args, **kwargs):
        raise AssertionError(f"{mode} must not fall back to chunked torch")

    monkeypatch.setattr(_kernel, "_flash_v0", fail_v0)
    monkeypatch.setattr(_kernel, "_flash_torch", fail_torch)

    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(1, 4, 67, 64, generator=g).to("mps", dtype)
    k = torch.randn(1, 2, 133, 64, generator=g).to("mps", dtype)
    v = torch.randn(1, 2, 133, 64, generator=g).to("mps", dtype)
    scale = 1.0 / math.sqrt(q.shape[-1])

    out = _kernel.flash_attn_forward(q, k, v, scale=scale, causal=causal)
    ref = ref_attention(q, k, v, scale=scale, causal=causal)

    assert out.shape == q.shape
    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), ref, **tol)


@pytest.mark.parametrize("causal", [False, True])
def test_forced_v2_fp32_matches_reference(monkeypatch, causal):
    run_forced_dtype_mode(
        monkeypatch,
        mode="v2_fp32",
        dtype=torch.float32,
        tol=dict(atol=2e-5, rtol=1e-4),
        seed=31,
        causal=causal,
    )


@pytest.mark.parametrize("causal", [False, True])
def test_forced_v2_bf16_matches_reference(monkeypatch, causal):
    run_forced_dtype_mode(
        monkeypatch,
        mode="v2_bf16",
        dtype=torch.bfloat16,
        tol=dict(atol=2e-2, rtol=2e-2),
        seed=37,
        causal=causal,
    )


@pytest.mark.parametrize(
    "dtype,tol,seed",
    [
        (torch.float16, dict(atol=4e-3, rtol=1e-2), 41),
        (torch.float32, dict(atol=2e-5, rtol=1e-4), 43),
        (torch.bfloat16, dict(atol=2e-2, rtol=2e-2), 47),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_forced_v2_dtype_matches_runtime_dtype(monkeypatch, dtype, tol, seed, causal):
    run_forced_dtype_mode(
        monkeypatch,
        mode="v2_dtype",
        dtype=dtype,
        tol=tol,
        seed=seed,
        causal=causal,
    )
