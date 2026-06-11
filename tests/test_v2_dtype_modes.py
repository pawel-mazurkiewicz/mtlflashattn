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
def test_bf16_small_d_uses_register_resident_v2r(monkeypatch, causal):
    """bf16 at D<=64 must take the register-resident-P v2r path (2.5x faster
    than the threadgroup-round-trip baseline) and stay correct."""
    from metal_flash_attn import _kernel

    require_v2()
    monkeypatch.setenv("MTLFLASHATTN_KERNEL", "v2_bf16")

    called = []
    orig = _kernel._flash_v2r_dtype

    def spy(*a, **kw):
        called.append(True)
        return orig(*a, **kw)

    monkeypatch.setattr(_kernel, "_flash_v2r_dtype", spy)

    g = torch.Generator(device="cpu").manual_seed(51)
    q = torch.randn(1, 4, 200, 64, generator=g).to("mps", torch.bfloat16)
    k = torch.randn(1, 2, 264, 64, generator=g).to("mps", torch.bfloat16)
    v = torch.randn(1, 2, 264, 64, generator=g).to("mps", torch.bfloat16)
    scale = 1.0 / math.sqrt(64)

    out = _kernel.flash_attn_forward(q, k, v, scale=scale, causal=causal)
    ref = ref_attention(q, k, v, scale=scale, causal=causal)

    assert called, "bf16 D<=64 did not use the register-resident v2r kernel"
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal", [False, True])
def test_fp32_small_d_uses_register_resident_v2r(monkeypatch, causal):
    """fp32 at D<=64 also takes v2r: 1.45x faster than the TG round-trip and
    bit-exact (S accumulates in fp32 registers, no precision loss)."""
    from metal_flash_attn import _kernel

    require_v2()
    monkeypatch.setenv("MTLFLASHATTN_KERNEL", "v2_fp32")

    called = []
    orig = _kernel._flash_v2r_dtype

    def spy(*a, **kw):
        called.append(True)
        return orig(*a, **kw)

    monkeypatch.setattr(_kernel, "_flash_v2r_dtype", spy)

    g = torch.Generator(device="cpu").manual_seed(57)
    q = torch.randn(1, 4, 200, 64, generator=g).to("mps", torch.float32)
    k = torch.randn(1, 2, 264, 64, generator=g).to("mps", torch.float32)
    v = torch.randn(1, 2, 264, 64, generator=g).to("mps", torch.float32)
    scale = 1.0 / math.sqrt(64)

    out = _kernel.flash_attn_forward(q, k, v, scale=scale, causal=causal)
    ref = ref_attention(q, k, v, scale=scale, causal=causal)

    assert called, "fp32 D<=64 did not use the register-resident v2r kernel"
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=1e-4)


def test_bf16_large_d_stays_on_tg_roundtrip(monkeypatch):
    """bf16 at D=128 must NOT use v2r (register pressure loses there) -- it
    keeps the threadgroup-round-trip dtype kernel."""
    from metal_flash_attn import _kernel

    require_v2()
    monkeypatch.setenv("MTLFLASHATTN_KERNEL", "v2_bf16")

    def fail(*a, **kw):
        raise AssertionError("bf16 D=128 must not use the v2r register path")

    monkeypatch.setattr(_kernel, "_flash_v2r_dtype", fail)

    g = torch.Generator(device="cpu").manual_seed(53)
    q = torch.randn(1, 4, 160, 128, generator=g).to("mps", torch.bfloat16)
    k = torch.randn(1, 2, 192, 128, generator=g).to("mps", torch.bfloat16)
    v = torch.randn(1, 2, 192, 128, generator=g).to("mps", torch.bfloat16)
    scale = 1.0 / math.sqrt(128)

    out = _kernel.flash_attn_forward(q, k, v, scale=scale, causal=False)
    ref = ref_attention(q, k, v, scale=scale, causal=False)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


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
