import math

import torch


def ref_attention(q, k, v, causal=False, scale=None):
    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[2]
    if Hkv != Hq:
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = (q.float() @ k.float().transpose(-1, -2)) * scale
    if causal:
        i = torch.arange(Lq, device=q.device)[:, None]
        j = torch.arange(Lk, device=q.device)[None, :]
        scores = scores.masked_fill(j > i + (Lk - Lq), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return probs @ v.float()


def test_auto_float32_routes_to_chunked_torch_fallback(monkeypatch):
    from metal_flash_attn import _kernel

    monkeypatch.setenv("MTLFLASHATTN_KERNEL", "auto")
    monkeypatch.setenv("MTLFLASHATTN_TORCH_CHUNK", "13")

    def fail_v0(*args, **kwargs):
        raise AssertionError("auto fp32 dispatch must not use scalar v0")

    monkeypatch.setattr(_kernel, "_flash_v0", fail_v0)

    g = torch.Generator(device="cpu").manual_seed(11)
    q = torch.randn(1, 4, 37, 32, generator=g, dtype=torch.float32)
    k = torch.randn(1, 2, 53, 32, generator=g, dtype=torch.float32)
    v = torch.randn(1, 2, 53, 32, generator=g, dtype=torch.float32)
    scale = 1.0 / math.sqrt(q.shape[-1])

    out = _kernel.flash_attn_forward(q, k, v, scale=scale, causal=True)
    ref = ref_attention(q, k, v, scale=scale, causal=True)

    assert out.dtype == torch.float32
    assert out.shape == q.shape
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=1e-4)


def test_auto_tier_selects_torch_for_float32():
    from metal_flash_attn import _kernel

    q = torch.randn(1, 2, 16, 64, dtype=torch.float32)
    assert _kernel._select_tier(q, q, q) == "torch"


def require_v2():
    from metal_flash_attn import _kernel

    if not _kernel._v2_supported():
        import pytest

        pytest.skip("v2 TensorOps kernel not supported on this machine")


def test_auto_tier_promotes_long_fp32_to_v2_fp32():
    from metal_flash_attn import _kernel

    require_v2()
    long_q = torch.randn(1, 2, 4096, 64, dtype=torch.float32)
    assert _kernel._select_tier(long_q, long_q, long_q) == "v2_fp32"


def test_auto_tier_keeps_short_fp32_on_torch():
    from metal_flash_attn import _kernel

    require_v2()
    short_q = torch.randn(1, 2, 512, 64, dtype=torch.float32)
    assert _kernel._select_tier(short_q, short_q, short_q) == "torch"


def test_v2_fp32_length_gate_is_env_configurable(monkeypatch):
    from metal_flash_attn import _kernel

    require_v2()
    q = torch.randn(1, 2, 256, 64, dtype=torch.float32)
    monkeypatch.setenv("MTLFLASHATTN_V2_FP32_MIN_SEQ", "128")
    assert _kernel._select_tier(q, q, q) == "v2_fp32"
    monkeypatch.setenv("MTLFLASHATTN_V2_FP32_MIN_SEQ", "100000")
    assert _kernel._select_tier(q, q, q) == "torch"


def test_long_fp32_with_ineligible_head_dim_stays_on_torch():
    from metal_flash_attn import _kernel

    require_v2()
    # D=33 is not a multiple of 8 -> TensorOps ineligible, must stay on torch
    q = torch.randn(1, 2, 4096, 33, dtype=torch.float32)
    assert _kernel._select_tier(q, q, q) == "torch"


def test_auto_long_fp32_dispatches_v2_fp32_and_matches_reference(monkeypatch):
    from metal_flash_attn import _kernel

    require_v2()
    monkeypatch.setenv("MTLFLASHATTN_KERNEL", "auto")
    monkeypatch.setenv("MTLFLASHATTN_V2_FP32_MIN_SEQ", "256")

    def fail_torch(*args, **kwargs):
        raise AssertionError("auto long fp32 must use v2_fp32, not chunked torch")

    def fail_v0(*args, **kwargs):
        raise AssertionError("auto long fp32 must use v2_fp32, not scalar v0")

    monkeypatch.setattr(_kernel, "_flash_torch", fail_torch)
    monkeypatch.setattr(_kernel, "_flash_v0", fail_v0)

    g = torch.Generator(device="cpu").manual_seed(17)
    q = torch.randn(1, 4, 384, 64, generator=g).to("mps", torch.float32)
    k = torch.randn(1, 2, 384, 64, generator=g).to("mps", torch.float32)
    v = torch.randn(1, 2, 384, 64, generator=g).to("mps", torch.float32)
    scale = 1.0 / math.sqrt(q.shape[-1])

    out = _kernel.flash_attn_forward(q, k, v, scale=scale, causal=True)
    ref = ref_attention(q, k, v, scale=scale, causal=True)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out.float(), ref, atol=2e-5, rtol=1e-4)


def test_forced_v0_keeps_debug_path(monkeypatch):
    from metal_flash_attn import _kernel

    monkeypatch.setenv("MTLFLASHATTN_KERNEL", "v0")

    q = torch.randn(1, 2, 5, 8, dtype=torch.float32)
    sentinel = torch.full_like(q, 7.0)
    calls = []

    def fake_v0(*args, **kwargs):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(_kernel, "_flash_v0", fake_v0)

    out = _kernel.flash_attn_forward(q, q, q, scale=0.25, causal=False)

    assert calls
    assert out is sentinel


def test_auto_bfloat16_stays_on_v0_until_tensorops_promotion(monkeypatch):
    from metal_flash_attn import _kernel

    monkeypatch.setenv("MTLFLASHATTN_KERNEL", "auto")

    q = torch.randn(1, 2, 5, 64, dtype=torch.bfloat16)
    sentinel = torch.full_like(q, 3.0)
    calls = []

    def fake_v0(*args, **kwargs):
        calls.append(args)
        return sentinel

    def fail_torch(*args, **kwargs):
        raise AssertionError("auto bf16 must not use the fp32 torch fallback")

    def fail_v1(*args, **kwargs):
        raise AssertionError("auto bf16 must not use fp16 v1")

    def fail_v2(*args, **kwargs):
        raise AssertionError("auto bf16 must not use fp16 v2")

    monkeypatch.setattr(_kernel, "_flash_v0", fake_v0)
    monkeypatch.setattr(_kernel, "_flash_torch", fail_torch)
    monkeypatch.setattr(_kernel, "_flash_v1", fail_v1)
    monkeypatch.setattr(_kernel, "_flash_v2", fail_v2)

    out = _kernel.flash_attn_forward(q, q, q, scale=0.125, causal=False)

    assert calls
    assert out is sentinel
