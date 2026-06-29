"""flash_attn_func vs a naive fp32 reference (bottom-right-aligned causal, like CUDA flash-attn)."""
import math

import pytest
import torch

mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires MPS"
)


def ref_attention(q, k, v, causal=False, scale=None, softcap=0.0, window_size=(-1, -1),
                  alibi_slopes=None):
    """Naive reference. q: [B,Hq,Lq,D], k/v: [B,Hkv,Lk,D] (heads-second). fp32.

    Causal is bottom-right aligned: query i attends key j iff j <= i + (Lk - Lq),
    matching CUDA flash-attn semantics (NOT torch sdpa's top-left is_causal).
    """
    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[2]
    if Hkv != Hq:
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    if softcap:
        s = softcap * torch.tanh(s / softcap)
    if alibi_slopes is not None:
        slopes = alibi_slopes.to(s.device, torch.float32)
        slopes = slopes.reshape(1, Hq, 1, 1) if slopes.dim() == 1 else slopes.reshape(B, Hq, 1, 1)
        ii = torch.arange(Lq, device=s.device)[:, None]
        jj = torch.arange(Lk, device=s.device)[None, :]
        s = s - slopes * (ii + (Lk - Lq) - jj).abs().float()
    wl, wr = window_size
    if causal or wl >= 0 or wr >= 0:
        i = torch.arange(Lq, device=q.device)[:, None]
        j = torch.arange(Lk, device=q.device)[None, :]
        center = i + (Lk - Lq)
        if causal:
            s = s.masked_fill(j > center, float("-inf"))
        if wl >= 0:
            s = s.masked_fill(j < center - wl, float("-inf"))
        if wr >= 0:
            s = s.masked_fill(j > center + wr, float("-inf"))
    p = torch.softmax(s, dim=-1)
    # fully-masked rows (Lq > Lk corner) produce nan; flash-attn outputs 0 there
    p = torch.nan_to_num(p, nan=0.0)
    return p @ v.float()


def make_qkv(B, Lq, Lk, Hq, Hkv, D, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, Lq, Hq, D, generator=g).to("mps", dtype)
    k = torch.randn(B, Lk, Hkv, D, generator=g).to("mps", dtype)
    v = torch.randn(B, Lk, Hkv, D, generator=g).to("mps", dtype)
    return q, k, v


# Tier-agnostic: fp16 may route through v1 (half-accumulated QK^T => ~1-2%
# worst-case on isolated elements); fp32 runs chunked PyTorch fallback, bf16 v0.
TOL = {
    torch.float32: dict(atol=2e-5, rtol=1e-4),
    torch.float16: dict(atol=1.5e-2, rtol=2e-2),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}


def check(q, k, v, causal=False, scale=None, softcap=0.0, window_size=(-1, -1),
          alibi_slopes=None, **tol_override):
    from metal_flash_attn import flash_attn_func

    out = flash_attn_func(
        q, k, v, softmax_scale=scale, causal=causal, softcap=softcap,
        window_size=window_size, alibi_slopes=alibi_slopes,
    )
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    ref = ref_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        causal=causal, scale=scale, softcap=softcap, window_size=window_size,
        alibi_slopes=alibi_slopes,
    ).transpose(1, 2)
    tol = {**TOL[q.dtype], **tol_override}
    torch.testing.assert_close(out.float(), ref, **tol)


@mps_only
class TestFlashAttnFunc:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_mha_self_attn(self, dtype):
        q, k, v = make_qkv(2, 128, 128, 4, 4, 64, dtype)
        check(q, k, v)

    @pytest.mark.parametrize("causal", [False, True])
    def test_cross_attn_lq_ne_lk(self, causal):
        q, k, v = make_qkv(1, 64, 192, 4, 4, 64, torch.float32)
        check(q, k, v, causal=causal)

    def test_causal_lq_gt_lk(self):
        q, k, v = make_qkv(1, 96, 48, 2, 2, 32, torch.float32)
        check(q, k, v, causal=True)

    @pytest.mark.parametrize("hq,hkv", [(8, 2), (8, 1)])
    def test_gqa_mqa(self, hq, hkv):
        q, k, v = make_qkv(2, 64, 64, hq, hkv, 64, torch.float16)
        check(q, k, v, causal=True)

    @pytest.mark.parametrize("d", [32, 64, 128])
    def test_head_dims(self, d):
        q, k, v = make_qkv(1, 64, 64, 2, 2, d, torch.float32)
        check(q, k, v)

    def test_custom_softmax_scale(self):
        q, k, v = make_qkv(1, 64, 64, 2, 2, 64, torch.float32)
        check(q, k, v, scale=0.25)
        # and confirm scale actually matters: wrong-scale ref must NOT match
        from metal_flash_attn import flash_attn_func

        out = flash_attn_func(q, k, v, softmax_scale=0.25)
        ref_wrong = ref_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=1.0
        ).transpose(1, 2)
        assert not torch.allclose(out.float(), ref_wrong, atol=1e-3)

    def test_head_dim_gt_128_raises(self):
        q, k, v = make_qkv(1, 16, 16, 1, 1, 160, torch.float32)
        from metal_flash_attn import flash_attn_func

        with pytest.raises(NotImplementedError):
            flash_attn_func(q, k, v)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(dropout_p=0.1),
            dict(deterministic=True),
            dict(return_attn_probs=True),
        ],
        ids=["dropout", "deterministic", "probs"],
    )
    def test_unsupported_kwargs_raise(self, kwargs):
        q, k, v = make_qkv(1, 32, 32, 4, 4, 64, torch.float16)
        from metal_flash_attn import flash_attn_func

        with pytest.raises(NotImplementedError):
            flash_attn_func(q, k, v, **kwargs)


@mps_only
class TestSoftcap:
    """Logit soft-capping s = c * tanh(s / c) applied to the scaled scores,
    honored by the auto path and every forced kernel tier."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_matches_reference(self, dtype):
        q, k, v = make_qkv(2, 128, 128, 4, 4, 64, dtype)
        check(q, k, v, softcap=30.0)

    def test_causal(self):
        q, k, v = make_qkv(1, 96, 96, 2, 2, 64, torch.float32)
        check(q, k, v, causal=True, softcap=20.0)

    def test_softcap_changes_output(self):
        # a tight cap must visibly saturate large logits vs no cap
        from metal_flash_attn import flash_attn_func

        q, k, v = make_qkv(1, 64, 64, 2, 2, 64, torch.float32)
        capped = flash_attn_func(q, k, v, softcap=3.0)
        uncapped = flash_attn_func(q, k, v)
        assert not torch.allclose(capped.float(), uncapped.float(), atol=1e-3)

    def test_negative_softcap_raises(self):
        from metal_flash_attn import flash_attn_func

        q, k, v = make_qkv(1, 32, 32, 2, 2, 64, torch.float16)
        with pytest.raises(ValueError):
            flash_attn_func(q, k, v, softcap=-1.0)

    # (kernel, dtype, D): D=64 exercises the register-resident v2r path, D=128 the
    # threadgroup-staged path — covering every shader that applies softcap.
    @pytest.mark.parametrize(
        "kernel,dtype,d",
        [
            ("torch", torch.float32, 64),
            ("v0", torch.float32, 64),
            ("v1", torch.float16, 64),
            ("v2", torch.float16, 64),
            ("v2", torch.float16, 128),
            ("v2_fp32", torch.float32, 64),
            ("v2_fp32", torch.float32, 128),
            ("v2_bf16", torch.bfloat16, 64),
            ("v2_bf16", torch.bfloat16, 128),
        ],
    )
    def test_per_tier_honors_softcap(self, monkeypatch, kernel, dtype, d):
        from metal_flash_attn import _kernel, flash_attn_func

        if kernel in ("v2", "v2_fp32", "v2_bf16") and not _kernel._v2_supported():
            pytest.skip("TensorOps v2 unavailable")
        monkeypatch.setenv("MTLFLASHATTN_KERNEL", kernel)
        q, k, v = make_qkv(1, 256, 256, 2, 2, d, dtype)
        out = flash_attn_func(q, k, v, softcap=5.0)
        ref = ref_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), softcap=5.0,
        ).transpose(1, 2)
        torch.testing.assert_close(out.float(), ref, **TOL[dtype])


@mps_only
class TestSlidingWindow:
    """Sliding-window attention: key j is valid for query i iff
    j in [center - left, center + right], center = i + (Lk - Lq); -1 = unbounded.
    Honored by the auto path and every forced kernel tier."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_matches_reference(self, dtype):
        q, k, v = make_qkv(2, 256, 256, 4, 4, 64, dtype)
        check(q, k, v, window_size=(48, 0))

    def test_symmetric_window(self):
        q, k, v = make_qkv(1, 128, 128, 2, 2, 64, torch.float32)
        check(q, k, v, window_size=(16, 16))

    def test_causal_plus_window(self):
        # Mistral-style: causal cap + bounded left context
        q, k, v = make_qkv(1, 200, 200, 2, 2, 64, torch.float32)
        check(q, k, v, causal=True, window_size=(32, -1))

    def test_cross_length_window(self):
        q, k, v = make_qkv(1, 64, 192, 2, 2, 64, torch.float32)
        check(q, k, v, window_size=(40, 8))

    def test_window_changes_output(self):
        from metal_flash_attn import flash_attn_func

        q, k, v = make_qkv(1, 128, 128, 2, 2, 64, torch.float32)
        narrow = flash_attn_func(q, k, v, window_size=(8, 0))
        full = flash_attn_func(q, k, v)
        assert not torch.allclose(narrow.float(), full.float(), atol=1e-3)

    def test_window_wider_than_seq_is_full(self):
        # a window spanning the whole sequence == unmasked attention
        q, k, v = make_qkv(1, 64, 64, 2, 2, 64, torch.float32)
        check(q, k, v, window_size=(1000, 1000))

    def test_bad_window_raises(self):
        from metal_flash_attn import flash_attn_func

        q, k, v = make_qkv(1, 32, 32, 2, 2, 64, torch.float16)
        with pytest.raises(ValueError):
            flash_attn_func(q, k, v, window_size=(-2, 0))

    @pytest.mark.parametrize(
        "kernel,dtype,d",
        [
            ("torch", torch.float32, 64),
            ("v0", torch.float32, 64),
            ("v1", torch.float16, 64),
            ("v2", torch.float16, 64),
            ("v2", torch.float16, 128),
            ("v2_fp32", torch.float32, 64),
            ("v2_fp32", torch.float32, 128),
            ("v2_bf16", torch.bfloat16, 64),
            ("v2_bf16", torch.bfloat16, 128),
        ],
    )
    def test_per_tier_honors_window(self, monkeypatch, kernel, dtype, d):
        from metal_flash_attn import _kernel, flash_attn_func

        if kernel in ("v2", "v2_fp32", "v2_bf16") and not _kernel._v2_supported():
            pytest.skip("TensorOps v2 unavailable")
        monkeypatch.setenv("MTLFLASHATTN_KERNEL", kernel)
        q, k, v = make_qkv(1, 256, 256, 2, 2, d, dtype)
        win = (64, 16)
        out = flash_attn_func(q, k, v, window_size=win)
        ref = ref_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), window_size=win,
        ).transpose(1, 2)
        torch.testing.assert_close(out.float(), ref, **TOL[dtype])


@mps_only
class TestAlibi:
    """ALiBi: per-head linear position bias s -= slope_h * |center - keypos|,
    center = qpos + (Lk - Lq); matches flash-attn for causal and non-causal.
    Honored by the auto path and every forced kernel tier."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_matches_reference(self, dtype):
        q, k, v = make_qkv(2, 128, 128, 4, 4, 64, dtype)
        slopes = torch.tensor([0.5, 0.25, 0.125, 0.0625], device="mps")
        check(q, k, v, alibi_slopes=slopes)

    def test_alibi_causal(self):
        q, k, v = make_qkv(1, 96, 96, 4, 4, 64, torch.float32)
        slopes = torch.tensor([0.5, 0.25, 0.125, 0.0625], device="mps")
        check(q, k, v, causal=True, alibi_slopes=slopes)

    def test_alibi_cross_length(self):
        q, k, v = make_qkv(1, 64, 192, 2, 2, 64, torch.float32)
        check(q, k, v, alibi_slopes=torch.tensor([0.3, 0.1], device="mps"))

    def test_batched_slopes(self):
        # alibi_slopes shaped [B, Hq]
        q, k, v = make_qkv(2, 96, 96, 2, 2, 64, torch.float32)
        slopes = torch.tensor([[0.5, 0.25], [0.1, 0.05]], device="mps")
        check(q, k, v, alibi_slopes=slopes)

    def test_alibi_changes_output(self):
        from metal_flash_attn import flash_attn_func

        q, k, v = make_qkv(1, 128, 128, 2, 2, 64, torch.float32)
        biased = flash_attn_func(q, k, v, alibi_slopes=torch.tensor([0.5, 0.5], device="mps"))
        plain = flash_attn_func(q, k, v)
        assert not torch.allclose(biased.float(), plain.float(), atol=1e-3)

    def test_bad_alibi_shape_raises(self):
        from metal_flash_attn import flash_attn_func

        q, k, v = make_qkv(1, 32, 32, 4, 4, 64, torch.float16)
        with pytest.raises(ValueError):
            flash_attn_func(q, k, v, alibi_slopes=torch.ones(3, device="mps"))  # Hq=4

    def test_alibi_non_tensor_raises(self):
        from metal_flash_attn import flash_attn_func

        q, k, v = make_qkv(1, 32, 32, 4, 4, 64, torch.float16)
        with pytest.raises(TypeError):
            flash_attn_func(q, k, v, alibi_slopes=[0.5, 0.25, 0.125, 0.0625])

    @pytest.mark.parametrize(
        "kernel,dtype,d",
        [
            ("torch", torch.float32, 64),
            ("v0", torch.float32, 64),
            ("v1", torch.float16, 64),
            ("v2", torch.float16, 64),
            ("v2", torch.float16, 128),
            ("v2_fp32", torch.float32, 64),
            ("v2_fp32", torch.float32, 128),
            ("v2_bf16", torch.bfloat16, 64),
            ("v2_bf16", torch.bfloat16, 128),
        ],
    )
    def test_per_tier_honors_alibi(self, monkeypatch, kernel, dtype, d):
        from metal_flash_attn import _kernel, flash_attn_func

        if kernel in ("v2", "v2_fp32", "v2_bf16") and not _kernel._v2_supported():
            pytest.skip("TensorOps v2 unavailable")
        monkeypatch.setenv("MTLFLASHATTN_KERNEL", kernel)
        q, k, v = make_qkv(1, 256, 256, 2, 2, d, dtype)
        slopes = torch.tensor([0.1, 0.05], device="mps")
        out = flash_attn_func(q, k, v, alibi_slopes=slopes)
        ref = ref_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), alibi_slopes=slopes,
        ).transpose(1, 2)
        torch.testing.assert_close(out.float(), ref, **TOL[dtype])
