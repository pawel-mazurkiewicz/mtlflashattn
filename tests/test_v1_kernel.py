"""v1 simdgroup_matrix kernel: forced via MTLFLASHATTN_KERNEL=v1, same semantics as v0.

v1 is fp16-only (half simdgroup matmuls, fp32 softmax state / O accumulation), so
tolerances are looser than the fp32-accumulating v0: QK^T accumulates in half
(simdgroup_matrix can't mix half inputs with fp32 accumulation), giving score
errors ~sqrt(D)*2^-11*scale that exp() turns into ~1-2% output error on a few
elements. Exact-path users force MTLFLASHATTN_KERNEL=v0; fp32 QK^T accumulation
on M1-M4 would need MFA-style explicit-layout simdgroup_matrix_storage (v1.5).
"""
import math

import pytest
import torch

from test_flash_attn_func import ref_attention

mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires MPS"
)

V1_TOL = dict(atol=1.5e-2, rtol=2e-2)


@pytest.fixture(autouse=True)
def force_v1(monkeypatch):
    monkeypatch.setenv("MTLFLASHATTN_KERNEL", "v1")


def run_v1(B, Lq, Lk, Hq, Hkv, D, causal=False, scale=None, seed=0):
    from metal_flash_attn._kernel import flash_attn_forward

    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, Hq, Lq, D, generator=g).to("mps", torch.float16)
    k = torch.randn(B, Hkv, Lk, D, generator=g).to("mps", torch.float16)
    v = torch.randn(B, Hkv, Lk, D, generator=g).to("mps", torch.float16)
    s = scale if scale is not None else 1.0 / math.sqrt(D)
    out = flash_attn_forward(q, k, v, scale=s, causal=causal)
    assert out.shape == (B, Hq, Lq, D)
    assert out.dtype == torch.float16
    ref = ref_attention(q, k, v, causal=causal, scale=s)
    torch.testing.assert_close(out.float(), ref, **V1_TOL)


@mps_only
class TestV1Kernel:
    def test_basic(self):
        run_v1(2, 128, 128, 4, 4, 64)

    @pytest.mark.parametrize("causal", [False, True])
    def test_unaligned_seqlens(self, causal):
        # Lq, Lk not multiples of 8/32 — exercises padding + in-kernel masking
        run_v1(1, 67, 133, 2, 2, 64, causal=causal)

    @pytest.mark.parametrize("causal", [False, True])
    def test_cross_attn(self, causal):
        run_v1(1, 64, 256, 2, 2, 64, causal=causal)

    def test_causal_lq_gt_lk(self):
        run_v1(1, 96, 48, 2, 2, 64, causal=True)

    @pytest.mark.parametrize("d", [32, 64, 80, 128])
    def test_head_dims(self, d):
        run_v1(1, 64, 64, 2, 2, d)

    @pytest.mark.parametrize("hq,hkv", [(8, 2), (8, 1)])
    def test_gqa_mqa(self, hq, hkv):
        run_v1(1, 64, 64, hq, hkv, 64, causal=True)

    def test_custom_scale(self):
        run_v1(1, 64, 64, 2, 2, 64, scale=0.25)

    def test_long_sequence(self):
        run_v1(1, 2048, 2048, 2, 2, 64, causal=True)

    def test_tiny_sequence(self):
        run_v1(1, 3, 5, 1, 1, 64)

    def test_v1_matches_v0(self, monkeypatch):
        from metal_flash_attn._kernel import flash_attn_forward

        g = torch.Generator(device="cpu").manual_seed(7)
        q = torch.randn(1, 4, 200, 64, generator=g).to("mps", torch.float16)
        k = torch.randn(1, 4, 200, 64, generator=g).to("mps", torch.float16)
        v = torch.randn(1, 4, 200, 64, generator=g).to("mps", torch.float16)
        monkeypatch.setenv("MTLFLASHATTN_KERNEL", "v1")
        out1 = flash_attn_forward(q, k, v, scale=0.125, causal=True)
        monkeypatch.setenv("MTLFLASHATTN_KERNEL", "v0")
        out0 = flash_attn_forward(q, k, v, scale=0.125, causal=True)
        torch.testing.assert_close(out1.float(), out0.float(), **V1_TOL)

    def test_force_v1_on_ineligible_dtype_raises(self):
        from metal_flash_attn._kernel import flash_attn_forward

        q = torch.randn(1, 2, 32, 64, device="mps", dtype=torch.float32)
        with pytest.raises(RuntimeError):
            flash_attn_forward(q, q, q, scale=0.125, causal=False)

    def test_auto_tier_selection(self, monkeypatch):
        monkeypatch.setenv("MTLFLASHATTN_KERNEL", "auto")
        from metal_flash_attn import _kernel

        q = torch.randn(1, 2, 64, 64, device="mps", dtype=torch.float16)
        expected = "v2" if _kernel._v2_supported() else "v1"
        assert _kernel._select_tier(q, q, q) == expected
        qf = q.float()
        assert _kernel._select_tier(qf, qf, qf) == "v0"
        q33 = torch.randn(1, 2, 64, 33, device="mps", dtype=torch.float16)
        assert _kernel._select_tier(q33, q33, q33) == "v0"  # D not multiple of 8
