"""v2 TensorOps (Metal 4 / MPP) kernel: forced via MTLFLASHATTN_KERNEL=v2.

Requires macOS 26+ with MetalPerformancePrimitives headers; runs the Neural
Accelerator on M5+, degrades to the simdgroup path on M1-M4. QK^T and PV
accumulate in fp32 cooperative tensors, so tolerances are v0-like for fp16.
"""
import math

import pytest
import torch

from test_flash_attn_func import ref_attention

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires MPS"
)

V2_TOL = dict(atol=4e-3, rtol=1e-2)


def v2_supported():
    from metal_flash_attn import _kernel

    return _kernel._v2_supported()


@pytest.fixture(autouse=True)
def force_v2(monkeypatch):
    if not v2_supported():
        pytest.skip("v2 TensorOps kernel not supported on this machine")
    monkeypatch.setenv("MTLFLASHATTN_KERNEL", "v2")


def run_v2(B, Lq, Lk, Hq, Hkv, D, causal=False, scale=None, seed=0):
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
    torch.testing.assert_close(out.float(), ref, **V2_TOL)


class TestV2Kernel:
    def test_basic(self):
        run_v2(2, 128, 128, 4, 4, 64)

    @pytest.mark.parametrize("causal", [False, True])
    def test_unaligned_seqlens(self, causal):
        run_v2(1, 67, 133, 2, 2, 64, causal=causal)

    @pytest.mark.parametrize("causal", [False, True])
    def test_cross_attn(self, causal):
        run_v2(1, 64, 256, 2, 2, 64, causal=causal)

    def test_causal_lq_gt_lk(self):
        run_v2(1, 96, 48, 2, 2, 64, causal=True)

    @pytest.mark.parametrize("d", [32, 64, 128])
    def test_head_dims(self, d):
        run_v2(1, 64, 64, 2, 2, d)

    @pytest.mark.parametrize("hq,hkv", [(8, 2), (8, 1)])
    def test_gqa_mqa(self, hq, hkv):
        run_v2(1, 64, 64, hq, hkv, 64, causal=True)

    def test_custom_scale(self):
        run_v2(1, 64, 64, 2, 2, 64, scale=0.25)

    def test_long_sequence(self):
        run_v2(1, 2048, 2048, 2, 2, 64, causal=True)

    def test_tiny_sequence(self):
        run_v2(1, 3, 5, 1, 1, 64)

    def test_preuse_paths_match(self, monkeypatch):
        """P-reuse path (no TG round-trip) must agree with the TG-round-trip path."""
        from metal_flash_attn import _kernel

        if not _kernel._v2_reuse_ok(64):
            pytest.skip("cooperative left-input reuse not supported here")
        g = torch.Generator(device="cpu").manual_seed(3)
        q = torch.randn(1, 4, 300, 64, generator=g).to("mps", torch.float16)
        k = torch.randn(1, 4, 300, 64, generator=g).to("mps", torch.float16)
        v = torch.randn(1, 4, 300, 64, generator=g).to("mps", torch.float16)
        monkeypatch.setenv("MTLFLASHATTN_V2_PREUSE", "1")
        out_r = _kernel.flash_attn_forward(q, k, v, scale=0.125, causal=True)
        monkeypatch.setenv("MTLFLASHATTN_V2_PREUSE", "0")
        out_t = _kernel.flash_attn_forward(q, k, v, scale=0.125, causal=True)
        # v2r accumulates S in half (API constraint) => v1-like tolerance
        torch.testing.assert_close(out_r.float(), out_t.float(), atol=1.5e-2, rtol=2e-2)

    def test_v2_matches_v0(self, monkeypatch):
        from metal_flash_attn._kernel import flash_attn_forward

        g = torch.Generator(device="cpu").manual_seed(7)
        q = torch.randn(1, 4, 200, 64, generator=g).to("mps", torch.float16)
        k = torch.randn(1, 4, 200, 64, generator=g).to("mps", torch.float16)
        v = torch.randn(1, 4, 200, 64, generator=g).to("mps", torch.float16)
        monkeypatch.setenv("MTLFLASHATTN_KERNEL", "v2")
        out2 = flash_attn_forward(q, k, v, scale=0.125, causal=True)
        monkeypatch.setenv("MTLFLASHATTN_KERNEL", "v0")
        out0 = flash_attn_forward(q, k, v, scale=0.125, causal=True)
        torch.testing.assert_close(out2.float(), out0.float(), **V2_TOL)
