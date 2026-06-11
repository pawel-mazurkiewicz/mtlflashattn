"""Gated F.scaled_dot_product_attention patch: fires only above the byte budget,
falls back to stock for anything it can't do, install/uninstall is clean."""
import pytest
import torch
import torch.nn.functional as F

from test_flash_attn_func import TOL, ref_attention

mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires MPS"
)


@pytest.fixture
def sdpa_patch():
    """Install with a tiny threshold so small test shapes route to the kernel."""
    from metal_flash_attn import sdpa

    assert sdpa.install(min_score_gb=1e-9)
    yield sdpa
    sdpa.uninstall()


def make_bhld(B, H, Lq, Lk, D, dtype=torch.float16, Hkv=None, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, H, Lq, D, generator=g).to("mps", dtype)
    k = torch.randn(B, Hkv or H, Lk, D, generator=g).to("mps", dtype)
    v = torch.randn(B, Hkv or H, Lk, D, generator=g).to("mps", dtype)
    return q, k, v


@mps_only
class TestSdpaPatch:
    def test_install_uninstall_restores_original(self):
        from metal_flash_attn import sdpa

        orig = F.scaled_dot_product_attention
        assert sdpa.install(min_score_gb=1e-9)
        assert F.scaled_dot_product_attention is not orig
        assert not sdpa.install(min_score_gb=1e-9)  # idempotent
        sdpa.uninstall()
        assert F.scaled_dot_product_attention is orig

    def test_eligible_routes_to_kernel_and_matches_reference(self, sdpa_patch):
        q, k, v = make_bhld(1, 4, 128, 128, 64)
        out = F.scaled_dot_product_attention(q, k, v)
        ref = ref_attention(q, k, v)
        torch.testing.assert_close(out.float(), ref, **TOL[torch.float16])

    def test_causal_same_length(self, sdpa_patch):
        q, k, v = make_bhld(1, 2, 96, 96, 64, dtype=torch.float32)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ref = ref_attention(q, k, v, causal=True)
        torch.testing.assert_close(out.float(), ref, **TOL[torch.float32])

    def test_below_threshold_uses_stock(self):
        from metal_flash_attn import sdpa

        q, k, v = make_bhld(1, 2, 64, 64, 64)
        stock = F.scaled_dot_product_attention(q, k, v)
        assert sdpa.install()  # default 12 GB threshold — far above this shape
        try:
            out = F.scaled_dot_product_attention(q, k, v)
        finally:
            sdpa.uninstall()
        assert torch.equal(out, stock)  # bitwise => stock path taken

    def test_attn_mask_falls_back_to_stock(self, sdpa_patch):
        q, k, v = make_bhld(1, 2, 64, 64, 64)
        mask = torch.zeros(64, 64, device="mps", dtype=torch.float16)
        mask[:, 32:] = float("-inf")
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        stock = sdpa_patch._orig(q, k, v, attn_mask=mask)
        assert torch.equal(out, stock)

    def test_causal_cross_length_falls_back(self, sdpa_patch):
        # sdpa is_causal is TOP-LEFT aligned; kernel is bottom-right — must not route
        q, k, v = make_bhld(1, 2, 32, 96, 64)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        stock = sdpa_patch._orig(q, k, v, is_causal=True)
        assert torch.equal(out, stock)

    def test_dropout_falls_back(self, sdpa_patch):
        q, k, v = make_bhld(1, 2, 64, 64, 64)
        # dropout is stochastic — just confirm it doesn't crash and has right shape
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.5)
        assert out.shape == q.shape

    def test_gqa_routes_and_matches(self, sdpa_patch):
        q, k, v = make_bhld(1, 8, 64, 64, 64, Hkv=2)
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        ref = ref_attention(q, k, v)
        torch.testing.assert_close(out.float(), ref, **TOL[torch.float16])

    def test_custom_scale(self, sdpa_patch):
        q, k, v = make_bhld(1, 2, 64, 64, 64, dtype=torch.float32)
        out = F.scaled_dot_product_attention(q, k, v, scale=0.5)
        ref = ref_attention(q, k, v, scale=0.5)
        torch.testing.assert_close(out.float(), ref, **TOL[torch.float32])

    def test_kill_switch_blocks_install(self, monkeypatch):
        from metal_flash_attn import sdpa

        monkeypatch.setenv("MTLFLASHATTN_SDPA", "off")
        orig = F.scaled_dot_product_attention
        assert not sdpa.install(min_score_gb=1e-9)
        assert F.scaled_dot_product_attention is orig


def _require_v2():
    from metal_flash_attn import _kernel

    if not _kernel._v2_supported():
        pytest.skip("v2 TensorOps kernel not supported on this machine")


@mps_only
class TestSdpaGate:
    """The patch fires for fast TensorOps tiers even when memory fits, and on
    correctness grounds at large sequences (stock MPS fused SDPA is silently
    wrong past ~4k tokens), while keeping tiny attention on stock."""

    def test_fast_tier_fires_when_memory_fits(self):
        from metal_flash_attn import sdpa

        _require_v2()
        # fp16 D=64 -> v2; L=2048 fits well under 12 GB but is 3-4x faster on v2
        q, k, v = make_bhld(1, 4, 2048, 2048, 64)
        assert sdpa.install()  # DEFAULT 12 GB threshold
        try:
            ok, reason = sdpa._eligibility(q, k, v, None, 0.0, False)
        finally:
            sdpa.uninstall()
        assert ok and reason == "fast-tier", reason

    def test_tiny_fast_tier_stays_stock(self):
        from metal_flash_attn import sdpa

        # below the fast floor and the correctness floor, and it fits -> stock
        q, k, v = make_bhld(1, 2, 64, 64, 64)
        assert sdpa.install()  # default threshold
        try:
            ok, reason = sdpa._eligibility(q, k, v, None, 0.0, False)
        finally:
            sdpa.uninstall()
        assert not ok and reason.startswith("fits"), reason

    def test_large_seq_fires_for_correctness_on_slow_tier(self):
        from metal_flash_attn import sdpa

        # D=100 is not a multiple of 8 -> slow v0 tier, but at L=4096 stock is
        # numerically wrong, so route to our (exact) kernel anyway.
        q, k, v = make_bhld(1, 2, 4096, 4096, 100)
        assert sdpa.install()  # default 12 GB; score here is ~67 MB
        try:
            ok, reason = sdpa._eligibility(q, k, v, None, 0.0, False)
        finally:
            sdpa.uninstall()
        assert ok and reason == "correctness-large-seq", reason

    def test_fast_floor_is_env_configurable(self, monkeypatch):
        from metal_flash_attn import sdpa

        _require_v2()
        q, k, v = make_bhld(1, 2, 512, 512, 64)  # fp16 v2, L=512
        monkeypatch.setenv("MTLFLASHATTN_SDPA_FAST_MIN_SEQ", "256")
        assert sdpa.install()
        try:
            ok, reason = sdpa._eligibility(q, k, v, None, 0.0, False)
        finally:
            sdpa.uninstall()
        assert ok and reason == "fast-tier", reason

    def test_correctness_floor_is_env_configurable(self, monkeypatch):
        from metal_flash_attn import sdpa

        # slow tier (D=100), L=512: only fires if the correctness floor is lowered
        q, k, v = make_bhld(1, 2, 512, 512, 100)
        monkeypatch.setenv("MTLFLASHATTN_SDPA_MIN_SEQ", "256")
        assert sdpa.install()
        try:
            ok, reason = sdpa._eligibility(q, k, v, None, 0.0, False)
        finally:
            sdpa.uninstall()
        assert ok and reason == "correctness-large-seq", reason
