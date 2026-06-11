"""qkvpacked / kvpacked thin wrappers must match the unpacked entry point."""
import pytest
import torch

mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires MPS"
)


@mps_only
class TestPackedFuncs:
    def test_qkvpacked_matches_unpacked(self):
        from metal_flash_attn import flash_attn_func, flash_attn_qkvpacked_func

        g = torch.Generator(device="cpu").manual_seed(0)
        qkv = torch.randn(2, 96, 3, 4, 64, generator=g).to("mps", torch.float16)
        out_p = flash_attn_qkvpacked_func(qkv, causal=True)
        q, k, v = qkv.unbind(dim=2)
        out_u = flash_attn_func(q, k, v, causal=True)
        assert out_p.shape == (2, 96, 4, 64)
        torch.testing.assert_close(out_p, out_u)

    def test_kvpacked_matches_unpacked_gqa(self):
        from metal_flash_attn import flash_attn_func, flash_attn_kvpacked_func

        g = torch.Generator(device="cpu").manual_seed(1)
        q = torch.randn(1, 64, 8, 64, generator=g).to("mps", torch.float16)
        kv = torch.randn(1, 128, 2, 2, 64, generator=g).to("mps", torch.float16)
        out_p = flash_attn_kvpacked_func(q, kv, softmax_scale=0.1)
        k, v = kv.unbind(dim=2)
        out_u = flash_attn_func(q, k, v, softmax_scale=0.1)
        assert out_p.shape == q.shape
        torch.testing.assert_close(out_p, out_u)

    def test_packed_shape_validation(self):
        from metal_flash_attn import flash_attn_kvpacked_func, flash_attn_qkvpacked_func

        bad_qkv = torch.randn(1, 32, 2, 4, 64, device="mps", dtype=torch.float16)
        with pytest.raises(ValueError):
            flash_attn_qkvpacked_func(bad_qkv)
        q = torch.randn(1, 32, 4, 64, device="mps", dtype=torch.float16)
        bad_kv = torch.randn(1, 32, 3, 4, 64, device="mps", dtype=torch.float16)
        with pytest.raises(ValueError):
            flash_attn_kvpacked_func(q, bad_kv)

    def test_unsupported_kwargs_propagate(self):
        from metal_flash_attn import flash_attn_qkvpacked_func

        qkv = torch.randn(1, 32, 3, 4, 64, device="mps", dtype=torch.float16)
        with pytest.raises(NotImplementedError):
            flash_attn_qkvpacked_func(qkv, dropout_p=0.5)

    def test_varlen_qkvpacked_matches_varlen(self):
        from metal_flash_attn import flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func

        g = torch.Generator(device="cpu").manual_seed(2)
        qkv = torch.randn(37 + 81, 3, 4, 64, generator=g).to("mps", torch.float16)
        cu = torch.tensor([0, 37, 118], dtype=torch.int32, device="mps")
        out_p = flash_attn_varlen_qkvpacked_func(qkv, cu, 81)
        q, k, v = qkv.unbind(dim=1)
        out_u = flash_attn_varlen_func(q, k, v, cu, cu, 81, 81)
        assert out_p.shape == (118, 4, 64)
        torch.testing.assert_close(out_p, out_u)

    def test_varlen_kvpacked_matches_varlen(self):
        from metal_flash_attn import flash_attn_varlen_func, flash_attn_varlen_kvpacked_func

        g = torch.Generator(device="cpu").manual_seed(3)
        q = torch.randn(50, 4, 64, generator=g).to("mps", torch.float16)
        kv = torch.randn(90, 2, 4, 64, generator=g).to("mps", torch.float16)
        cu_q = torch.tensor([0, 20, 50], dtype=torch.int32, device="mps")
        cu_k = torch.tensor([0, 60, 90], dtype=torch.int32, device="mps")
        out_p = flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_k, 30, 60)
        k, v = kv.unbind(dim=1)
        out_u = flash_attn_varlen_func(q, k, v, cu_q, cu_k, 30, 60)
        assert out_p.shape == q.shape
        torch.testing.assert_close(out_p, out_u)

    def test_varlen_packed_shape_validation(self):
        from metal_flash_attn import flash_attn_varlen_qkvpacked_func

        bad = torch.randn(32, 2, 4, 64, device="mps", dtype=torch.float16)
        cu = torch.tensor([0, 32], dtype=torch.int32, device="mps")
        with pytest.raises(ValueError):
            flash_attn_varlen_qkvpacked_func(bad, cu, 32)

    def test_shim_exposes_packed(self):
        import subprocess
        import sys

        r = subprocess.run(
            [sys.executable, "-c",
             "from flash_attn import flash_attn_qkvpacked_func, flash_attn_kvpacked_func, "
             "flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func; print('ok')"],
            capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "ok"
