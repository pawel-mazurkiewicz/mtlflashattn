"""Guarded flash_attn shim: real install always wins, lazy, kill-switchable.

Each test runs a fresh interpreter (the shim is import-machinery state).
"""
import os
import subprocess
import sys
import textwrap

import pytest
import torch

mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires MPS"
)


def run_py(code, env_extra=None):
    env = os.environ.copy()
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, env=env, timeout=120,
    )


@mps_only
class TestShim:
    def test_auto_import_via_pth(self):
        """`import flash_attn` works in a fresh interpreter with NO prior imports."""
        r = run_py("""
            import flash_attn
            print(flash_attn.__version__)
        """)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "2.7.4"

    def test_shim_exposes_package_functions(self):
        r = run_py("""
            import flash_attn
            import metal_flash_attn
            assert flash_attn.flash_attn_func is metal_flash_attn.flash_attn_func
            assert flash_attn.flash_attn_varlen_func is metal_flash_attn.flash_attn_varlen_func
            print("ok")
        """)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "ok"

    def test_flash_attn_interface_submodule(self):
        """SeedVR2-style import path: from flash_attn.flash_attn_interface import ..."""
        r = run_py("""
            from flash_attn.flash_attn_interface import flash_attn_varlen_func
            print("ok")
        """)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "ok"

    def test_real_install_wins(self, tmp_path):
        """A real flash_attn on sys.path must shadow the shim, never the reverse."""
        pkg = tmp_path / "flash_attn"
        pkg.mkdir()
        (pkg / "__init__.py").write_text('__version__ = "9.9.9-real"\n')
        r = run_py(
            """
            import flash_attn
            print(flash_attn.__version__)
            """,
            env_extra={"PYTHONPATH": str(tmp_path)},
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "9.9.9-real"

    def test_kill_switch(self):
        r = run_py(
            """
            import flash_attn
            """,
            env_extra={"MTLFLASHATTN_SHIM": "off"},
        )
        assert r.returncode != 0
        assert "ModuleNotFoundError" in r.stderr

    def test_packages_distributions_consistency(self):
        """transformers' is_flash_attn_2_available() does an UNGUARDED
        PACKAGE_DISTRIBUTION_MAPPING["flash_attn"] lookup (KeyError crash if the
        module is importable but no distribution provides it). The shim must
        seed the mapping so libraries see a consistent story — and correctly
        conclude this is NOT the CUDA flash-attn distribution."""
        r = run_py("""
            import importlib.metadata
            m = importlib.metadata.packages_distributions()
            assert "flash_attn" in m, "mapping not seeded"
            # transformers' exact FA2 check must cleanly evaluate to False:
            assert "flash-attn" not in [p.replace("_", "-") for p in m["flash_attn"]]
            # and the module itself still imports through the shim
            import flash_attn
            print(flash_attn.__version__)
        """)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "2.7.4"

    def test_shimmed_attention_runs(self):
        """End-to-end: a caller that only knows flash_attn gets working attention."""
        r = run_py("""
            import torch
            from flash_attn import flash_attn_func
            q = torch.randn(1, 64, 4, 64, device="mps", dtype=torch.float16)
            k = torch.randn(1, 64, 4, 64, device="mps", dtype=torch.float16)
            v = torch.randn(1, 64, 4, 64, device="mps", dtype=torch.float16)
            out = flash_attn_func(q, k, v, causal=True)
            assert out.shape == q.shape and torch.isfinite(out).all()
            print("ok")
        """)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "ok"
