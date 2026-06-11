"""Guarded meta-path finder providing `flash_attn` backed by metal_flash_attn.

Appended to the END of sys.meta_path: the stdlib PathFinder runs first, so a
real flash_attn install always wins and this finder never fires. Fully lazy —
torch / metal_flash_attn are imported only if something imports `flash_attn`.

Auto-activated at interpreter start by mtlflashattn_autoload.pth; can also be
activated explicitly via metal_flash_attn._shim.install().
Kill switch: MTLFLASHATTN_SHIM=off.
"""
import importlib.abc
import importlib.util
import os
import sys

_SUBMODULE_EXPORTS = (
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
)


class _MtlFlashAttnFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    _NAMES = frozenset({"flash_attn", "flash_attn.flash_attn_interface"})

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._NAMES:
            return None
        if os.environ.get("MTLFLASHATTN_SHIM", "auto").lower() in ("off", "0", "false"):
            return None
        try:
            import torch
            if not torch.backends.mps.is_available():
                return None
        except Exception:
            return None
        return importlib.util.spec_from_loader(
            fullname, self, is_package=(fullname == "flash_attn")
        )

    def create_module(self, spec):
        return None  # default module creation

    def exec_module(self, module):
        import metal_flash_attn as _mfa
        from metal_flash_attn._version import FLASH_ATTN_SHIM_VERSION

        for name in _SUBMODULE_EXPORTS:
            setattr(module, name, getattr(_mfa, name))
        if module.__name__ == "flash_attn":
            module.__version__ = FLASH_ATTN_SHIM_VERSION


def install():
    """Idempotently append the finder to sys.meta_path. Returns True if added."""
    if any(isinstance(f, _MtlFlashAttnFinder) for f in sys.meta_path):
        return False
    sys.meta_path.append(_MtlFlashAttnFinder())
    return True


install()
