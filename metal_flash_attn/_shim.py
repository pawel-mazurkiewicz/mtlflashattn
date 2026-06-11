"""Explicit opt-in registration of the guarded `flash_attn` shim.

Normally the shim self-registers at interpreter start via
mtlflashattn_autoload.pth. Call install() when that mechanism is unavailable
(e.g. embedded interpreters, PEX/zipapp environments).
"""


def install():
    import _mtlflashattn_finder

    return _mtlflashattn_finder.install()
