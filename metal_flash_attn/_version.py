__version__ = "0.1.0"

# Version the guarded `flash_attn` shim reports. Some libraries feature-gate on
# flash_attn.__version__ (e.g. ">=2.3" for window_size support), so it must
# parse as a plausible flash-attn 2.x release.
FLASH_ATTN_SHIM_VERSION = "2.7.4"
