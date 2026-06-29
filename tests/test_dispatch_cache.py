"""Per-call dispatch overhead: the v2 dispatch builds a small int32 `sh` shape
tensor and a float `pr` scale tensor for every attention call. Measured cost of
`torch.tensor([...], device='mps')` is ~270 us each (alloc + H2D + sync), which is
~90% of a tiny Lk=5 cross-attention call (~990 such calls in a real
diffusion workload). Caching the read-only sh/pr tensors by (values, device) / (scale,
device) drops it to ~0.1 us with identical kernel inputs.
"""
import math

import pytest
import torch

from metal_flash_attn import _kernel

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires MPS"
)


def test_pr_tensor_cached_and_correct():
    dev = torch.device("mps")
    t1 = _kernel._pr_tensor(0.1, 0.0, -1, -1, 0.0, dev)
    t2 = _kernel._pr_tensor(0.1, 0.0, -1, -1, 0.0, dev)
    assert t1 is t2, "same scale+softcap+window+alibi+device must reuse the cached tensor"
    assert t1.dtype == torch.float32 and t1.shape == (5,)
    assert abs(float(t1[0].item()) - 0.1) < 1e-6 and float(t1[1].item()) == 0.0
    assert t1[2].item() == -1.0 and t1[3].item() == -1.0 and t1[4].item() == 0.0
    assert _kernel._pr_tensor(0.2, 0.0, -1, -1, 0.0, dev) is not t1, "different scale -> different tensor"
    assert _kernel._pr_tensor(0.1, 30.0, -1, -1, 0.0, dev) is not t1, "different softcap -> different tensor"
    assert _kernel._pr_tensor(0.1, 0.0, 64, 0, 0.0, dev) is not t1, "different window -> different tensor"
    assert _kernel._pr_tensor(0.1, 0.0, -1, -1, 1.0, dev) is not t1, "different alibi flag -> different tensor"


def test_sh_tensor_cached_and_correct():
    dev = torch.device("mps")
    vals = [1, 16, 16, 4096, 5, 0]
    s1 = _kernel._sh_tensor(vals, dev)
    s2 = _kernel._sh_tensor(vals, dev)
    assert s1 is s2, "same shape values+device must reuse the cached tensor"
    assert s1.dtype == torch.int32
    assert s1.tolist() == vals
    assert _kernel._sh_tensor([1, 16, 16, 4096, 7, 0], dev) is not s1


def test_v2_dtype_dispatch_uses_cache_and_matches_reference():
    """The bf16 TG dispatch (tiny Lk stays off v2r) reuses cached sh/pr across
    repeated identical calls and still matches the fp32 reference."""
    if not _kernel._v2_supported():
        pytest.skip("v2 TensorOps kernel not supported on this machine")
    _kernel._sh_cache.clear()
    _kernel._pr_cache.clear()
    Hq, Lq, Lk, D = 8, 512, 5, 128
    q = torch.randn(1, Hq, Lq, D, device="mps", dtype=torch.bfloat16)
    k = torch.randn(1, Hq, Lk, D, device="mps", dtype=torch.bfloat16)
    v = torch.randn(1, Hq, Lk, D, device="mps", dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(D)
    o1 = _kernel._flash_v2_dtype(q, k, v, scale, False, _kernel._BiasParams(0.0, -1, -1, None), "v2_bf16")
    n_sh = len(_kernel._sh_cache)
    n_pr = len(_kernel._pr_cache)
    assert n_sh >= 1 and n_pr >= 1, "dispatch must populate the caches"
    o2 = _kernel._flash_v2_dtype(q, k, v, scale, False, _kernel._BiasParams(0.0, -1, -1, None), "v2_bf16")
    # identical shape/scale -> no new cache entries allocated
    assert len(_kernel._sh_cache) == n_sh
    assert len(_kernel._pr_cache) == n_pr
    ref = torch.softmax(q.float() @ k.float().transpose(-1, -2) * scale, dim=-1) @ v.float()
    assert (o1.float() - ref).abs().max().item() < 3e-2
    assert torch.equal(o1, o2)
