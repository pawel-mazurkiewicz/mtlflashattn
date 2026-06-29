"""flash_attn_varlen_func: packed [total_tokens, H, D] + cu_seqlens, per-sequence reference."""
import pytest
import torch

from test_flash_attn_func import TOL, ref_attention

mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires MPS"
)


def make_varlen(seqlens_q, seqlens_k, Hq, Hkv, D, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(sum(seqlens_q), Hq, D, generator=g).to("mps", dtype)
    k = torch.randn(sum(seqlens_k), Hkv, D, generator=g).to("mps", dtype)
    v = torch.randn(sum(seqlens_k), Hkv, D, generator=g).to("mps", dtype)
    cu_q = torch.tensor([0] + list(seqlens_q), dtype=torch.int32, device="mps").cumsum(0, dtype=torch.int32)
    cu_k = torch.tensor([0] + list(seqlens_k), dtype=torch.int32, device="mps").cumsum(0, dtype=torch.int32)
    return q, k, v, cu_q, cu_k


def ref_varlen(q, k, v, cu_q, cu_k, causal=False, scale=None, alibi_slopes=None):
    outs = []
    for i in range(len(cu_q) - 1):
        qs = q[cu_q[i]:cu_q[i + 1]]      # [Lq_i, Hq, D]
        ks = k[cu_k[i]:cu_k[i + 1]]
        vs = v[cu_k[i]:cu_k[i + 1]]
        slopes_i = alibi_slopes[i] if (alibi_slopes is not None and alibi_slopes.dim() == 2) \
            else alibi_slopes
        o = ref_attention(
            qs.permute(1, 0, 2)[None], ks.permute(1, 0, 2)[None],
            vs.permute(1, 0, 2)[None], causal=causal, scale=scale, alibi_slopes=slopes_i,
        )[0].permute(1, 0, 2)            # [Lq_i, Hq, D] fp32
        outs.append(o)
    return torch.cat(outs, dim=0)


def check_varlen(seqlens_q, seqlens_k, Hq=4, Hkv=4, D=64,
                 dtype=torch.float32, causal=False, scale=None):
    from metal_flash_attn import flash_attn_varlen_func

    q, k, v, cu_q, cu_k = make_varlen(seqlens_q, seqlens_k, Hq, Hkv, D, dtype)
    out = flash_attn_varlen_func(
        q, k, v, cu_q, cu_k,
        max_seqlen_q=max(seqlens_q), max_seqlen_k=max(seqlens_k),
        softmax_scale=scale, causal=causal,
    )
    assert out.shape == q.shape
    assert out.dtype == q.dtype
    ref = ref_varlen(q, k, v, cu_q, cu_k, causal=causal, scale=scale)
    torch.testing.assert_close(out.float(), ref, **TOL[dtype])


@mps_only
class TestFlashAttnVarlenFunc:
    def test_ragged_batch(self):
        check_varlen([37, 128, 5], [37, 128, 5])

    def test_ragged_causal_fp16(self):
        check_varlen([64, 17], [64, 17], dtype=torch.float16, causal=True)

    def test_ragged_cross_lens(self):
        # Lq != Lk per sequence (e.g. cross-attn packing)
        check_varlen([16, 33], [80, 7], causal=False)

    def test_gqa(self):
        check_varlen([40, 24], [40, 24], Hq=8, Hkv=2, causal=True, dtype=torch.float16)

    def test_batched_alibi_slopes(self):
        # [batch, Hq] slopes: each packed sequence must get its own row
        from metal_flash_attn import flash_attn_varlen_func

        seqlens = [40, 24]
        q, k, v, cu_q, cu_k = make_varlen(seqlens, seqlens, 4, 4, 64, torch.float32)
        slopes = torch.tensor(
            [[0.5, 0.25, 0.125, 0.0625], [0.1, 0.2, 0.3, 0.4]], device="mps"
        )  # [batch=2, Hq=4]
        out = flash_attn_varlen_func(
            q, k, v, cu_q, cu_k,
            max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens), alibi_slopes=slopes,
        )
        ref = ref_varlen(q, k, v, cu_q, cu_k, alibi_slopes=slopes)
        torch.testing.assert_close(out.float(), ref, **TOL[torch.float32])

    def test_per_head_alibi_slopes(self):
        # [Hq] slopes shared across all packed sequences
        from metal_flash_attn import flash_attn_varlen_func

        seqlens = [40, 24]
        q, k, v, cu_q, cu_k = make_varlen(seqlens, seqlens, 4, 4, 64, torch.float32)
        slopes = torch.tensor([0.5, 0.25, 0.125, 0.0625], device="mps")  # [Hq=4]
        out = flash_attn_varlen_func(
            q, k, v, cu_q, cu_k,
            max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens), alibi_slopes=slopes,
        )
        ref = ref_varlen(q, k, v, cu_q, cu_k, alibi_slopes=slopes)
        torch.testing.assert_close(out.float(), ref, **TOL[torch.float32])

    def test_single_sequence_matches_dense(self):
        from metal_flash_attn import flash_attn_func, flash_attn_varlen_func

        q, k, v, cu_q, cu_k = make_varlen([96], [96], 4, 4, 64, torch.float16)
        out_v = flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, max_seqlen_q=96, max_seqlen_k=96, causal=True
        )
        out_d = flash_attn_func(q[None], k[None], v[None], causal=True)[0]
        torch.testing.assert_close(out_v, out_d)

    def test_unsupported_kwargs_raise(self):
        from metal_flash_attn import flash_attn_varlen_func

        q, k, v, cu_q, cu_k = make_varlen([32], [32], 4, 4, 64, torch.float16)
        with pytest.raises(NotImplementedError):
            flash_attn_varlen_func(
                q, k, v, cu_q, cu_k, max_seqlen_q=32, max_seqlen_k=32, dropout_p=0.5
            )
        with pytest.raises(NotImplementedError):
            flash_attn_varlen_func(
                q, k, v, cu_q, cu_k, max_seqlen_q=32, max_seqlen_k=32,
                block_table=torch.zeros(1, 1, dtype=torch.int32, device="mps"),
            )

    def test_mismatched_batch_raises(self):
        from metal_flash_attn import flash_attn_varlen_func

        q, k, v, cu_q, _ = make_varlen([32, 32], [32, 32], 4, 4, 64, torch.float16)
        cu_k_bad = torch.tensor([0, 64], dtype=torch.int32, device="mps")
        with pytest.raises(ValueError):
            flash_attn_varlen_func(
                q, k, v, cu_q, cu_k_bad, max_seqlen_q=32, max_seqlen_k=64
            )
