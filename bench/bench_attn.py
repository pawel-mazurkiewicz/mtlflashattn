"""Benchmark: flash kernel tiers vs stock fused SDPA on MPS.

Usage: uv run python bench/bench_attn.py [--kernel v0|v1] [--causal]
Reports ms/call, effective TFLOP/s, and ratio vs stock fused SDPA.
"""
import argparse
import math
import time

import torch
import torch.nn.functional as F


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    args = ap.parse_args()

    from metal_flash_attn._kernel import flash_attn_forward

    B, H = args.batch, args.heads
    print(f"torch {torch.__version__}  B={B} H={H} fp16 causal={args.causal}")
    print(f"{'shape':>14} | {'stock ms':>9} {'TF/s':>6} | {'flash ms':>9} {'TF/s':>6} | ratio")
    for D in args.dims:
        for L in args.lens:
            q = torch.randn(B, H, L, D, device="mps", dtype=torch.float16)
            k = torch.randn(B, H, L, D, device="mps", dtype=torch.float16)
            v = torch.randn(B, H, L, D, device="mps", dtype=torch.float16)
            scale = 1.0 / math.sqrt(D)
            t_stock = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=args.causal))
            t_flash = bench(lambda: flash_attn_forward(q, k, v, scale=scale, causal=args.causal))
            flops = 4 * B * H * L * L * D * (0.5 if args.causal else 1.0)
            print(f"D={D:<4} L={L:<6} | {t_stock*1e3:9.2f} {flops/t_stock/1e12:6.2f} "
                  f"| {t_flash*1e3:9.2f} {flops/t_flash/1e12:6.2f} "
                  f"| {t_flash/t_stock:5.2f}x")


if __name__ == "__main__":
    main()
