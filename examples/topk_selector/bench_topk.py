"""Benchmark torch.topk vs TileLang topk_selector for msprof op comparison.

Usage:
    python bench_topk.py --mode torch        # benchmark torch.topk
    python bench_topk.py --mode tilelang     # benchmark TileLang original
    python bench_topk.py --mode tilelang-opt # benchmark TileLang improved
"""
import argparse

import torch
import torch_npu  # noqa: F401 — register NPU backend
import tilelang as tl

from example_topk_selector import simple_topk_selector
from example_topk_selector_improved import simple_topk_selector_improved


def main():
    parser = argparse.ArgumentParser(description="topk benchmark for msprof op")
    parser.add_argument("--mode", choices=["torch", "tilelang", "tilelang-opt"], required=True,
                        help="Which implementation to benchmark")
    parser.add_argument("--b", type=int, default=64, help="Batch size")
    parser.add_argument("--n", type=int, default=4096, help="Sequence length")
    parser.add_argument("--k", type=int, default=2048, help="Top-K value")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=1, help="Measurement iterations")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.npu.set_device(0)

    x = torch.randn(args.b, args.n, dtype=torch.float32, device="npu")

    if args.mode == "torch":
        for _ in range(args.warmup):
            torch.topk(x, args.k, dim=-1)
        torch.npu.synchronize()
        for _ in range(args.iters):
            torch.topk(x, args.k, dim=-1)
        torch.npu.synchronize()
    elif args.mode == "tilelang":
        block_n = max(args.k // 4, 1)
        kernel = simple_topk_selector(args.b, args.n, args.k, block_n)
        for _ in range(args.warmup):
            kernel(x)
        torch.npu.synchronize()
        for _ in range(args.iters):
            kernel(x)
        torch.npu.synchronize()
    else:
        kernel = simple_topk_selector_improved(args.b, args.n, args.k)
        for _ in range(args.warmup):
            kernel(x)
        torch.npu.synchronize()
        for _ in range(args.iters):
            kernel(x)
        torch.npu.synchronize()

    print(f"[{args.mode}] B={args.b} N={args.n} K={args.k} warmup={args.warmup} iters={args.iters} done")


if __name__ == "__main__":
    main()
