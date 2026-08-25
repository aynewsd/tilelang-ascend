"""msprof op target script: TileLang lightning_indexer (dynamic shape).

Run standalone:
    python bench_tilelang.py --S2 4096 --n-runs 20

Under msprof op:
    msprof op --application="python bench_tilelang.py --S2 4096 --n-runs 20" \
              --warm-up=5 --launch-count=10 --output=./msprof_output/tilelang

Uses the dynamic-shape kernel (T.symbolic B/S1/S2) so one compilation
covers all shapes. D=128 (baseline constraint), BLOCK_K=128 (>= D).

warmup and repeat are controlled by msprof op --warm-up / --launch-count.
This script only launches the kernel n-runs times.
"""

import argparse
import os
import sys

import torch
import torch_npu
import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_lightning_indexer_dynamic_shape import indexer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="lightning_indexer TileLang target for msprof op")
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--S1", type=int, default=512)
    parser.add_argument("--S2", type=int, default=4096)
    parser.add_argument("--G", type=int, default=32)
    parser.add_argument("--N2", type=int, default=1)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--n-runs", type=int, default=20)
    args = parser.parse_args()

    torch_npu.npu.set_device(0)

    tilelang.disable_cache()

    func = indexer(
        args.N2,
        args.G,
        args.D,
        args.top_k,
        256,
        16,
        64,
        64,
        128,
        MAX_S2=args.S2,
    )

    q = torch.randn(args.B, args.S1, args.G, args.D, dtype=torch.float16).npu()
    k = torch.randn(args.B, args.S2, args.N2, args.D, dtype=torch.float16).npu()
    w = torch.randn(args.B, args.S1, args.N2, args.G, dtype=torch.float32).npu()

    q = q.view(args.B, args.S1, args.N2, args.G * args.D)

    torch.npu.synchronize()

    for _ in range(args.n_runs):
        func(q, k, w)
    torch.npu.synchronize()
    print("done")


if __name__ == "__main__":
    main()
