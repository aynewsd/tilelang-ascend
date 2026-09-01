"""msprof op target script: torch_npu.npu_lightning_indexer baseline.

Run standalone:
    python bench_baseline.py --S2 4096 --n-runs 20

Under msprof op:
    msprof op --application="python bench_baseline.py --S2 4096 --n-runs 20" \
              --warm-up=5 --launch-count=10 --output=./msprof_output/baseline

warmup and repeat are controlled by msprof op --warm-up / --launch-count.
This script only launches the kernel n-runs times.
"""

import argparse

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser(description="lightning_indexer baseline target for msprof op")
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--S1", type=int, default=512)
    parser.add_argument("--S2", type=int, default=4096)
    parser.add_argument("--N1", type=int, default=32)
    parser.add_argument("--N2", type=int, default=1)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--sparse-mode", type=int, default=0, choices=[0, 3])
    parser.add_argument("--n-runs", type=int, default=20)
    args = parser.parse_args()

    torch_npu.npu.set_device(0)
    torch.set_default_device("npu:0")

    q = torch.randn(args.B, args.S1, args.N1, args.D, dtype=torch.float16)
    k = torch.randn(args.B, args.S2, args.N2, args.D, dtype=torch.float16)
    w = torch.randn(args.B, args.S1, args.N1, dtype=torch.float16).abs()
    asq = torch.full((args.B,), args.S1, dtype=torch.int32)
    ask = torch.full((args.B,), args.S2, dtype=torch.int32)

    torch.npu.synchronize()

    for _ in range(args.n_runs):
        torch_npu.npu_lightning_indexer(
            q,
            k,
            w,
            actual_seq_lengths_query=asq,
            actual_seq_lengths_key=ask,
            layout_query="BSND",
            layout_key="BSND",
            sparse_count=args.top_k,
            sparse_mode=args.sparse_mode,
        )
    torch.npu.synchronize()
    print("done")


if __name__ == "__main__":
    main()
