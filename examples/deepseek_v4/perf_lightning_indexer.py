"""Minimal perf comparison: TileLang lightning_indexer vs torch_npu.npu_lightning_indexer.

Usage (from repo root):
    python examples/deepseek_v4/perf_lightning_indexer.py                  # default BSND+BSND
    python examples/deepseek_v4/perf_lightning_indexer.py --preset sweep    # S2 sweep
    python examples/deepseek_v4/perf_lightning_indexer.py --s2 8192 --top-k 2048

Note: TileLang JIT compilation is excluded from the bench (warmup-before-bench).
Both sides run the same shape / dtype / sparse_count / sparse_mode.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch_npu

# Make the sibling lightning_indexer.py importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lightning_indexer import lightning_indexer as tl_lightning_indexer  # noqa: E402

from tilelang.profiler import do_bench  # noqa: E402


_DEV = 0
torch_npu.npu.set_device(_DEV)
torch.set_default_device(f"npu:{_DEV}")


def _make_rand(shape, dtype=torch.float16):
    return torch.tensor(np.random.uniform(-1, 1, shape), dtype=dtype)


def _make_rand_w(shape, dtype=torch.float16):
    # weights must be non-negative (matches deepseek_v4 example convention).
    return _make_rand(shape, dtype).abs()


def _set_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Top-K index set similarity per row (Counter multiset diff), averaged over rows.

    Both a and b: [..., K] int32. Returns fraction of matching indices (0..1).
    """
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    K = a.shape[-1]
    a_flat = a.cpu().reshape(-1, K)
    b_flat = b.cpu().reshape(-1, K)
    total = 0
    matched = 0
    for i in range(a_flat.shape[0]):
        ca = Counter(int(x) for x in a_flat[i].tolist() if int(x) >= 0)
        cb = Counter(int(x) for x in b_flat[i].tolist() if int(x) >= 0)
        diff = (ca - cb) + (cb - ca)
        total += K
        matched += K - sum(diff.values())
    return matched / total if total > 0 else 1.0


def _take_indices(out):
    """Normalize output to indices tensor only."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def build_inputs_bsnd(B, S1, S2, N1, N2, D):
    q = _make_rand((B, S1, N1, D))
    k = _make_rand((B, S2, N2, D))
    w = _make_rand_w((B, S1, N1))
    asq = torch.full((B,), S1, dtype=torch.int32)
    ask = torch.full((B,), S2, dtype=torch.int32)
    return q, k, w, asq, ask


def bench_fn(fn, *args, n_warmup=5, n_repeat=10):
    def f():
        fn(*args)
        torch.npu.synchronize()

    return do_bench(f, _n_warmup=n_warmup, _n_repeat=n_repeat, return_mode="median")


def run_one(name, B, S1, S2, N1, N2, D, top_k, sparse_mode, n_warmup=5, n_repeat=10):
    print(f"\n[{name}] B={B} S1={S1} S2={S2} N1={N1} N2={N2} D={D} top_k={top_k} mode={sparse_mode}")

    q, k, w, asq, ask = build_inputs_bsnd(B, S1, S2, N1, N2, D)

    # ----- TileLang: warmup compile (excluded from bench) -----
    print("  compiling TileLang kernel ...")
    tl_idx, _ = tl_lightning_indexer(
        q, k, w,
        actual_seq_lengths_query=asq,
        actual_seq_lengths_key=ask,
        layout_query="BSND",
        layout_key="BSND",
        sparse_count=top_k,
        sparse_mode=sparse_mode,
        return_value=False,
    )
    torch.npu.synchronize()

    # ----- torch_npu baseline -----
    print("  warming up torch_npu baseline ...")
    npu_idx = _take_indices(
        torch_npu.npu_lightning_indexer(
            q, k, w,
            actual_seq_lengths_query=asq,
            actual_seq_lengths_key=ask,
            layout_query="BSND",
            layout_key="BSND",
            sparse_count=top_k,
            sparse_mode=sparse_mode,
        )
    )
    torch.npu.synchronize()

    # ----- correctness gate -----
    sim = _set_similarity(tl_idx.to(torch.int32), npu_idx.to(torch.int32))
    print(f"  correctness: index-set similarity = {sim * 100:.2f}% (threshold 95%)")
    if sim < 0.95:
        print("  [ERROR] correctness check failed, skipping bench")
        return False

    # ----- bench -----
    # Both calls are keyword-heavy, so wrap in closure (do_bench takes a single callable).
    print("  benching ...")
    tl_ms = bench_fn(
        lambda: tl_lightning_indexer(
            q, k, w,
            actual_seq_lengths_query=asq,
            actual_seq_lengths_key=ask,
            layout_query="BSND",
            layout_key="BSND",
            sparse_count=top_k,
            sparse_mode=sparse_mode,
            return_value=False,
        ),
        n_warmup=n_warmup,
        n_repeat=n_repeat,
    )
    npu_ms = bench_fn(
        lambda: torch_npu.npu_lightning_indexer(
            q, k, w,
            actual_seq_lengths_query=asq,
            actual_seq_lengths_key=ask,
            layout_query="BSND",
            layout_key="BSND",
            sparse_count=top_k,
            sparse_mode=sparse_mode,
        ),
        n_warmup=n_warmup,
        n_repeat=n_repeat,
    )

    speedup = npu_ms / tl_ms if tl_ms > 0 else float("inf")
    print(f"  TileLang : {tl_ms:.4f} ms")
    print(f"  torch_npu: {npu_ms:.4f} ms")
    print(f"  speedup  : {speedup:.2f}x  (torch_npu / TileLang, >1.0 means TileLang faster)")
    return True


def main():
    parser = argparse.ArgumentParser(description="lightning_indexer perf: TileLang vs torch_npu baseline")
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--s1", type=int, default=3)
    parser.add_argument("--s2", type=int, default=32768)
    parser.add_argument("--n1", type=int, default=64)
    parser.add_argument("--n2", type=int, default=1)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--sparse-mode", type=int, default=3, choices=[0, 3])
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-repeat", type=int, default=10)
    parser.add_argument(
        "--preset",
        default="default",
        choices=["default", "sweep", "small"],
        help="preset suite (overrides individual args)",
    )
    args = parser.parse_args()

    print(f"Device: {torch.npu.get_device_name(_DEV)}")

    results = []

    if args.preset == "small":
        # quick smoke: smaller S2, fewer repeats
        results.append(run_one("small", 1, 3, 4096, 8, 1, 128, 1024, 0,
                               n_warmup=3, n_repeat=5))
    elif args.preset == "sweep":
        # vary S2 (main perf axis for indexer)
        for s2 in [4096, 8192, 16384, 32768]:
            results.append(run_one(f"s2={s2}", args.B, args.s1, s2, args.n1, args.n2, args.d,
                                   args.top_k, args.sparse_mode,
                                   n_warmup=args.n_warmup, n_repeat=args.n_repeat))
    else:
        results.append(run_one("default", args.B, args.s1, args.s2, args.n1, args.n2, args.d,
                               args.top_k, args.sparse_mode,
                               n_warmup=args.n_warmup, n_repeat=args.n_repeat))

    print("\n" + "=" * 60)
    print(f"Done. {sum(results)}/{len(results)} configs passed correctness gate.")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
