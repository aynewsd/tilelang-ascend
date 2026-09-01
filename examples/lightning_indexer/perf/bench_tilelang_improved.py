"""TileLang Lightning Indexer 的 msprof 单次启动目标脚本。

应用侧只启动一次 kernel；预热与重复采样完全由 msprof 的
``--warm-up`` 和 ``--launch-count`` 控制。
"""

import argparse
import os
import sys

import torch
import torch_npu
import tilelang

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from example_lightning_indexer_dynamic_shape_improved import indexer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="供 msprof 采集的 Lightning Indexer TileLang 单次启动目标")
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--S1", type=int, default=512)
    parser.add_argument("--S2", type=int, default=4096)
    parser.add_argument("--G", type=int, default=32)
    parser.add_argument("--N2", type=int, default=1)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1024)
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
        256,
        128,
        MAX_S2=args.S2,
    )

    q = torch.randn(args.B, args.S1, args.G, args.D, dtype=torch.float16).npu()
    k = torch.randn(args.B, args.S2, args.N2, args.D, dtype=torch.float16).npu()
    w = torch.randn(args.B, args.S1, args.N2, args.G, dtype=torch.float32).npu()

    q = q.view(args.B, args.S1, args.N2, args.G * args.D)

    torch.npu.synchronize()

    func(q, k, w)
    torch.npu.synchronize()
    print("done")


if __name__ == "__main__":
    main()
