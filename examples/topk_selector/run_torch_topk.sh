#!/bin/bash
cd /mnt/workspace/tilelang-ascend/examples/topk_selector
python bench_topk.py --mode torch --warmup=5 --iters=1
