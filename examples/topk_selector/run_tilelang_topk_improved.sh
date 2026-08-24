#!/bin/bash
cd /mnt/workspace/tilelang-ascend/examples/topk_selector
python bench_topk.py --mode tilelang-opt --warmup=5 --iters=1
