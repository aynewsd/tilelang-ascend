"""AOT compilation script for RoPE kernel.

Lowers the RoPE @T.prim_func to C++ source via tilelang.engine.lower,
then compiles to a shared library using LibraryGenerator.

Usage:
    python aot_rope.py --shape 16 64 512 256 --layout half --dtype float16
    python aot_rope.py --target pto --output rope_pto.so
"""

import argparse
import shutil
import sys

import tilelang
import tvm
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.utils.target import determine_platform

sys.path.insert(0, ".")
from rope_half_interleaved import rope_kernel, select_block_M, pass_configs as rope_pass_configs  # noqa: E402

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="RoPE AOT Compilation")
parser.add_argument("--shape", type=int, nargs=4, default=[16, 64, 512, 256], metavar=("BS", "H", "HS", "RD"))
parser.add_argument("--layout", default="half", choices=["half", "interleaved"])
parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
parser.add_argument("--target", default="ascendc", choices=["ascendc", "pto"])
parser.add_argument("--platform", default="auto")
parser.add_argument("-o", "--output", default="./rope_lib.so")
args = parser.parse_args()

bs, head_num, hidden_size, rope_dim = args.shape

# Compute kernel params (same logic as tilelang_rope wrapper)
NUM_CORES = 48


block_M = select_block_M(head_num, rope_dim, args.layout)
M = bs * head_num
sc_rows = bs

m_num_full = M // block_M
tail_rows = M % block_M
has_tail = 1 if tail_rows > 0 else 0
total_chunks = m_num_full + has_tail
num_blocks = min(total_chunks, NUM_CORES)

# Get the raw @T.prim_func (bypass @tilelang.jit decorator)
prim_func = rope_kernel.__wrapped__(
    M, block_M, num_blocks, total_chunks, sc_rows, hidden_size, rope_dim, head_num, args.layout, dtype=args.dtype
)

# Set pass context (same config as @tilelang.jit)
pass_ctx_map = {k.value: v for k, v in rope_pass_configs.items()}
with tvm.transform.PassContext(opt_level=3, config=pass_ctx_map):
    # Lower to C++ source
    platform = determine_platform(args.platform)
    artifact = tilelang.engine.lower(prim_func, target=args.target, platform=platform)

# Compile to .so
lib_generator = LibraryGenerator(target=args.target, platform=platform)
lib_generator.update_lib_code(artifact.kernel_source)
lib_generator.compile_lib()
shutil.copy(lib_generator.get_lib_path(), args.output)

print(f"Built {args.output} (target={args.target}, platform={platform})")
print(f"  shape={args.shape} layout={args.layout} dtype={args.dtype}")
print(f"  M={M} block_M={block_M} num_blocks={num_blocks} total_chunks={total_chunks}")
