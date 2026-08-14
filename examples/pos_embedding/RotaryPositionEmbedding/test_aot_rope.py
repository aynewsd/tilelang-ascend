"""Test AOT-compiled RoPE kernel via ctypes.

Loads rope_lib.so, calls the kernel directly, and compares against
the CANN golden (torch_npu.npu_rotary_mul).

Usage:
    python test_aot_rope.py --shape 16 64 512 256 --layout half --dtype float16
"""

import argparse
import ctypes
import sys

import torch
import torch_npu

sys.path.insert(0, ".")
from rope_half_interleaved import cann_rope_ref, check_precision  # noqa: E402

torch.manual_seed(42)

parser = argparse.ArgumentParser(description="Test AOT RoPE kernel")
parser.add_argument("--shape", type=int, nargs=4, default=[16, 64, 512, 256], metavar=("BS", "H", "HS", "RD"))
parser.add_argument("--layout", default="half", choices=["half", "interleaved"])
parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
parser.add_argument("--lib", default="./rope_lib.so")
args = parser.parse_args()

bs, head_num, hidden_size, rope_dim = args.shape
dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
torch_dtype = dtype_map[args.dtype]

# Inputs on NPU
x = torch.randn(bs, head_num, hidden_size, dtype=torch_dtype, device="npu")
sin = torch.randn(bs, 1, rope_dim, dtype=torch_dtype, device="npu")
cos = torch.randn(bs, 1, rope_dim, dtype=torch_dtype, device="npu")

# Golden via CANN op
out_ref = cann_rope_ref(x.cpu().clone(), sin.cpu(), cos.cpu(), args.layout, args.dtype)

# Reshape for kernel: x=[M, HS], sin/cos=[sc_rows, RD]
x_2d = x.view(-1, hidden_size).contiguous()
sin_2d = sin.view(-1, rope_dim).contiguous()
cos_2d = cos.view(-1, rope_dim).contiguous()

# Load AOT kernel
lib = ctypes.CDLL(args.lib)
stream = torch.npu.current_stream()._as_parameter_

# Call: kernel(x, sin, cos, stream) — in-place on x
lib.call(
    ctypes.c_void_p(x_2d.data_ptr()),
    ctypes.c_void_p(sin_2d.data_ptr()),
    ctypes.c_void_p(cos_2d.data_ptr()),
    stream,
)
torch.npu.synchronize()

# Reshape back and compare
out_npu = x_2d.view(bs, head_num, hidden_size).cpu()

passed, ratio, max_abs = check_precision(out_npu, out_ref, args.dtype)
tag = "PASS" if passed else "FAIL"
print(f"[AOT_{tag}] shape={args.shape} layout={args.layout} dtype={args.dtype} matched_ratio={ratio:.4f} max_abs={max_abs:.3e}")

if passed:
    print("AOT Kernel Output Match!")
else:
    print("AOT Kernel Output MISMATCH!")
    sys.exit(1)
