# RoPE 基准测试：TileLang vs torch_npu.npu_rotary_mul

TileLang RoPE（`rope_half_interleaved.py`）对比 CANN 官方算子
`aclnnRotaryPositionEmbedding`（通过 `torch_npu.npu_rotary_mul` 调用），
涵盖**性能**与**精度**两部分。

## 一、性能测试

### 测量方法

- **工具**：`msprof` device 侧 Task Duration
- **两侧入口**：`python rope_half_interleaved.py --perf --side {tl|cann}`，单次 kernel launch
- **加速比**：`cann_us / tl_us`（>1.0 表示 TileLang 更快）
- 两侧使用相同 shape / dtype / layout / seed=0

### 性能结果

| 场景 | Shape | Layout | Dtype | TileLang (us) | CANN (us) | 加速比 |
|------|-------|--------|-------|---------------|-----------|--------|
| decode_bs1 | 1 32 128 128 | half | float16 | 7.98 | 7.63 | 0.96x |
| decode_bs64 | 64 64 128 128 | half | float16 | 14.61 | 19.82 | 1.36x |
| prefill_bs4_h64 | 4 64 128 128 | half | float16 | 6.99 | 18.89 | 2.70x |
| prefill_bs8_h64 | 8 64 128 128 | half | float16 | 7.34 | 19.90 | 2.71x |
| prefill_bs32_h64 | 32 64 128 128 | half | float16 | 10.09 | 20.08 | 1.99x |
| prefill_bs4_d256 | 4 64 256 256 | half | float16 | 9.43 | 18.99 | 2.01x |
| prefill_bs4_d512 | 4 64 512 512 | half | float16 | 12.98 | 18.68 | 1.44x |
| prefill_bs4_bf16 | 4 64 128 128 | half | bfloat16 | 7.61 | 18.81 | 2.47x |
| prefill_bs8_bf16 | 8 64 128 128 | half | bfloat16 | 9.90 | 19.88 | 2.01x |
| prefill_bs4_inter | 4 64 128 128 | interleaved | float16 | 15.06 | 19.00 | 1.26x |
| prefill_bs8_inter | 8 64 128 128 | interleaved | float16 | 15.03 | 19.32 | 1.29x |
| prefill_bs4_inter_bf16 | 4 64 128 128 | interleaved | bfloat16 | 15.44 | 18.40 | 1.19x |
| bsnd_bs4_s4 | 4 4 64 128 128 | half | float16 | 9.38 | 21.54 | 2.30x |

### 关键观察

- **TileLang 在 13 个场景中胜出 12 个**，加速比 1.19x–2.71x。
- **half layout 收益最大**（1.36x–2.70x）：TileLang 的 copy-swap 路径避免了 CANN `RotateHalf` 实现中的 gather 开销。
- **interleaved layout 收益较小**（1.19x–1.29x）：两侧均采用 gather-based rotate，差距主要来自 TileLang 的 UB 调度与多核切分。
- **decode_bs1 是唯一 CANN 略胜的场景**（0.96x）：该 shape 极小（M=32，单核），启动开销占主导。
- CANN 延迟在不同 shape 下非常稳定（~18–21 us），说明其 tiling 策略对 shape 变化自适应较差；TileLang 随实际工作量缩放，中大 shape 收益更大。

## 二、精度测试

### Golden 来源

精度 golden 采用 CANN 官方算子 `torch_npu.npu_rotary_mul`（底层调 `aclnnRotaryPositionEmbedding`），与性能对比同一算子。

### 精度标准

混合容差双门限（`check_precision`）：

| dtype | atol | rtol | max_abs_limit | 要求匹配率 |
|-------|------|------|---------------|-----------|
| float16 | 2^-14 (6.10e-5) | 2^-9 (1.95e-3) | 1e-1 | 99% |
| bfloat16 | 2^-10 (9.77e-4) | 2^-6 (1.56e-2) | 1e0 | 99% |

### 测试覆盖

| 级别 | 用例数 | 覆盖范围 |
|------|--------|---------|
| L0 门槛 | 6 | half/interleaved × fp16/bf16 × TND/BSND 核心组合 |
| L1 功能 | 12 | 部分/全旋转、变 head_num/rope_dim、tail 行、大 batch、最小用例 |
| L2 负向 | 4 | 奇数 rope_dim、1D/5D 输入、rope_dim > hidden_size（须抛异常） |
| boundary | 6 | 大值、零值、全旋转大 batch、最小 rope_dim、单行、BSND tail |

### 语义对齐

CANN op（`npu_rotary_mul`）是 full-rope + 非原地；TileLang 是 partial-rope + 原地。对齐方式：
- 两侧都只对 `x[..., dim_start:]` slice 做完整 RoPE，仅对比该 slice。
- BSND interleave 时 CANN op 不支持 sin/cos broadcast，自动 reshape 为等效 TND。

### mode 映射注意事项

`aclnn_rotary_position_embedding.h` 注释声称 `2=interleave` 是**错误的**，实际枚举（`op_host/rotary_position_embedding_tiling.h`）：
- `0=half, 1=interleave, 2=quarter, 3=deepseek_interleave`

`npu_rotary_mul` 的 `rotary_mode` 参数用字符串 `"half"` / `"interleave"`（注意无尾随 d）。
