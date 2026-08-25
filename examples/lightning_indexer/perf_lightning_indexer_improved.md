# Lightning Indexer 优化报告：TileLang Improved vs Baseline

Generated: 2026-08-25

## 1. 优化目标

对 `example_lightning_indexer_dynamic_shape.py` 进行性能优化，目标达到 torch_npu baseline 的 0.8x 加速比。

## 2. 优化版本

`example_lightning_indexer_dynamic_shape_improved.py`

### 优化策略

| # | 优化点 | 原始 | 优化 | 效果 |
|---|--------|------|------|------|
| 1 | **Block Dim 提升** | `T.Kernel(B*N2)` = 2 blocks | `T.Kernel(B*N2*S1_TILES)` = 16 blocks | 并行度 8x |
| 2 | **Cube barrier 去除** | 每次 copy/gemm 前后都有 `T.barrier_all()` | 仅保留 Cube→Vector cross flag 同步 | 减少 barrier 开销 |
| 3 | **K hoisting** | 每个 g 迭代都重新搬运 K | K 搬运提到 g 循环外 | MTE2 22%→4.3% |
| 4 | **BLOCK_N 增大** | 64 | 128 | S2 循环次数减半 |
| 5 | **VECTOR_BASEN 增大** | 256 | 512 | Vector 循环次数减半 |
| 6 | **Python for→T.serial** | `for i in range(256)` 展开 256 次 | `T.serial(256)` | 减少 icache 压力 |

## 3. 性能对比

### 3.1 原始 vs 优化 vs Baseline (msprof op, median Task Duration)

| S2 | Original (us) | Improved (us) | Baseline (us) | Orig→Imp | vs Baseline |
|----|---------------|---------------|---------------|----------|-------------|
| 512 | 5134.67 | 749.65 | 94.52 | 6.85x | 0.126x |
| 1024 | 10147.04 | 1478.76 | 143.13 | 6.86x | 0.097x |
| 2048 | 20628.23 | 2985.59 | 236.42 | 6.91x | 0.079x |
| 4096 | 42231.39 | 6188.93 | 408.99 | 6.82x | 0.066x |

### 3.2 Block Dim 对比

| S2 | Original Block Dim | Improved Block Dim | Baseline Block Dim |
|----|--------------------|--------------------|--------------------|
| 所有 | 2 | 16 | 24 |

### 3.3 Cube/MTE2 指标对比 (S2=4096)

| Metric | Original | Improved v1 | Improved v2 | Baseline |
|--------|----------|-------------|-------------|----------|
| Task Duration (us) | 42231 | 8754 | 6189 | 409 |
| Block Dim | 2 | 16 | 16 | 24 |
| Cube Ratio | 3.59% | 2.16% | 2.35% | 23.61% |
| MTE2 Ratio | 14.95% | 22.11% | 4.31% | 10.46% |
| L2 Hit Rate | 99.02% | 99.22% | 93.73% | 93.56% |

## 4. 瓶颈分析

### 4.1 已优化部分（6.8x 提升）

- **并行度**：Block Dim 2→16（8x），是最大的提升来源
- **数据搬运**：K hoisting + 增大 BLOCK_N，MTE2 从 22%→4.3%
- **Vector 循环**：VECTOR_BASEN 256→512，减少 s2 循环次数

### 4.2 无法绕过的架构瓶颈（剩余 8-12x 差距）

**核心问题：Cube 写 GM → Vector 读 GM 的 512MB 数据往返**

教学版 kernel 的架构是两阶段串行：
1. **Cube 阶段**：GEMM → 写 `QK_RES[B, N2, S1, G, S2]` 到 GM
2. **Vector 阶段**：从 GM 读 `QK_RES` → 加权 → reduce → topk

`QK_RES` shape = `[2, 1, 512, 32, 4096]` × float32 = **512MB**

这意味着：
- Cube 写 512MB 到 GM（fixpipe + MTE3）
- Vector 从 GM 读 512MB 到 UB（MTE2）
- **仅数据搬运就需 ~512MB / ~100GB/s ≈ 5ms**，已超过 baseline 总耗时（0.4ms）

Baseline (aclnnLightningIndexer) 的 410us 表明它**避免了完整的 GM 往返**，可能策略：
- Cube+Vector 流水化（fixpipe 直达 UB，不经 GM）
- 分块处理（每次只处理一个 S2 tile，不写完整 QK_RES）
- L1/UB 级别的数据复用

### 4.3 Cube 利用率极低（2.35% vs baseline 23.61%）

即使优化后，Cube ratio 仍仅 2.35%。原因：
- 每个 GEMM tile 只有 `64×128×128` 的计算量
- Cube 被 fixpipe（L0C→GM 写入）阻塞
- GEMM 和 fixpipe 串行执行，无 double buffer

### 4.4 Vector 阶段串行处理 64 行 S1

每个 block 的 Vector 核需串行处理 BLOCK_M=64 行 S1，每行遍历 S2×G 次。即使 Cube 提速，Vector 仍是瓶颈。

## 5. 进一步优化方向（需架构级改动）

| 优先级 | 方向 | 预期效果 | 难度 |
|--------|------|----------|------|
| P0 | **Cube+Vector 流水化**：fixpipe 直达 UB，不经 GM | 消除 512MB 往返，~10x | 高（需重构内存层级） |
| P0 | **分块 topk**：每个 S2 tile 做 local topk → merge | 避免全量 score_accum | 中 |
| P1 | **L0C double buffer**：GEMM 和 fixpipe 并行 | Cube ratio 翻倍 | 中 |
| P1 | **Cube+Vector 跨核流水**：Cube 写一个 tile，Vector 立即处理 | 隐藏 Vector 延迟 | 高 |
| P2 | **增大 BLOCK_M**：128 或 256 | 提高 GEMM 效率 | 低（受 L1 限制） |

## 6. 结论

| 指标 | 结果 |
|------|------|
| 优化版相对原始版 | **6.8x 提升** |
| 优化版相对 baseline | **0.066x ~ 0.126x**（未达到 0.8x 目标） |
| 主要瓶颈 | Cube→GM→Vector 的 512MB 数据往返（架构级） |
| 可行路径 | 需重写为 Cube+Vector 流水化架构（参考 `deepseek_v4/lightning_indexer.py`） |

**当前教学版架构有本质限制**：Cube 和 Vector 通过 GM 中转数据，搬运量 512MB 远超 baseline 总耗时。达到 0.8x 目标需要架构级重写，消除 GM 往返。

## 7. 文件清单

| File | Description |
|------|-------------|
| `example_lightning_indexer_dynamic_shape_improved.py` | 优化版 kernel |
| `bench_tilelang_improved.py` | msprof op target 脚本（优化版） |
| `perf_lightning_indexer_improved.md` | 本报告 |
