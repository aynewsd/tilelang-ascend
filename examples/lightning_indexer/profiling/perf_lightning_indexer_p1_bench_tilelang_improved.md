# Lightning Indexer P1 性能采集：bench_tilelang_improved.py

Generated: 2026-09-04 11:02:47

## 环境

| Item | Value |
|------|-------|
| Device | Ascend910_9392 |
| CANN | /usr/local/Ascend/cann-9.1.0-beta.1 |
| Profiler | msprof op (device Task Duration) |

## 采集方法

- 仅执行本次 `--target`，不会运行或刷新 baseline。
- 目标脚本每次执行仅启动一次 kernel，不提供应用侧重复控制。
- msprof 的 warmup 与采样次数分别由 `--warm-up`、`--launch-count` 显式控制。
- msprof 发生异常时保留其完整输出，便于分析设备报错与运行时日志。
- 历史 baseline 仅来自 `perf_lightning_indexer.json`，并按配置标记可比性。

## 结果

| S2 | P1 采集状态 | P1 target median (us) | 历史 baseline median (us) | 比值 | baseline 可比性 |
|----|------------|-----------------------|--------------------------|------|----------------|
| 4096 | 失败（无有效 kernel 目录） | N/A | 408.99 | N/A | 仅供参考 |
> S2=4096 baseline 注意事项：历史 baseline 的 msprof warm-up 设置不同；历史 baseline 的 msprof launch-count 设置不同。

> 比值 = 历史 baseline / P1 target；仅在“可比”时可用于性能结论。

## Kernel 详情

### S2=4096

**P1 target**: No data；目标应用触发 `507014` AICore timeout，且 `OPPROF` 目录没有有效 kernel 子目录。

**历史 baseline（未重跑）** kernel `LightningIndexer_468c104183baa47aac33ce774c8668e0_196865_mix_aic`:

- Op Type: mix
- Instances: 10
- Task Duration (min): 408.22 us
- Task Duration (median): 408.99 us
- Task Duration (mean): 408.91 us
- Task Duration (max): 409.84 us
- Block Dim: 24
- Cube Ratio: 23.67345%
- MTE2 Ratio: 9.96595%
- VEC Ratio: 0.0%
- L2 Read Hit Rate: 93.770962%


## 原始数据

msprof output: `/root/workspace/tilelang-ascend/examples/lightning_indexer/msprof_output/`
Structured JSON: `/root/workspace/tilelang-ascend/examples/lightning_indexer/profiling/perf_lightning_indexer_p1_bench_tilelang_improved.json`
