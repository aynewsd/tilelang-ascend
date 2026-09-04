# Lightning Indexer P1：Cube-Vector 在线 TopK

## 状态

- 分支：`opt/lightning-indexer-p1-online-topk`
- 目标实现：`example_lightning_indexer_dynamic_shape_improved.py`
- 目标平台：A3
- 范围：仅实施 P1 流水优化。由于 A3 不提供该硬件路径，P0（L0C Fixpipe 直接写 UB）不在范围内。

## 当前验证结论

- 当前保留稳定的 Expert 在线 TopK 实现：`threads=1`、256 宽 trunk、独立排序和选中结果缓冲区。
- 保留已验证的低风险修复：Q L1 在 GEMM 完成后归还，S2 循环外不追加无后继的 L1 drain wait。
- 本地事件 ID 已按有向管线对重编码到 `0..2`；quick 与 `B=1,D=64` 默认精度通过，但 `B=2,D=128` 单次 benchmark 仍出现 `507014`，问题已从普通 flag ID 越界收敛到大配置的 AICore/AIV 运行时异常。
- 目标脚本只启动一次 kernel；重复次数由 `msprof` 的 `--warm-up` 与 `--launch-count` 控制。当前没有有效的 P1 `Task Duration`，不能宣称性能收益。

## 当前验证结论

- 当前稳定实现保留 `threads=1`、256 宽 trunk、独立排序与选中结果缓冲区；未采用后续实验中的 UB 激进复用和设备 DumpTensor 探针。
- 已验证的低风险修复包括：Q L1 仅在 GEMM 完成后归还、移除 S2 循环结束后的无后继 L1 drain wait，以及避免两个 AIV 同时执行未按 `vid` 切分的 Vector 路径。
- `msprof` 目标脚本只启动一次 kernel；重复次数由 `--warm-up` 和 `--launch-count` 控制。此前采集没有产生有效 P1 `Task Duration`，因此当前不能宣称性能收益。

## 本地事件 ID 重编码

TileLang 的普通 `set_flag/wait_flag` 会直接生成 AscendC `HardEvent` 事件调用。A3/DAV 2201
的本地事件表按有向管线对独立分配，每个管线对的 ID 合法范围为 `0..7`，不能把 ID 作为
全局编号递增。当前实现按管线对复用小范围 ID：

```text
 FIX -> M       : L0C=1
 M -> FIX       : L0C=1
MTE2 -> M      : K ready=0, Q ready=1
M -> MTE2      : K free=0, Q free=1
V -> MTE2      : G reduce=0, history load=1, slot release=2
MTE2 -> V      : G ready=0, history ready=1
V -> MTE3      : history store=0, output=1
MTE3 -> V      : history store=0, output=1
```

跨核 `READY/FREE` 使用独立 namespace，当前只使用 `0..3`；不能用普通本地事件 ID 的规则
直接替代跨核 flag 的判断。每次修改事件表后都必须检查生成 AscendC，而不能只检查 Python
常量。

### 与 DeepSeek V4 实现的同步差异

参考实现 `examples/deepseek_v4/lightning_indexer.py:573` 使用 `MIX_AIC_1_2`，即一个 AIC
配两个 AIV，并在 `:847-850` 通过 `vid` 将 S1 行切分给两个 AIV。它使用固定的
`SYNC_C1V1=0` 和 `SYNC_V1C1=1`（`:476-478`），但每个物理 `cid` 只处理自己通过
`_real_start/_real_end` 划分的任务区（`:653-656`）。

当前实现使用 `MIX_AIC_1_1`（`threads=1`），不能直接照搬两个 AIV 的计数假设；当前
Vector 路径不使用 `vid`，所有 history、QK 槽和输出均按单个 task 处理。`mode=2` 的公开
语义是同组 AIC 与 AIV 同步，并没有文档证明必须存在两个 AIV；但 DAV2201 的跨核消息只
编码 mode 和 flagId，不编码 cid，故固定 `READY/FREE` 在多 task 下的隔离依赖运行时拓扑，
不能仅由 flag 数值合法性证明安全。未来若改回 `threads=2`，必须同时完成 vid 分工、初始
FREE token 计数和每个 AIV 的 workspace/输出范围切分。

## 基线评估

当前 kernel 为每个 `(B, N2, S1 tile)` 分配一个 Cube/Vector 对。

1. Cube 遍历每个 `S2` 分块和每个 GQA 组，计算 `Q @ K^T`，执行 ReLU，并将所有部分结果写入完整的 `QK_RES` GM 工作区。
2. Vector 等待 Cube 完成全部 `S2` 分块后，将完整工作区读回 UB，应用组权重、沿 G 归约、累积完整 score 行，再对 `S2` 一次性调用 `T.tile.topk`。

这会串行化 Cube 和 Vector 工作，并使完整中间张量经过 GM 往返。现有报告将完整 `QK_RES` 的 GM 流量识别为剩余的主要瓶颈；即使已有改进实现，仍显著慢于 ACLNN 基线。

参考资料：

- 当前 kernel：`example_lightning_indexer_dynamic_shape_improved.py`
- 现有性能报告：`perf_lightning_indexer_improved.md`
- 在线 TopK 参考：`ops-transformer/attention/lightning_indexer_v2/op_kernel/arch35/vf/lightning_indexer_v2_topk.h`

## 参考算法概述

AscendC LightningIndexer V2 在 S2 trunk 上执行精确在线 TopK：

1. 计算一行的一个 score trunk。
2. 选择该 trunk 的局部 TopK。
3. 第一个 trunk 保留其值和全局索引作为历史。
4. 后续每个 trunk 从先前 K 个历史候选与当前 trunk 候选的拼接中选择 TopK。
5. 历史缓冲区 ping-pong，并在最后一个 trunk 后输出最终 K 个索引。

参考实现使用四趟 radix histogram 阈值算法选择候选，并压缩高于或等于阈值的元素；它不保证 score 排序输出。当前 TileLang 测试校验 TopK 索引多重集合，符合此契约。TileLang 已提供排序/TopK 原语，但没有已确认的等价 histogram 与寄存器 gather API，因此 P1 优先用 TileLang 排序保证正确性，同时保留在线候选归约结构。

## P1 设计

### 执行归属

- 保持现有网格：一个逻辑任务拥有 `(b_id, n2_id, m_id)`。
- P1 有意为每个逻辑任务只启动一个 AIV（`threads=1`）：当前 Vector 路径的 QK 槽、history、输出和跨 Scope 标志均按 task 而非 `vid` 切分。未来 P2 只有在将全部 V 状态、处理范围和标志按 `vid` 分区后，才可使用 `threads=2`。
- Cube 一次生成一个 `(BLOCK_M, G, BLOCK_N)` QK trunk。
- Vector 处理同一任务的 trunk 及其 `BLOCK_M` 条 score 行。
- 每条 score 行仅保留 K 个历史值与 K 个全局索引；不会在 UB 或 GM 中物化完整 S2 score 行。

### 合法 A3 数据路径

P1 不假设 L0C 到 UB 的 Fixpipe 路径。每个 trunk 的路径为：

```text
GM Query/Key -> L1 -> L0C -> GM qk_slot[slot] -> UB -> 加权归约 -> 在线 TopK
```

`qk_slot` 是一个用于单个 Cube 结果 trunk 的小型双缓冲 GM 工作区，不是完整的 `QK_RES[B, N2, S1, G, S2]` 张量。Vector 消费 trunk 后复用两个槽。

### Cube-Vector 流水

对每个 S2 trunk `n` 和 ping-pong 槽 `slot = n % 2`：

1. Cube 等待 Vector 标记 `slot` 可复用。
2. Cube 将当前 K tile 拷入 L1，计算所有 GQA 组的 `(BLOCK_M, BLOCK_N)` GEMM，在 L0C 到 GM 的拷贝中执行 ReLU，并保存到 `qk_slot[slot]`。
3. Cube 发出 `slot` 已就绪信号给 Vector。
4. Vector 等待就绪信号，将组切片和权重加载到 UB，执行加权与 G 归约；它在每行处理该 trunk 前从紧凑 GM history 读取上一个 trunk 的 K 个候选（首 trunk 直接以 `-inf` 初始化），合并后立即将更新后的 history 写回 GM。
5. 仅在 `BLOCK_M` 全部行完成在线合并后，Vector 才标记 `slot` 可覆盖。

使用独立的 Cube→Vector 就绪标志对和 Vector→Cube 可复用标志对；Vector 在启动时建立两个槽的初始可复用状态，且不得在相应等待完成前复用槽。

### Expert 模式同步边

自动同步已关闭，Cube 必须在每个 trunk 的 K GM→L1 DMA 后执行 `MTE2→M` 等待，并在每个
GEMM 前对该 G 的 Q GM→L1 DMA 执行独立的 `MTE2→M` 等待。每次 history GM→UB 读取必须用
`V→MTE2` 与 `MTE2→V` 成对同步；每次 history UB→GM 写回以及最终索引 UB→GM 输出都必须用
`V→MTE3` 与 `MTE3→V` 成对同步，确保 UB 在下一行或下一 trunk 复用前 DMA 已完成。这些事件 ID
与 L0C、组归约和槽协议使用的事件 ID 分离，不改变在线 TopK 合并或双槽协议。

#### L1 所有权生命周期

`k_l1` 和 `q_l1` 都是单缓冲，MTE2 不能在 M 仍读取时重写。使用独立的
`K_L1_FREE_FLAG=0` 和 `Q_L1_FREE_FLAG=1` 作为 `M→MTE2` 的归还令牌；二者在 S2
循环开始前各建立一次初始空闲令牌。

1. 每个 K trunk：MTE2 先等待 `K_L1_FREE_FLAG`，再执行 `GM→k_l1`；DMA 完成后以既有
   `K_L1_READY_FLAG` 通知 M。M 完成该 trunk 的最后一个 G GEMM、确定不再读取 `k_l1` 后，
   才以 `K_L1_FREE_FLAG` 将所有权归还 MTE2。
2. 每个 G GEMM：MTE2 先等待 `Q_L1_FREE_FLAG`，再执行 `GM→q_l1`；DMA 完成后以既有
   `Q_L1_READY_FLAG` 通知 M。M 发射并完成消费该 Q tile 的 GEMM 后，立即以
   `Q_L1_FREE_FLAG` 归还 `q_l1`，下一组才能装载 Q。
3. 归还令牌只由 M 发送、只由 MTE2 等待；M 不等待归还令牌。因此不会形成 M 等待 MTE2、而
    MTE2 又等待 M 的循环。L0C 的 `FIX→M`、`M→FIX` 顺序及 QK 槽的跨核协议保持不变。
     L1 空闲令牌只在下一次对应的 GM→L1 DMA 前由 MTE2 消费；S2 循环结束时不再额外等待或
    清空这些令牌。

#### L1 末尾 drain 修正

曾经在 S2 循环结束后追加以下等待：

```python
T.wait_flag("M", "MTE2", K_L1_FREE_FLAG)
T.wait_flag("M", "MTE2", Q_L1_FREE_FLAG)
```

这些等待位于最后一次生产之后，循环已经没有下一次 GM 到 L1 的消费者。它们不再提供可复用缓冲区所需的顺序约束，却可能让 Cube 等待一个没有后继业务的事件，导致 C/V 流水无法正常退出并表现为 `507014`。正确做法是保留每次下一轮 DMA 前的空闲等待，以及 GEMM 消费完成后的 `M -> MTE2` 归还；只删除循环外的两条终止等待。

此外，Q L1 的归还必须位于 `M -> FIX` 的 GEMM 完成等待之后。若在 `gemm_v0` 发射后立即归还，下一次 Q DMA 可能覆盖仍被 Cube 读取的 L1 数据。

### 在线候选状态

`TOPK_HISTORY_WORKSPACE` 是第二个自动分配的 GM workspace，形状为
`[TASK_COUNT, BLOCK_M, 2 * TOP_K]`，每个 task 的每条 S1 行保存一组交错的
`[value, index]` 候选。它取代 `BLOCK_M * 2 * TOP_K` 的 UB 累积器；workspace 不保存完整
score 或完整 QK 结果。

Vector 在 UB 中一次只分配并处理一条行 history，外加以下 merge 临时状态：

```text
history_values_and_indices[2 * TOP_K]
sorted_trunk_values_and_indices[2 * BLOCK_N]
padded_trunk_values_and_indices[2 * TOP_K]
merged_values_and_indices[4 * TOP_K]
```

昇腾 AIV 的 `MrgSort` 要求两个输入源的元素数量相等。局部静态块排序生成交错的 `(score, local_index)` lane；随后对整个交错块加 `n * BLOCK_N`，使索引成为全局 S2 索引。每次排序并完成索引偏移后，先将 `2 * TOP_K` lane 的 trunk UB 缓冲区全部填为 `-inf`，再把有效的 `2 * BLOCK_N` lane 拷贝到其前缀。这样历史源和 trunk 源均以 `(TOP_K, TOP_K)` 元素配置进入归并，归并输出缓冲区须容纳 `4 * TOP_K` scalar lane。填充和拷贝完成后、归并前，以及归并完成后、历史写回前，均使用 V 管线屏障隔离生产者和消费者。第一个 trunk 不读取未初始化的 GM history，而是在 UB 将交错 history lane 全部填为 `-inf`；每次 merge 后将前 `2 * TOP_K` lane 写回该行 workspace。最终读取最后一份 history，用 `P1010` 抽取索引并转换为 `int32`。

### 目录与性能脚本约定

性能目标脚本统一放在 `perf/`，历史和新采集结果放在 `profiling/`。`msprof_run.py` 只运行命令行指定的目标脚本，并复用 `profiling/perf_lightning_indexer.json` 中已有的 baseline，不重复执行 baseline。目标脚本本身不得再实现应用侧重复 launch，避免把 warmup、采样和应用循环叠加。

### P1 默认 trunk 宽度

默认完整用例使用 `BLOCK_N=256` 和 `VECTOR_BASEN=256`；`test_indexer` 保留参数化宽度，以便小型回归用例仍可显式使用 64 宽。已测得默认 `BLOCK_N=64` 时形成 32,768 次在线 merge 并发生超时。对固定完整用例（`S1=512`、`S2=4096`、`BLOCK_M=64`），将 trunk 宽度增至 256 后，S2 trunk 数由 64 降为 16，默认在线 merge 数降至 8,192。对应 L0C 累加器占用为 `64 * 256 * 4 = 64KB`，在 128KB L0C 容量内；该调整不改变在线合并算法、C/V 标志、workspace 布局公式或 A3 数据路径。

### 空间复杂度与 UB 风险

以 `B=2`、`S1=512`、`S2=4096`、`N2=1`、`G=32`、`D=64`、`BLOCK_M=64`、`BLOCK_N=256`、`TOP_K=1024` 为例，`TASK_COUNT = B * N2 * (S1 / BLOCK_M) = 16`，并按 `float32` workspace 计算：

- GM `QK_SLOT`：`16 * 2 * 64 * 32 * 256 * 4 = 67,108,864 B = 64 MiB`。
- GM history：`16 * 64 * (2 * 1024) * 4 = 8,388,608 B = 8 MiB`。
- P1 GM workspace 合计：`64 MiB + 8 MiB = 72 MiB`。
- 旧版完整 `QK_RES`：`2 * 1 * 512 * 32 * 4096 * 4 = 536,870,912 B = 512 MiB`。
- P1 相对旧完整 workspace 减少 `440 MiB`，空间缩小约 `7.11 倍`（占旧版约 `14.06%`）。

Vector UB 的显式缓冲区总量约为 `85 KiB`：`mm_res_ub` 16 KiB、`weight_ub` 64 B、`reduce_tmp_ub` 16 KiB、`reduce_g_ub` 1 KiB、`history_ub` 8 KiB、`sorted_block_ub` 2 KiB、`padded_trunk_ub` 8 KiB、`merged_ub` 16 KiB、`index_lane_ub` 2 KiB、`selected_ub` 8 KiB、`topk_index_ub` 4 KiB、`output_ub` 4 KiB。实际 lowering 后的 UB 保留量为 `196352 B`，而 192 KiB 容量为 `196608 B`，仅余 `256 B` 余量；这是一项高风险资源条件，任何新增 UB 临时量、对齐膨胀或 pass 行为变化都可能导致溢出。

### 末尾令牌与同步闭合

所有真实复用边均有成对的 `set/wait`：QK 双槽的 Cube→Vector 就绪与 Vector→Cube 释放、K/Q L1 的 MTE2 装载与 M 归还、history 的 GM↔UB DMA，以及最终输出的 UB→GM DMA 都遵循该规则。S2 最后一个 producer 完成消费后，L0C、K L1、Q L1 的空闲令牌可以保留未被后续消费者等待；这是有意的，因为已经不存在未来复用者。此前追加末尾等待会造成超时，因此不得为“清空”这些终端令牌添加终止等待。

### 支持域

当前实现明确保持既有约束：`S1 % BLOCK_M == 0`、`S2 % BLOCK_N == 0`、`G % VECTOR_BASEG == 0`，且 `TOP_K <= S2`。未实现或验证尾块，因此不得把该实现用于非整除 S1/S2/G 或 `TOP_K > S2` 的输入。A3 UB 上限为 192 KB；禁止恢复 `BLOCK_M × 2 × TOP_K` 的 history UB 分配，Vector history 必须保持为单行 `2 × TOP_K`，跨 trunk 状态仅保存在上述 task/row GM workspace 中。

## 验证门禁

- 正确性 oracle：`example_lightning_indexer_dynamic_shape_improved.py` 中的 `index_golden`。
- 按输出行比较 TopK 索引多重集合，因为 AscendC 在线算法与候选合并契约均不保证并列分数的排序顺序。
- 当前验收门禁为索引多重集合匹配率严格大于 `0.99`；测试必须持续打印匹配率和不匹配索引数。目标域 FP16 边界分数可导致少量候选差异，例如已观测匹配率为 `0.999695`，这不应被误判为同步竞争。
- 对唯一 score 的精确索引相等仅作为诊断信号，用于排查算法或同步问题；它不是当前 P1 的通过条件，也不得以此隐藏或替代多重集合匹配率结果。
- 运行小型整除用例以及当前固定用例（`B=1, S1=512, S2=4096, N2=1, G=32, D=64, TOP_K=1024`）后才可进行基准测试。
- 性能验收使用 `msprof op`，记录 kernel 时间、基线时间以及相对 P1 前 improved 实现的变化。

## 风险与决策

- 在线合并减少长期 GM 工作区并允许重叠，但 A3 仍要求每个 QK trunk 执行一次短暂的 L0C→GM→UB 传输；不得宣称存在直接 Cube→Vector UB 传输或 P0 收益。
- `T.tile.sort` 与 `T.tile.merge_sort` 的内部临时存储可能多于 AscendC radix selector。若 profiling 证明局部选择器主导，应在 P1 正确性完成后单独决定 P2 是否新增或暴露 radix-selection 原语。
- 跨 Scope 标志与按任务私有的工作区归属必须通过生成的 AscendC 验证；错误握手可能覆盖槽或导致死锁。
- 临时分支保持为开发分支，直到 P1 通过精度和性能审查；只有之后才可将 P1 提交压缩到 `feat/lightning_indexer_new`，失败版本保留在此分支用于诊断。
