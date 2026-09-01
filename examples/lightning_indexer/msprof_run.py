"""lightning_indexer 的 P1 目标脚本 msprof 采集器。

默认仅采集 ``bench_tilelang_improved.py``，不会重新运行 baseline。历史 baseline
从 ``perf_lightning_indexer.json`` 读取；若配置不一致，报告会明确标记不可直接横比。
目标脚本每次执行只启动一次 kernel；采样次数仅由显式的 ``--warm-up`` 和
``--launch-count`` 控制。
"""

import argparse
import csv
import json
import os
import signal
import shutil
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILING_DIR = os.path.join(SCRIPT_DIR, "profiling")
PERF_DIR = os.path.join(SCRIPT_DIR, "perf")
DEFAULT_TARGET = os.path.join(PERF_DIR, "bench_tilelang_improved.py")
LEGACY_PERF_JSON = os.path.join(PROFILING_DIR, "perf_lightning_indexer.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "msprof_output")


AIC_METRICS = "ArithmeticUtilization,PipeUtilization,Memory,L2Cache,MemoryUB,ResourceConflictRatio"


def find_latest_opprof(base_dir):
    """查找 base_dir 下最新的 OPPROF_* 目录。"""
    if not os.path.exists(base_dir):
        return None
    dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("OPPROF_")])
    return os.path.join(base_dir, dirs[-1]) if dirs else None


def find_kernel_dirs(opprof_dir):
    """查找所有 kernel 子目录（OPPROF_*/<kernel_name>/<id>）。"""
    results = []
    if not opprof_dir or not os.path.exists(opprof_dir):
        return results
    for entry in os.listdir(opprof_dir):
        entry_path = os.path.join(opprof_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        for sub in os.listdir(entry_path):
            sub_path = os.path.join(entry_path, sub)
            if os.path.isdir(sub_path):
                results.append((entry, sub, sub_path))
    return results


def read_csv_rows(csv_path):
    """读取 CSV 文件的全部行。"""
    if not csv_path or not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_csv(kernel_dir, prefix):
    """在 kernel_dir 中查找指定前缀的首个 CSV 文件。"""
    if not os.path.exists(kernel_dir):
        return None
    for f in os.listdir(kernel_dir):
        if f.startswith(prefix) and f.endswith(".csv"):
            return os.path.join(kernel_dir, f)
    return None


def read_cube_row(kernel_dir, prefix):
    """从指标 CSV 中读取 sub_block_id 为 cube0 的行。"""
    csv_path = find_csv(kernel_dir, prefix)
    rows = read_csv_rows(csv_path)
    for row in rows:
        if row.get("sub_block_id") == "cube0":
            return row
    return rows[0] if rows else {}


def parse_metrics(opprof_dir, label):
    """解析 msprof op 输出中的任务时长和关键比率。"""
    kernel_dirs = find_kernel_dirs(opprof_dir)
    if not kernel_dirs:
        print(f"  [{label}] no kernel dirs found under {opprof_dir}")
        return None

    all_metrics = []
    for kernel_name, kid, kdir in kernel_dirs:
        metrics = {"kernel_name": kernel_name, "kernel_id": kid}

        # OpBasicInfo：任务时长和 Block Dim。
        basic_csv = find_csv(kdir, "OpBasicInfo")
        basic_rows = read_csv_rows(basic_csv)
        if basic_rows:
            row = basic_rows[0]
            metrics["task_duration_us"] = float(row.get("Task Duration(us)", 0))
            metrics["block_dim"] = int(row.get("Block Dim", 0))
            metrics["op_type"] = row.get("Op Type", row.get("OP Type", ""))

        # ArithmeticUtilization：cube_ratio。
        arith = read_cube_row(kdir, "ArithmeticUtilization")
        if arith:
            metrics["cube_ratio"] = float(arith.get("aic_cube_ratio", 0)) * 100

        # PipeUtilization：mte2_ratio、vec_ratio。
        pipe = read_cube_row(kdir, "PipeUtilization")
        if pipe:
            metrics["mte2_ratio"] = float(pipe.get("aic_mte2_ratio", 0)) * 100
            metrics["vec_ratio"] = float(pipe.get("aic_vec_ratio", 0)) * 100

        # L2Cache：read_hit_rate。
        l2 = read_cube_row(kdir, "L2Cache")
        if l2:
            metrics["l2_read_hit_rate"] = float(l2.get("aic_read_hit_rate(%)", 0))

        all_metrics.append(metrics)
        print(
            f"  [{label}] kernel={kernel_name} id={kid} "
            f"task={metrics.get('task_duration_us', 'N/A')}us "
            f"block_dim={metrics.get('block_dim', 'N/A')} "
            f"op_type={metrics.get('op_type', 'N/A')}"
        )

    return all_metrics


def aggregate_metrics(all_metrics, label):
    """将多个 kernel 实例的指标聚合为汇总统计。"""
    if not all_metrics:
        return None

    durations = [m.get("task_duration_us", 0) for m in all_metrics if m.get("task_duration_us")]
    if not durations:
        return None

    import statistics

    summary = {
        "n_instances": len(durations),
        "task_duration_min_us": min(durations),
        "task_duration_median_us": statistics.median(durations),
        "task_duration_mean_us": statistics.mean(durations),
        "task_duration_max_us": max(durations),
        "block_dim": all_metrics[0].get("block_dim"),
        "op_type": all_metrics[0].get("op_type"),
        "kernel_name": all_metrics[0].get("kernel_name"),
    }

    # 聚合比率：各实例通常相近，取中位数。
    for key in ("cube_ratio", "mte2_ratio", "vec_ratio", "l2_read_hit_rate"):
        vals = [m.get(key) for m in all_metrics if m.get(key) is not None]
        if vals:
            summary[key] = statistics.median(vals)

    print(
        f"  [{label}] aggregated {summary['n_instances']} instances: "
        f"min={summary['task_duration_min_us']:.2f}us "
        f"median={summary['task_duration_median_us']:.2f}us "
        f"mean={summary['task_duration_mean_us']:.2f}us"
    )
    return summary


def run_msprof(target_script, target_args, output_subdir, warm_up, launch_count, kernel_name=None, timeout=120):
    """运行 msprof op，并在超时时终止整个进程组。"""
    output_path = os.path.join(OUTPUT_DIR, output_subdir)
    if os.path.exists(output_path):
        shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)

    app_cmd = f"{sys.executable} {target_script} {target_args}"
    cmd = [
        "msprof",
        "op",
        f"--application={app_cmd}",
        f"--output={output_path}",
        f"--aic-metrics={AIC_METRICS}",
        f"--launch-count={launch_count}",
        f"--warm-up={warm_up}",
    ]
    if kernel_name:
        cmd.append(f"--kernel-name={kernel_name}")

    print(f"  运行: msprof op --application=python {os.path.basename(target_script)} {target_args}")
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"  [错误] msprof 在 {timeout}s 后超时，正在终止整个进程组")
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                print("  [错误] SIGTERM 后进程组仍未退出，发送 SIGKILL")
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            return {"status": "timeout", "returncode": process.returncode, "stdout": stdout,
                    "stderr": stderr, "message": f"msprof 在 {timeout}s 后超时，已终止进程组"}
        if process.returncode != 0:
            print(f"  [错误] msprof 以退出码 {process.returncode} 结束")
            if stderr:
                print(f"  stderr（末尾 500 字符）: {stderr[-500:]}")
            return {"status": "failed", "returncode": process.returncode, "stdout": stdout,
                    "stderr": stderr, "message": "msprof 非零退出"}
    except FileNotFoundError:
        print("  [错误] 未找到 msprof 命令")
        return {"status": "failed", "returncode": None, "message": "未找到 msprof 命令"}

    opprof_dir = find_latest_opprof(output_path)
    if not opprof_dir:
        message = f"{output_path} 下未找到 OPPROF_* 目录"
        print(f"  [错误] {message}")
        return {"status": "failed", "returncode": process.returncode, "message": message}

    return {"status": "success", "returncode": process.returncode, "opprof_dir": opprof_dir}


def resolve_target(target):
    """解析目录内目标脚本，拒绝意外执行目录外文件。"""
    target_path = target if os.path.isabs(target) else os.path.join(SCRIPT_DIR, target)
    target_path = os.path.abspath(target_path)
    if os.path.commonpath([SCRIPT_DIR, target_path]) != SCRIPT_DIR:
        raise ValueError("--target 必须指向 SCRIPT_DIR目录内的脚本")
    if not target_path.endswith(".py") or not os.path.isfile(target_path):
        raise ValueError(f"目标脚本不存在或不是 Python 文件: {target}")
    return target_path


def load_legacy_baseline():
    """读取历史 baseline，不在 P1 采集中重复执行它。"""
    try:
        with open(LEGACY_PERF_JSON, encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except (OSError, json.JSONDecodeError) as error:
        print(f"  [warn] 无法读取历史 baseline: {error}")
        return []


def reference_baseline(legacy_results, config):
    """按计算配置匹配历史 baseline，并返回可比性说明。"""
    config_keys = ("B", "S1", "S2", "G", "N2", "D", "top_k")
    for item in legacy_results:
        legacy_config = item.get("config", {})
        if all(legacy_config.get(key) == config[key] for key in config_keys):
            caveats = []
            if legacy_config.get("warm_up") != config["warm_up"]:
                caveats.append("历史 baseline 的 msprof warm-up 设置不同")
            if legacy_config.get("launch_count") != config["launch_count"]:
                caveats.append("历史 baseline 的 msprof launch-count 设置不同")
            return item.get("baseline"), {
                "compatible": not caveats,
                "caveats": caveats,
                "legacy_config": legacy_config,
            }
    return None, {
        "compatible": False,
        "caveats": ["历史 baseline 中没有相同的 B/S1/S2/G/N2/D/top_k 配置"],
        "legacy_config": None,
    }


def run_one_config(
    target_script,
    target_label,
    legacy_results,
    B,
    S1,
    S2,
    G,
    N2,
    D,
    top_k,
    warm_up,
    launch_count,
    kernel_name,
    timeout,
):
    """只采集一个目标脚本，并关联已有历史 baseline。"""
    config_name = f"B{B}_S1{S1}_S2{S2}_G{G}_D{D}_K{top_k}"
    print(f"\n{'=' * 60}")
    print(f"Config: {config_name}")
    print(f"{'=' * 60}")

    config = {
        "B": B, "S1": S1, "S2": S2, "G": G, "N2": N2, "D": D, "top_k": top_k,
        "warm_up": warm_up, "launch_count": launch_count,
    }
    target_args = (
        f"--B {B} --S1 {S1} --S2 {S2} --G {G} --N2 {N2} --D {D} "
        f"--top-k {top_k}"
    )

    print(f"\n[target] {os.path.basename(target_script)}（仅此目标，不运行 baseline）")
    target_run = run_msprof(
        target_script,
        target_args,
        f"p1_{target_label}_{config_name}",
        warm_up,
        launch_count,
        kernel_name=kernel_name,
        timeout=timeout,
    )
    target_opprof = target_run.get("opprof_dir")
    target_raw = parse_metrics(target_opprof, target_label) if target_run["status"] == "success" else None
    target_metrics = aggregate_metrics(target_raw, target_label) if target_raw else None
    baseline_metrics, compatibility = reference_baseline(legacy_results, config)

    return {
        "config": config,
        "target_run": target_run,
        "target": target_metrics,
        "baseline_reference": baseline_metrics,
        "baseline_compatibility": compatibility,
    }


def generate_markdown(results, output_path, target_name):
    """生成 P1 目标采集报告，并说明历史 baseline 的可比性。"""
    lines = []
    lines.append(f"# Lightning Indexer P1 性能采集：{target_name}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 环境。
    lines.append("## 环境")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
    try:
        import torch_npu

        dev_name = torch_npu.npu.get_device_name(0)
        lines.append(f"| Device | {dev_name} |")
    except Exception:
        lines.append("| Device | N/A |")
    lines.append(f"| CANN | {os.environ.get('ASCEND_HOME_PATH', 'N/A')} |")
    lines.append("| Profiler | msprof op (device Task Duration) |")
    lines.append("")

    # 采集方法。
    lines.append("## 采集方法")
    lines.append("")
    lines.append("- 仅执行本次 `--target`，不会运行或刷新 baseline。")
    lines.append("- 目标脚本每次执行仅启动一次 kernel，不提供应用侧重复控制。")
    lines.append("- msprof 的 warmup 与采样次数分别由 `--warm-up`、`--launch-count` 显式控制。")
    lines.append("- `--timeout` 超时会终止 msprof 及其启动的整个进程组，并记录失败状态。")
    lines.append("- 历史 baseline 仅来自 `perf_lightning_indexer.json`，并按配置标记可比性。")
    lines.append("")

    # 结果表。
    lines.append("## 结果")
    lines.append("")
    lines.append("| S2 | P1 采集状态 | P1 target median (us) | 历史 baseline median (us) | 比值 | baseline 可比性 |")
    lines.append("|----|------------|-----------------------|--------------------------|------|----------------|")

    for r in results:
        cfg = r["config"]
        target = r.get("target")
        target_run = r["target_run"]
        baseline = r.get("baseline_reference")

        def fmt(val):
            if val is None:
                return "N/A"
            if isinstance(val, float):
                return f"{val:.2f}"
            return str(val)

        target_dur = fmt(target.get("task_duration_median_us") if target else None)
        baseline_dur = fmt(baseline.get("task_duration_median_us") if baseline else None)

        target_dur_f = target.get("task_duration_median_us") if target else 0
        baseline_dur_f = baseline.get("task_duration_median_us") if baseline else 0
        speedup = f"{baseline_dur_f / target_dur_f:.2f}x" if target_dur_f and baseline_dur_f else "N/A"
        compatibility = r["baseline_compatibility"]
        comparison = "可比" if compatibility["compatible"] else "仅供参考"

        status = target_run["status"]
        status_text = {"success": "成功", "failed": "失败", "timeout": "超时"}[status]
        lines.append(f"| {cfg['S2']} | {status_text} | {target_dur} | {baseline_dur} | {speedup} | {comparison} |")
        if status != "success":
            lines.append(f"> S2={cfg['S2']} P1 采集{status_text}：{target_run.get('message', '无详细信息')}。")
        if compatibility["caveats"]:
            lines.append(f"> S2={cfg['S2']} baseline 注意事项：{'；'.join(compatibility['caveats'])}。")

    lines.append("")
    lines.append("> 比值 = 历史 baseline / P1 target；仅在“可比”时可用于性能结论。")
    lines.append("")

    # Kernel 详情。
    lines.append("## Kernel 详情")
    lines.append("")
    for r in results:
        cfg = r["config"]
        target_run = r["target_run"]
        lines.append(f"### S2={cfg['S2']}")
        lines.append("")
        if target_run["status"] != "success":
            lines.append(f"**P1 采集状态**: {target_run['status']}（{target_run.get('message', '无详细信息')}）")
            lines.append("")
        for side, label in [("target", "P1 target"), ("baseline_reference", "历史 baseline（未重跑）")]:
            m = r.get(side)
            if not m:
                lines.append(f"**{label}**: No data")
                lines.append("")
                continue
            lines.append(f"**{label}** kernel `{m.get('kernel_name', 'N/A')}`:")
            lines.append("")
            lines.append(f"- Op Type: {m.get('op_type', 'N/A')}")
            lines.append(f"- Instances: {m.get('n_instances', 'N/A')}")
            lines.append(f"- Task Duration (min): {m.get('task_duration_min_us', 'N/A'):.2f} us" if isinstance(m.get('task_duration_min_us'), (int, float)) else f"- Task Duration (min): N/A")
            lines.append(f"- Task Duration (median): {m.get('task_duration_median_us', 'N/A'):.2f} us" if isinstance(m.get('task_duration_median_us'), (int, float)) else f"- Task Duration (median): N/A")
            lines.append(f"- Task Duration (mean): {m.get('task_duration_mean_us', 'N/A'):.2f} us" if isinstance(m.get('task_duration_mean_us'), (int, float)) else f"- Task Duration (mean): N/A")
            lines.append(f"- Task Duration (max): {m.get('task_duration_max_us', 'N/A'):.2f} us" if isinstance(m.get('task_duration_max_us'), (int, float)) else f"- Task Duration (max): N/A")
            lines.append(f"- Block Dim: {m.get('block_dim', 'N/A')}")
            lines.append(f"- Cube Ratio: {m.get('cube_ratio', 'N/A')}%" if isinstance(m.get('cube_ratio'), (int, float)) else f"- Cube Ratio: N/A")
            lines.append(f"- MTE2 Ratio: {m.get('mte2_ratio', 'N/A')}%" if isinstance(m.get('mte2_ratio'), (int, float)) else f"- MTE2 Ratio: N/A")
            lines.append(f"- VEC Ratio: {m.get('vec_ratio', 'N/A')}%" if isinstance(m.get('vec_ratio'), (int, float)) else f"- VEC Ratio: N/A")
            lines.append(f"- L2 Read Hit Rate: {m.get('l2_read_hit_rate', 'N/A')}%" if isinstance(m.get('l2_read_hit_rate'), (int, float)) else f"- L2 Read Hit Rate: N/A")
            lines.append("")
        lines.append("")

    # 原始数据位置。
    lines.append("## 原始数据")
    lines.append("")
    lines.append(f"msprof output: `{OUTPUT_DIR}/`")
    lines.append(f"Structured JSON: `{output_path.replace('.md', '.json')}`")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="仅采集 lightning_indexer 的指定 P1 目标脚本")
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="目标脚本名称或路径（必须位于本目录；默认：perf/bench_tilelang_improved.py）",
    )
    parser.add_argument("--kernel-name", default="main_kernel", help="传给 msprof 的 kernel 名称过滤器")
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--S1", type=int, default=512)
    parser.add_argument("--S2", type=int, default=4096)
    parser.add_argument("--G", type=int, default=32)
    parser.add_argument("--N2", type=int, default=1)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--warm-up", type=int, default=5)
    parser.add_argument("--launch-count", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=120, help="msprof 最大运行秒数；默认 120")
    parser.add_argument("--preset", default="default", choices=["default", "sweep"])
    parser.add_argument("--dry-run", action="store_true", help="仅检查目标与历史 baseline 匹配，不启动 msprof")
    args = parser.parse_args()

    if args.timeout < 1:
        parser.error("--timeout 必须至少为 1")
    try:
        target_script = resolve_target(args.target)
    except ValueError as error:
        parser.error(str(error))

    target_label = os.path.splitext(os.path.basename(target_script))[0]
    legacy_results = load_legacy_baseline()
    if args.dry_run:
        dry_config = {
            "B": args.B, "S1": args.S1, "S2": args.S2, "G": args.G, "N2": args.N2,
            "D": args.D, "top_k": args.top_k, "warm_up": args.warm_up,
            "launch_count": args.launch_count,
        }
        _, compatibility = reference_baseline(legacy_results, dry_config)
        print(f"目标脚本: {target_script}")
        print("应用侧 kernel launch: 1")
        print(f"历史 baseline 可比性: {'可比' if compatibility['compatible'] else '仅供参考'}")
        for caveat in compatibility["caveats"]:
            print(f"- {caveat}")
        return
    results = []

    if args.preset == "sweep":
        for s2 in [512, 1024, 2048, 4096]:
            results.append(
                run_one_config(
                    target_script, target_label, legacy_results, args.B, args.S1, s2, args.G,
                    args.N2, args.D, args.top_k, args.warm_up, args.launch_count,
                    args.kernel_name, args.timeout,
                )
            )
    else:
        results.append(
            run_one_config(
                target_script, target_label, legacy_results, args.B, args.S1, args.S2, args.G,
                args.N2, args.D, args.top_k, args.warm_up, args.launch_count,
                args.kernel_name, args.timeout,
            )
        )

    output_stem = f"perf_lightning_indexer_p1_{target_label}"
    json_path = os.path.join(PROFILING_DIR, f"{output_stem}.json")
    with open(json_path, "w") as f:
        json.dump({"target": os.path.basename(target_script), "results": results}, f, indent=2)
    print(f"\nJSON results: {json_path}")

    md_path = os.path.join(PROFILING_DIR, f"{output_stem}.md")
    generate_markdown(results, md_path, os.path.basename(target_script))


if __name__ == "__main__":
    main()
