"""msprof op runner: TileLang lightning_indexer vs torch_npu baseline.

Usage:
    # single shape
    python msprof_run.py --S2 4096

    # sweep S2
    python msprof_run.py --preset sweep

    # custom
    python msprof_run.py --S2 8192 --B 1 --warm-up 5 --launch-count 10

msprof op --warm-up / --launch-count control warmup and capture.
Target scripts (bench_baseline.py / bench_tilelang.py) only launch kernels.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_BASELINE = os.path.join(SCRIPT_DIR, "bench_baseline.py")
BENCH_TILELANG = os.path.join(SCRIPT_DIR, "bench_tilelang.py")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "msprof_output")

AIC_METRICS = "ArithmeticUtilization,PipeUtilization,Memory,L2Cache,MemoryUB,ResourceConflictRatio"


def find_latest_opprof(base_dir):
    """Find the latest OPPROF_* directory under base_dir."""
    if not os.path.exists(base_dir):
        return None
    dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("OPPROF_")])
    return os.path.join(base_dir, dirs[-1]) if dirs else None


def find_kernel_dirs(opprof_dir):
    """Find all kernel subdirectories (OPPROF_*/<kernel_name>/<id>)."""
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
    """Read all rows from a CSV file."""
    if not csv_path or not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_csv(kernel_dir, prefix):
    """Find the first CSV file with given prefix in kernel_dir."""
    if not os.path.exists(kernel_dir):
        return None
    for f in os.listdir(kernel_dir):
        if f.startswith(prefix) and f.endswith(".csv"):
            return os.path.join(kernel_dir, f)
    return None


def read_cube_row(kernel_dir, prefix):
    """Read the cube0 sub_block_id row from a metric CSV."""
    csv_path = find_csv(kernel_dir, prefix)
    rows = read_csv_rows(csv_path)
    for row in rows:
        if row.get("sub_block_id") == "cube0":
            return row
    return rows[0] if rows else {}


def parse_metrics(opprof_dir, label):
    """Parse msprof op output for Task Duration and key ratios."""
    kernel_dirs = find_kernel_dirs(opprof_dir)
    if not kernel_dirs:
        print(f"  [{label}] no kernel dirs found under {opprof_dir}")
        return None

    all_metrics = []
    for kernel_name, kid, kdir in kernel_dirs:
        metrics = {"kernel_name": kernel_name, "kernel_id": kid}

        # OpBasicInfo -> Task Duration, Block Dim
        basic_csv = find_csv(kdir, "OpBasicInfo")
        basic_rows = read_csv_rows(basic_csv)
        if basic_rows:
            row = basic_rows[0]
            metrics["task_duration_us"] = float(row.get("Task Duration(us)", 0))
            metrics["block_dim"] = int(row.get("Block Dim", 0))
            metrics["op_type"] = row.get("Op Type", row.get("OP Type", ""))

        # ArithmeticUtilization -> cube_ratio
        arith = read_cube_row(kdir, "ArithmeticUtilization")
        if arith:
            metrics["cube_ratio"] = float(arith.get("aic_cube_ratio", 0)) * 100

        # PipeUtilization -> mte2_ratio, vec_ratio
        pipe = read_cube_row(kdir, "PipeUtilization")
        if pipe:
            metrics["mte2_ratio"] = float(pipe.get("aic_mte2_ratio", 0)) * 100
            metrics["vec_ratio"] = float(pipe.get("aic_vec_ratio", 0)) * 100

        # L2Cache -> read_hit_rate
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
    """Aggregate multiple kernel instance metrics into summary stats."""
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

    # Aggregate ratios (take first instance, they're usually similar)
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


def run_msprof(target_script, target_args, output_subdir, warm_up, launch_count, kernel_name=None, timeout=600):
    """Run msprof op on a target script and parse results."""
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

    print(f"  running: msprof op --application=python {os.path.basename(target_script)} {target_args}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"  [warn] msprof exited with code {result.returncode}")
            if result.stderr:
                print(f"  stderr (last 500 chars): {result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        print(f"  [error] msprof timed out after {timeout}s")
        return None
    except FileNotFoundError:
        print("  [error] msprof command not found")
        return None

    opprof_dir = find_latest_opprof(output_path)
    if not opprof_dir:
        print(f"  [error] no OPPROF_* dir found under {output_path}")
        return None

    return opprof_dir


def run_one_config(B, S1, S2, G, N2, D, top_k, warm_up, launch_count, n_runs):
    """Run msprof op for both baseline and tilelang on one config."""
    config_name = f"B{B}_S1{S1}_S2{S2}_G{G}_D{D}_K{top_k}"
    print(f"\n{'=' * 60}")
    print(f"Config: {config_name}")
    print(f"{'=' * 60}")

    args_str_common = f"--B {B} --S1 {S1} --S2 {S2} --N2 {N2} --D {D} --top-k {top_k} --n-runs {n_runs}"
    baseline_args = f"{args_str_common} --N1 {G}"
    tilelang_args = f"{args_str_common} --G {G}"

    # --- baseline ---
    print("\n[baseline] torch_npu.npu_lightning_indexer")
    baseline_opprof = run_msprof(
        BENCH_BASELINE, baseline_args, f"baseline_{config_name}", warm_up, launch_count,
        kernel_name="LightningIndexer",
    )
    baseline_raw = parse_metrics(baseline_opprof, "baseline") if baseline_opprof else None
    baseline_metrics = aggregate_metrics(baseline_raw, "baseline") if baseline_raw else None

    # --- tilelang ---
    print("\n[tilelang] lightning_indexer (dynamic)")
    tilelang_opprof = run_msprof(
        BENCH_TILELANG, tilelang_args, f"tilelang_{config_name}", warm_up, launch_count,
        kernel_name="main_kernel",
    )
    tilelang_raw = parse_metrics(tilelang_opprof, "tilelang") if tilelang_opprof else None
    tilelang_metrics = aggregate_metrics(tilelang_raw, "tilelang") if tilelang_raw else None

    return {
        "config": {
            "B": B, "S1": S1, "S2": S2, "G": G, "N2": N2, "D": D, "top_k": top_k,
            "warm_up": warm_up, "launch_count": launch_count,
        },
        "baseline": baseline_metrics,
        "tilelang": tilelang_metrics,
    }


def generate_markdown(results, output_path):
    """Generate a markdown summary report from results."""
    lines = []
    lines.append("# Lightning Indexer Performance: TileLang vs torch_npu Baseline")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Environment
    lines.append("## Environment")
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

    # Method
    lines.append("## Method")
    lines.append("")
    lines.append("- **Profiler**: `msprof op` with `--warm-up` and `--launch-count`")
    lines.append("- **Metric**: Device-side `Task Duration(us)` from `OpBasicInfo.csv`")
    lines.append("- **Warmup**: Controlled by msprof `--warm-up` (not in target script)")
    lines.append("- **Capture**: Controlled by msprof `--launch-count`")
    lines.append("- **TileLang kernel**: `example_lightning_indexer_dynamic_shape.py` (D=128, BLOCK_K=128)")
    lines.append("- **Baseline**: `torch_npu.npu_lightning_indexer` (aclnnLightningIndexer)")
    lines.append("")

    # Results table
    lines.append("## Results")
    lines.append("")
    lines.append("| S2 | TileLang median (us) | Baseline median (us) | Speedup | TL Block Dim | BL Block Dim | TL Cube% | BL Cube% | TL MTE2% | BL MTE2% | TL L2 Hit% | BL L2 Hit% |")
    lines.append("|----|-----------------------|-----------------------|---------|--------------|--------------|----------|----------|----------|----------|------------|------------|")

    for r in results:
        cfg = r["config"]
        tl = r.get("tilelang")
        bl = r.get("baseline")

        def fmt(val):
            if val is None:
                return "N/A"
            if isinstance(val, float):
                return f"{val:.2f}"
            return str(val)

        tl_dur = fmt(tl.get("task_duration_median_us") if tl else None)
        bl_dur = fmt(bl.get("task_duration_median_us") if bl else None)

        tl_dur_f = tl.get("task_duration_median_us") if tl else 0
        bl_dur_f = bl.get("task_duration_median_us") if bl else 0
        speedup = f"{bl_dur_f / tl_dur_f:.2f}x" if tl_dur_f and tl_dur_f > 0 and bl_dur_f and bl_dur_f > 0 else "N/A"

        lines.append(
            f"| {cfg['S2']} | {tl_dur} | {bl_dur} | {speedup} | "
            f"{fmt(tl.get('block_dim') if tl else None)} | {fmt(bl.get('block_dim') if bl else None)} | "
            f"{fmt(tl.get('cube_ratio') if tl else None)} | {fmt(bl.get('cube_ratio') if bl else None)} | "
            f"{fmt(tl.get('mte2_ratio') if tl else None)} | {fmt(bl.get('mte2_ratio') if bl else None)} | "
            f"{fmt(tl.get('l2_read_hit_rate') if tl else None)} | {fmt(bl.get('l2_read_hit_rate') if bl else None)} |"
        )

    lines.append("")
    lines.append("> Speedup = Baseline / TileLang. > 1.0x means TileLang is faster.")
    lines.append("> Task Duration: device-side median from msprof op `OpBasicInfo.csv`.")
    lines.append("")

    # Kernel details
    lines.append("## Kernel Details")
    lines.append("")
    for r in results:
        cfg = r["config"]
        lines.append(f"### S2={cfg['S2']}")
        lines.append("")
        for side, label in [("tilelang", "TileLang"), ("baseline", "Baseline")]:
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

    # Raw data location
    lines.append("## Raw Data")
    lines.append("")
    lines.append(f"msprof output: `{OUTPUT_DIR}/`")
    lines.append(f"Structured JSON: `{output_path.replace('.md', '.json')}`")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="msprof op runner: lightning_indexer TileLang vs baseline")
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--S1", type=int, default=512)
    parser.add_argument("--S2", type=int, default=4096)
    parser.add_argument("--G", type=int, default=32)
    parser.add_argument("--N2", type=int, default=1)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--warm-up", type=int, default=5)
    parser.add_argument("--launch-count", type=int, default=10)
    parser.add_argument("--n-runs", type=int, default=20, help="kernel launches in target script (>= warm-up + launch-count)")
    parser.add_argument("--preset", default="default", choices=["default", "sweep"])
    args = parser.parse_args()

    results = []

    if args.preset == "sweep":
        for s2 in [512, 1024, 2048, 4096]:
            results.append(
                run_one_config(
                    args.B, args.S1, s2, args.G, args.N2, args.D, args.top_k,
                    args.warm_up, args.launch_count, args.n_runs,
                )
            )
    else:
        results.append(
            run_one_config(
                args.B, args.S1, args.S2, args.G, args.N2, args.D, args.top_k,
                args.warm_up, args.launch_count, args.n_runs,
            )
        )

    # Save JSON
    json_path = os.path.join(SCRIPT_DIR, "perf_lightning_indexer.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results: {json_path}")

    # Generate markdown
    md_path = os.path.join(SCRIPT_DIR, "perf_lightning_indexer.md")
    generate_markdown(results, md_path)


if __name__ == "__main__":
    main()
