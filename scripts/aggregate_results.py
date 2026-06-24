#!/usr/bin/env python3
"""
PageRank 实验结果聚合工具

读取 lab_res/ 下所有 CSV 文件，输出:
  1. 合并后的完整 CSV (summary.csv)
  2. 终端可读的 Markdown 格式汇总表

用法:
    python3 scripts/aggregate_results.py lab_res/
    python3 scripts/aggregate_results.py lab_res/ --output lab_res/summary.csv --markdown lab_res/report.md
"""

import sys
import os
import csv
import argparse
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate PageRank experiment CSVs")
    p.add_argument("input_dir", help="Directory containing CSV files")
    p.add_argument("--output", "-o",
                   help="Output merged CSV path (default: <input_dir>/summary.csv)")
    p.add_argument("--markdown", "-m",
                   help="Output Markdown report path (default: <input_dir>/report.md)")
    return p.parse_args()


def load_all_csvs(input_dir):
    """加载所有 CSV 并合并为一个行列表"""
    all_rows = []
    fieldnames = None
    csv_files = sorted([
        f for f in os.listdir(input_dir) if f.endswith('.csv') and f != 'summary.csv'
    ])

    for fname in csv_files:
        fpath = os.path.join(input_dir, fname)
        with open(fpath, 'r') as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                all_rows.append(row)

    return fieldnames, all_rows


def write_merged_csv(output_path, fieldnames, rows):
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Aggregate] Wrote {output_path} ({len(rows)} rows)")


def fmt_float(val, prec=4):
    try:
        return f"{float(val):.{prec}f}"
    except (ValueError, TypeError):
        return str(val)


def fmt_pct(val):
    try:
        return f"{float(val):.1f}%"
    except (ValueError, TypeError):
        return str(val)


def generate_markdown(rows):
    """生成可读 Markdown 报告，按 engine 分组"""
    lines = []
    lines.append("# PageRank Experiment Results")
    lines.append("")

    # 分组
    by_engine = defaultdict(list)
    for r in rows:
        by_engine[r['engine']].append(r)

    # 表头
    header = ("| Engine | Dataset | MPI | OMP | Iters | Total(s) | SpMV(s) | "
              "Comm(s) | Comm% | Comp(s) | AvgSpMV(ms) | PeakRSS(KB) | L1 vs Ref | Top10 |")
    sep    = "|" + "|".join(["--------"] * 14) + "|"

    for eng in sorted(by_engine.keys()):
        lines.append(f"## {eng}")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for r in sorted(by_engine[eng],
                         key=lambda x: (int(x.get('mpi_ranks', 1)),
                                        int(x.get('omp_threads', 1)))):
            lines.append(
                f"| {r['engine']} | {r['dataset']} | {r['mpi_ranks']} | {r['omp_threads']} | "
                f"{r['iterations']} | {fmt_float(r['total_time_s'])} | {fmt_float(r['spmv_time_s'])} | "
                f"{fmt_float(r['comm_time_s'])} | {fmt_pct(r['comm_pct'])} | "
                f"{fmt_float(r['comp_time_s'])} | {fmt_float(r.get('avg_spmv_ms', 0), 2)} | "
                f"{r.get('mem_peak_rss_kb', 0)} | {r.get('l1_vs_ref', '')} | {r.get('top10_match', '')} |")
        lines.append("")

    # 全局汇总
    lines.append("## Summary")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for r in sorted(rows, key=lambda x: (x['dataset'],
                                          int(x.get('mpi_ranks', 1)),
                                          int(x.get('omp_threads', 1)))):
        lines.append(
            f"| {r['engine']} | {r['dataset']} | {r['mpi_ranks']} | {r['omp_threads']} | "
            f"{r['iterations']} | {fmt_float(r['total_time_s'])} | {fmt_float(r['spmv_time_s'])} | "
            f"{fmt_float(r['comm_time_s'])} | {fmt_pct(r['comm_pct'])} | "
            f"{fmt_float(r['comp_time_s'])} | {fmt_float(r.get('avg_spmv_ms', 0), 2)} | "
            f"{r.get('mem_peak_rss_kb', 0)} | {r.get('l1_vs_ref', '')} | {r.get('top10_match', '')} |")

    return "\n".join(lines)


def main():
    args = parse_args()
    input_dir = args.input_dir

    if not os.path.isdir(input_dir):
        print(f"ERROR: directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    fieldnames, rows = load_all_csvs(input_dir)
    if not rows:
        print("No CSV data found.")
        sys.exit(0)

    print(f"[Aggregate] Loaded {len(rows)} rows from {input_dir}")

    # output merged CSV
    csv_out = args.output or os.path.join(input_dir, "summary.csv")
    write_merged_csv(csv_out, fieldnames, rows)

    # output Markdown report
    md_out = args.markdown or os.path.join(input_dir, "report.md")
    md_content = generate_markdown(rows)
    with open(md_out, 'w') as f:
        f.write(md_content)
    print(f"[Aggregate] Wrote {md_out}")

    # terminal summary
    print(f"\n{'--- Quick Summary':<110s}")
    print(f"  {'Engine':<22s} {'MPI':>4s} {'OMP':>4s} {'Iters':>6s}  {'Total(s)':>9s}  {'SpMV(s)':>9s}  {'Comm%':>7s}  {'Comps':>9s}  {'Mem(KB)':>9s}")
    print(f"  {'-'*90}")
    for r in sorted(rows, key=lambda x: (x['engine'],
                                          int(x.get('mpi_ranks', 1)),
                                          int(x.get('omp_threads', 1)))):
        eng = r['engine'][:21]
        print(f"  {eng:<22s} {r['mpi_ranks']:>4s} {r['omp_threads']:>4s} "
              f"{r['iterations']:>6s}  {fmt_float(r['total_time_s']):>9s}  "
              f"{fmt_float(r['spmv_time_s']):>9s}  {fmt_pct(r['comm_pct']):>7s}  "
              f"{fmt_float(r['comp_time_s']):>9s}  {r.get('mem_peak_rss_kb','0'):>9s}")


if __name__ == "__main__":
    main()
