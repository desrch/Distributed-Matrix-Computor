#!/usr/bin/env python3
"""
PageRank 实验结果全面分析

读取 4 个数据集的 CSV, 输出:
  1. Experimental Setup 表
  2. Correctness 验证表 (本地 OMP=2 MPI=2 对拍 scipy)
  3. 扩展性图 / MPI Speedup / OMP Speedup
  4. Hybrid Heatmaps / Comm Stacked / Memory
  5. 全部图表 → analysis/

用法:
    python3 scripts/analyze_results.py
"""

import os, csv, sys, math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── config ──
PROJECT = Path(__file__).resolve().parents[1]
DATA_DIRS = {
    "Epinions":    PROJECT / "Soc-Epinions1",
    "Stanford":    PROJECT / "Stanford",
    "Google":      PROJECT / "Google",
    "LiveJournal": PROJECT / "LiveJournal",
}
OUT_DIR = PROJECT / "analysis"
OUT_DIR.mkdir(exist_ok=True)

DS_ORDER = ["Epinions", "Stanford", "Google", "LiveJournal"]
DS_SHORT  = {"Epinions": "Epinions", "Stanford": "Stanford", "Google": "Google", "LiveJournal": "LiveJrnl"}
DS_LABELS = {
    "Epinions":    "soc-Epinions1\n(76K, 0.5M)",
    "Stanford":    "web-Stanford\n(282K, 2.3M)",
    "Google":      "web-Google\n(916K, 5.1M)",
    "LiveJournal": "soc-LiveJournal1\n(4.8M, 69M)",
}

# ── local correctness verification results ──
LOCAL_CORRECTNESS = {
    "Epinions":    {"iters": 54, "l1": 1.47e-15, "top10": 10},
    "Stanford":    {"iters": 63, "l1": 4.46e-15, "top10": 10},
    "Google":      {"iters": 62, "l1": 2.79e-14, "top10": 10},
    "LiveJournal": {"iters": 51, "l1": 2.37e-14, "top10": 10},
}

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150})
C = plt.cm.tab10.colors
DS_COLORS = [C[0], C[1], C[2], C[3]]

ALL_ROWS = []; BY_DS = defaultdict(list)

# ── helpers ──
def sf(k): return float(k) if k else 0.0
def si(k): return int(float(k)) if k else 0

def load():
    global ALL_ROWS, BY_DS
    ALL_ROWS, BY_DS = [], defaultdict(list)
    for ds, d in DATA_DIRS.items():
        for f in sorted(d.glob("*.csv")):
            if "summary" in f.name: continue
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    row["_ds"] = ds; ALL_ROWS.append(row); BY_DS[ds].append(row)
    print(f"Loaded {len(ALL_ROWS)} rows")

def eng_match(row, prefix, p=None, t=None):
    """Check engine starts with prefix and has given mpi_ranks/omp_threads."""
    if not row.get("engine","").startswith(prefix): return False
    if p is not None and si(row.get("mpi_ranks")) != p: return False
    if t is not None and si(row.get("omp_threads")) != t: return False
    return True

# ── 1. Setup & Dataset Stats ──
def part1_setup():
    print("\n═══ Part 1: Experimental Setup ═══")
    for ds in DS_ORDER:
        rows = BY_DS[ds]
        if not rows: continue
        r = rows[0]; N = si(r["num_nodes"]); E = si(r["num_edges"])
        nnz = si(r["csr_nnz"]); density = 100.0*E/(N*N) if N else 0
        print(f"  {DS_LABELS[ds].split(chr(10))[0]:<22s}  N={N:>10,d}  E={E:>11,d}  "
              f"nnz={nnz:>11,d}  density={density:.6f}%  "
              f"iters={rows[0]['iterations']}  conv={rows[0]['converged']}")

# ── 2. Correctness ──
def part2_correctness():
    print("\n═══ Part 2: Correctness (Local: OMP=2, MPI=2 vs scipy ref) ═══")
    print(f"  {'Dataset':<14s} {'Iters':>6s}  {'L1 Error':>12s}  {'Top10':>6s}")
    print(f"  {'-'*46}")
    for ds in DS_ORDER:
        c = LOCAL_CORRECTNESS[ds]
        print(f"  {DS_SHORT[ds]:<14s} {c['iters']:>6d}  {c['l1']:>12.2e}  {c['top10']:>5d}/10")

# ── 3. Scalability (best single-node: OMP_CSR_t4) ──
def part3_scalability():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── 3a: best single-node total times ──
    names, ttl, spv, comm, comp, damp = [], [], [], [], [], []
    for ds in DS_ORDER:
        cand = [r for r in BY_DS[ds] if eng_match(r, "OMP_CSR", p=1, t=4)]
        if not cand: cand = [r for r in BY_DS[ds] if eng_match(r, "CSR_Serial")]
        if not cand: continue
        r = cand[0]; names.append(DS_SHORT[ds])
        ttl.append(sf(r["total_time_s"])); spv.append(sf(r["spmv_time_s"]))
        comm.append(sf(r["comm_time_s"])); comp.append(sf(r["comp_time_s"]))
        damp.append(sf(r["damp_time_s"]))

    x = range(len(names))
    bars = ax1.bar(x, ttl, color=DS_COLORS[:len(names)], edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(names)
    ax1.set_ylabel("Total Time (s)"); ax1.set_title("Best Single-Node PageRank Time\n(CSR + OMP=4)")
    for i, b in enumerate(bars):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+max(ttl)*0.015,
                 f"{ttl[i]:.3f}s", ha="center", fontsize=8)

    # ── 3b: wall-clock breakdown ──
    bottom_comp = [0]*len(names)
    bottom_damp = comp
    bottom_comm = [c+d for c,d in zip(comp, damp)]
    ax2.bar(x, comp,  label="Compute (SpMV)", color="#4CAF50")
    ax2.bar(x, damp,  bottom=bottom_comp,  label="Damping", color="#FFC107")
    ax2.bar(x, comm,  bottom=bottom_comm,  label="Comm (MPI)", color="#F44336")
    ax2.set_xticks(x); ax2.set_xticklabels(names)
    ax2.set_ylabel("Time (s)"); ax2.set_title("Wall-Clock Breakdown (OMP=4)")
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR/"fig1_scalability.png", bbox_inches="tight"); plt.close(fig)
    print("  fig1_scalability.png")

# ── 4. MPI Scalability + Speedup ──
def part4_mpi():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    mpi_v = [1,2,4,8]

    for idx, ds in enumerate(DS_ORDER):
        times = []
        for p in mpi_v:
            cand = [r for r in BY_DS[ds] if eng_match(r, "MPI_CSR", p=p, t=1)]
            times.append(sf(cand[0]["total_time_s"]) if cand else np.nan)
        ax1.plot(mpi_v, times, "o-", c=DS_COLORS[idx], lw=2, ms=7, label=DS_SHORT[ds])

    ax1.set_xlabel("MPI Processes"); ax1.set_ylabel("Total Time (s)")
    ax1.set_title("MPI Scalability (OMP=1)"); ax1.set_xticks(mpi_v)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # speedup (baseline = MPI_CSR_p1)
    ax2.plot(mpi_v, mpi_v, "k--", lw=1, alpha=0.4, label="Ideal")
    for idx, ds in enumerate(DS_ORDER):
        base = [r for r in BY_DS[ds] if eng_match(r, "MPI_CSR", p=1, t=1)]
        t1 = sf(base[0]["total_time_s"]) if base else np.nan
        su = []
        for p in mpi_v:
            cand = [r for r in BY_DS[ds] if eng_match(r, "MPI_CSR", p=p, t=1)]
            tp = sf(cand[0]["total_time_s"]) if cand else np.nan
            su.append(t1/tp if tp>0 else np.nan)
        ax2.plot(mpi_v, su, "s-", c=DS_COLORS[idx], lw=2, ms=7, label=DS_SHORT[ds])

    ax2.set_xlabel("MPI Processes"); ax2.set_ylabel("Speedup")
    ax2.set_title("MPI Strong Scaling Speedup\n(baseline = MPI_CSR_p1)"); ax2.set_xticks(mpi_v)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR/"fig2_mpi_speedup.png", bbox_inches="tight"); plt.close(fig)
    print("  fig2_mpi_speedup.png")

# ── 5. OMP Scalability + Speedup ──
def part5_omp():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    omp_v = [1,2,4]

    for idx, ds in enumerate(DS_ORDER):
        times = []
        for t in omp_v:
            cand = [r for r in BY_DS[ds] if eng_match(r, "OMP_CSR", p=1, t=t)]
            times.append(sf(cand[0]["total_time_s"]) if cand else np.nan)
        ax1.plot(omp_v, times, "o-", c=DS_COLORS[idx], lw=2, ms=7, label=DS_SHORT[ds])

    ax1.set_xlabel("OpenMP Threads"); ax1.set_ylabel("Total Time (s)")
    ax1.set_title("OpenMP Scalability (MPI=1)"); ax1.set_xticks(omp_v)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # speedup (baseline = OMP_CSR_t1)
    ax2.plot(omp_v, omp_v, "k--", lw=1, alpha=0.4, label="Ideal")
    for idx, ds in enumerate(DS_ORDER):
        base = [r for r in BY_DS[ds] if eng_match(r, "OMP_CSR", p=1, t=1)]
        t1 = sf(base[0]["total_time_s"]) if base else np.nan
        su = []
        for t in omp_v:
            cand = [r for r in BY_DS[ds] if eng_match(r, "OMP_CSR", p=1, t=t)]
            tp = sf(cand[0]["total_time_s"]) if cand else np.nan
            su.append(t1/tp if tp>0 else np.nan)
        ax2.plot(omp_v, su, "s-", c=DS_COLORS[idx], lw=2, ms=7, label=DS_SHORT[ds])

    ax2.set_xlabel("OpenMP Threads"); ax2.set_ylabel("Speedup")
    ax2.set_title("OpenMP Strong Scaling Speedup\n(baseline = OMP_CSR_t1)"); ax2.set_xticks(omp_v)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR/"fig3_omp_speedup.png", bbox_inches="tight"); plt.close(fig)
    print("  fig3_omp_speedup.png")

# ── 6. Hybrid Heatmaps ──
def part6_hybrid():
    mpi_v, omp_v = [1,2,4,8], [1,2,4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, ds in enumerate(DS_ORDER):
        data = np.full((4, 3), np.nan)
        annot = [[""]*3 for _ in range(4)]
        for i, p in enumerate(mpi_v):
            for j, t in enumerate(omp_v):
                cand = [r for r in BY_DS[ds] if eng_match(r, "Hybrid", p=p, t=t)]
                if cand:
                    data[i,j] = sf(cand[0]["total_time_s"])
                    annot[i][j] = f"{data[i,j]:.2f}s"

        im = axes[idx].imshow(data, cmap="RdYlGn_r", aspect="auto",
                               vmin=np.nanmin(data), vmax=np.nanmax(data))
        axes[idx].set_xticks(range(3)); axes[idx].set_xticklabels([f"OMP={t}" for t in omp_v])
        axes[idx].set_yticks(range(4)); axes[idx].set_yticklabels([f"MPI={p}" for p in mpi_v])
        axes[idx].set_title(DS_SHORT[ds])
        plt.colorbar(im, ax=axes[idx], label="Total Time (s)", shrink=0.82)
        for i in range(4):
            for j in range(3):
                if annot[i][j]:
                    mid = (np.nanmin(data)+np.nanmax(data))/2
                    c = "white" if data[i,j] > mid else "black"
                    axes[idx].text(j, i, annot[i][j], ha="center", va="center", fontsize=8, color=c)

    fig.suptitle("Hybrid MPI×OpenMP Execution Time (s)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR/"fig4_hybrid_heatmaps.png", bbox_inches="tight"); plt.close(fig)
    print("  fig4_hybrid_heatmaps.png")

# ── 7. Communication Analysis ──
def part7_comm():
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten(); mpi_v = [1,2,4,8]

    for idx, ds in enumerate(DS_ORDER):
        comps, comms = [], []
        for p in mpi_v:
            cand = [r for r in BY_DS[ds] if eng_match(r, "MPI_CSR", p=p, t=1)]
            if cand:
                comps.append(sf(cand[0]["comp_time_s"]))
                comms.append(sf(cand[0]["comm_time_s"]))
            else:
                comps.append(0); comms.append(0)

        x = range(4)
        axes[idx].bar(x, comps, label="Compute", color="#4CAF50", edgecolor="white")
        axes[idx].bar(x, comms, bottom=comps, label="Comm", color="#F44336", edgecolor="white")
        for i, (c, m) in enumerate(zip(comps, comms)):
            if c+m > 0: axes[idx].text(i, c+m+max(comps)*0.02, f"{100*m/(c+m):.0f}%", ha="center", fontsize=8)
        axes[idx].set_xticks(x); axes[idx].set_xticklabels([f"MPI={p}" for p in mpi_v])
        axes[idx].set_ylabel("Time (s)"); axes[idx].set_title(DS_SHORT[ds])
        axes[idx].legend(fontsize=8); axes[idx].grid(True, alpha=0.2, axis="y")

    fig.suptitle("Communication vs Computation (MPI_CSR, OMP=1)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR/"fig5_communication.png", bbox_inches="tight"); plt.close(fig)
    print("  fig5_communication.png")

# ── 8. Memory ──
def part8_memory():
    fig, ax = plt.subplots(figsize=(9, 5))
    names, mems_mb, dense_mb = [], [], []
    for ds in DS_ORDER:
        cand = [r for r in BY_DS[ds] if eng_match(r, "CSR_Serial")]
        if not cand: continue
        r = cand[0]; N = si(r["num_nodes"])
        names.append(DS_SHORT[ds])
        mems_mb.append(sf(r["mem_peak_rss_kb"])/1024)
        dense_mb.append(N*N*8/1024/1024)

    bars = ax.bar(range(len(names)), mems_mb, color=DS_COLORS[:len(names)], edgecolor="white")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel("Peak RSS Memory (MB)"); ax.set_title("Memory Usage — CSR_Serial Baseline\n(dense equivalent annotated)")
    for i, (b, m, d) in enumerate(zip(bars, mems_mb, dense_mb)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(mems_mb)*0.02,
                f"{m:.0f} MB", ha="center", fontsize=9)
        ax.text(b.get_x()+b.get_width()/2, m/3, f"Dense={d:.0f} MB", ha="center",
                fontsize=7, color="white", fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR/"fig6_memory.png", bbox_inches="tight"); plt.close(fig)
    print("  fig6_memory.png")

# ── summary tables ──
def print_summary():
    print("\n═══ Summary Tables ═══")

    # Best per dataset
    print("\n  Best config per dataset:")
    for ds in DS_ORDER:
        best, best_t = None, 1e9
        for r in BY_DS[ds]:
            t = sf(r["total_time_s"])
            if t > 0 and t < best_t:
                best_t, best = t, r
        if best:
            print(f"  {DS_SHORT[ds]:<14s}  {best['engine']:<22s}  total={sf(best['total_time_s']):.4f}s  "
                  f"spmv={sf(best['spmv_time_s']):.4f}s  comm_pct={sf(best['comm_pct']):.0f}%")

    # MPI speedup table
    print("\n  MPI Speedup (MPI_CSR, OMP=1, baseline=MPI_CSR_p1):")
    print(f"  {'Dataset':<14s}    p=1     p=2     p=4     p=8")
    for ds in DS_ORDER:
        base = [r for r in BY_DS[ds] if eng_match(r, "MPI_CSR", p=1, t=1)]
        t1 = sf(base[0]["total_time_s"]) if base else 0
        vals = []
        for p in [1,2,4,8]:
            cand = [r for r in BY_DS[ds] if eng_match(r, "MPI_CSR", p=p, t=1)]
            tp = sf(cand[0]["total_time_s"]) if cand else 0
            vals.append(f"{t1/tp:6.2f}x" if tp>0 else "   N/A")
        print(f"  {DS_SHORT[ds]:<14s} " + " ".join(vals))

    # OMP speedup table
    print("\n  OMP Speedup (OMP_CSR, MPI=1, baseline=OMP_CSR_t1):")
    print(f"  {'Dataset':<14s}    t=1     t=2     t=4")
    for ds in DS_ORDER:
        base = [r for r in BY_DS[ds] if eng_match(r, "OMP_CSR", p=1, t=1)]
        t1 = sf(base[0]["total_time_s"]) if base else 0
        vals = []
        for t in [1,2,4]:
            cand = [r for r in BY_DS[ds] if eng_match(r, "OMP_CSR", p=1, t=t)]
            tp = sf(cand[0]["total_time_s"]) if cand else 0
            vals.append(f"{t1/tp:6.2f}x" if tp>0 else "   N/A")
        print(f"  {DS_SHORT[ds]:<14s} " + " ".join(vals))

    print(f"\n═══ Done. Charts → {OUT_DIR}/ ═══")

# ── main ──
def main():
    load()
    part1_setup()
    part2_correctness()
    part3_scalability()
    part4_mpi()
    part5_omp()
    part6_hybrid()
    part7_comm()
    part8_memory()
    print_summary()

if __name__ == "__main__":
    main()
