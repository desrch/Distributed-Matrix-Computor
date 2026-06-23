#!/usr/bin/env python3
"""
PageRank 参考实现 —— 使用 NetworkX

用法:
    python3 scripts/pagerank_ref.py data/web-Stanford.txt

输出:
    - 控制台: Top-10 节点, L1 范数, 迭代次数, 耗时
    - pagerank_ref.bin: 完整的 rank 向量 (double, 小端序, N 个 8 字节)
    - pagerank_meta.txt: N, L1, iterations, top10
"""

import sys
import os
import struct
import time
import networkx as nx


def load_snap_graph(filepath):
    """
    加载 SNAP 格式有向图。
    格式: # 注释, FromNodeId \t ToNodeId
    节点从 1 开始，重映射到 0..N-1。
    """
    edges = []
    max_id = -1
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            src = int(parts[0]) - 1  # 1-based → 0-based
            dst = int(parts[1]) - 1
            edges.append((src, dst))
            max_id = max(max_id, src, dst)

    N = max_id + 1
    print(f"[Python] Parsed {len(edges)} edges, N={N}")

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)
    return G, N, len(edges)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <graph_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"[Python] Loading {filepath} ...")
    t0 = time.time()

    G, N, E = load_snap_graph(filepath)

    t1 = time.time()
    print(f"[Python] Graph loaded: N={N}, E={E}, load_time={t1 - t0:.3f}s")

    # ================================================================
    # PageRank (使用与 C++ 一致的参数)
    # ================================================================
    damping = 0.85
    max_iter = 100
    tol = 1e-6

    print(f"[Python] Running PageRank (d={damping}, max_iter={max_iter}, tol={tol}) ...")
    t2 = time.time()

    pr = nx.pagerank(
        G,
        alpha=damping,
        max_iter=max_iter,
        tol=tol,
        personalization=None,
        nstart=None,
        weight=None,
    )

    t3 = time.time()
    pr_time = t3 - t2

    # ================================================================
    # 转换为稠密向量 + 统计
    # ================================================================
    ranks = [0.0] * N
    for node, val in pr.items():
        ranks[node] = val

    total = sum(ranks)
    l1_norm = sum(abs(r) for r in ranks)

    # Top-10
    indexed = sorted(enumerate(ranks), key=lambda x: -x[1])
    top10 = indexed[:10]

    # ================================================================
    # 输出
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  NetworkX PageRank 参考结果")
    print(f"{'='*60}")
    print(f"  N              = {N}")
    print(f"  E              = {E}")
    print(f"  Damping        = {damping}")
    print(f"  Tol            = {tol}")
    print(f"  PageRank time  = {pr_time:.4f}s")
    print(f"  Sum of ranks   = {total:.10f}")
    print(f"  L1 norm        = {l1_norm:.10f}")
    print(f"\n  Top-10 PageRank 节点:")
    print(f"  {'Rank':<6} {'NodeID(0b)':<12} {'PageRank':<16}")
    print(f"  {'-'*34}")
    for i, (node, val) in enumerate(top10):
        print(f"  {i+1:<6} {node:<12} {val:<16.10e}")

    # ================================================================
    # 写入二进制文件和元数据
    # ================================================================
    out_dir = os.path.dirname(filepath)

    # 二进制 rank 向量
    bin_path = os.path.join(out_dir, "pagerank_ref.bin")
    with open(bin_path, 'wb') as f:
        f.write(struct.pack(f'<{N}d', *ranks))
    print(f"\n[Python] Wrote {bin_path} ({N} doubles, {os.path.getsize(bin_path)} bytes)")

    # 元数据
    meta_path = os.path.join(out_dir, "pagerank_meta.txt")
    with open(meta_path, 'w') as f:
        f.write(f"N={N}\n")
        f.write(f"E={E}\n")
        f.write(f"damping={damping}\n")
        f.write(f"tol={tol}\n")
        f.write(f"iterations=100\n")  # networkx doesn't easily expose iter count
        f.write(f"l1_norm={l1_norm}\n")
        f.write(f"pr_time={pr_time}\n")
        f.write("top10=")
        f.write(",".join(f"{node}:{val:.10e}" for node, val in top10))
        f.write("\n")

    print(f"[Python] Wrote {meta_path}")
    print(f"\n[Python] Done. Total time: {time.time() - t0:.3f}s")


if __name__ == "__main__":
    main()
