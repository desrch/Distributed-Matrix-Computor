#!/usr/bin/env python3
"""
PageRank 参考实现 —— 使用 scipy.sparse CSR 匹配 C++ 算法

与 C++ 版本算法完全一致:
  - 列随机转换矩阵 M(dst, src) = 1/out_degree[src]
  - 相同的 Power Method 迭代逻辑
  - 相同的悬挂节点处理

用法:
    python3 scripts/pagerank_ref_sparse.py data/web-Stanford.txt
"""

import sys
import os
import struct
import time
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter


def load_snap_sparse(filepath):
    """加载 SNAP 格式图，构建列随机 CSR 矩阵。自动检测 0-based / 1-based。"""
    print(f"[Python] Loading {filepath} ...")

    edges_src = []
    edges_dst = []
    max_id = -1
    min_id = 2**31

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            src = int(parts[0])
            dst = int(parts[1])
            edges_src.append(src)
            edges_dst.append(dst)
            max_id = max(max_id, src, dst)
            min_id = min(min_id, src, dst)

    # auto-detect: 如果出现节点 0 则为 0-based
    one_based = (min_id >= 1)
    offset = 1 if one_based else 0
    print(f"[Python] Detected: {'1' if one_based else '0'}-based "
          f"(min_id={min_id}, max_id={max_id})")

    if one_based:
        edges_src = [s - 1 for s in edges_src]
        edges_dst = [d - 1 for d in edges_dst]
        max_id -= 1

    N = max_id + 1
    src_arr = np.array(edges_src, dtype=np.int32)
    dst_arr = np.array(edges_dst, dtype=np.int32)

    # 计算出度
    out_degree = np.bincount(src_arr, minlength=N)
    dangling = np.where(out_degree == 0)[0]

    # 列随机权重: M(dst, src) = 1/out_degree[src]
    weights = 1.0 / out_degree[src_arr]

    # 构建 CSR: rows=dst, cols=src
    M = csr_matrix((weights, (dst_arr, src_arr)), shape=(N, N))
    
    print(f"[Python] N={N}, E={len(edges_src)}, nnz={M.nnz}, "
          f"dangling={len(dangling)}, load_time={time.time() - t0:.3f}s")
    return M, N, dangling


def pagerank_power(M, N, dangling, damping=0.85, max_iter=100, tol=1e-6):
    """
    Power Method — 与 C++ PageRank::run 算法完全一致。
    用于和 C++ 结果做逐位对拍。
    """
    r = np.full(N, 1.0 / N, dtype=np.float64)
    one_minus_d = 1.0 - damping
    converged = False
    iters = 0

    for iters in range(1, max_iter + 1):
        # SpMV: y = M @ r
        y = M.dot(r)

        # 悬挂节点贡献
        danglesum = r[dangling].sum() if len(dangling) > 0 else 0.0
        personalization = (damping * danglesum + one_minus_d) / N

        # 更新
        r_new = damping * y + personalization

        # L1 误差
        error = np.abs(r_new - r).sum()

        r = r_new
        if error < tol:
            converged = True
            break

    return r, iters, error, converged


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <graph_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    t0 = time.time()
    M, N, dangling = load_snap_sparse(filepath)

    damping = 0.85
    max_iter = 100
    tol = 1e-6

    print(f"[Python] Running PageRank (d={damping}, tol={tol}) ...")
    t2 = time.time()
    ranks, iters, error, converged = pagerank_power(
        M, N, dangling, damping, max_iter, tol)
    t3 = time.time()

    # 转为 Python list
    ranks_list = ranks.tolist()

    # Top-10
    indexed = sorted(enumerate(ranks_list), key=lambda x: -x[1])
    top10 = indexed[:10]

    print(f"\n{'='*60}")
    print(f"  Python Power-Method PageRank 参考结果")
    print(f"{'='*60}")
    print(f"  N              = {N}")
    print(f"  Damping        = {damping}")
    print(f"  Iterations     = {iters}")
    print(f"  Converged      = {converged}")
    print(f"  Final L1 err   = {error:.8e}")
    print(f"  PageRank time  = {t3 - t2:.4f}s")
    print(f"  Sum of ranks   = {sum(ranks_list):.10f}")

    print(f"\n  Top-10 PageRank 节点:")
    print(f"  {'Rank':<6} {'NodeID(0b)':<12} {'PageRank':<16}")
    print(f"  {'-'*34}")
    for i, (node, val) in enumerate(top10):
        print(f"  {i+1:<6} {node:<12} {val:<16.10e}")

    # 写入参考二进制
    out_dir = os.path.dirname(os.path.abspath(filepath))
    bin_path = os.path.join(out_dir, "pagerank_ref.bin")
    with open(bin_path, 'wb') as f:
        f.write(struct.pack(f'<{N}d', *ranks_list))
    print(f"\n[Python] Wrote {bin_path} ({N} doubles, {os.path.getsize(bin_path)} bytes)")

    # 元数据
    meta_path = os.path.join(out_dir, "pagerank_meta.txt")
    with open(meta_path, 'w') as f:
        f.write(f"N={N}\n")
        f.write(f"damping={damping}\n")
        f.write(f"tol={tol}\n")
        f.write(f"iterations={iters}\n")
        f.write(f"converged={converged}\n")
        f.write(f"final_error={error}\n")
        f.write(f"pr_time={t3 - t2}\n")
        f.write("top10=")
        f.write(",".join(f"{node}:{val:.10e}" for node, val in top10))
        f.write("\n")
    print(f"[Python] Wrote {meta_path}")
    print(f"\n[Python] Done. Total time: {time.time() - t0:.3f}s")
