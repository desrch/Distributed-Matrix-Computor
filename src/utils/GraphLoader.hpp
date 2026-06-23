#pragma once

#include "../matrix/CSRMatrix.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace dmc {

/**
 * @brief SNAP 格式图数据加载器
 *
 * SNAP 格式: # 注释行、FromNodeId\tToNodeId
 *
 * 加载后构建列随机 CSR 矩阵（用于 PageRank）：
 *   M(dst, src) = 1.0 / out_degree[src]
 *
 * 即对每条边 src→dst，在 CSR 的行 dst、列 src 处存储该值。
 */
class GraphLoader {
public:
    /**
     * @brief 加载结果
     */
    struct Graph {
        CSRMatrix csr;                  // 列随机转换矩阵 M (N×N)
        int num_nodes;                  // 节点数（已重映射为 0..N-1）
        int num_edges;                  // 边数
        std::vector<int> dangling;      // 悬挂节点列表 (out-degree == 0)
        std::vector<int> out_degree;    // 每个节点的出度
    };

    /**
     * @brief 从 SNAP 格式文件加载有向图
     *
     * @param filepath  文件路径
     * @param one_based  true 如果节点 ID 从 1 开始 (SNAP 惯例)
     * @return Graph 包含列随机 CSR 矩阵
     */
    static Graph loadSNAP(const std::string& filepath, bool one_based = true)
    {
        // ============================================================
        // Pass 1: 统计节点、边和出度
        // ============================================================
        std::ifstream fin(filepath);
        if (!fin.is_open()) {
            throw std::runtime_error("GraphLoader: cannot open " + filepath);
        }

        std::vector<std::pair<int,int>> edges;
        edges.reserve(3000000);
        int max_id = 0;

        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            int src, dst;
            if (std::sscanf(line.c_str(), "%d\t%d", &src, &dst) != 2) {
                // 尝试空格分隔
                if (std::sscanf(line.c_str(), "%d %d", &src, &dst) != 2)
                    continue;
            }
            if (one_based) { src--; dst--; }
            edges.emplace_back(src, dst);
            if (src > max_id) max_id = src;
            if (dst > max_id) max_id = dst;
        }
        fin.close();

        int N = max_id + 1;
        int E = static_cast<int>(edges.size());

        printf("[GraphLoader] Parsed %d edges, max_node_id=%d → N=%d\n",
               E, max_id, N);

        // ============================================================
        // Pass 2: 计算出度 & 悬挂节点
        // ============================================================
        std::vector<int> out_degree(N, 0);
        for (const auto& e : edges) {
            out_degree[e.first]++;
        }

        std::vector<int> dangling;
        for (int i = 0; i < N; ++i) {
            if (out_degree[i] == 0) {
                dangling.push_back(i);
            }
        }
        printf("[GraphLoader] dangling nodes: %zu / %d\n",
               dangling.size(), N);

        // ============================================================
        // Pass 3: 构建列随机 CSR
        //   M(dst, src) = 1.0 / out_degree[src]
        //
        //   两步: (a) 统计每行（即每个 dst）有多少入边
        //         (b) 填充 values 和 col_idx
        // ============================================================

        // (a) 统计 CSR 各行的 nnz
        std::vector<int> csr_row_counts(N, 0);
        for (const auto& e : edges) {
            int dst = e.second;
            csr_row_counts[dst]++;
        }

        // row_ptr
        std::vector<int> csr_row_ptr(N + 1, 0);
        for (int i = 0; i < N; ++i) {
            csr_row_ptr[i + 1] = csr_row_ptr[i] + csr_row_counts[i];
        }
        int total_nnz = csr_row_ptr[N];

        std::vector<double> csr_vals(total_nnz);
        std::vector<int>    csr_cidx(total_nnz);

        // (b) 填充: 需要将 edges 按 dst 分组
        //    使用 csr_row_counts 作为游标
        std::fill(csr_row_counts.begin(), csr_row_counts.end(), 0);
        for (const auto& e : edges) {
            int src = e.first;
            int dst = e.second;
            int pos = csr_row_ptr[dst] + csr_row_counts[dst];
            double val = 1.0 / static_cast<double>(out_degree[src]);
            csr_vals[pos] = val;
            csr_cidx[pos] = src;  // col = source node
            csr_row_counts[dst]++;
        }

        // 验证游标
        for (int i = 0; i < N; ++i) {
            assert(csr_row_counts[i] == csr_row_ptr[i + 1] - csr_row_ptr[i]);
        }

        printf("[GraphLoader] CSR built: N=%d, nnz=%d, M=%d×%d\n",
               N, total_nnz, N, N);

        CSRMatrix M(N, N,
                    std::move(csr_vals),
                    std::move(csr_cidx),
                    std::move(csr_row_ptr));

        Graph g;
        g.csr        = std::move(M);
        g.num_nodes  = N;
        g.num_edges  = E;
        g.dangling   = std::move(dangling);
        g.out_degree = std::move(out_degree);
        return g;
    }
};

} // namespace dmc
