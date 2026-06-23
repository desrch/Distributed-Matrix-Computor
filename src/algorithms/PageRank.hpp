#pragma once

#include "../matrix/CSRMatrix.hpp"
#include "../operators/MatrixVector.hpp"
#include "../utils/GraphLoader.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

namespace dmc {

/**
 * @brief PageRank 算法 (Power Method)
 *
 * 公式:
 *   r_{k+1} = d × M × r_k  +  (d × danglesum + (1-d)) / N × 1
 *
 * 其中:
 *   M       列随机转换矩阵 (CSR: rows=dst, cols=src, val=1/out[src])
 *   d       阻尼因子 (0.85)
 *   danglesum = Σ_{悬挂节点 j} r_k[j]
 *   N       节点总数
 *
 * 收敛条件: L1(r_{k+1} - r_k) < tol
 *
 * 内部使用 MatrixVector::multiply(CSR, Vector) 进行 SpMV，
 * 该函数已集成 OpenMP 并行加速。
 */
class PageRank {
public:
    struct Result {
        std::vector<double> ranks;     // 最终 PageRank 值 (长度 N)
        int    iterations;              // 实际迭代次数
        double final_error;             // 最终 L1 误差
        bool   converged;               // 是否收敛
    };

    /**
     * @brief 执行 PageRank
     *
     * @param graph    图加载结果 (含列随机 CSR 矩阵)
     * @param damping  阻尼因子，默认 0.85
     * @param max_iter 最大迭代次数
     * @param tol      收敛容差 (L1 范数)
     */
    static Result run(const GraphLoader::Graph& graph,
                      double damping = 0.85,
                      int    max_iter = 100,
                      double tol = 1e-6)
    {
        const CSRMatrix& M   = graph.csr;
        int N                = graph.num_nodes;
        const auto& dangling = graph.dangling;

        // ---- 初始化 r = 1/N ----
        std::vector<double> r(N, 1.0 / N);
        std::vector<double> r_new(N);

        int    iter        = 0;
        double final_error = 0.0;
        bool   converged   = false;

        for (iter = 0; iter < max_iter; ++iter) {
            // ---- SpMV: y = M × r ----
            std::vector<double> y = MatrixVector::multiply(M, r);

            // ---- 计算悬挂节点贡献 ----
            double danglesum = 0.0;
            for (int j : dangling) {
                danglesum += r[j];
            }
            double personalization = (damping * danglesum + (1.0 - damping)) / N;

            // ---- 更新 r_new 并计算 L1 误差 ----
            double error = 0.0;
            for (int i = 0; i < N; ++i) {
                r_new[i] = damping * y[i] + personalization;
                error += std::abs(r_new[i] - r[i]);
            }

            // ---- 交换缓冲区 ----
            r.swap(r_new);

            final_error = error;
            if (error < tol) {
                converged = true;
                ++iter; // 计入当前迭代
                break;
            }
        }

        printf("[PageRank] converged=%s  iterations=%d  final_L1_error=%.8e\n",
               converged ? "yes" : "no", iter, final_error);

        return {std::move(r), iter, final_error, converged};
    }

    /**
     * @brief 获取 Top-K 节点 (按 PageRank 值降序)
     *
     * @return vector of (node_id, rank)
     */
    static std::vector<std::pair<int, double>> topK(
        const std::vector<double>& ranks, int k = 10)
    {
        std::vector<std::pair<int, double>> indexed;
        indexed.reserve(ranks.size());
        for (std::size_t i = 0; i < ranks.size(); ++i) {
            indexed.emplace_back(static_cast<int>(i), ranks[i]);
        }

        std::partial_sort(
            indexed.begin(),
            indexed.begin() + std::min(k, static_cast<int>(indexed.size())),
            indexed.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        indexed.resize(std::min(k, static_cast<int>(indexed.size())));
        return indexed;
    }

    /**
     * @brief 计算 L1 距离
     */
    static double l1Distance(const std::vector<double>& a,
                             const std::vector<double>& b)
    {
        assert(a.size() == b.size());
        double sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            sum += std::abs(a[i] - b[i]);
        }
        return sum;
    }
};

} // namespace dmc
