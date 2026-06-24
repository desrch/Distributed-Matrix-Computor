#pragma once

#include "../matrix/CSRMatrix.hpp"
#include "../engines/SpMVEngine.hpp"
#include "../runtime/DistributedContext.hpp"
#include "../communication/MPIWrapper.hpp"
#include "../utils/GraphLoader.hpp"
#include "../utils/Memory.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace dmc {

/**
 * @brief PageRank 性能指标
 */
struct PerfStats {
    // 输入参数
    int    num_nodes    = 0;
    int    num_edges    = 0;
    int    csr_nnz      = 0;
    int    mpi_ranks    = 1;
    int    omp_threads  = 1;
    double damping     = 0.85;
    double tol         = 1e-6;

    // 运行结果
    int    iterations   = 0;
    double final_error  = 0.0;
    bool   converged    = false;

    // 时间 (秒)
    double load_time    = 0.0;   // 图加载+CSR构建
    double total_time   = 0.0;   // PageRank 总时间 (不含加载)
    double spmv_time    = 0.0;   // SpMV 总时间
    double comm_time    = 0.0;   // MPI 通信总时间
    double comp_time    = 0.0;   // 纯计算总时间
    double damp_time    = 0.0;   // damping+error 计算时间

    // 内存 (KB)
    long   mem_before   = 0;     // PageRank 前
    long   mem_after    = 0;     // PageRank 后
    long   mem_peak_rss = 0;     // 峰值 RSS

    // 每轮平均
    double avg_spmv_ms()  const { return (iterations > 0) ? (spmv_time * 1000.0 / iterations) : 0; }
    double avg_comm_ms()  const { return (iterations > 0) ? (comm_time * 1000.0 / iterations) : 0; }
    double avg_comp_ms()  const { return (iterations > 0) ? (comp_time * 1000.0 / iterations) : 0; }
    double comm_pct()     const { return (spmv_time > 0) ? (comm_time / spmv_time * 100.0) : 0; }
    double comp_pct()     const { return (spmv_time > 0) ? (comp_time / spmv_time * 100.0) : 0; }
};

class PageRank {
public:
    struct Result {
        std::vector<double> ranks;
        PerfStats stats;
    };

    /**
     * @brief 执行 PageRank (含完整性能指标采集)
     *
     * @param engine    注入的 SpMV 引擎
     * @param graph     图数据 (CSR 仅在 root 上有效)
     * @param load_sec  图加载耗时 (外部传入)
     */
    static Result run(ISpMVEngine& engine,
                      const GraphLoader::Graph& graph,
                      double load_sec = 0.0,
                      double damping = 0.85,
                      int    max_iter = 100,
                      double tol = 1e-6)
    {
        auto& ctx = DistributedContext::instance();
        bool is_master = ctx.isMaster();

        const CSRMatrix& M   = graph.csr;
        int N                = graph.num_nodes;
        const auto& dangling = graph.dangling;

        PerfStats stats;
        stats.num_nodes   = N;
        stats.num_edges   = graph.num_edges;
        stats.csr_nnz     = M.nnz();
        stats.mpi_ranks   = ctx.size();
        stats.damping     = damping;
        stats.tol         = tol;
        stats.load_time   = load_sec;
#if defined(_OPENMP)
        stats.omp_threads = omp_get_max_threads();
#endif

        // 重置引擎的累计计时
        engine.resetTiming();

        // ---- 内存 snapshot ----
        long mem0 = Memory::currentRSS();
        stats.mem_before = mem0;

        // ---- 初始化 rank 向量 ----
        std::vector<double> r;
        if (is_master) r.assign(N, 1.0 / N);

        std::vector<double> r_new;
        if (is_master) r_new.resize(N);

        int    iter        = 0;
        double final_error = 0.0;
        bool   converged   = false;
        double damp_acc    = 0.0;

        auto total_t0 = std::chrono::high_resolution_clock::now();

        for (iter = 0; iter < max_iter; ++iter) {
            // ---- SpMV ----
            std::vector<double> y = engine.multiply(M, r);

            int stop_flag = 0;

            if (is_master) {
                auto dt0 = std::chrono::high_resolution_clock::now();

                double danglesum = 0.0;
                for (int j : dangling) danglesum += r[j];
                double personalization =
                    (damping * danglesum + (1.0 - damping)) / N;

                double error = 0.0;
                for (int i = 0; i < N; ++i) {
                    r_new[i] = damping * y[i] + personalization;
                    error += std::abs(r_new[i] - r[i]);
                }

                r.swap(r_new);
                final_error = error;

                auto dt1 = std::chrono::high_resolution_clock::now();
                damp_acc += std::chrono::duration<double>(dt1 - dt0).count();

                if (error < tol) {
                    converged = true;
                    ++iter;
                    stop_flag = 1;
                }
            }

            MPIWrapper::broadcastInt(&stop_flag, 1, 0);
            if (stop_flag) break;
        }

        auto total_t1 = std::chrono::high_resolution_clock::now();
        stats.total_time = std::chrono::duration<double>(total_t1 - total_t0).count();

        // ---- 从引擎获取 SpMV 计时 ----
        SpMVTiming spmv_tm = engine.getTiming();
        stats.spmv_time = spmv_tm.total();
        stats.comm_time = spmv_tm.comm_sec;
        stats.comp_time = spmv_tm.comp_sec;
        stats.damp_time = damp_acc;

        // ---- 内存 snapshot ----
        long mem1 = Memory::currentRSS();
        stats.mem_after  = mem1;
        stats.mem_peak_rss = std::max(mem0, mem1);

        // ---- 填充结果 ----
        stats.iterations  = iter;
        stats.final_error = final_error;
        stats.converged   = converged;

        if (is_master) {
            printf("[PageRank] engine=%-22s  iters=%d  total=%.4fs  SpMV=%.4fs  "
                   "comm=%.4fs(%.1f%%)  comp=%.4fs(%.1f%%)  damp=%.4fs  mem=%ld KB\n",
                   engine.name().c_str(), iter,
                   stats.total_time, stats.spmv_time,
                   stats.comm_time, stats.comm_pct(),
                   stats.comp_time, stats.comp_pct(),
                   damp_acc, stats.mem_peak_rss);
        }

        Result res;
        res.stats = stats;
        if (is_master) {
            res.ranks = std::move(r);
        }
        return res;
    }

    static std::vector<std::pair<int, double>> topK(
        const std::vector<double>& ranks, int k = 10)
    {
        std::vector<std::pair<int, double>> indexed;
        indexed.reserve(ranks.size());
        for (std::size_t i = 0; i < ranks.size(); ++i)
            indexed.emplace_back(static_cast<int>(i), ranks[i]);

        std::partial_sort(
            indexed.begin(),
            indexed.begin() + std::min(k, static_cast<int>(indexed.size())),
            indexed.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        indexed.resize(std::min(k, static_cast<int>(indexed.size())));
        return indexed;
    }

    static double l1Distance(const std::vector<double>& a,
                             const std::vector<double>& b)
    {
        assert(a.size() == b.size());
        double sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
            sum += std::abs(a[i] - b[i]);
        return sum;
    }
};

} // namespace dmc
