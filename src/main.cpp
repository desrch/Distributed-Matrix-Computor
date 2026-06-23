/**
 * @file main.cpp
 * @brief 分布式矩阵计算框架 —— PageRank 正确性验证 (实验一)
 *
 * 流程:
 *   1. 从 SNAP 格式文件加载有向图
 *   2. 构建列随机 CSR 转换矩阵
 *   3. PageRank Power Method (d=0.85, max_iter=100, tol=1e-6)
 *   4. 输出 Top-10 节点 + L1 范数
 *   5. 若存在 NetworkX 参考文件则计算 L1 Error
 *
 * 运行:
 *   ./build/matrix_app data/web-Stanford.txt
 *   mpirun -np 4 ./build/matrix_app data/web-Stanford.txt
 */

#include "matrix/CSRMatrix.hpp"
#include "runtime/DistributedContext.hpp"
#include "operators/MatrixVector.hpp"
#include "algorithms/PageRank.hpp"
#include "utils/GraphLoader.hpp"
#include "utils/Timer.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

// ========== 辅助 ==========

/** 从 NetworkX 生成的二进制文件加载 rank 向量 */
static bool loadRefRanks(const std::string& path,
                         std::vector<double>& ranks)
{
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) return false;

    fin.seekg(0, std::ios::end);
    std::streamsize sz = fin.tellg();
    fin.seekg(0, std::ios::beg);

    std::size_t count = static_cast<std::size_t>(sz) / sizeof(double);
    if (count == 0) return false;

    ranks.resize(count);
    fin.read(reinterpret_cast<char*>(ranks.data()), sz);
    return fin.good();
}

/** 打印 Top-K */
static void printTopK(const std::vector<double>& ranks,
                      int k = 10,
                      int label_offset = 0) // 0-based → 1-based for display
{
    auto top = dmc::PageRank::topK(ranks, k);
    printf("  %-6s %-14s %-18s\n", "Rank", "NodeID(1b)", "PageRank");
    printf("  %s\n", std::string(36, '-').c_str());
    for (std::size_t i = 0; i < top.size(); ++i) {
        printf("  %-6zu %-14d %-18.10e\n",
               i + 1, top[i].first + label_offset, top[i].second);
    }
}

// ========== main ==========

int main(int argc, char** argv)
{
    using namespace dmc;

    // ---- 参数解析 ----
    std::string filepath;
    if (argc >= 2) {
        filepath = argv[1];
    } else {
        fprintf(stderr, "Usage: %s <graph_file>\n", argv[0]);
        return 1;
    }

    // ---- 初始化 MPI ----
    DistributedContext::initialize(argc, argv);
    auto& ctx = DistributedContext::instance();
    int rank = ctx.rank();
    int world = ctx.size();

    bool is_master = ctx.isMaster();

#if defined(_OPENMP)
    int nthr = omp_get_max_threads();
#else
    int nthr = 1;
#endif

    if (is_master) {
        printf("========================================\n");
        printf("  PageRank 正确性验证 (实验一)\n");
        printf("  MPI=%d  OMP=%d\n", world, nthr);
        printf("  Data: %s\n", filepath.c_str());
        printf("========================================\n\n");
    }

    // 非 master 等待，仅 master 加载
    ctx.barrier();

    // ================================================================
    // Phase 1: 加载图 & 构建 CSR
    // ================================================================
    Timer timer;
    GraphLoader::Graph graph;
    int N = 0, E = 0;

    if (is_master) {
        timer.start("load_and_build");
        graph = GraphLoader::loadSNAP(filepath, /*one_based=*/true);
        timer.stop("load_and_build");

        N = graph.num_nodes;
        E = graph.num_edges;

        printf("\n--- Graph Stats ---\n");
        printf("  Nodes:       %d\n", N);
        printf("  Edges:       %d\n", E);
        printf("  CSR nnz:     %d\n", graph.csr.nnz());
        printf("  Dangling:    %zu (%.2f%%)\n",
               graph.dangling.size(),
               100.0 * graph.dangling.size() / N);
        printf("  Load+Build:  %.4f s\n", timer.get("load_and_build"));
    }

    ctx.barrier();

    // ================================================================
    // Phase 2: PageRank
    // ================================================================
    PageRank::Result pr_result;

    if (is_master) {
        printf("\n--- PageRank (d=0.85, max_iter=100, tol=1e-6) ---\n");

        timer.start("pagerank");
        pr_result = PageRank::run(graph, 0.85, 100, 1e-6);
        timer.stop("pagerank");

        printf("  Iterations:   %d\n", pr_result.iterations);
        printf("  Final L1 err: %.8e\n", pr_result.final_error);
        printf("  Converged:    %s\n", pr_result.converged ? "yes" : "no");
        printf("  PageRank time: %.4f s\n", timer.get("pagerank"));

        // 验证 rank 和为 1
        double sum = 0.0;
        for (double v : pr_result.ranks) sum += v;
        printf("  Sum(ranks):   %.10f\n", sum);

        // Top-10
        printf("\n--- Top-10 PageRank Nodes ---\n");
        printTopK(pr_result.ranks, 10, /*1-based display*/ 1);
    }

    // ================================================================
    // Phase 3: 与 NetworkX 参考对比 (若存在)
    // ================================================================
    if (is_master) {
        // 推断参考文件路径: data/pagerank_ref.bin
        std::string dir;
        {
            auto pos = filepath.rfind('/');
            dir = (pos != std::string::npos) ? filepath.substr(0, pos) : ".";
        }
        std::string ref_path = dir + "/pagerank_ref.bin";

        std::vector<double> ref_ranks;
        if (loadRefRanks(ref_path, ref_ranks)) {
            printf("\n--- 与 NetworkX 参考对比 ---\n");
            printf("  参考文件: %s\n", ref_path.c_str());
            printf("  参考 N:   %zu\n", ref_ranks.size());

            if (ref_ranks.size() == pr_result.ranks.size()) {
                double l1_err = PageRank::l1Distance(pr_result.ranks, ref_ranks);
                printf("  L1 Error (ours vs NetworkX): %.10e\n", l1_err);

                // Top-10 对拍
                auto top_ours = PageRank::topK(pr_result.ranks, 10);
                auto top_ref  = PageRank::topK(ref_ranks, 10);
                printf("  %-8s %-16s %-16s\n", "", "Ours (1b)", "Ref (1b)");
                for (int i = 0; i < 10; ++i) {
                    printf("  Top%-4d %-16d %-16d",
                           i + 1, top_ours[i].first + 1, top_ref[i].first + 1);
                    if (top_ours[i].first == top_ref[i].first)
                        printf("  ✓\n");
                    else
                        printf("  ✗\n");
                }

                if (l1_err < 1e-4) {
                    printf("\n  ✅ PASS: L1 Error 在可接受范围内 (< 1e-4)\n");
                } else {
                    printf("\n  ⚠️  WARNING: L1 Error 较大 (%.2e)\n", l1_err);
                }
            } else {
                printf("  ⚠️  维度不匹配: ours=%zu, ref=%zu\n",
                       pr_result.ranks.size(), ref_ranks.size());
            }
        } else {
            printf("\n--- NetworkX 参考 ---\n");
            printf("  未找到参考文件: %s\n", ref_path.c_str());
            printf("  运行: python3 scripts/pagerank_ref.py %s\n", filepath.c_str());
        }
    }

    // ================================================================
    // 总结
    // ================================================================
    if (is_master) {
        printf("\n--- 时间摘要 ---\n");
        timer.printAll();

        printf("\n========================================\n");
        printf("  实验一完成\n");
        printf("========================================\n");
    }

    ctx.barrier();
    DistributedContext::finalize();
    return 0;
}
