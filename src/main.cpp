/**
 * @file main.cpp
 * @brief PageRank 多引擎对比 + 性能指标采集
 *
 * 每个引擎运行后采集:
 *  - total_time, spmv_time, comm_time, comp_time, damp_time
 *  - memory (RSS KB)
 *  - iterations, convergence 等
 *
 * 输出 CSV 到 lab_res/ 目录。
 *
 * 运行:
 *   OMP_NUM_THREADS=4 mpirun -np 4 ./build/matrix_app data/web-Stanford.txt
 */

#include "matrix/CSRMatrix.hpp"
#include "runtime/DistributedContext.hpp"
#include "communication/MPIWrapper.hpp"
#include "utils/GraphLoader.hpp"
#include "utils/Timer.hpp"
#include "utils/Memory.hpp"
#include "algorithms/PageRank.hpp"

#include "engines/SpMVEngine.hpp"
#include "engines/DenseSerialEngine.hpp"
#include "engines/CSRSerialEngine.hpp"
#include "engines/OpenMPCSREngine.hpp"
#include "engines/MPICSREngine.hpp"
#include "engines/HybridSpMVEngine.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

// ========== 辅助 ==========

static bool loadRefRanks(const std::string& path, std::vector<double>& ranks)
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

/** 确保目录存在 */
static void ensureDir(const std::string& dir)
{
    mkdir(dir.c_str(), 0755);
}

/** 生成时间戳 */
static std::string timestamp()
{
    time_t now = time(nullptr);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&now));
    return buf;
}

/** 写入 CSV 表头 */
static void writeCSVHeader(std::ofstream& csv)
{
    csv << "engine,dataset,num_nodes,num_edges,csr_nnz,"
        << "mpi_ranks,omp_threads,damping,tol,"
        << "iterations,converged,final_error,"
        << "load_time_s,total_time_s,spmv_time_s,"
        << "comm_time_s,comp_time_s,damp_time_s,"
        << "comm_pct,comp_pct,"
        << "avg_spmv_ms,avg_comm_ms,avg_comp_ms,"
        << "mem_before_kb,mem_after_kb,mem_peak_rss_kb,"
        << "l1_vs_ref,top10_match\n";
}

/** 写入一行性能数据 */
static void writeCSVRow(std::ofstream& csv, const dmc::PerfStats& s,
                        const std::string& engine, const std::string& dataset,
                        double l1_err, int top_match)
{
    csv << engine << ","
        << dataset << ","
        << s.num_nodes << ","
        << s.num_edges << ","
        << s.csr_nnz << ","
        << s.mpi_ranks << ","
        << s.omp_threads << ","
        << s.damping << ","
        << s.tol << ","
        << s.iterations << ","
        << (s.converged ? "yes" : "no") << ","
        << s.final_error << ","
        << s.load_time << ","
        << s.total_time << ","
        << s.spmv_time << ","
        << s.comm_time << ","
        << s.comp_time << ","
        << s.damp_time << ","
        << s.comm_pct() << ","
        << s.comp_pct() << ","
        << s.avg_spmv_ms() << ","
        << s.avg_comm_ms() << ","
        << s.avg_comp_ms() << ","
        << s.mem_before << ","
        << s.mem_after << ","
        << s.mem_peak_rss << ","
        << l1_err << ","
        << top_match << "\n";
}

/** 提取数据集名称 (从路径) */
static std::string datasetName(const std::string& filepath)
{
    auto pos = filepath.rfind('/');
    std::string fname = (pos != std::string::npos) ? filepath.substr(pos + 1) : filepath;
    auto dot = fname.rfind('.');
    return (dot != std::string::npos) ? fname.substr(0, dot) : fname;
}

// ========== main ==========

int main(int argc, char** argv)
{
    using namespace dmc;

    DistributedContext::initialize(argc, argv);
    auto& ctx = DistributedContext::instance();
    int rank  = ctx.rank();
    int world = ctx.size();

#if defined(_OPENMP)
    int nthr = omp_get_max_threads();
#else
    int nthr = 1;
#endif

    std::string filepath;
    if (argc >= 2) {
        filepath = argv[1];
    } else {
        if (rank == 0) fprintf(stderr, "Usage: %s <graph_file>\n", argv[0]);
        DistributedContext::finalize();
        return 1;
    }

    std::string dname = datasetName(filepath);

    if (rank == 0) {
        printf("============================================================\n");
        printf("  PageRank 性能指标采集\n");
        printf("  Dataset: %s  MPI=%d  OMP=%d\n", dname.c_str(), world, nthr);
        printf("============================================================\n\n");
    }

    // ================================================================
    // 加载图
    // ================================================================
    Timer timer;
    GraphLoader::Graph graph;
    int N = 0;
    double load_sec = 0.0;

    if (rank == 0) {
        timer.start("load");
        graph = GraphLoader::loadSNAP(filepath);  // auto-detect 0/1-based
        load_sec = timer.stop("load");
        N = graph.num_nodes;
        printf("  Graph: N=%d  E=%d  nnz=%d  dangling=%zu (%.2f%%)\n",
               N, graph.num_edges, graph.csr.nnz(),
               graph.dangling.size(), 100.0 * graph.dangling.size() / N);
        printf("  Load time: %.4f s\n", load_sec);
        printf("  Mem after load: %ld KB\n\n", Memory::currentRSS());
    }

    // broadcast N
    int global_n[1] = {N};
    MPIWrapper::broadcastInt(global_n, 1, 0);
    N = global_n[0];

    ctx.barrier();

    // ---- 参考值 ----
    std::vector<double> ref_ranks;
    bool have_ref = false;
    if (rank == 0) {
        auto pos = filepath.rfind('/');
        std::string dir = (pos != std::string::npos) ? filepath.substr(0, pos) : ".";
        have_ref = loadRefRanks(dir + "/pagerank_ref.bin", ref_ranks);
    }

    // ================================================================
    // 引擎列表
    // ================================================================
    DenseSerialEngine  e_dense;
    CSRSerialEngine    e_csr_serial;
    OpenMPCSREngine    e_omp;
    MPICSREngine       e_mpi;
    HybridSpMVEngine   e_hybrid;

    struct EngineEntry {
        ISpMVEngine* engine;
        bool skip_if_mpi_gt_1;
        int  max_nodes;
    };
    const int DENSE_MAX_N = 20000;

    std::vector<EngineEntry> engines = {
        {&e_dense,       true,  DENSE_MAX_N},
        {&e_csr_serial,  true,  0},
        {&e_omp,         true,  0},
        {&e_mpi,         false, 0},
        {&e_hybrid,      false, 0},
    };

    constexpr double DAMPING  = 0.85;
    constexpr int    MAX_ITER = 100;
    constexpr double TOL      = 1e-6;

    struct Row {
        std::string engine_name;
        PerfStats   stats;
        double      l1_err;
        int         top_match;
        bool        skipped;
    };
    std::vector<Row> rows;

    for (const auto& entry : engines) {
        if (entry.skip_if_mpi_gt_1 && world > 1) continue;

        if (entry.max_nodes > 0 && N > entry.max_nodes) {
            if (rank == 0)
                printf("--- %s (SKIP: N=%d > max=%d) ---\n\n",
                       entry.engine->name().c_str(), N, entry.max_nodes);
            rows.push_back({entry.engine->name(), {}, -1, 0, true});
            continue;
        }

        ctx.barrier();

        if (rank == 0) {
            printf("--- %s ---\n", entry.engine->name().c_str());
            fflush(stdout);
        }

        PageRank::Result pr = PageRank::run(*entry.engine, graph,
                                             load_sec, DAMPING, MAX_ITER, TOL);

        ctx.barrier();

        if (rank == 0) {
            double l1_err = -1.0;
            int top_match = 0;
            if (have_ref && pr.stats.num_nodes == static_cast<int>(ref_ranks.size())) {
                l1_err = PageRank::l1Distance(pr.ranks, ref_ranks);
                auto topO = PageRank::topK(pr.ranks, 10);
                auto topR = PageRank::topK(ref_ranks, 10);
                for (int i = 0; i < 10; ++i)
                    if (topO[i].first == topR[i].first) top_match++;
            }

            rows.push_back({entry.engine->name(), pr.stats, l1_err, top_match, false});

            printf("  iters=%d  ttl=%.4fs  spmv=%.4fs  comm=%.4fs(%.0f%%)  "
                   "comp=%.4fs(%.0f%%)  mem=%ldKB\n",
                   pr.stats.iterations, pr.stats.total_time, pr.stats.spmv_time,
                   pr.stats.comm_time, pr.stats.comm_pct(),
                   pr.stats.comp_time, pr.stats.comp_pct(),
                   pr.stats.mem_peak_rss);
            if (have_ref)
                printf("  ✅ L1=%.2e  Top10=%d/10\n\n", l1_err, top_match);
            else
                printf("\n");
        }
    }

    // ================================================================
    // 写入 CSV
    // ================================================================
    if (rank == 0) {
        ensureDir("lab_res");
        std::string ts = timestamp();
        std::string csv_path = "lab_res/pagerank_"
                             + dname + "_p" + std::to_string(world)
                             + "t" + std::to_string(nthr)
                             + "_" + ts + ".csv";

        std::ofstream csv(csv_path);
        if (!csv.is_open()) {
            fprintf(stderr, "ERROR: Cannot write %s\n", csv_path.c_str());
        } else {
            writeCSVHeader(csv);
            for (const auto& r : rows) {
                if (r.skipped) continue;
                writeCSVRow(csv, r.stats, r.engine_name, dname,
                            r.l1_err, r.top_match);
            }
            csv.close();
            printf("============================================================\n");
            printf("  指标已保存: %s  (%zu 行)\n", csv_path.c_str(),
                   rows.size() - std::count_if(rows.begin(), rows.end(),
                       [](const Row& r) { return r.skipped; }));
            printf("============================================================\n");
        }

        // 同时打印终端汇总表
        printf("\n  %-26s %6s %8s %8s %8s %8s %6s %8s %8s\n",
               "Engine", "Iters", "Total", "SpMV", "Comm", "Comp",
               "C%%", "L1Err", "Top10");
        printf("  %s\n", std::string(90, '-').c_str());
        for (const auto& r : rows) {
            if (r.skipped) {
                printf("  %-26s %s\n", r.engine_name.c_str(), "SKIP");
                continue;
            }
            const auto& s = r.stats;
            char l1buf[16];
            snprintf(l1buf, sizeof(l1buf), "%.1e", r.l1_err);
            char topbuf[16];
            snprintf(topbuf, sizeof(topbuf), "%d/10", r.top_match);
            printf("  %-26s %6d %8.4f %8.4f %8.4f %8.4f %5.0f%% %8s %8s\n",
                   r.engine_name.c_str(), s.iterations,
                   s.total_time, s.spmv_time, s.comm_time, s.comp_time,
                   s.comm_pct(), l1buf, topbuf);
        }
        printf("\n");
    }

    ctx.barrier();
    DistributedContext::finalize();
    return 0;
}
