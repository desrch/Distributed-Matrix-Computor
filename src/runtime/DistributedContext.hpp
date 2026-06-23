#pragma once

#if defined(DMC_USE_MPI)
#include <mpi.h>
#endif

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmc {

/**
 * @brief MPI 运行时上下文的轻量封装
 *
 * 职责:
 *  - 管理 MPI_Init / MPI_Finalize 生命周期
 *  - 提供 rank / world_size / isMaster 查询
 *  - 提供 barrier 等常用集合操作
 *
 * 设计动机: 避免项目各处直接调用 MPI C API，便于日后替换通信后端。
 *
 * 典型用法:
 * @code
 *   int main(int argc, char** argv) {
 *       dmc::DistributedContext::initialize(argc, argv);
 *       auto& ctx = dmc::DistributedContext::instance();
 *       // 使用 ctx.rank(), ctx.size(), ...
 *       dmc::DistributedContext::finalize();
 *   }
 * @endcode
 *
 * 当编译时未定义 DMC_USE_MPI 时，退化为单进程模式 (rank==0, size==1)。
 */
class DistributedContext {
public:
    // ===================================================================
    // 生命周期管理
    // ===================================================================

    /**
     * @brief 初始化 MPI 并创建全局单例
     *
     * 必须在所有其它 MPI 调用之前、且在线程安全的上下文中调用一次。
     * 重复调用是安全的（幂等）。
     *
     * @param argc  来自 main 的 argc 引用
     * @param argv  来自 main 的 argv 引用
     */
    static void initialize(int& argc, char**& argv)
    {
        if (instance_) return; // 幂等

#if defined(DMC_USE_MPI)
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (provided < MPI_THREAD_FUNNELED) {
            // 仅警告，不阻止运行 —— 多数集群实现支持 FUNNELED
            fprintf(stderr,
                    "[DistributedContext] WARNING: MPI_THREAD_FUNNELED not "
                    "available (got level %d).\n",
                    provided);
        }
#else
        (void)argc; (void)argv; // 无 MPI 时抑制未使用警告
#endif

        instance_.reset(new DistributedContext());
        instance_->init();
    }

    /**
     * @brief 初始化 MPI（不带命令行参数版本）
     *
     * 用于 MPI 实现不需要 argc/argv 的场景（MPI-2 及以后允许）。
     */
    static void initialize()
    {
        if (instance_) return;

#if defined(DMC_USE_MPI)
        int provided;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
        if (provided < MPI_THREAD_FUNNELED) {
            fprintf(stderr,
                    "[DistributedContext] WARNING: MPI_THREAD_FUNNELED not "
                    "available (got level %d).\n",
                    provided);
        }
#endif

        instance_.reset(new DistributedContext());
        instance_->init();
    }

    /**
     * @brief 清理 MPI 并销毁全局单例
     *
     * 幂等调用；如果从未 initialize 则为 no-op。
     */
    static void finalize()
    {
        if (!instance_) return;

#if defined(DMC_USE_MPI)
        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
#endif

        instance_.reset();
    }

    // ===================================================================
    // 全局单例访问
    // ===================================================================

    /**
     * @brief 获取全局 DistributedContext 引用
     *
     * 调用前必须已经 initialize()，否则抛出异常。
     */
    static DistributedContext& instance()
    {
        if (!instance_) {
            throw std::logic_error(
                "DistributedContext: not initialized. "
                "Call DistributedContext::initialize() first.");
        }
        return *instance_;
    }

    // ===================================================================
    // 进程信息
    // ===================================================================

    /** 当前进程在 MPI_COMM_WORLD 中的 rank (从 0 开始) */
    int rank() const { return rank_; }

    /** MPI_COMM_WORLD 中的进程总数 */
    int size() const { return world_size_; }

    /** 当前进程是否为 master 进程 (rank == 0) */
    bool isMaster() const { return rank_ == 0; }

    // ===================================================================
    // 集合操作
    // ===================================================================

    /** MPI_Barrier 封装 */
    void barrier() const
    {
#if defined(DMC_USE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    // ===================================================================
    // 辅助: master 进程执行 lambda
    // ===================================================================

    /**
     * @brief 仅在 master 进程上执行给定的函数
     *
     * 方便初始化 / 打印 / I/O 等只应由 rank 0 执行的逻辑。
     */
    template <typename F>
    static void runOnMaster(F&& f)
    {
        if (instance_ && instance_->isMaster()) {
            f();
        }
    }

    // ===================================================================
    // 分布式数据辅助
    // ===================================================================

    /**
     * @brief 计算按行均匀划分时，当前 rank 负责的局部行范围
     *
     * @param total_rows 全局总行数
     * @param local_start [out] 当前进程起始行 (0-based)
     * @param local_count [out] 当前进程负责的行数
     */
    void rowPartition(int total_rows,
                      int& local_start,
                      int& local_count) const
    {
        int base = total_rows / world_size_;
        int rem  = total_rows % world_size_;

        local_start = rank_ * base + std::min(rank_, rem);
        local_count = base + (rank_ < rem ? 1 : 0);
    }

    /**
     * @brief 计算按列均匀划分时，当前 rank 负责的局部列范围
     */
    void colPartition(int total_cols,
                      int& local_start,
                      int& local_count) const
    {
        rowPartition(total_cols, local_start, local_count); // 逻辑相同
    }

private:
    // ===================================================================
    // 内部
    // ===================================================================

    DistributedContext() = default;

    void init()
    {
#if defined(DMC_USE_MPI)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
#else
        rank_       = 0;
        world_size_ = 1;
#endif
    }

    int rank_{0};
    int world_size_{1};

    static std::unique_ptr<DistributedContext> instance_;
};

// 静态成员定义
inline std::unique_ptr<DistributedContext> DistributedContext::instance_;

} // namespace dmc
