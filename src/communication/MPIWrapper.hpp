#pragma once

#if defined(DMC_USE_MPI)
#include <mpi.h>
#else
// 无 MPI 时的占位类型
using MPI_Comm = int;
#endif

#include <cassert>
#include <cstring>
#include <vector>

namespace dmc {

/**
 * @brief MPI 通信原语封装
 *
 * 将常用集合操作包装为静态方法。
 * 业务层不应直接调用 MPI C API，应通过此类完成所有通信。
 *
 * 当 DMC_USE_MPI 未定义时退化为单进程实现（本地拷贝）。
 */
class MPIWrapper {
public:
    // ===================================================================
    // Broadcast
    // ===================================================================

    /**
     * @brief 广播整个稠密矩阵
     *
     * @param mat  根进程传入待广播的矩阵；非根进程需预先分配相同维度。
     *             调用后所有进程拥有相同数据。
     * @param root 广播源 rank
     */
    static void broadcastDense(double* data, int rows, int cols, int root);

    // ===================================================================
    // Scatter (按行分发稠密矩阵)
    // ===================================================================

    /**
     * @brief 按行分散稠密矩阵
     *
     * 根进程将 global_data 按 send_counts / displs 切分，
     * 每个进程收到属于自己的一段连续行数据。
     *
     * @param global_data  根进程上的完整矩阵数据 (行优先)
     * @param local_data   各进程接收缓冲区，需预先分配 local_count 个 double
     * @param send_counts  每进程发送的元素个数 (仅根进程有效)
     * @param displs       每进程起始偏移 (仅根进程有效)
     * @param root         数据源 rank
     */
    static void scatterDense(const double* global_data,
                             double* local_data,
                             const std::vector<int>& send_counts,
                             const std::vector<int>& displs,
                             int root);

    // ===================================================================
    // Gather (按行收集稠密矩阵)
    // ===================================================================

    /**
     * @brief 按行收集稠密矩阵
     *
     * 每个进程发送自己的 local_data，根进程按 recv_counts / displs
     * 拼装出完整矩阵。
     *
     * @param local_data    各进程上的局部数据
     * @param global_data   根进程输出缓冲区，需预先分配
     * @param recv_counts   每进程接收的元素个数 (仅根进程有效)
     * @param displs        每进程在全局缓冲区中的偏移 (仅根进程有效)
     * @param root          收集目标 rank
     */
    static void gatherDense(const double* local_data,
                            double* global_data,
                            const std::vector<int>& recv_counts,
                            const std::vector<int>& displs,
                            int root);

    // ===================================================================
    // Allgather
    // ===================================================================

    /**
     * @brief 全收集 —— 每个进程向所有进程广播自己的局部数据
     *
     * @param local_data   各进程上的局部数据 (发送)
     * @param global_data  所有进程的输出缓冲区 (接收拼接结果)
     * @param send_count   本进程发送的元素数
     * @param recv_counts  每进程接收的元素数
     * @param displs       每进程在全局缓冲区中的偏移
     */
    static void allgatherDense(const double* local_data,
                               double* global_data,
                               int send_count,
                               const std::vector<int>& recv_counts,
                               const std::vector<int>& displs);

    // ===================================================================
    // Integer 数组通信 (CSR 元数据)
    // ===================================================================

    /** 广播 int 数组 */
    static void broadcastInt(int* data, int count, int root);

    /** 按元素分发 int 数组 */
    static void scatterInt(const int* global_data,
                           int* local_data,
                           const std::vector<int>& send_counts,
                           const std::vector<int>& displs,
                           int root);

    /** 按元素收集 int 数组 */
    static void gatherInt(const int* local_data,
                          int* global_data,
                          const std::vector<int>& recv_counts,
                          const std::vector<int>& displs,
                          int root);

    /** 全收集 int 数组 */
    static void allgatherInt(const int* local_data,
                             int* global_data,
                             int send_count,
                             const std::vector<int>& recv_counts,
                             const std::vector<int>& displs);
};

// =======================================================================
// 实现
// =======================================================================

inline void MPIWrapper::broadcastDense(double* data, int rows, int cols, int root)
{
    int count = rows * cols;
#if defined(DMC_USE_MPI)
    MPI_Bcast(data, count, MPI_DOUBLE, root, MPI_COMM_WORLD);
#else
    (void)data; (void)count; (void)root;
#endif
}

inline void MPIWrapper::scatterDense(const double* global_data,
                                     double* local_data,
                                     const std::vector<int>& send_counts,
                                     const std::vector<int>& displs,
                                     int root)
{
#if defined(DMC_USE_MPI)
    int local_count = 0;
    MPI_Scatter(send_counts.data(), 1, MPI_INT,
                &local_count, 1, MPI_INT,
                root, MPI_COMM_WORLD);

    MPI_Scatterv(global_data, send_counts.data(), displs.data(), MPI_DOUBLE,
                 local_data, local_count, MPI_DOUBLE,
                 root, MPI_COMM_WORLD);
#else
    // 单进程模式: 拷贝全部数据
    assert(!send_counts.empty());
    int total = 0;
    for (int c : send_counts) total += c;
    (void)displs; (void)root;
    std::memcpy(local_data, global_data, total * sizeof(double));
#endif
}

inline void MPIWrapper::gatherDense(const double* local_data,
                                    double* global_data,
                                    const std::vector<int>& recv_counts,
                                    const std::vector<int>& displs,
                                    int root)
{
#if defined(DMC_USE_MPI)
    int local_count = 0;
    // 先获取自己的数据量（等于 recv_counts[rank]）
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    local_count = recv_counts[static_cast<std::size_t>(rank)];

    MPI_Gatherv(local_data, local_count, MPI_DOUBLE,
                global_data, recv_counts.data(), displs.data(), MPI_DOUBLE,
                root, MPI_COMM_WORLD);
#else
    // 单进程模式: 拷贝全部数据
    assert(!recv_counts.empty());
    int total = 0;
    for (int c : recv_counts) total += c;
    (void)displs; (void)root;
    std::memcpy(global_data, local_data, total * sizeof(double));
#endif
}

inline void MPIWrapper::allgatherDense(const double* local_data,
                                       double* global_data,
                                       int send_count,
                                       const std::vector<int>& recv_counts,
                                       const std::vector<int>& displs)
{
#if defined(DMC_USE_MPI)
    MPI_Allgatherv(local_data, send_count, MPI_DOUBLE,
                   global_data, recv_counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
#else
    // 单进程模式: 拷贝
    assert(!recv_counts.empty());
    int total = 0;
    for (int c : recv_counts) total += c;
    (void)send_count; (void)displs;
    std::memcpy(global_data, local_data, total * sizeof(double));
#endif
}

// ===================================================================
// Integer 数组通信实现
// ===================================================================

inline void MPIWrapper::broadcastInt(int* data, int count, int root)
{
#if defined(DMC_USE_MPI)
    MPI_Bcast(data, count, MPI_INT, root, MPI_COMM_WORLD);
#else
    (void)data; (void)count; (void)root;
#endif
}

inline void MPIWrapper::scatterInt(const int* global_data,
                                   int* local_data,
                                   const std::vector<int>& send_counts,
                                   const std::vector<int>& displs,
                                   int root)
{
#if defined(DMC_USE_MPI)
    int local_count = 0;
    MPI_Scatter(send_counts.data(), 1, MPI_INT,
                &local_count, 1, MPI_INT,
                root, MPI_COMM_WORLD);

    MPI_Scatterv(global_data, send_counts.data(), displs.data(), MPI_INT,
                 local_data, local_count, MPI_INT,
                 root, MPI_COMM_WORLD);
#else
    assert(!send_counts.empty());
    int total = 0;
    for (int c : send_counts) total += c;
    (void)displs; (void)root;
    std::memcpy(local_data, global_data, total * sizeof(int));
#endif
}

inline void MPIWrapper::gatherInt(const int* local_data,
                                  int* global_data,
                                  const std::vector<int>& recv_counts,
                                  const std::vector<int>& displs,
                                  int root)
{
#if defined(DMC_USE_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int local_count = recv_counts[static_cast<std::size_t>(rank)];

    MPI_Gatherv(local_data, local_count, MPI_INT,
                global_data, recv_counts.data(), displs.data(), MPI_INT,
                root, MPI_COMM_WORLD);
#else
    assert(!recv_counts.empty());
    int total = 0;
    for (int c : recv_counts) total += c;
    (void)displs; (void)root;
    std::memcpy(global_data, local_data, total * sizeof(int));
#endif
}

inline void MPIWrapper::allgatherInt(const int* local_data,
                                     int* global_data,
                                     int send_count,
                                     const std::vector<int>& recv_counts,
                                     const std::vector<int>& displs)
{
#if defined(DMC_USE_MPI)
    MPI_Allgatherv(local_data, send_count, MPI_INT,
                   global_data, recv_counts.data(), displs.data(), MPI_INT,
                   MPI_COMM_WORLD);
#else
    assert(!recv_counts.empty());
    int total = 0;
    for (int c : recv_counts) total += c;
    (void)send_count; (void)displs;
    std::memcpy(global_data, local_data, total * sizeof(int));
#endif
}

} // namespace dmc
