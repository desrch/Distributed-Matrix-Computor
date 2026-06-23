#pragma once

#include "../matrix/DenseMatrix.hpp"
#include "../partition/RowPartitioner.hpp"
#include "../runtime/DistributedContext.hpp"
#include "../communication/MPIWrapper.hpp"
#include "../operators/DenseMultiply.hpp"

#include <vector>

namespace dmc {

/**
 * @brief 分布式稠密矩阵乘法
 *
 * 完整流水线:
 *
 *   Master (rank 0) 持有完整 A(M×K) 和 B(K×N)
 *        │
 *        ├─ 1. RowPartitioner 确定每个 rank 负责的 A 行范围
 *        ├─ 2. MPI_Scatterv  按行分发 A → 各 rank 得到 local_A
 *        ├─ 3. MPI_Bcast     广播完整 B → 所有 rank
 *        ├─ 4. 各 rank 独立计算 local_C = local_A × B
 *        └─ 5. MPI_Gatherv   按行收集 local_C → Master 拼出完整 C(M×N)
 *
 * 用法:
 * @code
 *   // 所有 rank 上构造相同维度的 A, B（数据仅 master 有效）
 *   DenseMatrix A(M, K, ...);  // master 填数据
 *   DenseMatrix B(K, N, ...);  // master 填数据
 *   DenseMatrix C = DistributedDenseMultiply::multiply(A, B);
 *   // rank 0 上的 C 是完整结果
 * @endcode
 */
class DistributedDenseMultiply {
public:
    /**
     * @brief 执行分布式稠密矩阵乘法
     *
     * 所有 rank 必须传入相同维度的 A 和 B。
     * A 的数据仅在 root(0) 上有效；B 的数据仅在 root 上有效。
     * 返回的 C 仅在 root 上包含完整结果（非 root 上为空矩阵）。
     */
    static DenseMatrix multiply(const DenseMatrix& A, const DenseMatrix& B)
    {
        auto& ctx = DistributedContext::instance();
        int rank   = ctx.rank();
        int world  = ctx.size();
        constexpr int root = 0;

        // 全局维度
        int M = A.rows();
        int K = A.cols();
        int N = B.cols();

        // ===============================================================
        // 1. 计算本 rank 负责的行范围
        // ===============================================================
        PartitionInfo info = RowPartitioner::compute(M, rank, world);
        int local_rows = info.local_rows();

        // ===============================================================
        // 2. 构建 Scatter 参数: send_counts / displs (仅 root 使用)
        // ===============================================================
        std::vector<int> send_counts(world);
        std::vector<int> displs(world);
        for (int r = 0; r < world; ++r) {
            PartitionInfo pi = RowPartitioner::compute(M, r, world);
            send_counts[r] = pi.local_rows() * K;  // 每个 rank 拿到的 double 数
            displs[r]      = pi.start_row * K;      // 在全局 A 数据中的起始偏移
        }

        // ===============================================================
        // 3. Scatter: 按行分发 A
        // ===============================================================
        DenseMatrix local_A(local_rows, K);
        MPIWrapper::scatterDense(A.raw_ptr(),
                                 local_A.raw_ptr(),
                                 send_counts, displs, root);

        // ===============================================================
        // 4. Broadcast: 将完整 B 发给所有 rank
        // ===============================================================
        DenseMatrix local_B(K, N);       // 非 root 预先分配相同维度
        if (rank == root) {
            local_B = B;                 // root 拷贝自己的 B
        }
        MPIWrapper::broadcastDense(local_B.raw_ptr(), K, N, root);

        // ===============================================================
        // 5. 局部乘法: local_C = local_A × B
        // ===============================================================
        DenseMatrix local_C(local_rows, N);
        if (local_rows > 0) {
            local_C = DenseMultiply::multiply(local_A, local_B);
        }

        // ===============================================================
        // 6. 构建 Gather 参数: recv_counts / displs (仅 root 使用)
        // ===============================================================
        std::vector<int> recv_counts(world);
        std::vector<int> rdispls(world);
        for (int r = 0; r < world; ++r) {
            PartitionInfo pi = RowPartitioner::compute(M, r, world);
            recv_counts[r] = pi.local_rows() * N;
            rdispls[r]     = pi.start_row * N;
        }

        // ===============================================================
        // 7. Gather: 按行收集 C
        // ===============================================================
        DenseMatrix C(M, N);
        MPIWrapper::gatherDense(local_C.raw_ptr(),
                                C.raw_ptr(),
                                recv_counts, rdispls, root);

        return C;
    }
};

} // namespace dmc
