#pragma once

#include "../matrix/CSRMatrix.hpp"
#include "../partition/RowPartitioner.hpp"
#include "../runtime/DistributedContext.hpp"
#include "../communication/MPIWrapper.hpp"
#include "../operators/MatrixVector.hpp"

#include <vector>

namespace dmc {

/**
 * @brief 分布式 CSR × Vector (SpMV)
 *
 * 完整流水线:
 *
 *   Master (rank 0) 持有完整 CSR 矩阵 A 和向量 x
 *        │
 *        ├─ 1. RowPartitioner 确定每个 rank 负责的行范围
 *        ├─ 2. 按行切分 CSR: 各 rank 收到局部 values / col_idx / row_ptr
 *        ├─ 3. MPI_Bcast     广播向量 x → 所有 rank
 *        ├─ 4. 各 rank 独立计算 y_local = A_local × x (SpMV)
 *        └─ 5. MPI_Gatherv   收集 y_local → Master 拼出完整 y
 *
 * 通信策略:
 *   - row_ptr: 全局广播 (O(M) ints, 远小于 O(nnz) doubles)
 *   - values / col_idx: 按 nnz 量分散 (MPI_Scatterv)
 *   - x: 广播 (O(N) doubles)
 *   - y: 按行收集 (MPI_Gatherv)
 *
 * 用法:
 * @code
 *   // 所有 rank 上构造相同维度的 A
 *   CSRMatrix A = CSRMatrix::fromDense(denseA);  // 仅 master 数据有效
 *   std::vector<double> x(N, ...);                // 仅 master 数据有效
 *   std::vector<double> y = DistributedCSRVector::multiply(A, x);
 *   // rank 0 上的 y 是完整结果
 * @endcode
 */
class DistributedCSRVector {
public:
    /**
     * @brief 执行分布式 CSR × Vector
     *
     * 所有 rank 必须传入相同维度的 A 和 x。
     * A 的 CSR 数据仅在 root(0) 上有效；x 的数据仅在 root 上有效。
     * 返回的 y 仅在 root 上包含完整结果。
     */
    static std::vector<double> multiply(const CSRMatrix& A,
                                        const std::vector<double>& x)
    {
        auto& ctx = DistributedContext::instance();
        int rank   = ctx.rank();
        int world  = ctx.size();
        constexpr int root = 0;

        // ===========================================================
        // 0. 广播维度 (非 root 上的 A 可能是空壳)
        // ===========================================================
        int dims[2] = {A.rows(), A.cols()};
        MPIWrapper::broadcastInt(dims, 2, root);
        int M = dims[0];
        int N = dims[1];

        // ===========================================================
        // 1. 广播 nnz 和全局 row_ptr
        // ===========================================================
        int nnz_global = (rank == root) ? A.nnz() : 0;
        MPIWrapper::broadcastInt(&nnz_global, 1, root);

        std::vector<int> global_row_ptr(M + 1);
        if (rank == root) {
            global_row_ptr = A.rowPtr();
        }
        MPIWrapper::broadcastInt(global_row_ptr.data(), M + 1, root);

        // ===========================================================
        // 2. 计算各 rank 的行范围 & CSR 分片参数
        // ===========================================================
        PartitionInfo info = RowPartitioner::compute(M, rank, world);
        int local_rows = info.local_rows();
        int nnz_start  = global_row_ptr[info.start_row];
        int nnz_end    = global_row_ptr[info.end_row];
        int local_nnz  = nnz_end - nnz_start;

        // 各 rank 的 send_counts/displs (仅 root 需要发送)
        std::vector<int> send_counts_val(world);
        std::vector<int> displs_val(world);
        std::vector<int> send_counts_cidx(world);
        std::vector<int> displs_cidx(world);
        for (int r = 0; r < world; ++r) {
            PartitionInfo pi = RowPartitioner::compute(M, r, world);
            int ns = global_row_ptr[pi.start_row];
            int ne = global_row_ptr[pi.end_row];
            send_counts_val[r]  = ne - ns;
            displs_val[r]       = ns;
            send_counts_cidx[r] = ne - ns;
            displs_cidx[r]      = ns;
        }

        // ===========================================================
        // 3. 散射 values 和 col_idx
        // ===========================================================
        std::vector<double> local_values(local_nnz);
        MPIWrapper::scatterDense(
            (rank == root) ? A.values().data() : nullptr,
            local_values.data(),
            send_counts_val, displs_val, root);

        std::vector<int> local_cidx(local_nnz);
        MPIWrapper::scatterInt(
            (rank == root) ? A.colIdx().data() : nullptr,
            local_cidx.data(),
            send_counts_cidx, displs_cidx, root);

        // ===========================================================
        // 4. 各 rank 构建局部 row_ptr（归零调整）
        // ===========================================================
        std::vector<int> local_row_ptr(local_rows + 1);
        for (int i = 0; i <= local_rows; ++i) {
            local_row_ptr[i] = global_row_ptr[info.start_row + i] - nnz_start;
        }

        // 构造局部 CSR 矩阵
        CSRMatrix local_A(local_rows, N,
                          std::move(local_values),
                          std::move(local_cidx),
                          std::move(local_row_ptr));

        // ===========================================================
        // 5. 广播向量 x
        // ===========================================================
        std::vector<double> local_x(N);
        if (rank == root) {
            local_x = x;
        }
        MPIWrapper::broadcastDense(local_x.data(), N, 1, root);

        // ===========================================================
        // 6. 局部 SpMV
        // ===========================================================
        std::vector<double> local_y;
        if (local_rows > 0) {
            local_y = MatrixVector::multiply(local_A, local_x);
        } else {
            local_y.resize(local_rows); // 空
        }
        assert(static_cast<int>(local_y.size()) == local_rows);

        // ===========================================================
        // 7. 构建 Gather 参数
        // ===========================================================
        std::vector<int> recv_counts(world);
        std::vector<int> rdispls(world);
        for (int r = 0; r < world; ++r) {
            PartitionInfo pi = RowPartitioner::compute(M, r, world);
            recv_counts[r] = pi.local_rows();
            rdispls[r]     = pi.start_row;
        }

        // ===========================================================
        // 8. 收集 y
        // ===========================================================
        std::vector<double> y_global(M);
        MPIWrapper::gatherDense(local_y.data(),
                                y_global.data(),
                                recv_counts, rdispls, root);

        return y_global;
    }
};

} // namespace dmc
