#pragma once

#include "../matrix/CSRMatrix.hpp"
#include "../matrix/DenseMatrix.hpp"
#include "../partition/RowPartitioner.hpp"
#include "../runtime/DistributedContext.hpp"
#include "../communication/MPIWrapper.hpp"
#include "../operators/SparseMultiply.hpp"

#include <vector>

namespace dmc {

/**
 * @brief 分布式 CSR × Dense → Dense
 *
 * 流水线:
 *   Master 持有完整 CSR A(M×K) 和 Dense B(K×N)
 *        │
 *        ├─ 1. 广播 M,K,N + row_ptr, 按行 Scatter CSR A
 *        ├─ 2. Broadcast  完整 Dense B
 *        ├─ 3. 各 rank CSR×Dense: local_C = local_A × B
 *        └─ 4. Gather     local_C 行数据 → Master 拼出完整 C(M×N)
 *
 * 适用场景: PageRank、Graph Embedding、GNN 中大矩阵 × 稠密特征矩阵
 */
class DistributedCSRDense {
public:
    static DenseMatrix multiply(const CSRMatrix& A, const DenseMatrix& B)
    {
        auto& ctx = DistributedContext::instance();
        int rank   = ctx.rank();
        int world  = ctx.size();
        constexpr int root = 0;

        // ---- 0. 广播维度 ----
        int dims[3] = {A.rows(), A.cols(), B.cols()};
        MPIWrapper::broadcastInt(dims, 3, root);
        int M = dims[0];
        int K = dims[1];
        int N = dims[2];

        // ---- 1. 广播 row_ptr, 散射 CSR A ----
        int nnz_global = (rank == root) ? A.nnz() : 0;
        MPIWrapper::broadcastInt(&nnz_global, 1, root);

        std::vector<int> global_row_ptr(M + 1);
        if (rank == root) global_row_ptr = A.rowPtr();
        MPIWrapper::broadcastInt(global_row_ptr.data(), M + 1, root);

        PartitionInfo info = RowPartitioner::compute(M, rank, world);
        int local_rows = info.local_rows();
        int nnz_start  = global_row_ptr[info.start_row];
        int nnz_end    = global_row_ptr[info.end_row];
        int local_nnz  = nnz_end - nnz_start;

        std::vector<int> send_val_counts(world), displs_val(world);
        std::vector<int> send_cidx_counts(world), displs_cidx(world);
        for (int r = 0; r < world; ++r) {
            PartitionInfo pi = RowPartitioner::compute(M, r, world);
            int ns = global_row_ptr[pi.start_row];
            int ne = global_row_ptr[pi.end_row];
            send_val_counts[r]  = ne - ns;  displs_val[r]  = ns;
            send_cidx_counts[r] = ne - ns;  displs_cidx[r] = ns;
        }

        std::vector<double> local_vals(local_nnz);
        MPIWrapper::scatterDense((rank == root) ? A.values().data() : nullptr,
                                 local_vals.data(), send_val_counts, displs_val, root);
        std::vector<int> local_cidx(local_nnz);
        MPIWrapper::scatterInt((rank == root) ? A.colIdx().data() : nullptr,
                               local_cidx.data(), send_cidx_counts, displs_cidx, root);

        std::vector<int> local_row_ptr(local_rows + 1);
        for (int i = 0; i <= local_rows; ++i)
            local_row_ptr[i] = global_row_ptr[info.start_row + i] - nnz_start;

        CSRMatrix local_A(local_rows, K,
                          std::move(local_vals), std::move(local_cidx),
                          std::move(local_row_ptr));

        // ---- 2. 广播 Dense B ----
        DenseMatrix local_B(K, N);
        if (rank == root) local_B = B;
        MPIWrapper::broadcastDense(local_B.raw_ptr(), K, N, root);

        // ---- 3. 局部 CSR×Dense ----
        DenseMatrix local_C;
        if (local_rows > 0)
            local_C = SparseMultiply::multiply(local_A, local_B);
        else
            local_C = DenseMatrix(0, N);

        // ---- 4. 收集 Dense C 的行 ----
        std::vector<int> recv_counts(world), rdispls(world);
        for (int r = 0; r < world; ++r) {
            PartitionInfo pi = RowPartitioner::compute(M, r, world);
            recv_counts[r] = pi.local_rows() * N;
            rdispls[r]     = pi.start_row * N;
        }

        DenseMatrix C(M, N);
        MPIWrapper::gatherDense(local_C.raw_ptr(), C.raw_ptr(),
                                recv_counts, rdispls, root);
        return C;
    }
};

} // namespace dmc
