#pragma once

#include "../matrix/CSRMatrix.hpp"
#include "../partition/RowPartitioner.hpp"
#include "../runtime/DistributedContext.hpp"
#include "../communication/MPIWrapper.hpp"
#include "../operators/SparseMultiply.hpp"

#include <algorithm>
#include <vector>

namespace dmc {

/**
 * @brief 分布式 CSR × CSR (SpGEMM)
 *
 * 流水线:
 *   Master 持有完整 CSR A(M×K) 和 CSR B(K×N)
 *        │
 *        ├─ 1. 广播维度 + row_ptr, 按行 Scatter CSR A
 *        ├─ 2. Broadcast  完整 CSR B (row_ptr + values + col_idx)
 *        ├─ 3. 各 rank 局部 SpGEMM: local_C = local_A × B
 *        └─ 4. Gather     CSR 三数组 → Master 拼出完整 C
 *
 * Gather 策略（处理 CSR 结果的非均匀 nnz）：
 *   allgather local_nnz → root 计算偏移
 *   gather    per-row nnz → root 重建 row_ptr
 *   gather    values, col_idx
 */
class DistributedCSRCSR {
public:
    static CSRMatrix multiply(const CSRMatrix& A, const CSRMatrix& B)
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

        // ===========================================================
        // 1. 散射 CSR A（行分片）
        // ===========================================================
        int nnzA_global = (rank == root) ? A.nnz() : 0;
        MPIWrapper::broadcastInt(&nnzA_global, 1, root);

        std::vector<int> global_row_ptr_A(M + 1);
        if (rank == root) global_row_ptr_A = A.rowPtr();
        MPIWrapper::broadcastInt(global_row_ptr_A.data(), M + 1, root);

        PartitionInfo info = RowPartitioner::compute(M, rank, world);
        int local_rows = info.local_rows();
        int nnz_start  = global_row_ptr_A[info.start_row];
        int nnz_end    = global_row_ptr_A[info.end_row];
        int local_nnz  = nnz_end - nnz_start;

        std::vector<int> send_val_counts(world), displs_val(world);
        std::vector<int> send_cidx_counts(world), displs_cidx(world);
        for (int r = 0; r < world; ++r) {
            PartitionInfo pi = RowPartitioner::compute(M, r, world);
            int ns = global_row_ptr_A[pi.start_row];
            int ne = global_row_ptr_A[pi.end_row];
            send_val_counts [r] = ne - ns;  displs_val [r] = ns;
            send_cidx_counts[r] = ne - ns;  displs_cidx[r] = ns;
        }

        std::vector<double> local_vals_a(local_nnz);
        MPIWrapper::scatterDense((rank == root) ? A.values().data() : nullptr,
                                 local_vals_a.data(),
                                 send_val_counts, displs_val, root);
        std::vector<int> local_cidx_a(local_nnz);
        MPIWrapper::scatterInt((rank == root) ? A.colIdx().data() : nullptr,
                               local_cidx_a.data(),
                               send_cidx_counts, displs_cidx, root);

        std::vector<int> local_row_ptr_a(local_rows + 1);
        for (int i = 0; i <= local_rows; ++i)
            local_row_ptr_a[i] = global_row_ptr_A[info.start_row + i] - nnz_start;

        CSRMatrix local_A(local_rows, K,
                          std::move(local_vals_a), std::move(local_cidx_a),
                          std::move(local_row_ptr_a));

        // ===========================================================
        // 2. 广播完整 CSR B
        // ===========================================================
        int nnzB_global = (rank == root) ? B.nnz() : 0;
        MPIWrapper::broadcastInt(&nnzB_global, 1, root);

        std::vector<int> global_row_ptr_B(K + 1);
        if (rank == root) global_row_ptr_B = B.rowPtr();
        MPIWrapper::broadcastInt(global_row_ptr_B.data(), K + 1, root);

        std::vector<double> b_vals(nnzB_global);
        if (rank == root) b_vals = B.values();
        MPIWrapper::broadcastDense(b_vals.data(), nnzB_global, 1, root);

        std::vector<int> b_cidx(nnzB_global);
        if (rank == root) b_cidx = B.colIdx();
        MPIWrapper::broadcastInt(b_cidx.data(), nnzB_global, root);

        CSRMatrix local_B(K, N,
                          std::move(b_vals), std::move(b_cidx),
                          std::move(global_row_ptr_B));

        // ===========================================================
        // 3. 局部 SpGEMM
        // ===========================================================
        CSRMatrix local_C;
        if (local_rows > 0) {
            local_C = SparseMultiply::multiply(local_A, local_B);
        } else {
            local_C = CSRMatrix(0, N);
        }
        int local_c_nnz = local_C.nnz();

        // ===========================================================
        // 4. 收集 CSR 结果
        // ===========================================================

        // 4a. allgather 各 rank 的 nnz 总量 → root 计算偏移
        std::vector<int> all_nnz(world);
        std::vector<int> one_per_rank(world, 1);
        std::vector<int> displs_one_per_rank(world);
        for (int r = 0; r < world; ++r) displs_one_per_rank[r] = r;
        MPIWrapper::allgatherInt(&local_c_nnz, all_nnz.data(), 1,
                                 one_per_rank, displs_one_per_rank);

        std::vector<int> recv_val_counts(world), recv_val_displs(world);
        int cum_offset = 0;
        for (int r = 0; r < world; ++r) {
            recv_val_counts [r] = all_nnz[r];
            recv_val_displs[r] = cum_offset;
            cum_offset += all_nnz[r];
        }
        int total_c_nnz = cum_offset;

        // 4b. 计算 per-row nnz，gather 到 root 重建 row_ptr
        std::vector<int> per_row_nnz(local_rows);
        {
            const auto& lrptr = local_C.rowPtr();
            for (int i = 0; i < local_rows; ++i)
                per_row_nnz[i] = lrptr[i + 1] - lrptr[i];
        }

        std::vector<int> recv_rptr_counts(world), recv_rptr_displs(world);
        int rptr_offset = 0;
        for (int r = 0; r < world; ++r) {
            PartitionInfo pi = RowPartitioner::compute(M, r, world);
            recv_rptr_counts[r] = pi.local_rows();
            recv_rptr_displs[r] = rptr_offset;
            rptr_offset += recv_rptr_counts[r];
        }

        std::vector<int> global_per_row_nnz(M);
        MPIWrapper::gatherInt(per_row_nnz.data(),
                               global_per_row_nnz.data(),
                               recv_rptr_counts, recv_rptr_displs, root);

        // 4c. gather values, col_idx
        std::vector<double> c_vals(total_c_nnz);
        MPIWrapper::gatherDense(local_C.values().data(), c_vals.data(),
                                recv_val_counts, recv_val_displs, root);

        std::vector<int> c_cidx(total_c_nnz);
        MPIWrapper::gatherInt(local_C.colIdx().data(), c_cidx.data(),
                               recv_val_counts, recv_val_displs, root);

        // 4d. root 重建全局 row_ptr
        if (rank == root) {
            std::vector<int> global_c_rptr(M + 1, 0);
            for (int i = 0; i < M; ++i)
                global_c_rptr[i + 1] = global_c_rptr[i] + global_per_row_nnz[i];

            return CSRMatrix(M, N,
                             std::move(c_vals), std::move(c_cidx),
                             std::move(global_c_rptr));
        }

        return CSRMatrix();
    }
};

} // namespace dmc
