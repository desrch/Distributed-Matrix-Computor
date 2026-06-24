#pragma once

#include "SpMVEngine.hpp"
#include "../partition/RowPartitioner.hpp"
#include "../runtime/DistributedContext.hpp"
#include "../communication/MPIWrapper.hpp"

#include <cassert>
#include <vector>

namespace dmc {

class MPICSREngine : public ISpMVEngine {
public:
    std::string name() const override {
        auto& ctx = DistributedContext::instance();
        return "MPI_CSR_p" + std::to_string(ctx.size());
    }

    std::vector<double> multiply(const CSRMatrix& A,
                                 const std::vector<double>& x) override
    {
        auto& ctx = DistributedContext::instance();
        int rank  = ctx.rank();
        int world = ctx.size();
        constexpr int root = 0;

        double comm_acc = 0.0;
        double comp_acc = 0.0;
        double t0, t1;

        // ===========================================================
        // Communication phase 1: broadcast metadata
        // ===========================================================
        int dims[2] = {A.rows(), A.cols()};
        ctx.barrier(); t0 = MPI_Wtime();
        MPIWrapper::broadcastInt(dims, 2, root);
        t1 = MPI_Wtime(); comm_acc += (t1 - t0);
        int M = dims[0];
        int N = dims[1];

        int nnz_global = (rank == root) ? A.nnz() : 0;
        t0 = MPI_Wtime();
        MPIWrapper::broadcastInt(&nnz_global, 1, root);
        t1 = MPI_Wtime(); comm_acc += (t1 - t0);

        std::vector<int> global_rptr(M + 1);
        if (rank == root) global_rptr = A.rowPtr();
        t0 = MPI_Wtime();
        MPIWrapper::broadcastInt(global_rptr.data(), M + 1, root);
        t1 = MPI_Wtime(); comm_acc += (t1 - t0);

        // ---- compute scatter params (cheap, counts as comp) ----
        PartitionInfo info = RowPartitioner::compute(M, rank, world);
        int local_rows = info.local_rows();
        int nnz_start  = global_rptr[info.start_row];
        int nnz_end    = global_rptr[info.end_row];
        int local_nnz  = nnz_end - nnz_start;

        std::vector<int> send_val_cnt(world), displs_val(world);
        std::vector<int> send_cidx_cnt(world), displs_cidx(world);
        for (int r = 0; r < world; ++r) {
            PartitionInfo pi = RowPartitioner::compute(M, r, world);
            int ns = global_rptr[pi.start_row];
            int ne = global_rptr[pi.end_row];
            send_val_cnt [r] = ne - ns;  displs_val [r] = ns;
            send_cidx_cnt[r] = ne - ns;  displs_cidx[r] = ns;
        }

        // ===========================================================
        // Communication phase 2: scatter A data
        // ===========================================================
        std::vector<double> local_vals(local_nnz);
        std::vector<int>    local_cidx(local_nnz);

        ctx.barrier(); t0 = MPI_Wtime();
        MPIWrapper::scatterDense((rank == root) ? A.values().data() : nullptr,
                                 local_vals.data(), send_val_cnt, displs_val, root);
        MPIWrapper::scatterInt((rank == root) ? A.colIdx().data() : nullptr,
                               local_cidx.data(), send_cidx_cnt, displs_cidx, root);
        t1 = MPI_Wtime(); comm_acc += (t1 - t0);

        // ---- build local row_ptr (cheap, counts as comp) ----
        t0 = MPI_Wtime();
        std::vector<int> local_rptr(static_cast<std::size_t>(local_rows) + 1);
        for (int i = 0; i <= local_rows; ++i)
            local_rptr[i] = global_rptr[info.start_row + i] - nnz_start;
        t1 = MPI_Wtime(); comp_acc += (t1 - t0);

        // ===========================================================
        // Communication phase 3: broadcast x
        // ===========================================================
        std::vector<double> local_x(N);
        if (rank == root) local_x = x;
        ctx.barrier(); t0 = MPI_Wtime();
        MPIWrapper::broadcastDense(local_x.data(), N, 1, root);
        t1 = MPI_Wtime(); comm_acc += (t1 - t0);

        // ===========================================================
        // Computation: local SpMV
        // ===========================================================
        std::vector<double> local_y(local_rows, 0.0);
        t0 = MPI_Wtime();
        {
            const double* vals = local_vals.data();
            const int*    cidx = local_cidx.data();
            const int*    rptr = local_rptr.data();
            for (int i = 0; i < local_rows; ++i) {
                double sum = 0.0;
                for (int p = rptr[i]; p < rptr[i + 1]; ++p)
                    sum += vals[p] * local_x[cidx[p]];
                local_y[i] = sum;
            }
        }
        t1 = MPI_Wtime(); comp_acc += (t1 - t0);

        // ===========================================================
        // Communication phase 4: gather y
        // ===========================================================
        std::vector<int> recv_cnt(world), rdispls(world);
        for (int r = 0; r < world; ++r) {
            PartitionInfo pi = RowPartitioner::compute(M, r, world);
            recv_cnt[r] = pi.local_rows();
            rdispls [r] = pi.start_row;
        }

        std::vector<double> y_global(M);
        ctx.barrier(); t0 = MPI_Wtime();
        MPIWrapper::gatherDense(local_y.data(), y_global.data(),
                                recv_cnt, rdispls, root);
        t1 = MPI_Wtime(); comm_acc += (t1 - t0);

        timing_.comm_sec += comm_acc;
        timing_.comp_sec += comp_acc;

        return y_global;
    }
};

} // namespace dmc
