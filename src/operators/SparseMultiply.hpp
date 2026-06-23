#pragma once

#include "../matrix/CSRMatrix.hpp"
#include "../matrix/CSCMatrix.hpp"
#include "../matrix/DenseMatrix.hpp"
#include "../operators/DenseMultiply.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace dmc {

/**
 * @brief 稀疏矩阵乘法算子 (OpenMP 加速)
 *
 * 统一管理:
 *  - CSR × Dense   → Dense   (OpenMP: 按行并行)
 *  - CSR × CSR     → CSR     (SpGEMM, Gustavson + OpenMP 两趟法)
 *  - CSR × CSC     → CSR     (朴素归并, OpenMP: 按行并行)
 */
class SparseMultiply {
public:
    // ===================================================================
    // CSR × Dense → Dense
    // ===================================================================

    /**
     * @brief C = A_csr × B_dense
     *
     * 对 A 的每行 i，遍历其非零元 A(i,k)，将 aik * B(k,*) 累加到 C(i,*)。
     * OpenMP 按 A 的行并行，各线程写 C 的不同行，无数据竞争。
     */
    static DenseMatrix multiply(const CSRMatrix& A, const DenseMatrix& B)
    {
        int M = A.rows();
        int K = A.cols();
        int N = B.cols();
        assert(K == B.rows());

        DenseMatrix C(M, N, 0.0);

        const auto& vals = A.values();
        const auto& cidx = A.colIdx();
        const auto& rptr = A.rowPtr();

#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < M; ++i) {
            for (int p = rptr[i]; p < rptr[i + 1]; ++p) {
                int k = cidx[p];
                double aik = vals[p];
                for (int j = 0; j < N; ++j) {
                    C(i, j) += aik * B(k, j);
                }
            }
        }
        return C;
    }

    // ===================================================================
    // CSR × CSR (朴素, 用于对拍验证)
    // ===================================================================

    /** 朴素版: 通过 Dense 往返，结果用于对拍 */
    static CSRMatrix multiplyNaive(const CSRMatrix& A, const CSRMatrix& B)
    {
        assert(A.cols() == B.rows());
        DenseMatrix Ad = A.toDense();
        DenseMatrix Bd = B.toDense();
        DenseMatrix Cd = DenseMultiply::multiply(Ad, Bd);
        return CSRMatrix::fromDense(Cd);
    }

    // ===================================================================
    // CSR × CSR (SpGEMM, Gustavson + OpenMP 两趟法)
    // ===================================================================

    /**
     * @brief 高效并行 SpGEMM: C = A × B (均为 CSR)
     *
     * OpenMP 两趟策略:
     *
     *   Pass 1 (parallel): 计数每输出行的 nnz。
     *     每个线程维护私有 mark[] 数组，用 row_counter 标记"首次访问"。
     *
     *   Pass 2 (serial):   前缀和构建 c_rptr。
     *
     *   Pass 3 (parallel): 填充值。
     *     每个线程维护私有 SPA (稠密累加器) + dirty 列表。
     *     利用 Pass2 的 c_rptr 直接写入全局输出缓冲区，
     *     线程间无竞争（不同线程负责不同的输出行）。
     */
    static CSRMatrix multiply(const CSRMatrix& A, const CSRMatrix& B)
    {
        int M = A.rows();
        int K = A.cols();
        int N = B.cols();
        assert(K == B.rows());

        const auto& a_vals = A.values();
        const auto& a_cidx = A.colIdx();
        const auto& a_rptr = A.rowPtr();

        const auto& b_vals = B.values();
        const auto& b_cidx = B.colIdx();
        const auto& b_rptr = B.rowPtr();

        // ---- 输出缓冲区 ----
        std::vector<int> row_counts(M, 0);          // Pass1 输出

#if defined(_OPENMP)
        // ===========================================================
        // Pass 1 (parallel): 计数每输出行的 nnz
        // ===========================================================
        {
            int nthr = omp_get_max_threads();

            // 每个线程的私有 mark 数组: mark[j] == my_row_counter → column j 已被本行访问
            // 线程间完全隔离，不需要 atomic。
            std::vector<std::vector<int>> thread_marks(nthr);
            std::vector<int> thread_row_counters(nthr, 0);
            for (int t = 0; t < nthr; ++t) {
                thread_marks[t].assign(N, -1);
            }

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto& mark = thread_marks[tid];
                int& row_ctr = thread_row_counters[tid];

                #pragma omp for schedule(static)
                for (int i = 0; i < M; ++i) {
                    int local_nnz = 0;
                    for (int pa = a_rptr[i]; pa < a_rptr[i + 1]; ++pa) {
                        int k = a_cidx[pa];
                        for (int pb = b_rptr[k]; pb < b_rptr[k + 1]; ++pb) {
                            int j = b_cidx[pb];
                            if (mark[j] != row_ctr) {
                                mark[j] = row_ctr;
                                ++local_nnz;
                            }
                        }
                    }
                    row_counts[i] = local_nnz;
                    ++row_ctr;
                }
            }
        }
#else
        // ---- 串行回退 ----
        {
            std::vector<int> mark(N, -1);
            int row_ctr = 0;
            for (int i = 0; i < M; ++i) {
                int local_nnz = 0;
                for (int pa = a_rptr[i]; pa < a_rptr[i + 1]; ++pa) {
                    int k = a_cidx[pa];
                    for (int pb = b_rptr[k]; pb < b_rptr[k + 1]; ++pb) {
                        int j = b_cidx[pb];
                        if (mark[j] != row_ctr) {
                            mark[j] = row_ctr;
                            ++local_nnz;
                        }
                    }
                }
                row_counts[i] = local_nnz;
                ++row_ctr;
            }
        }
#endif

        // ===========================================================
        // Pass 2 (serial): 构建 row_ptr
        // ===========================================================
        std::vector<int> c_rptr(static_cast<std::size_t>(M) + 1, 0);
        for (int i = 0; i < M; ++i) {
            c_rptr[i + 1] = c_rptr[i] + row_counts[i];
        }
        int total_nnz = c_rptr[M];

        std::vector<double> c_vals(total_nnz);
        std::vector<int>    c_cidx(total_nnz);

#if defined(_OPENMP)
        // ===========================================================
        // Pass 3 (parallel): 填充值
        // ===========================================================
        {
            int nthr = omp_get_max_threads();

            // 每个线程的私有 SPA + dirty
            std::vector<std::vector<double>> thread_acc(nthr);
            std::vector<std::vector<int>>    thread_dirty(nthr);
            std::vector<std::vector<int>>    thread_mark3(nthr);
            std::vector<int>                 thread_row_ctr3(nthr, 0);

            for (int t = 0; t < nthr; ++t) {
                thread_acc[t].assign(N, 0.0);
                thread_dirty[t].reserve(N);
                thread_mark3[t].assign(N, -1);
            }

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto& acc   = thread_acc[tid];
                auto& dirty = thread_dirty[tid];
                auto& mark  = thread_mark3[tid];
                int& row_ctr = thread_row_ctr3[tid];

                #pragma omp for schedule(static)
                for (int i = 0; i < M; ++i) {
                    int offset = c_rptr[i];       // 本输出行在全局数组中的起始位置

                    // ---- 累加 ----
                    for (int pa = a_rptr[i]; pa < a_rptr[i + 1]; ++pa) {
                        int k = a_cidx[pa];
                        double aik = a_vals[pa];
                        for (int pb = b_rptr[k]; pb < b_rptr[k + 1]; ++pb) {
                            int j = b_cidx[pb];
                            if (mark[j] != row_ctr) {
                                mark[j] = row_ctr;
                                dirty.push_back(j);
                            }
                            acc[j] += aik * b_vals[pb];
                        }
                    }

                    // ---- 排序 & 刷出到全局数组 ----
                    std::sort(dirty.begin(), dirty.end());
                    int cnt = 0;
                    for (int j : dirty) {
                        double val = acc[j];
                        if (std::abs(val) > 1e-15) {
                            c_vals[offset + cnt] = val;
                            c_cidx[offset + cnt] = j;
                            ++cnt;
                        }
                        acc[j] = 0.0;          // 重置 SPA
                    }

                    dirty.clear();
                    ++row_ctr;
                }
            }
        }
#else
        // ---- 串行 SpGEMM (原版) ----
        {
            std::vector<double> acc(N, 0.0);
            std::vector<int>    dirty;
            dirty.reserve(N);

            for (int i = 0; i < M; ++i) {
                int offset = c_rptr[i];
                for (int pa = a_rptr[i]; pa < a_rptr[i + 1]; ++pa) {
                    int k = a_cidx[pa];
                    double aik = a_vals[pa];
                    for (int pb = b_rptr[k]; pb < b_rptr[k + 1]; ++pb) {
                        int j = b_cidx[pb];
                        if (acc[j] == 0.0) dirty.push_back(j);
                        acc[j] += aik * b_vals[pb];
                    }
                }

                std::sort(dirty.begin(), dirty.end());
                int cnt = 0;
                for (int j : dirty) {
                    double val = acc[j];
                    if (std::abs(val) > 1e-15) {
                        c_vals[offset + cnt] = val;
                        c_cidx[offset + cnt] = j;
                        ++cnt;
                    }
                    acc[j] = 0.0;
                }
                dirty.clear();
            }
        }
#endif

        return CSRMatrix(M, N,
                         std::move(c_vals),
                         std::move(c_cidx),
                         std::move(c_rptr));
    }

    // ===================================================================
    // CSR × CSC (朴素归并, OpenMP: 按行并行 + 两趟法)
    // ===================================================================

    /**
     * @brief 朴素 CSR × CSC
     *
     * 对每对 (i, j)，用 CSR 行 + CSC 列的已排序索引做归并点积。
     * 两趟法: Pass1 并发计数每行 nnz；Pass2 串行构建 row_ptr；
     *         Pass3 并发填充值到正确偏移（各线程写不同行，无竞争）。
     */
    static CSRMatrix multiplyCSR_CSC_naive(const CSRMatrix& A, const CSCMatrix& B)
    {
        int M = A.rows();
        int N = B.cols();
        int K = A.cols();
        assert(K == B.rows());

        const auto& a_vals = A.values();
        const auto& a_cidx = A.colIdx();
        const auto& a_rptr = A.rowPtr();

        const auto& b_vals = B.values();
        const auto& b_ridx = B.rowIdx();
        const auto& b_cptr = B.colPtr();

        (void)K; // 通过 b_cptr 间接使用

        // ---- Pass 1 (parallel): 计数每行 nnz ----
        std::vector<int> row_counts(M, 0);

#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < M; ++i) {
            int cnt = 0;
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                int pa = a_rptr[i];
                int pa_end = a_rptr[i + 1];
                int pb = b_cptr[j];
                int pb_end = b_cptr[j + 1];

                while (pa < pa_end && pb < pb_end) {
                    int ca = a_cidx[pa];
                    int rb = b_ridx[pb];
                    if (ca == rb) {
                        sum += a_vals[pa] * b_vals[pb];
                        pa++; pb++;
                    } else if (ca < rb) {
                        pa++;
                    } else {
                        pb++;
                    }
                }
                if (std::abs(sum) > 1e-15) ++cnt;
            }
            row_counts[i] = cnt;
        }

        // ---- Pass 2 (serial): 构建 row_ptr ----
        std::vector<int> c_rptr(static_cast<std::size_t>(M) + 1, 0);
        for (int i = 0; i < M; ++i) {
            c_rptr[i + 1] = c_rptr[i] + row_counts[i];
        }
        int total_nnz = c_rptr[M];

        std::vector<double> c_vals(total_nnz);
        std::vector<int>    c_cidx(total_nnz);

        // ---- Pass 3 (parallel): 填充值 ----
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < M; ++i) {
            int offset = c_rptr[i];
            int cnt = 0;

            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                int pa = a_rptr[i];
                int pa_end = a_rptr[i + 1];
                int pb = b_cptr[j];
                int pb_end = b_cptr[j + 1];

                while (pa < pa_end && pb < pb_end) {
                    int ca = a_cidx[pa];
                    int rb = b_ridx[pb];
                    if (ca == rb) {
                        sum += a_vals[pa] * b_vals[pb];
                        pa++; pb++;
                    } else if (ca < rb) {
                        pa++;
                    } else {
                        pb++;
                    }
                }

                if (std::abs(sum) > 1e-15) {
                    c_vals[offset + cnt] = sum;
                    c_cidx[offset + cnt] = j;
                    ++cnt;
                }
            }
        }

        return CSRMatrix(M, N,
                         std::move(c_vals),
                         std::move(c_cidx),
                         std::move(c_rptr));
    }
};

} // namespace dmc
