#pragma once

#include "../matrix/DenseMatrix.hpp"
#include "../matrix/CSRMatrix.hpp"

#include <cassert>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace dmc {

/**
 * @brief 矩阵-向量乘法算子
 *
 * 支持:
 *  - DenseMatrix × Vector   (y = A * x)     — OpenMP 并行化 i 循环
 *  - CSRMatrix  × Vector   (y = A * x, SpMV) — OpenMP 按行并行
 *
 * SpMV 复杂度 O(nnz)，远优于先 toDense() 再乘的 O(n²)。
 */
class MatrixVector {
public:
    // ===================================================================
    // Dense × Vector
    // ===================================================================

    /** y = A * x (稠密) */
    static std::vector<double> multiply(const DenseMatrix& A,
                                        const std::vector<double>& x)
    {
        int M = A.rows();
        int N = A.cols();
        assert(static_cast<int>(x.size()) == N);

        std::vector<double> y(M, 0.0);

#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < M; ++i) {
            double sum = 0.0;
            for (int j = 0; j < N; ++j) {
                sum += A(i, j) * x[j];
            }
            y[i] = sum;
        }
        return y;
    }

    // ===================================================================
    // CSR × Vector (SpMV) —— 核心稀疏运算
    // ===================================================================

    /** y = A * x (CSR SpMV)
     *
     * 每行独立计算 y[i] = Σ A(i,k) * x[k]，
     * OpenMP 按行并行，线程间无数据竞争（各线程写不同的 y[i]）。
     */
    static std::vector<double> multiply(const CSRMatrix& A,
                                        const std::vector<double>& x)
    {
        int M = A.rows();
        int N = A.cols();
        assert(static_cast<int>(x.size()) == N);
        (void)N;

        const auto& values  = A.values();
        const auto& col_idx = A.colIdx();
        const auto& row_ptr = A.rowPtr();

        std::vector<double> y(M, 0.0);

#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < M; ++i) {
            double sum = 0.0;
            int start = row_ptr[i];
            int end   = row_ptr[i + 1];
            for (int p = start; p < end; ++p) {
                sum += values[p] * x[col_idx[p]];
            }
            y[i] = sum;
        }
        return y;
    }

    // ===================================================================
    // multiplyAdd — 用于分布式场景
    // ===================================================================

    /** 局部 SpMV 结果写入 y[dest_start .. dest_start+M)
     *  各 rank 输出区域不重叠，线程安全。
     */
    static void multiplyAdd(const CSRMatrix& A,
                            const std::vector<double>& x,
                            std::vector<double>& y,
                            int dest_start)
    {
        int M = A.rows();
        assert(static_cast<int>(x.size()) == A.cols());
        assert(dest_start + M <= static_cast<int>(y.size()));

        const auto& values  = A.values();
        const auto& col_idx = A.colIdx();
        const auto& row_ptr = A.rowPtr();

#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < M; ++i) {
            double sum = 0.0;
            int start = row_ptr[i];
            int end   = row_ptr[i + 1];
            for (int p = start; p < end; ++p) {
                sum += values[p] * x[col_idx[p]];
            }
            y[dest_start + i] = sum;
        }
    }
};

} // namespace dmc
