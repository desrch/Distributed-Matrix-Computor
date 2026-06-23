#pragma once

#include "../matrix/DenseMatrix.hpp"

#include <cassert>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace dmc {

/**
 * @brief 单机稠密矩阵乘法 —— i-k-j loop order + OpenMP
 *
 * C = A × B
 *
 * A: M × K
 * B: K × N
 * C: M × N
 *
 * 优化:
 *  - i-k-j 循环顺序提升缓存命中率 (B(k,j) 在 k 循环内连续)
 *  - #pragma omp parallel for on i-loop
 */
class DenseMultiply {
public:
    // ===================================================================
    // multiply (returns new matrix)
    // ===================================================================

    static DenseMatrix multiply(const DenseMatrix& A, const DenseMatrix& B)
    {
        int M = A.rows();
        int K = A.cols();
        int N = B.cols();

        assert(K == B.rows() && "DenseMultiply: dimension mismatch");

        DenseMatrix C(M, N, 0.0);

        // i-k-j loop order: innermost loop over j strides contiguous on C(i,:)
        // and B(k,:).  Eliminates the separate sum variable.
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                double aik = A(i, k);
                // 如果 aik==0 可跳过——对稠密矩阵通常不跳过
                for (int j = 0; j < N; ++j) {
                    C(i, j) += aik * B(k, j);
                }
            }
        }

        return C;
    }

    // ===================================================================
    // multiplyAdd (writes into pre-allocated C starting at dest_row)
    // ===================================================================

    static void multiplyAdd(const DenseMatrix& A,
                            const DenseMatrix& B,
                            DenseMatrix& C,
                            int dest_row)
    {
        int M = A.rows();
        int K = A.cols();
        int N = B.cols();

        assert(K == B.rows());
        assert(dest_row + M <= C.rows() && N == C.cols());

#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                double aik = A(i, k);
                for (int j = 0; j < N; ++j) {
                    C(dest_row + i, j) += aik * B(k, j);
                }
            }
        }
    }
};

} // namespace dmc
