#pragma once

#include "Matrix.hpp"
#include "DenseMatrix.hpp"
#include "CSRMatrix.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <unordered_map>
#include <vector>

namespace dmc {

// 前向声明: CSRMatrix::toCSC() 在本文件末尾定义

/**
 * @brief CSC (Compressed Sparse Column) 稀疏矩阵
 *
 * 存储格式 (列优先):
 *   - values[i]   : 第 i 个非零元的值
 *   - row_idx[i]  : 第 i 个非零元所在的行号
 *   - col_ptr[j]  : 第 j 列第一个非零元在 values/row_idx 中的起始位置
 *                    col_ptr[cols] = nnz (哨兵)
 *
 * CSC 是 CSR 的转置视角: CSC = CSR(A^T)
 * 用途: 当需要高效访问某一列的所有非零元时使用 CSC，
 *       例如 CSR×CSC SpGEMM 中 B 的列访问。
 */
class CSCMatrix : public Matrix {
public:
    // ===================================================================
    // 构造
    // ===================================================================

    CSCMatrix() = default;

    /** 构造 rows×cols 空稀疏矩阵 */
    CSCMatrix(int rows, int cols)
        : rows_(rows)
        , cols_(cols)
        , col_ptr_(static_cast<std::size_t>(cols) + 1, 0)
    {
        assert(rows >= 0 && cols >= 0);
    }

    /** 从完整 CSC 三数组构造 */
    CSCMatrix(int rows, int cols,
              std::vector<double> vals,
              std::vector<int> row_idx,
              std::vector<int> col_ptr)
        : rows_(rows)
        , cols_(cols)
        , values_(std::move(vals))
        , row_idx_(std::move(row_idx))
        , col_ptr_(std::move(col_ptr))
    {
        assert(rows >= 0 && cols >= 0);
        assert(col_ptr_.size() == static_cast<std::size_t>(cols) + 1);
        assert(values_.size() == row_idx_.size());
        assert(static_cast<int>(values_.size()) == col_ptr_.back());
    }

    // 拷贝 / 移动
    CSCMatrix(const CSCMatrix&) = default;
    CSCMatrix& operator=(const CSCMatrix&) = default;
    CSCMatrix(CSCMatrix&&) noexcept = default;
    CSCMatrix& operator=(CSCMatrix&&) noexcept = default;

    // ===================================================================
    // Matrix 接口
    // ===================================================================

    int rows() const override { return rows_; }
    int cols() const override { return cols_; }
    bool isSparse() const override { return true; }

    void printInfo(std::ostream& os) const override
    {
        os << "CSCMatrix(" << rows_ << "x" << cols_
           << ", nnz=" << nnz()
           << ", density=" << (cols_ > 0 ? 100.0 * nnz() / (rows_ * cols_) : 0.0)
           << "%)";
    }

    // ===================================================================
    // 只读访问
    // ===================================================================

    const std::vector<double>& values()  const { return values_; }
    const std::vector<int>&    rowIdx()  const { return row_idx_; }
    const std::vector<int>&    colPtr()  const { return col_ptr_; }

    int nnz() const { return static_cast<int>(values_.size()); }

    /** 第 c 列的非零元个数 */
    int colNnz(int c) const
    {
        assert(c >= 0 && c < cols_);
        return col_ptr_[c + 1] - col_ptr_[c];
    }

    // ===================================================================
    // 元素访问
    // ===================================================================

    /** 获取 A(i, j) —— 在第 j 列中线性扫描，O(nnz_per_col) */
    double get(int i, int j) const
    {
        assert(i >= 0 && i < rows_);
        assert(j >= 0 && j < cols_);

        int start = col_ptr_[j];
        int end   = col_ptr_[j + 1];
        for (int p = start; p < end; ++p) {
            if (row_idx_[p] == i) return values_[p];
        }
        return 0.0;
    }

    double operator()(int i, int j) const { return get(i, j); }

    // ===================================================================
    // 转换
    // ===================================================================

    /** 从稠密矩阵构造 CSC */
    static CSCMatrix fromDense(const DenseMatrix& dense, double eps = 1e-15)
    {
        int M = dense.rows();
        int N = dense.cols();

        std::vector<double> vals;
        std::vector<int>    ridx;
        std::vector<int>    cptr(static_cast<std::size_t>(N) + 1, 0);

        for (int j = 0; j < N; ++j) {
            cptr[j] = static_cast<int>(vals.size());
            for (int i = 0; i < M; ++i) {
                double v = dense(i, j);
                if (std::abs(v) > eps) {
                    vals.push_back(v);
                    ridx.push_back(i);
                }
            }
        }
        cptr[N] = static_cast<int>(vals.size());

        return CSCMatrix(M, N, std::move(vals), std::move(ridx), std::move(cptr));
    }

    /** 转为稠密矩阵 */
    DenseMatrix toDense() const
    {
        DenseMatrix dense(rows_, cols_, 0.0);
        for (int j = 0; j < cols_; ++j) {
            for (int p = col_ptr_[j]; p < col_ptr_[j + 1]; ++p) {
                dense(row_idx_[p], j) = values_[p];
            }
        }
        return dense;
    }

    /** 转为 CSR */
    CSRMatrix toCSR() const;

    // ===================================================================
    // 列切片 (用于分布式按列分片)
    // ===================================================================

    /** 提取列范围 [start_col, end_col) */
    CSCMatrix colSlice(int start_col, int end_col) const
    {
        assert(start_col >= 0 && end_col <= cols_ && start_col <= end_col);

        int local_cols = end_col - start_col;
        int nnz_start  = col_ptr_[start_col];
        int nnz_end    = col_ptr_[end_col];

        std::vector<double> local_vals(values_.begin() + nnz_start,
                                        values_.begin() + nnz_end);
        std::vector<int> local_ridx(row_idx_.begin() + nnz_start,
                                     row_idx_.begin() + nnz_end);
        std::vector<int> local_cptr(static_cast<std::size_t>(local_cols) + 1);

        for (int j = 0; j <= local_cols; ++j) {
            local_cptr[j] = col_ptr_[start_col + j] - nnz_start;
        }

        return CSCMatrix(rows_, local_cols,
                         std::move(local_vals),
                         std::move(local_ridx),
                         std::move(local_cptr));
    }

    // ===================================================================
    // 调试
    // ===================================================================

    void print(std::ostream& os = std::cout,
               int max_display_cols = 8,
               int max_display_rows = 10) const
    {
        os << "CSCMatrix[" << rows_ << "x" << cols_
           << "], nnz=" << nnz() << "\n";
        int clim = std::min(cols_, max_display_cols);
        int rlim = std::min(rows_, max_display_rows);

        os << std::fixed << std::setprecision(4);
        for (int j = 0; j < clim; ++j) {
            os << "col " << j << ": ";
            for (int p = col_ptr_[j]; p < col_ptr_[j + 1]; ++p) {
                if (row_idx_[p] < rlim) {
                    os << "(" << row_idx_[p] << "," << values_[p] << ") ";
                }
            }
            if (col_ptr_[j + 1] - col_ptr_[j] == 0) {
                os << "(empty)";
            }
            os << "\n";
        }
        if (clim < cols_) {
            os << "  … (" << (cols_ - clim) << " more cols)\n";
        }
    }

private:
    int rows_{0};
    int cols_{0};
    std::vector<double> values_;
    std::vector<int>    row_idx_;
    std::vector<int>    col_ptr_;   // 长度 cols+1
};

// =======================================================================
// CSRMatrix::toCSC() 实现 (在这里定义，避免 CSRMatrix.hpp 中的循环依赖)
// =======================================================================

inline CSCMatrix CSRMatrix::toCSC() const
{
    // 1. 统计每列非零元数
    std::vector<int> col_counts(cols_, 0);
    for (int i = 0; i < rows_; ++i) {
        for (int p = row_ptr_[i]; p < row_ptr_[i + 1]; ++p) {
            col_counts[col_idx_[p]]++;
        }
    }

    // 2. 构建 col_ptr
    std::vector<int> csc_col_ptr(static_cast<std::size_t>(cols_) + 1, 0);
    for (int j = 0; j < cols_; ++j) {
        csc_col_ptr[j + 1] = csc_col_ptr[j] + col_counts[j];
    }

    int total_nnz = csc_col_ptr[cols_];
    std::vector<double> csc_values(total_nnz);
    std::vector<int>    csc_row_idx(total_nnz);

    // 3. 第二遍: 填值 (使用 col_counts 作为插入游标)
    std::fill(col_counts.begin(), col_counts.end(), 0);
    for (int i = 0; i < rows_; ++i) {
        for (int p = row_ptr_[i]; p < row_ptr_[i + 1]; ++p) {
            int j = col_idx_[p];
            int pos = csc_col_ptr[j] + col_counts[j];
            csc_values[pos]  = values_[p];
            csc_row_idx[pos] = i;
            col_counts[j]++;
        }
    }

    return CSCMatrix(rows_, cols_,
                     std::move(csc_values),
                     std::move(csc_row_idx),
                     std::move(csc_col_ptr));
}

// =======================================================================
// CSCMatrix::toCSR()
// =======================================================================

inline CSRMatrix CSCMatrix::toCSR() const
{
    // 对称于 toCSC(): 将 CSC 三数组转换为 CSR 三数组

    // 1. 统计每行非零元数
    std::vector<int> row_counts(rows_, 0);
    for (int j = 0; j < cols_; ++j) {
        for (int p = col_ptr_[j]; p < col_ptr_[j + 1]; ++p) {
            row_counts[row_idx_[p]]++;
        }
    }

    // 2. 构建 row_ptr
    std::vector<int> csr_row_ptr(static_cast<std::size_t>(rows_) + 1, 0);
    for (int i = 0; i < rows_; ++i) {
        csr_row_ptr[i + 1] = csr_row_ptr[i] + row_counts[i];
    }

    int total_nnz = csr_row_ptr[rows_];
    std::vector<double> csr_values(total_nnz);
    std::vector<int>    csr_col_idx(total_nnz);

    // 3. 第二遍: 按行填入 (使用 row_counts 作为游标)
    std::fill(row_counts.begin(), row_counts.end(), 0);
    for (int j = 0; j < cols_; ++j) {
        for (int p = col_ptr_[j]; p < col_ptr_[j + 1]; ++p) {
            int i = row_idx_[p];
            int pos = csr_row_ptr[i] + row_counts[i];
            csr_values[pos]  = values_[p];
            csr_col_idx[pos] = j;
            row_counts[i]++;
        }
    }

    return CSRMatrix(rows_, cols_,
                     std::move(csr_values),
                     std::move(csr_col_idx),
                     std::move(csr_row_ptr));
}

} // namespace dmc
