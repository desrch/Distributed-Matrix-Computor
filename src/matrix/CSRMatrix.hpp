#pragma once

#include "Matrix.hpp"
#include "DenseMatrix.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <vector>

namespace dmc {

class CSCMatrix; // 前向声明，toCSC() 实现在 CSCMatrix.hpp 末尾

/**
 * @brief CSR (Compressed Sparse Row) 稀疏矩阵
 *
 * 存储格式:
 *   - values[i]   : 第 i 个非零元的值
 *   - col_idx[i]  : 第 i 个非零元所在的列号
 *   - row_ptr[r]   : 第 r 行第一个非零元在 values/col_idx 中的起始位置
 *                    row_ptr[rows] = nnz (哨兵)
 *
 * 非零元按行连续存储，每行内部列号不要求有序。
 *
 * 空间复杂度: O(2*nnz + rows)
 */
class CSRMatrix : public Matrix {
public:
    // ===================================================================
    // 构造
    // ===================================================================

    /** 默认构造: 0×0 空矩阵 */
    CSRMatrix() = default;

    /** 构造 rows×cols 空稀疏矩阵（已分配 row_ptr，无元素） */
    CSRMatrix(int rows, int cols)
        : rows_(rows)
        , cols_(cols)
        , row_ptr_(static_cast<std::size_t>(rows) + 1, 0)
    {
        assert(rows >= 0 && cols >= 0);
    }

    /**
     * @brief 从 CSR 三数组构造
     * @param rows     行数
     * @param cols     列数
     * @param vals     非零元值
     * @param col_idx  列索引
     * @param row_ptr  行指针 (长度 rows+1, 最后元素 == vals.size())
     */
    CSRMatrix(int rows, int cols,
              std::vector<double> vals,
              std::vector<int> col_idx,
              std::vector<int> row_ptr)
        : rows_(rows)
        , cols_(cols)
        , values_(std::move(vals))
        , col_idx_(std::move(col_idx))
        , row_ptr_(std::move(row_ptr))
    {
        assert(rows >= 0 && cols >= 0);
        assert(row_ptr_.size() == static_cast<std::size_t>(rows) + 1);
        assert(values_.size() == col_idx_.size());
        assert(static_cast<int>(values_.size()) == row_ptr_.back());
    }

    // 拷贝 / 移动
    CSRMatrix(const CSRMatrix&) = default;
    CSRMatrix& operator=(const CSRMatrix&) = default;
    CSRMatrix(CSRMatrix&&) noexcept = default;
    CSRMatrix& operator=(CSRMatrix&&) noexcept = default;

    // ===================================================================
    // Matrix 接口实现
    // ===================================================================

    int rows() const override { return rows_; }
    int cols() const override { return cols_; }
    bool isSparse() const override { return true; }

    void printInfo(std::ostream& os) const override
    {
        os << "CSRMatrix(" << rows_ << "x" << cols_
           << ", nnz=" << nnz()
           << ", density=" << (cols_ > 0 ? 100.0 * nnz() / (rows_ * cols_) : 0.0)
           << "%)";
    }

    // ===================================================================
    // CSR 三数组只读访问
    // ===================================================================

    const std::vector<double>& values()  const { return values_; }
    const std::vector<int>&    colIdx()  const { return col_idx_; }
    const std::vector<int>&    rowPtr()  const { return row_ptr_; }

    /** 非零元总数 */
    int nnz() const { return static_cast<int>(values_.size()); }

    /** 第 r 行的非零元个数 */
    int rowNnz(int r) const
    {
        assert(r >= 0 && r < rows_);
        return row_ptr_[r + 1] - row_ptr_[r];
    }

    // ===================================================================
    // 元素访问
    // ===================================================================

    /**
     * @brief 获取元素 A(i, j)
     *
     * 对第 i 行的非零元做线性扫描，复杂度 O(nnz_per_row)。
     * 对于大规模重复随机访问应考虑先用 toDense() 转为稠密。
     */
    double get(int i, int j) const
    {
        assert(i >= 0 && i < rows_);
        assert(j >= 0 && j < cols_);

        int start = row_ptr_[i];
        int end   = row_ptr_[i + 1];
        for (int p = start; p < end; ++p) {
            if (col_idx_[p] == j) return values_[p];
        }
        return 0.0;
    }

    /** 下标运算符便捷形式 */
    double operator()(int i, int j) const { return get(i, j); }

    // ===================================================================
    // 转换
    // ===================================================================

    /** 从稠密矩阵构造 CSR（只保存 |val| > eps 的元素） */
    static CSRMatrix fromDense(const DenseMatrix& dense, double eps = 1e-15)
    {
        int M = dense.rows();
        int N = dense.cols();

        std::vector<double> vals;
        std::vector<int>    cidx;
        std::vector<int>    rptr(static_cast<std::size_t>(M) + 1, 0);

        for (int i = 0; i < M; ++i) {
            rptr[i] = static_cast<int>(vals.size());
            for (int j = 0; j < N; ++j) {
                double v = dense(i, j);
                if (std::abs(v) > eps) {
                    vals.push_back(v);
                    cidx.push_back(j);
                }
            }
        }
        rptr[M] = static_cast<int>(vals.size());

        return CSRMatrix(M, N, std::move(vals), std::move(cidx), std::move(rptr));
    }

    /** 转为稠密矩阵 */
    DenseMatrix toDense() const
    {
        DenseMatrix dense(rows_, cols_, 0.0);
        for (int i = 0; i < rows_; ++i) {
            for (int p = row_ptr_[i]; p < row_ptr_[i + 1]; ++p) {
                dense(i, col_idx_[p]) = values_[p];
            }
        }
        return dense;
    }

    /** 转为 CSC (列优先) 稀疏矩阵 —— 实现在 CSCMatrix.hpp 末尾 */
    CSCMatrix toCSC() const;

    // ===================================================================
    // 子矩阵提取 (用于分布式分片)
    // ===================================================================

    /**
     * @brief 提取行范围 [start_row, end_row) 的局部 CSR
     *
     * 返回的局部 CSR 中 row_ptr 已重新归零，
     * 即 local.row_ptr[0] == 0, local.row_ptr.back() == local.nnz()。
     */
    CSRMatrix rowSlice(int start_row, int end_row) const
    {
        assert(start_row >= 0 && end_row <= rows_ && start_row <= end_row);

        int local_rows = end_row - start_row;
        int nnz_start  = row_ptr_[start_row];
        int nnz_end    = row_ptr_[end_row];
        std::vector<double> local_vals(values_.begin() + nnz_start,
                                        values_.begin() + nnz_end);
        std::vector<int> local_cidx(col_idx_.begin() + nnz_start,
                                     col_idx_.begin() + nnz_end);
        std::vector<int> local_rptr(static_cast<std::size_t>(local_rows) + 1);

        for (int i = 0; i <= local_rows; ++i) {
            local_rptr[i] = row_ptr_[start_row + i] - nnz_start;
        }

        return CSRMatrix(local_rows, cols_,
                         std::move(local_vals),
                         std::move(local_cidx),
                         std::move(local_rptr));
    }

    // ===================================================================
    // 调试
    // ===================================================================

    void print(std::ostream& os = std::cout,
               int max_display_rows = 20,
               int max_display_cols = 10) const
    {
        os << "CSRMatrix[" << rows_ << "x" << cols_
           << "], nnz=" << nnz() << "\n";
        int rlim = std::min(rows_, max_display_rows);
        int clim = std::min(cols_, max_display_cols);

        os << std::fixed << std::setprecision(4);
        for (int i = 0; i < rlim; ++i) {
            os << "row " << i << ": ";
            for (int p = row_ptr_[i]; p < row_ptr_[i + 1]; ++p) {
                if (col_idx_[p] < clim) {
                    os << "(" << col_idx_[p] << "," << values_[p] << ") ";
                }
            }
            if (row_ptr_[i + 1] - row_ptr_[i] == 0) {
                os << "(empty)";
            }
            os << "\n";
        }
        if (rlim < rows_) {
            os << "  … (" << (rows_ - rlim) << " more rows)\n";
        }
    }

private:
    int rows_{0};
    int cols_{0};
    std::vector<double> values_;
    std::vector<int>    col_idx_;
    std::vector<int>    row_ptr_;   // 长度 rows+1
};

} // namespace dmc
