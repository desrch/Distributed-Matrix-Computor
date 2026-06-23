#pragma once

#include "Matrix.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace dmc {

/**
 * @brief 稠密矩阵 —— 行优先 (row-major) 连续内存存储
 *
 * 数据布局: data[i * cols + j]
 *
 * 此类提供:
 *  - 构造 / 拷贝 / 移动
 *  - 元素访问 (operator() / at )
 *  - 基础线性代数工具 (setZero, fill, block, norm, transpose)
 *  - OpenMP 可并行的数据指针访问
 */
class DenseMatrix : public Matrix {
public:
    // =======================================================================
    // 构造
    // =======================================================================

    /** 默认构造：0×0 空矩阵 */
    DenseMatrix() = default;

    /** 构造 rows×cols 矩阵，所有元素初始化为 0 */
    DenseMatrix(int rows, int cols)
        : rows_(rows)
        , cols_(cols)
        , data_(static_cast<std::size_t>(rows) * cols, 0.0)
    {
        assert(rows >= 0 && cols >= 0);
    }

    /** 构造 rows×cols 矩阵，所有元素初始化为 init_val */
    DenseMatrix(int rows, int cols, double init_val)
        : rows_(rows)
        , cols_(cols)
        , data_(static_cast<std::size_t>(rows) * cols, init_val)
    {
        assert(rows >= 0 && cols >= 0);
    }

    /** 从 vector 构造（拷贝） */
    DenseMatrix(int rows, int cols, const std::vector<double>& data)
        : rows_(rows)
        , cols_(cols)
        , data_(data)
    {
        assert(rows >= 0 && cols >= 0);
        assert(data_.size() == static_cast<std::size_t>(rows) * cols);
    }

    /** 从 vector 构造（移动） */
    DenseMatrix(int rows, int cols, std::vector<double>&& data)
        : rows_(rows)
        , cols_(cols)
        , data_(std::move(data))
    {
        assert(rows >= 0 && cols >= 0);
        assert(data_.size() == static_cast<std::size_t>(rows) * cols);
    }

    /** 从 initializer_list 构造
     *  @code DenseMatrix m(2, 3, {1,2,3, 4,5,6}); @endcode
     */
    DenseMatrix(int rows, int cols, std::initializer_list<double> il)
        : rows_(rows)
        , cols_(cols)
        , data_(il)
    {
        assert(rows >= 0 && cols >= 0);
        assert(static_cast<int>(il.size()) == rows * cols);
    }

    // =======================================================================
    // 拷贝 / 移动 (default 即可，vector 已管理内存)
    // =======================================================================

    DenseMatrix(const DenseMatrix&) = default;
    DenseMatrix& operator=(const DenseMatrix&) = default;
    DenseMatrix(DenseMatrix&&) noexcept = default;
    DenseMatrix& operator=(DenseMatrix&&) noexcept = default;

    // =======================================================================
    // Matrix 接口实现
    // =======================================================================

    int rows() const override { return rows_; }
    int cols() const override { return cols_; }
    bool isSparse() const override { return false; }

    void printInfo(std::ostream& os) const override
    {
        os << "DenseMatrix(" << rows_ << "x" << cols_
           << ", " << (rows_ * cols_ * sizeof(double)) << " bytes)";
    }

    // =======================================================================
    // 元素访问
    // =======================================================================

    /** 行优先索引 (i, j)，无边界检查 */
    double& operator()(int i, int j)
    {
        return data_[static_cast<std::size_t>(i) * cols_ + j];
    }

    /** const 版本 */
    const double& operator()(int i, int j) const
    {
        return data_[static_cast<std::size_t>(i) * cols_ + j];
    }

    /** 带边界检查的访问 */
    double& at(int i, int j)
    {
        if (i < 0 || i >= rows_ || j < 0 || j >= cols_) {
            throw std::out_of_range("DenseMatrix::at: index out of range");
        }
        return data_[static_cast<std::size_t>(i) * cols_ + j];
    }

    /** const 带边界检查 */
    const double& at(int i, int j) const
    {
        if (i < 0 || i >= rows_ || j < 0 || j >= cols_) {
            throw std::out_of_range("DenseMatrix::at: index out of range");
        }
        return data_[static_cast<std::size_t>(i) * cols_ + j];
    }

    // =======================================================================
    // 原始数据访问 (OpenMP 并行区域中常用)
    // =======================================================================

    const std::vector<double>& data() const { return data_; }
    std::vector<double>& data() { return data_; }

    const double* raw_ptr() const { return data_.data(); }
    double* raw_ptr() { return data_.data(); }

    // =======================================================================
    // 原地操作
    // =======================================================================

    void setZero()
    {
        std::fill(data_.begin(), data_.end(), 0.0);
    }

    void fill(double val)
    {
        std::fill(data_.begin(), data_.end(), val);
    }

    void setIdentity()
    {
        setZero();
        int n = std::min(rows_, cols_);
        for (int i = 0; i < n; ++i) {
            (*this)(i, i) = 1.0;
        }
    }

    // =======================================================================
    // 基本运算 (返回新矩阵)
    // =======================================================================

    /** 提取子矩阵 [start_row, start_row+block_rows) × [start_col, start_col+block_cols) */
    DenseMatrix block(int start_row, int start_col,
                      int block_rows, int block_cols) const
    {
        assert(start_row >= 0 && start_row + block_rows <= rows_);
        assert(start_col >= 0 && start_col + block_cols <= cols_);

        DenseMatrix result(block_rows, block_cols);
        for (int i = 0; i < block_rows; ++i) {
            for (int j = 0; j < block_cols; ++j) {
                result(i, j) = (*this)(start_row + i, start_col + j);
            }
        }
        return result;
    }

    /** 矩阵转置 */
    DenseMatrix transpose() const
    {
        DenseMatrix result(cols_, rows_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    // =======================================================================
    // 范数
    // =======================================================================

    /** Frobenius 范数 */
    double normF() const
    {
        double sum = 0.0;
        for (const auto v : data_) {
            sum += v * v;
        }
        return std::sqrt(sum);
    }

    /** 最大范数 (所有元素绝对值最大值) */
    double normMax() const
    {
        double max_val = 0.0;
        for (const auto v : data_) {
            max_val = std::max(max_val, std::abs(v));
        }
        return max_val;
    }

    // =======================================================================
    // 比较
    // =======================================================================

    bool operator==(const DenseMatrix& other) const
    {
        return rows_ == other.rows_ && cols_ == other.cols_ && data_ == other.data_;
    }

    bool operator!=(const DenseMatrix& other) const
    {
        return !(*this == other);
    }

    // =======================================================================
    // 调试输出
    // =======================================================================

    void print(std::ostream& os = std::cout,
               int precision = 4,
               int max_display_rows = 20,
               int max_display_cols = 10) const
    {
        os << "DenseMatrix[" << rows_ << "x" << cols_ << "]\n";
        int rlim = std::min(rows_, max_display_rows);
        int clim = std::min(cols_, max_display_cols);

        os << std::fixed << std::setprecision(precision);
        for (int i = 0; i < rlim; ++i) {
            os << (i == 0 ? "⎡" : (i == rlim - 1 && rlim == rows_ ? "⎣" : "⎢"));
            for (int j = 0; j < clim; ++j) {
                os << std::setw(precision + 5) << (*this)(i, j);
            }
            if (clim < cols_) os << " …";
            os << (i == 0 ? " ⎤" : (i == rlim - 1 && rlim == rows_ ? " ⎦" : " ⎥"));
            os << "\n";
        }
        if (rlim < rows_) {
            os << "  … (" << (rows_ - rlim) << " more rows)\n";
        }
    }

private:
    int rows_{0};
    int cols_{0};
    std::vector<double> data_;
};

} // namespace dmc
