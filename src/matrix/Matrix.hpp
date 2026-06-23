#pragma once

#include <cstddef>
#include <ostream>

namespace dmc {

/**
 * @brief 统一矩阵抽象基类
 *
 * 所有矩阵类型（稠密、CSR、CSC）都由此接口派生，
 * 使得上层算子可以以多态方式操作不同存储格式的矩阵。
 */
class Matrix {
public:
    virtual ~Matrix() = default;

    // -----------------------------------------------------------------------
    // 维度查询
    // -----------------------------------------------------------------------

    /** 矩阵行数 */
    virtual int rows() const = 0;

    /** 矩阵列数 */
    virtual int cols() const = 0;

    /** 是否为稀疏存储格式 */
    virtual bool isSparse() const = 0;

    // -----------------------------------------------------------------------
    // 派生接口
    // -----------------------------------------------------------------------

    /** 矩阵元素总数 (rows × cols) */
    int size() const { return rows() * cols(); }

    /** 判断矩阵是否为空 (rows == 0 || cols == 0) */
    bool empty() const { return rows() == 0 || cols() == 0; }

    /** 判断是否为方阵 */
    bool isSquare() const { return rows() == cols(); }

    // -----------------------------------------------------------------------
    // 调试输出 (纯虚接口，子类必须实现)
    // -----------------------------------------------------------------------

    /** 向输出流打印矩阵摘要信息 */
    virtual void printInfo(std::ostream& os) const = 0;

protected:
    Matrix() = default;

    // 禁止拷贝 / 移动 —— 矩阵通过指针/引用传递
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;
};

} // namespace dmc
