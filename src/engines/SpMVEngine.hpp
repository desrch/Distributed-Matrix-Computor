#pragma once

#include "../matrix/CSRMatrix.hpp"
#include "../matrix/DenseMatrix.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace dmc {

/**
 * @brief SpMV 单次调用计时数据
 */
struct SpMVTiming {
    double comm_sec = 0.0;  // MPI 通信耗时
    double comp_sec = 0.0;  // 纯计算耗时
    double total() const { return comm_sec + comp_sec; }
};

/**
 * @brief SpMV 抽象接口
 *
 * y = A * x
 *
 * 所有实现（串行 / OpenMP / MPI / Hybrid）均遵循此接口。
 */
class ISpMVEngine {
public:
    virtual ~ISpMVEngine() = default;

    /** 引擎名称 */
    virtual std::string name() const = 0;

    /**
     * @brief 执行 SpMV: y = A * x
     *
     * 所有 MPI rank 必须同时调用。
     * A 的 CSR 数据仅在 rank==0 上有效。
     * 返回的 y 在 rank==0 上包含完整结果，其它 rank 可为空。
     *
     * 同时累计内部 comm/comp 计时，通过 getTiming() 获取。
     */
    virtual std::vector<double> multiply(const CSRMatrix& A,
                                         const std::vector<double>& x) = 0;

    /** 重置累计计时 */
    virtual void resetTiming() { timing_ = SpMVTiming{}; }

    /** 获取累计计时 */
    virtual SpMVTiming getTiming() const { return timing_; }

protected:
    SpMVTiming timing_;
};

} // namespace dmc
