#pragma once

namespace dmc {

/**
 * @brief 分块信息 —— 描述一个 rank 负责的行范围
 *
 * 半开区间 [start_row, end_row)
 */
struct PartitionInfo {
    int start_row{0};
    int end_row{0};

    /** 本块行数 */
    int local_rows() const { return end_row - start_row; }
};

/**
 * @brief 分块器抽象基类
 *
 * 子类实现不同的切分策略（按行 / 二维分块等）。
 */
class Partitioner {
public:
    virtual ~Partitioner() = default;

    /**
     * @param rows       全局总行数
     * @param cols       全局总列数（RowPartitioner 不使用此参数）
     * @param world_size 进程总数
     * @return 当前 rank 负责的分块信息
     */
    virtual PartitionInfo partition(int rows, int cols, int world_size) const = 0;
};

} // namespace dmc
