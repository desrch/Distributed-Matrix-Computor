#pragma once

#include "Partitioner.hpp"

#include <algorithm>
#include <cassert>

namespace dmc {

/**
 * @brief 按行均匀切分
 *
 * 将 M 行矩阵均匀分配给 N 个进程，余数行分配给前几个 rank。
 *
 * 例: 10000 行, 4 进程 → Rank0:[0,2500), Rank1:[2500,5000), ...
 */
class RowPartitioner : public Partitioner {
public:
    /**
     * @brief 实例方法 —— 符合 Partitioner 接口；需要运行时提供 rank
     *
     * @param rank 调用方需额外传入当前进程 rank
     */
    PartitionInfo partition(int total_rows, int /*cols*/,
                            int world_size) const override
    {
        return compute(total_rows, rank_, world_size);
    }

    /** 设置该实例对应的 rank */
    void setRank(int rank) { rank_ = rank; }

    // ------------------------------------------------------------------
    // 静态便捷方法
    // ------------------------------------------------------------------

    /**
     * @brief 静态方法 —— 可直接使用，无需实例化
     *
     * @param total_rows 全局总行数
     * @param rank       当前进程 rank (0-based)
     * @param world_size 进程总数
     * @return PartitionInfo 当前 rank 的行范围 [start_row, end_row)
     */
    static PartitionInfo compute(int total_rows, int rank, int world_size)
    {
        assert(total_rows > 0);
        assert(world_size > 0);
        assert(rank >= 0 && rank < world_size);

        int base = total_rows / world_size;
        int rem  = total_rows % world_size;

        int start = rank * base + std::min(rank, rem);
        int count = base + (rank < rem ? 1 : 0);

        return {start, start + count};
    }

private:
    int rank_{0};
};

} // namespace dmc
