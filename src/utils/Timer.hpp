#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(DMC_USE_MPI)
#include <mpi.h>
#endif

namespace dmc {

/**
 * @brief 高精度计时器
 *
 * 用法:
 * @code
 *   Timer t;
 *   t.start("load");
 *   // ... work ...
 *   t.stop("load");
 *   t.printAll();
 * @endcode
 */
class Timer {
public:
    using Clock     = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration  = std::chrono::duration<double>; // seconds

    /** 开始计时一个命名阶段 */
    void start(const std::string& name)
    {
        starts_[name] = Clock::now();
    }

    /** 停止计时，返回耗时（秒） */
    double stop(const std::string& name)
    {
        auto it = starts_.find(name);
        if (it == starts_.end()) return 0.0;
        double elapsed = Duration(Clock::now() - it->second).count();
        records_[name] = elapsed;
        starts_.erase(it);
        return elapsed;
    }

    /** 查询已记录的耗时 */
    double get(const std::string& name) const
    {
        auto it = records_.find(name);
        return (it != records_.end()) ? it->second : 0.0;
    }

    /** 打印所有阶段耗时 */
    void printAll() const
    {
        for (const auto& kv : records_) {
            printf("[Timer] %-20s : %8.4f s\n", kv.first.c_str(), kv.second);
        }
    }

    /** 返回记录的阶段名和耗时列表 */
    std::vector<std::pair<std::string, double>> all() const
    {
        std::vector<std::pair<std::string, double>> result;
        for (const auto& kv : records_)
            result.emplace_back(kv.first, kv.second);
        return result;
    }

    /** 重置所有记录 */
    void reset() { records_.clear(); starts_.clear(); }

private:
    std::unordered_map<std::string, TimePoint> starts_;
    std::unordered_map<std::string, double>    records_;
};

} // namespace dmc
