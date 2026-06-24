#pragma once

#include <cstddef>
#include <cstdio>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#elif defined(__linux__)
#include <unistd.h>
#include <fstream>
#include <string>
#endif

namespace dmc {

/**
 * @brief 跨平台内存统计
 *
 * 返回当前进程的驻留集大小 (RSS)，单位 KB。
 */
class Memory {
public:
    /** 当前进程 RSS (KB) */
    static long currentRSS()
    {
#if defined(__APPLE__)
        struct mach_task_basic_info info;
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                      (task_info_t)&info, &count) == KERN_SUCCESS) {
            return static_cast<long>(info.resident_size / 1024);
        }
        return -1;
#elif defined(__linux__)
        long rss = 0;
        std::ifstream f("/proc/self/status");
        std::string line;
        while (std::getline(f, line)) {
            if (line.compare(0, 6, "VmRSS:") == 0) {
                // "VmRSS:   12345 kB"
                sscanf(line.c_str() + 6, "%ld", &rss);
                break;
            }
        }
        return rss;
#else
        return -1;
#endif
    }

    /** 当前进程虚拟内存 (KB) */
    static long currentVM()
    {
#if defined(__APPLE__)
        struct task_vm_info info;
        mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
        if (task_info(mach_task_self(), TASK_VM_INFO,
                      (task_info_t)&info, &count) == KERN_SUCCESS) {
            return static_cast<long>(info.virtual_size / 1024);
        }
        return -1;
#elif defined(__linux__)
        long vm = 0;
        std::ifstream f("/proc/self/status");
        std::string line;
        while (std::getline(f, line)) {
            if (line.compare(0, 7, "VmSize:") == 0) {
                sscanf(line.c_str() + 7, "%ld", &vm);
                break;
            }
        }
        return vm;
#else
        return -1;
#endif
    }
};

} // namespace dmc
