#!/bin/bash
# =============================================================================
# HPC 集群运行时加载脚本
#
# 仅在检测到 module 命令可用时加载（本地 macOS 自动跳过）。
#
# 设计:
#   - C++ 构建依赖: compiler/gnu/14.3.0 + mpi/openmpi/4.1.6
#   - Python 参考:  仅调用 load_python_ref() 时临时加载 miniconda3
#   - 显式设置 LD_LIBRARY_PATH 确保使用 GCC 自带的 libstdc++ 和 libgomp
#     (避免被 miniconda 的旧版 libstdc++.so.6 覆盖导致 GLIBCXX_3.4.32 not found)
#
# 用法:
#   source scripts/load_modules.sh            # 加载 C++ 运行时
#   source scripts/load_modules.sh python     # 额外加载 Python
# =============================================================================

# 检测 module 命令
if ! command -v module &>/dev/null; then
    # macOS / 无 LMOD 环境: 静默跳过
    return 0 2>/dev/null || exit 0
fi

echo "[load_modules] Detected LMOD/Environment Modules, loading HPC stack..."

# ============================
# 1. C++ 编译器: GCC 14.3.0
# ============================
module load compiler/gnu/14.3.0 2>/dev/null || {
    module load compiler/gnu/12.2.0 2>/dev/null || {
        echo "[load_modules] WARNING: No GCC module found" >&2
    }
}

# ============================
# 2. MPI: Open MPI 4.1.6
# ============================
module load mpi/openmpi/4.1.6 2>/dev/null || {
    module load mpi/openmpi/3.1.6 2>/dev/null || {
        echo "[load_modules] WARNING: No OpenMPI module found" >&2
    }
}

# ============================
# 3. 修复 LD_LIBRARY_PATH: 确保 GCC 的 libstdc++/libgomp 优先
# ============================
# 不同集群上 GCC lib64 路径可能不同，按优先级搜索
GCC_LIB64=""
for candidate in \
    /public/software/compiler/gnu/gcc-14.3.0/lib64 \
    /public/software/compiler/gnu/gcc-14.3.0/lib \
    /public/software/compiler/gnu/gcc-12.2.0/lib64 \
    /public/software/compiler/gnu/gcc-12.2.0/lib; do
    if [ -f "$candidate/libstdc++.so.6" ]; then
        GCC_LIB64="$candidate"
        break
    fi
done

if [ -n "$GCC_LIB64" ]; then
    # 优先级: GCC 路径 > 已有 LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="${GCC_LIB64}:${LD_LIBRARY_PATH:-}"
    echo "[load_modules] LD_LIBRARY_PATH prepended: $GCC_LIB64"
fi

# ============================
# 4. Python (仅当需要生成参考数据时)
# ============================
if [ "${1:-}" = "python" ] || [ "${1:-}" = "py" ]; then
    module load apps/envs/miniconda3/25.5.1 2>/dev/null || {
        echo "[load_modules] INFO: miniconda3 module not found, using system python3" >&2
    }
    # 注意: miniconda 可能再次覆盖 LD_LIBRARY_PATH，因此重新前置 GCC lib64
    if [ -n "$GCC_LIB64" ] && [[ ":$LD_LIBRARY_PATH:" != *":$GCC_LIB64:"* ]]; then
        export LD_LIBRARY_PATH="${GCC_LIB64}:${LD_LIBRARY_PATH:-}"
    fi
fi

# ============================
# 5. 诊断输出
# ============================
echo "[load_modules] gcc    = $(gcc --version 2>/dev/null | head -1)"
echo "[load_modules] g++    = $(g++ --version 2>/dev/null | head -1)"
echo "[load_modules] mpic++ = $(which mpic++ 2>/dev/null)"
echo "[load_modules] python = $(which python3 2>/dev/null)"
