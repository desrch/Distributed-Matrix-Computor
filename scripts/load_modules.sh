#!/bin/bash
# =============================================================================
# HPC 集群运行时加载脚本
#
# 仅在检测到 module 命令可用时加载（本地 macOS 自动跳过）。
#
# 用法:
#   source scripts/load_modules.sh
# 或:
#   bash scripts/load_modules.sh  (独立执行, echo 到 stdout)
# =============================================================================

# 检测 module 命令是否可用
if command -v module &>/dev/null; then
    echo "[load_modules] Detected LMOD/Environment Modules, loading HPC stack..."

    # 1. GCC 14.3.0 (最新, C++17 + OpenMP 原生支持)
    module load compiler/gnu/14.3.0 2>/dev/null || \
    module load compiler/gnu/12.2.0 2>/dev/null || \
    { echo "[load_modules] WARNING: No GCC module found, using system gcc ($(gcc --version 2>/dev/null | head -1))"; }

    # 2. Open MPI 4.1.6
    module load mpi/openmpi/4.1.6 2>/dev/null || \
    module load mpi/openmpi/3.1.6 2>/dev/null || \
    { echo "[load_modules] WARNING: No OpenMPI module found, using system mpic++ ($(which mpic++ 2>/dev/null))"; }

    # 3. Miniconda3 (Python + scipy, 仅生成参考数据时需要)
    module load apps/envs/miniconda3/25.5.1 2>/dev/null || \
    { echo "[load_modules] INFO: miniconda3 not loaded (Python ref may use system python3)"; }

    echo "[load_modules] Loaded: gcc=$(gcc --version 2>/dev/null | head -1)"
    echo "[load_modules] Loaded: mpic++=$(which mpic++ 2>/dev/null)"
    echo "[load_modules] Loaded: python3=$(which python3 2>/dev/null)"
elif [ "${SLURM_JOB_ID:-}" != "" ] || [ "${HOSTNAME:-}" != "" ]; then
    # 在 HPC 集群上但无 module 命令的退化情况
    echo "[load_modules] WARNING: module command not found on cluster node ${HOSTNAME:-unknown}" >&2
fi
