#!/bin/bash
# =============================================================================
# 分布式矩阵计算框架 — 编译脚本
#
# 自动检测平台 (macOS / Linux)，使用合适的 OpenMP 编译选项。
#
# 用法:
#   bash scripts/build.sh [--no-omp] [--debug]
#   bash scripts/build.sh                # Release + OpenMP
#   bash scripts/build.sh --no-omp        # 纯 MPI，无 OpenMP
#   bash scripts/build.sh --debug         # Debug 模式
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
mkdir -p build

# ------ 默认配置 ------
USE_OMP=1
BUILD_TYPE="-O2"

# ------ 解析参数 ------
for arg in "$@"; do
    case "$arg" in
        --no-omp) USE_OMP=0 ;;
        --debug)  BUILD_TYPE="-O0 -g" ;;
    esac
done

# ------ 查找 MPI 编译器 ------
if command -v mpic++ &>/dev/null; then
    CXX=mpic++
elif command -v mpicxx &>/dev/null; then
    CXX=mpicxx
else
    echo "ERROR: mpic++ not found in PATH" >&2
    exit 1
fi

CXXFLAGS="-std=c++17 -DDMC_USE_MPI -Wall -Wextra ${BUILD_TYPE}"

# ------ 平台检测 (OpenMP) ------
UNAME_S=$(uname -s)
if [ "$USE_OMP" = "1" ]; then
    if [ "$UNAME_S" = "Darwin" ]; then
        # macOS: libomp via Homebrew
        if [ -d "/opt/homebrew/opt/libomp" ]; then
            CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp"
        elif [ -d "/usr/local/opt/libomp" ]; then
            CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include -L/usr/local/opt/libomp/lib -lomp"
        else
            echo "WARNING: libomp not found via Homebrew; building without OpenMP" >&2
        fi
    else
        # Linux: GCC / standard Clang
        CXXFLAGS="$CXXFLAGS -fopenmp"
    fi
fi

# ------ 编译 ------
echo "========================================"
echo "  Building matrix_app"
echo "  CXX     = $CXX"
echo "  CXXFLAGS= $CXXFLAGS"
echo "  OMP     = $USE_OMP"
echo "========================================"

$CXX $CXXFLAGS -o build/matrix_app src/main.cpp

echo ""
echo "  ✅ Build succeeded: build/matrix_app"
ls -lh build/matrix_app
