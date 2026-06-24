#!/bin/bash
# =============================================================================
# PageRank 实验运行脚本 (SLURM + 本地模式)
#
# 用法:
#   # 单次本地运行:
#   bash scripts/run_experiments.sh --dataset web-Stanford --mpi 4 --omp 4
#
#   # 参数扫描 + 本地运行:
#   bash scripts/run_experiments.sh --dataset web-Stanford \
#       --mpi-list "1,2,4" --omp-list "1,2,4"
#
#   # SLURM 提交 (分配 4 节点, 每节点 16 核):
#   bash scripts/run_experiments.sh --slurm --nodes 4 --partition compute \
#       --time 01:00:00 --dataset web-Stanford --mpi 4 --omp 4
#
#   # 运行全部实验 (多数据集 + 多配置):
#   bash scripts/run_experiments.sh --all
#
#   # 仅编译并运行 Python 参考:
#   bash scripts/run_experiments.sh --dataset web-Stanford --python-ref
# =============================================================================

set -euo pipefail

# ------ 默认值 ------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DATASET="web-Stanford"
DATASET_PATH=""

MPI_LIST=""           # MPI 进程数列表
OMP_LIST=""           # OpenMP 线程数列表
SINGLE_MPI=""         # 单个 MPI 值
SINGLE_OMP=""         # 单个 OMP 值

USE_SLURM=0
SLURM_NODES=2
SLURM_PARTITION="compute"
SLURM_TIME="02:00:00"
SLURM_ACCOUNT=""
SLURM_QOS=""

RUN_ALL=0
PYTHON_REF=0
BUILD_ONLY=0
SKIP_BUILD=0
MAX_ITER=100
TOLERANCE="1e-6"
DAMPING="0.85"

LAB_DIR="$PROJECT_DIR/lab_res"

# ------ 帮助 ------
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --dataset NAME       Dataset: web-Stanford | web-Google | soc-Epinions1
                        (default: web-Stanford)

  --dataset-path PATH  Direct path to dataset file (overrides --dataset)

  Single-run mode:
  --mpi N              MPI processes (e.g. 4)
  --omp N              OpenMP threads (e.g. 4)

  Sweep mode:
  --mpi-list "1,2,4"    Comma-separated MPI process counts
  --omp-list "1,2,4,8"  Comma-separated OpenMP thread counts

  SLURM mode:
  --slurm               Submit via sbatch
  --nodes N             Node count (default: 2)
  --partition NAME      Partition (default: compute)
  --time HH:MM:SS       Wall time (default: 02:00:00)
  --account NAME        SLURM account
  --qos NAME            SLURM QoS

  Misc:
  --all                 Run all datasets × (1,2,4) MPI × (1,2,4) OMP
  --python-ref          Generate Python scipy reference (before experiments)
  --build-only          Only build, don't run
  --skip-build          Don't rebuild
  --max-iter N          Max PageRank iterations (default: 100)
  --tol VAL             Convergence tolerance (default: 1e-6)
  --help                Show this help

Output:
  CSV files → lab_res/
  SLURM log → logs/

Examples:
  $0 --dataset web-Stanford --mpi 4 --omp 4
  $0 --dataset web-Google --mpi-list "1,2,4,8" --omp-list "1,2,4"
  $0 --all --slurm --nodes 4 --time 04:00:00
EOF
    exit 0
}

# ------ 参数解析 ------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)      DATASET="$2";        shift 2 ;;
        --dataset-path) DATASET_PATH="$2";    shift 2 ;;
        --mpi)          SINGLE_MPI="$2";      shift 2 ;;
        --omp)          SINGLE_OMP="$2";      shift 2 ;;
        --mpi-list)     MPI_LIST="$2";        shift 2 ;;
        --omp-list)     OMP_LIST="$2";        shift 2 ;;
        --slurm)        USE_SLURM=1;          shift 1 ;;
        --nodes)        SLURM_NODES="$2";      shift 2 ;;
        --partition)    SLURM_PARTITION="$2";  shift 2 ;;
        --time)         SLURM_TIME="$2";       shift 2 ;;
        --account)      SLURM_ACCOUNT="$2";    shift 2 ;;
        --qos)          SLURM_QOS="$2";        shift 2 ;;
        --all)          RUN_ALL=1;            shift 1 ;;
        --python-ref)   PYTHON_REF=1;         shift 1 ;;
        --build-only)   BUILD_ONLY=1;         shift 1 ;;
        --skip-build)   SKIP_BUILD=1;         shift 1 ;;
        --max-iter)     MAX_ITER="$2";        shift 2 ;;
        --tol)          TOLERANCE="$2";        shift 2 ;;
        --help)         usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

cd "$PROJECT_DIR"
mkdir -p "$LAB_DIR" logs

# ------ 解析数据集路径 ------
if [ -z "$DATASET_PATH" ]; then
    case "$DATASET" in
        web-Stanford)   DATASET_PATH="data/web-Stanford.txt" ;;
        web-Google)     DATASET_PATH="data/web-Google.txt" ;;
        soc-Epinions1)  DATASET_PATH="data/soc-Epinions1.txt" ;;
        *) echo "ERROR: Unknown dataset '$DATASET'. Use --dataset-path." >&2; exit 1 ;;
    esac
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset file not found: $DATASET_PATH" >&2
    exit 1
fi

DATASET_NAME="$(basename "$DATASET_PATH" .txt)"

# ------ 构建 MPI/OMP 组合列表 ------
if [ -n "$SINGLE_MPI" ] && [ -n "$SINGLE_OMP" ]; then
    MPI_VALS=("$SINGLE_MPI")
    OMP_VALS=("$SINGLE_OMP")
elif [ -n "$MPI_LIST" ] && [ -n "$OMP_LIST" ]; then
    IFS=',' read -ra MPI_VALS <<< "$MPI_LIST"
    IFS=',' read -ra OMP_VALS <<< "$OMP_LIST"
elif [ "$RUN_ALL" = "1" ]; then
    MPI_VALS=(1 2 4)
    OMP_VALS=(1 2 4)
else
    echo "ERROR: Specify --mpi/--omp for single run, --mpi-list/--omp-list for sweep, or --all." >&2
    usage
fi

# ------ 生成 Python 参考 ------
if [ "$PYTHON_REF" = "1" ]; then
    echo "========================================"
    echo "  Generating Python scipy reference"
    echo "  Dataset: $DATASET_PATH"
    echo "========================================"
    python3 scripts/pagerank_ref_sparse.py "$DATASET_PATH"
    echo ""
fi

# ------ 编译 ------
if [ "$SKIP_BUILD" = "0" ]; then
    echo "========================================"
    echo "  Building matrix_app"
    echo "========================================"
    bash scripts/build.sh
    echo ""
fi

if [ "$BUILD_ONLY" = "1" ]; then
    echo "Build complete. Exiting (--build-only)."
    exit 0
fi

# ------ 生成 SLURM 脚本或直接运行 ------
generate_slurm_script() {
    local MPI_P=$1
    local OMP_T=$2
    local JOB_NAME="pr_${DATASET_NAME}_p${MPI_P}t${OMP_T}"

    cat <<SBSCRIPT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=${SLURM_NODES}
#SBATCH --ntasks-per-node=$(( MPI_P / SLURM_NODES ))
#SBATCH --cpus-per-task=${OMP_T}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --time=${SLURM_TIME}
#SBATCH --output=logs/${JOB_NAME}_%j.out
#SBATCH --error=logs/${JOB_NAME}_%j.err
$( [ -n "$SLURM_ACCOUNT" ] && echo "#SBATCH --account=$SLURM_ACCOUNT")
$( [ -n "$SLURM_QOS" ]     && echo "#SBATCH --qos=$SLURM_QOS")

export OMP_NUM_THREADS=${OMP_T}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "================================================"
echo "  SLURM Job: \${SLURM_JOB_ID}"
echo "  Dataset:   ${DATASET_NAME}"
echo "  MPI procs: ${MPI_P}"
echo "  OMP threads: ${OMP_T}"
echo "  Nodes:     ${SLURM_NODES}"
echo "================================================"

cd "$PROJECT_DIR"

mpirun -np ${MPI_P} ./build/matrix_app "$DATASET_PATH"

echo "Done: ${JOB_NAME}"
SBSCRIPT
}

run_local() {
    local MPI_P=$1
    local OMP_T=$2
    local JOB_NAME="pr_${DATASET_NAME}_p${MPI_P}t${OMP_T}"

    echo "========================================"
    echo "  Local Run: $JOB_NAME"
    echo "  MPI=$MPI_P  OMP=$OMP_T"
    echo "  Dataset: $DATASET_PATH"
    echo "========================================"

    export OMP_NUM_THREADS=$OMP_T
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores

    mpirun -np "$MPI_P" ./build/matrix_app "$DATASET_PATH"
    echo ""
}

# ------ 执行 ------
TOTAL=${#MPI_VALS[@]}
CURRENT=0
for MPI_P in "${MPI_VALS[@]}"; do
    for OMP_T in "${OMP_VALS[@]}"; do
        CURRENT=$((CURRENT + 1))

        if [ "$USE_SLURM" = "1" ]; then
            SLURM_SCRIPT="logs/slurm_${DATASET_NAME}_p${MPI_P}t${OMP_T}.sh"
            generate_slurm_script "$MPI_P" "$OMP_T" > "$SLURM_SCRIPT"
            chmod +x "$SLURM_SCRIPT"
            echo "[$CURRENT/$TOTAL] Submitting: sbatch $SLURM_SCRIPT"
            sbatch "$SLURM_SCRIPT"
        else
            echo "[$CURRENT/$TOTAL] MPI=$MPI_P OMP=$OMP_T"
            run_local "$MPI_P" "$OMP_T"
        fi
    done
done

# ------ 汇总 ------
if [ "$USE_SLURM" = "0" ]; then
    echo "========================================"
    echo "  Experiment runs complete."
    echo "  CSV files → lab_res/"
    echo "========================================"
    ls -lh "$LAB_DIR"/*.csv 2>/dev/null | tail -20 || echo "  (no CSV files yet)"

    echo ""
    echo "To aggregate results, run:"
    echo "  python3 scripts/aggregate_results.py lab_res/ --output lab_res/summary.csv"
else
    echo "All SLURM jobs submitted. Monitor with: squeue -u \$USER"
    echo "After completion, aggregate with:"
    echo "  python3 scripts/aggregate_results.py lab_res/ --output lab_res/summary.csv"
fi
