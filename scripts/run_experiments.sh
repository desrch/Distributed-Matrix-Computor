#!/bin/bash
# =============================================================================
# PageRank 实验运行脚本 (SLURM + 本地模式)
#
# 自动加载 HPC 运行时模块:
#   compiler/gnu/14.3.0  +  mpi/openmpi/4.1.6  +  apps/envs/miniconda3/25.5.1
#
# 用法:
#   bash scripts/run_experiments.sh --dataset web-Stanford --mpi 4 --omp 4
#   bash scripts/run_experiments.sh --dataset web-Stanford --mpi-list "1,2,4" --omp-list "1,2,4"
#   bash scripts/run_experiments.sh --slurm --nodes 2 --mpi-list "1,2,4,8" --omp-list "1,2,4"
#   bash scripts/run_experiments.sh --all --slurm --nodes 4 --time 04:00:00
#   bash scripts/run_experiments.sh --dataset web-Stanford --python-ref
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LAB_DIR="$PROJECT_DIR/lab_res"

# ------ 加载 HPC 运行时 (集群自动生效, macOS 跳过) ------
source "$SCRIPT_DIR/load_modules.sh"

cd "$PROJECT_DIR"
mkdir -p "$LAB_DIR" logs

# ------ 默认值 ------
DATASET="web-Stanford"
DATASET_PATH=""
MPI_LIST=""
OMP_LIST=""
SINGLE_MPI=""
SINGLE_OMP=""
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

# ------ 帮助 ------
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --dataset NAME        web-Stanford | web-Google (default: web-Stanford)
  --dataset-path PATH   Direct path to dataset file
  --mpi N / --omp N     Single MPI processes / OMP threads
  --mpi-list "1,2,4"    Sweep MPI counts
  --omp-list "1,2,4"    Sweep OMP thread counts
  --slurm               Submit via sbatch
  --nodes N             SLURM node count (default: 2)
  --partition NAME      SLURM partition (default: compute)
  --time HH:MM:SS       SLURM wall time (default: 02:00:00)
  --account NAME        SLURM account
  --qos NAME            SLURM QoS
  --all                 All datasets x (1,2,4) MPI x (1,2,4) OMP
  --python-ref          Generate Python scipy reference first
  --build-only          Compile only, don't run
  --skip-build          Skip compilation
  --help                Show this help
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
        --help)         usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ------ 解析数据集路径 ------
if [ -z "$DATASET_PATH" ]; then
    case "$DATASET" in
        web-Stanford)   DATASET_PATH="data/web-Stanford.txt" ;;
        web-Google)     DATASET_PATH="data/web-Google.txt" ;;
        *) echo "ERROR: Unknown dataset '$DATASET'." >&2; exit 1 ;;
    esac
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset file not found: $DATASET_PATH" >&2
    exit 1
fi

DATASET_NAME="$(basename "$DATASET_PATH" .txt)"

# ------ 构建 MPI/OMP 组合 ------
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
    echo "ERROR: Specify --mpi/--omp, --mpi-list/--omp-list, or --all." >&2
    usage
fi

# ------ Python 参考 ------
if [ "$PYTHON_REF" = "1" ]; then
    echo "========================================"
    echo "  Generating Python scipy reference"
    echo "========================================"
    python3 scripts/pagerank_ref_sparse.py "$DATASET_PATH"
    echo ""
fi

# ------ 编译 ------
if [ "$SKIP_BUILD" = "0" ]; then
    bash scripts/build.sh
    echo ""
fi
[ "$BUILD_ONLY" = "1" ] && { echo "Build complete."; exit 0; }

# ------ SLURM 脚本生成器 ------
generate_slurm_script() {
    local MPI_P=$1
    local OMP_T=$2
    local JOB_NAME="pr_${DATASET_NAME}_p${MPI_P}t${OMP_T}"

    # 每节点任务数: 推荐 = MPI_P (让 SLURM 自行分配到节点)
    local NTASKS="${MPI_P}"

    cat <<SBSCRIPT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=${SLURM_NODES}
#SBATCH --ntasks=${NTASKS}
#SBATCH --cpus-per-task=${OMP_T}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --time=${SLURM_TIME}
#SBATCH --output=logs/${JOB_NAME}_%j.out
#SBATCH --error=logs/${JOB_NAME}_%j.err
$( [ -n "$SLURM_ACCOUNT" ] && echo "#SBATCH --account=$SLURM_ACCOUNT")
$( [ -n "$SLURM_QOS" ]     && echo "#SBATCH --qos=$SLURM_QOS")

# ===== HPC 运行时 =====
module load compiler/gnu/14.3.0  2>/dev/null
module load mpi/openmpi/4.1.6    2>/dev/null
module load apps/envs/miniconda3/25.5.1 2>/dev/null

export OMP_NUM_THREADS=${OMP_T}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "================================================"
echo "  SLURM Job: \${SLURM_JOB_ID}"
echo "  Dataset:   ${DATASET_NAME}"
echo "  MPI: ${MPI_P}  OMP: ${OMP_T}  Nodes: ${SLURM_NODES}"
echo "  gcc:       \$(gcc --version 2>/dev/null | head -1)"
echo "  mpic++:    \$(which mpic++)"
echo "================================================"

cd "$PROJECT_DIR"
mpirun -np ${MPI_P} ./build/matrix_app "$DATASET_PATH"

echo "Done: ${JOB_NAME}"
SBSCRIPT
}

# ------ 本地运行器 ------
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

    # 集群: 确认模块已加载 (幂等)
    module load compiler/gnu/14.3.0  2>/dev/null || true
    module load mpi/openmpi/4.1.6    2>/dev/null || true

    mpirun -np "$MPI_P" ./build/matrix_app "$DATASET_PATH"
    echo ""
}

# ------ 执行 ------
TOTAL=$((${#MPI_VALS[@]} * ${#OMP_VALS[@]}))
CURRENT=0
for MPI_P in "${MPI_VALS[@]}"; do
    for OMP_T in "${OMP_VALS[@]}"; do
        CURRENT=$((CURRENT + 1))

        if [ "$USE_SLURM" = "1" ]; then
            SLURM_SCRIPT="logs/slurm_${DATASET_NAME}_p${MPI_P}t${OMP_T}.sh"
            generate_slurm_script "$MPI_P" "$OMP_T" > "$SLURM_SCRIPT"
            chmod +x "$SLURM_SCRIPT"
            echo "[$CURRENT/$TOTAL] Submitting: sbatch $SLURM_SCRIPT"
            sbatch "$SLURM_SCRIPT" 2>&1 || echo "  (sbatch not available — SLURM script written to $SLURM_SCRIPT)"
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
    echo "  CSV files -> $LAB_DIR/"
    echo "========================================"
    ls -lh "$LAB_DIR"/*.csv 2>/dev/null | tail -20 || echo "  (no CSV files yet)"
    echo ""
    echo "Aggregate: python3 scripts/aggregate_results.py $LAB_DIR/"
else
    echo ""
    echo "All SLURM jobs submitted.  Monitor:  squeue -u \$USER"
    echo "After completion, aggregate:"
    echo "  python3 scripts/aggregate_results.py $LAB_DIR/"
fi
