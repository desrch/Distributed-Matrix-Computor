#!/bin/bash
#SBATCH --job-name=pr_web-Stanford_p4t4
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --output=logs/pr_web-Stanford_p4t4_%j.out
#SBATCH --error=logs/pr_web-Stanford_p4t4_%j.err



# ===== HPC 运行时 =====
module load compiler/gnu/14.3.0  2>/dev/null
module load mpi/openmpi/4.1.6    2>/dev/null
module load apps/envs/miniconda3/25.5.1 2>/dev/null

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "================================================"
echo "  SLURM Job: ${SLURM_JOB_ID}"
echo "  Dataset:   web-Stanford"
echo "  MPI: 4  OMP: 4  Nodes: 2"
echo "  gcc:       $(gcc --version 2>/dev/null | head -1)"
echo "  mpic++:    $(which mpic++)"
echo "================================================"

cd "/Users/desrchfriedrich/Code/C++/HPC/DMC"
mpirun -np 4 ./build/matrix_app "data/web-Stanford.txt"

echo "Done: pr_web-Stanford_p4t4"
