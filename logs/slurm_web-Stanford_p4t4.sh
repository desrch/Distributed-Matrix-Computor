#!/bin/bash
#SBATCH --job-name=pr_web-Stanford_p4t4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --output=logs/pr_web-Stanford_p4t4_%j.out
#SBATCH --error=logs/pr_web-Stanford_p4t4_%j.err



export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "================================================"
echo "  SLURM Job: ${SLURM_JOB_ID}"
echo "  Dataset:   web-Stanford"
echo "  MPI procs: 4"
echo "  OMP threads: 4"
echo "  Nodes:     2"
echo "================================================"

cd "/Users/desrchfriedrich/Code/C++/HPC/DMC"

mpirun -np 4 ./build/matrix_app "data/web-Stanford.txt"

echo "Done: pr_web-Stanford_p4t4"
