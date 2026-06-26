[eduhpc1@login1 DMC]$ bash scripts/run_experiments.sh --slurm \
>     --dataset soc-LiveJournal1 \
>     --mpi-list "1,2,4,8" --omp-list "1,2,4" \
>     --time 02:00:0
[load_modules] Detected LMOD/Environment Modules, loading HPC stack...
[load_modules] LD_LIBRARY_PATH prepended: /public/software/compiler/gnu/gcc-14.3.0/lib64
[load_modules] gcc    = gcc (GCC) 14.3.0
[load_modules] g++    = g++ (GCC) 14.3.0
[load_modules] mpic++ = /public/software/mpi/openmpi/4.1.6/bin/mpic++
[load_modules] python = /usr/bin/python3
ERROR: Unknown dataset 'soc-LiveJournal1'.
[eduhpc1@login1 DMC]$ 