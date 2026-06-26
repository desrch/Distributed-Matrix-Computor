ssh eduhpc1@59.78.189.132 -p 2323 


scp ./DMC.zip eduhpc1@59.78.189.132:/public/home/eduhpc1/51285903021/

bash scripts/run_experiments.sh --dataset web-Stanford --mpi 1 --omp 1

bash scripts/run_experiments.sh --slurm \
    --dataset web-Stanford \
    --mpi-list "1,2,4,8" --omp-list "1,2,4" \
    --nodes 4 --time 02:00:00


bash scripts/run_experiments.sh --slurm \
    --dataset web-Google \
    --mpi-list "1,2,4,8" --omp-list "1,2,4" \
    --time 02:00:00
s


bash scripts/run_experiments.sh --slurm \
    --dataset web-Google \
    --mpi-list "1,2,4,8" --omp-list "1,2,4" \
    --time 02:00:0

bash scripts/run_experiments.sh --slurm \
    --dataset soc-LiveJournal1 \
    --mpi-list "1,2,4,8" --omp-list "1,2,4" \
    --time 02:00:0


    bash scripts/run_experiments.sh --slurm \
    --dataset soc-LiveJournal1 \
    --mpi-list "1,2,4,8" --omp-list "1,2,4" \
    --time 04:00:00