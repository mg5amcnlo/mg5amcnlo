#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=3700
#SBATCH --time=02:00:00
#SBATCH --partition=devel

module purge
module load GCC/10.2.0 OpenMPI/4.0.5 CUDA/11.1.1
module load GCCcore/10.2.0
module load Python/3.8.6
module load dask/2021.2.0

srun ~/code/mg5amcnlo/bin/mg5_aMC ~/code/mg5amcnlo/MY_FIRST_MG5_RUN/Cards/run_card.dat

