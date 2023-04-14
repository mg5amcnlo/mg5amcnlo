#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3700
#SBATCH --time=00:30:00

module purge
module load GCC/10.2.0 OpenMPI/4.0.5 CUDA/11.1.1
module load GCCcore/10.2.0
module load Python/3.8.6
module load dask/2021.2.0

# Below is the correct launch command for the daskmpi mode

# Memory per worker
export MEM=$((${SLURM_CPUS_PER_TASK}*${SLURM_MEM_PER_CPU}))

# Scheduler file - the Python code will need to refer to this
export SCHED_FILE=${SLURM_JOB_ID}.sched

# Launch the scheduler 
dask-scheduler --scheduler-file=$SCHED_FILE  &

# Launch the worker processes using the SLURM-allocated resources
srun  dask-worker --local-directory='/tmp' --no-nanny --nthreads=${SLURM_CPUS_PER_TASK} \
--scheduler-file=$SCHED_FILE  --memory-limit=${MEM}M &

# Launch the master/client script
python <path_to>/bin/mg5_aMC <path_to>/<input>.mg