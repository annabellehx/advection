#!/bin/bash
#SBATCH --job-name=advection              # Job name
#SBATCH --nodes=16                        # Request 1 node
#SBATCH --ntasks-per-node=1               # Request 1 task (job step)
#SBATCH --cpus-per-task=16                # Number of cores per processor
# #SBATCH --exclusive                     # Exclusive access to the node
#SBATCH --time=00:05:00                   # Time limit (hh:mm:ss)
#SBATCH --partition=caslake               # Specify partition if needed
#SBATCH --output=advection_output.txt     # File for standard output
#SBATCH --error=advection_error.txt       # File for standard error
#SBATCH --account=mpcs51087               # File for standard output

module load openmpi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun ./advection_hybrid 4000 1 1 16
