#!/bin/bash
#SBATCH -A @ACCOUNT@
#SBATCH -J @JOB_NAME@
#SBATCH -o slurm-%j.out
#SBATCH -t @TIME@
#SBATCH -p work
#SBATCH -N @NODES@
#SBATCH -n @NTASKS@

# CPU build required (the gpu profile's hipcc build will not run here)
module load PrgEnv-gnu cray-mpich hdf5/1.14.5-parallel-api-v112
export OMP_NUM_THREADS=1

srun -n @NTASKS@ @COMMAND@ > run.log 2>&1

sacct -j ${SLURM_JOBID} -o jobid%20,Start%20,elapsed%20
