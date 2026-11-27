#!/bin/bash
#SBATCH -A @ACCOUNT@
#SBATCH -J @JOB_NAME@
#SBATCH -o @LOG@
#SBATCH -t @TIME@
#SBATCH -p @PARTITION@
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1

# One GCD (with its 8 cores; Setonix rejects explicit CPU requests for GPU jobs), one rank. Each `run` line runs one run directory to completion in turn (runs whose run.log already ends
# with "Finished." are skipped, so a batch cut off by the walltime can be resubmitted as is).
. @QUOKKA@/scripts/hpc_profiles/setonix-gpu.profile
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1

run() {
	local dir=$1
	shift
	cd "$dir" || return
	if grep -q '^Finished\.' run.log 2>/dev/null; then
		echo "$(date -Is) skip $dir (finished)"
		return
	fi
	echo "$(date -Is) start $dir"
	srun -N 1 -n 1 --gres=gpu:1 "$@" > run.log 2>&1
	local status=$?
	echo "$(date -Is) end $dir (exit $status)"
}

@RUNS@

sacct -j ${SLURM_JOBID} -o jobid%20,Start%20,elapsed%20
