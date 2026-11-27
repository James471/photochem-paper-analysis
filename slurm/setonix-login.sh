#!/bin/bash
# Serial runs on a Setonix login node, one after another, without Slurm (started with nohup by `run --batch` for partition
# "login"). Runs whose run.log already has "Finished." are skipped. device.txt records the node and CPU.
source @ENV_SCRIPT@ > /dev/null 2>&1
export OMP_NUM_THREADS=1

run() {
	local dir=$1
	shift
	cd "$dir" || return
	if grep -q '^Finished\.' run.log 2>/dev/null; then
		echo "$(date -Is) skip $dir (finished)"
		return
	fi
	echo "$(date -Is) start $dir on $(hostname)"
	echo "$(hostname): $(lscpu | sed -n 's/^Model name: *//p')" > device.txt
	"$@" > run.log 2>&1
	local status=$?
	echo "$(date -Is) end $dir (exit $status)"
}

@RUNS@
