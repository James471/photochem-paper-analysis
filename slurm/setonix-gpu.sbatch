#!/bin/bash
#SBATCH -A @ACCOUNT@
#SBATCH -J @JOB_NAME@
#SBATCH -o slurm-%j.out
#SBATCH -t @TIME@
#SBATCH -p @PARTITION@
#SBATCH --exclusive
#SBATCH -N @NODES@

# Adapted from quokka/scripts/slurm/setonix-1node.submit
. @QUOKKA@/scripts/hpc_profiles/setonix-gpu.profile
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=NUMA
export OMP_NUM_THREADS=1

cat > selectGPU_X.sh << 'EOF'
#!/bin/bash
export ROCR_VISIBLE_DEVICES="$SLURM_LOCALID"
exec "$@"
EOF
chmod 755 ./selectGPU_X.sh
CPU_BIND=$(generate_CPU_BIND.sh map_cpu)

# continue from the newest checkpoint, if there is one (chained jobs)
if [ @CHAIN@ -gt 0 ]; then
	CHK=$(ls -d chk[0-9]* 2> /dev/null | sort -V | tail -n 1)
	RESTART=${CHK:+restartfile=$CHK}
	echo "$(date -Is) $SLURM_JOB_ID start ${RESTART:-from scratch}" >> chain.log
fi

srun -N @NODES@ -n @NTASKS@ -c 8 --gres=gpu:8 --cpu-bind=${CPU_BIND} ./selectGPU_X.sh @COMMAND@ $RESTART >> run.log 2>&1
STATUS=$?

sacct -j ${SLURM_JOBID} -o jobid%20,Start%20,elapsed%20

# Resubmit (at most @CHAIN@ times) only if every job so far, this one included, ended cleanly ("Simulation ended" once
# per start; a crash breaks the count), this one stopped short of the end time (walltime limit), and it wrote a new checkpoint.
if [ @CHAIN@ -gt 0 ]; then
	echo "$(date -Is) $SLURM_JOB_ID exit $STATUS" >> chain.log
	STARTS=$(grep -c " start " chain.log)
	ENDED=$(grep -c "Simulation ended at t" run.log)
	LAST=$(grep "Simulation ended at t" run.log | tail -n 1)
	NEW=$(ls -d chk[0-9]* 2> /dev/null | sort -V | tail -n 1)
	if [ "$ENDED" -eq "$STARTS" ] && ! echo "$LAST" | grep -q "(100%)" && [ "$NEW" != "$CHK" ] && [ "$STARTS" -le @CHAIN@ ]; then
		ls -d chk[0-9]* | sort -V | head -n -2 | xargs -r rm -rf # keep the two newest checkpoints
		sbatch job.sbatch >> chain.log
	fi
fi
