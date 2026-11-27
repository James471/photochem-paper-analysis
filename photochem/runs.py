import datetime
import json
import shlex
import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .build import build_dir, built_executable, env_script, source_dir, variant_dir
from .config import ANALYSIS, PAPER_BRANCH, PAPER_COMMIT, host, load_config
from .source import dirty, resolve

CHAIN_CHECKPOINT_INTERVAL = 1_000_000  # steps; chained runs rely on the checkpoint written when max_walltime stops a job


@dataclass
class Run:
	name: str  # unique within a figure; becomes the run directory name
	problem: str  # Quokka problem, e.g. "StromgrenSphere"
	inputs: str  # toml, relative to the quokka repo (e.g. "inputs/StromgrenSphere.toml")
	dim: str = "3d"  # key into the host's `builds`
	overrides: list = field(default_factory=list)  # runtime overrides, "key=value"
	exe: str = ""  # executable name if it differs from `problem`
	quokka: str = ""  # Quokka source tree override (absolute), e.g. a patched worktree; used for configure, inputs and provenance
	# Pin the source to `commit` of `branch` (instead of `quokka`): a detached worktree quokka-<commit[:9]> next to the host's
	# checkout, created at configure time; every stage warns if the branch head has moved on from the commit. Default: the paper's
	# pin (config.yaml `paper`)
	branch: str = PAPER_BRANCH
	commit: str = PAPER_COMMIT
	build: str = ""  # build dir override (absolute)
	env: str = ""  # environment script (relative to this repository) for build and run, instead of the host's `env_script`
	cmake_args: list = field(default_factory=list)  # extra configure-time flags, e.g. "-DFOO=1"; needs its own `variant` or `build`
	variant: str = ""  # name of the cmake_args set; built in the shared build dir, executable kept in <build dir>/variants/<variant>
	ntasks: int = 0  # MPI ranks for local / cpu runs; 0 = host default
	host: str = ""  # machine this run belongs to ("" = any); configure/build/run/reduce skip it elsewhere
	runs: str = ""  # runs root override (absolute), e.g. on another project's /scratch
	# Setonix job settings (gpu-dev: <= 2 nodes, <= 4 h).
	# "login": Setonix login node, no Slurm (serial runs started with nohup by `run --batch`)
	partition: str = "gpu-dev"  # gpu-dev | gpu | work | login
	nodes: int = 1
	# GPUs: 0 = whole nodes, 1 = one GCD (one rank), which is also what `run --batch` uses.
	ngpus: int = 0
	time: str = "01:00:00"
	# Setonix whole-node GPU jobs: resubmit up to `chain` more times, each continuing from the newest checkpoint, until
	# the run reaches its end time. Adds max_walltime (10 min short of `time`, unless overridden) and checkpoint_interval overrides
	# so each job ends on a checkpoint. Quokka stops at max(0.9 max_walltime, max_walltime - 5 min).
	chain: int = 0


def run_dir(figure, run, hcfg=None):
	hcfg = hcfg or load_config()[1]
	return Path(run.runs or hcfg["runs"]) / figure / run.name


def executable(run, hcfg):
	bdir = build_dir(run, hcfg)
	return variant_dir(bdir, run.variant) / built_executable(run, bdir).name if run.variant else built_executable(run, bdir)


def _git(quokka, *args):
	return subprocess.run(["git", "-C", str(quokka), *args], capture_output=True, text=True, check=True).stdout.strip()


def finished(rdir):
	log = rdir / "run.log"
	return log.exists() and "\nFinished." in log.read_text(errors="replace")


def prepared_commit(rdir):
	"""(commit, dirty) recorded in an existing run.json, else None."""
	path = rdir / "run.json"
	if not path.exists():
		return None
	meta = json.loads(path.read_text())
	return meta.get("quokka_commit"), meta.get("quokka_dirty")


def prepare(figure, run, force=False):
	"""Create the run directory with inputs, provenance (run.json) and the launch script."""
	_, hcfg = load_config()
	rdir = run_dir(figure, run, hcfg)
	if (rdir / "run.json").exists() and not force:
		raise SystemExit(f"{rdir} already prepared (use --force to overwrite)")
	exe = executable(run, hcfg)
	if not exe.exists():
		raise SystemExit(f"executable not found: {exe} (run the `build` stage first)")
	quokka = source_dir(run, hcfg)
	if run.commit:
		head = _git(quokka, "rev-parse", "HEAD")
		if head != resolve(quokka, run.commit):
			raise SystemExit(f"{quokka} is at {head[:9]}, not the pinned {run.commit[:9]}")
		if dirty(quokka):
			print(f"WARNING: {quokka} has uncommitted changes; the run records them in quokka.diff, but the build may not match the pinned commit")
	rdir.mkdir(parents=True, exist_ok=True)
	if force and (rdir / "run.log").exists():  # keep the old log, but out of the way of the batch scripts' "Finished." check
		stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		(rdir / "run.log").rename(rdir / f"run.log.{stamp}")
		print(f"moved the previous run.log of {run.name} to run.log.{stamp}")
	shutil.copy(quokka / run.inputs, rdir / "inputs.toml")

	diff = _git(quokka, "diff", "HEAD")
	# untracked files (e.g. a new problem) as additions; `git diff --no-index` exits 1 when the files differ
	for path in _git(quokka, "ls-files", "--others", "--exclude-standard").splitlines():
		new = subprocess.run(["git", "-C", str(quokka), "diff", "--no-index", "/dev/null", path], capture_output=True, text=True).stdout
		diff = f"{diff}\n{new.strip()}".strip()
	if diff:
		(rdir / "quokka.diff").write_text(diff + "\n")
	meta = {
		"figure": figure,
		"run": asdict(run),
		"host": host(),
		"executable": str(exe),
		"quokka": str(quokka),
		"quokka_commit": _git(quokka, "rev-parse", "HEAD"),
		"quokka_branch": _git(quokka, "branch", "--show-current") or run.branch,
		"quokka_dirty": bool(diff),
		"prepared": datetime.datetime.now().isoformat(timespec="seconds"),
	}
	cmd = [str(exe), "inputs.toml", *run.overrides]
	if run.chain:
		hours, minutes, seconds = (int(x) for x in run.time.split(":"))
		stop = max(hours * 3600 + minutes * 60 + seconds - 600, 600)
		if not any(o.startswith("max_walltime=") for o in run.overrides):  # e.g. a short override to test the chain
			cmd.append(f"max_walltime={stop // 3600:02d}:{stop % 3600 // 60:02d}:{stop % 60:02d}")
		cmd.append(f"checkpoint_interval={CHAIN_CHECKPOINT_INTERVAL}")
	meta["command"] = cmd
	if run.partition == "login":
		env = env_script(run, hcfg)
		source = f"source {shlex.quote(str(env))} > /dev/null 2>&1\n" if env else ""
		(rdir / "launch.sh").write_text(f"#!/bin/bash\ncd \"$(dirname \"$0\")\"\n{source}export OMP_NUM_THREADS=1\n{shlex.join(cmd)} > run.log 2>&1\n")
	elif host() == "setonix":
		(rdir / "job.sbatch").write_text(_sbatch(figure, run, hcfg, cmd))
	else:
		launch = f"mpirun -n {run.ntasks or hcfg['ntasks']} " + shlex.join(cmd)
		(rdir / "launch.sh").write_text(f"#!/bin/bash\ncd \"$(dirname \"$0\")\"\n{launch} > run.log 2>&1\n")
	(rdir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
	return rdir


def _check_gpu_dev(name, partition, nodes, time):
	if partition == "gpu-dev":
		hours, minutes, seconds = (int(x) for x in time.split(":"))
		if nodes > 2 or hours * 3600 + minutes * 60 + seconds > 4 * 3600:
			raise SystemExit(f"{name}: gpu-dev allows <= 2 nodes and <= 04:00:00 (got {nodes}, {time})")


def _sbatch(figure, run, hcfg, cmd):
	gpu = run.partition.startswith("gpu")
	_check_gpu_dev(run.name, run.partition, run.nodes, run.time)
	if gpu and run.ngpus:
		if run.ngpus != 1 or run.nodes != 1:
			raise SystemExit(f"{run.name}: on Setonix ngpus is 0 (whole nodes) or 1 (one GCD)")
		if run.chain:
			raise SystemExit(f"{run.name}: chained runs on Setonix need whole nodes (ngpus = 0)")
		return _sbatch_gcd(f"{figure}.{run.name}", run, hcfg, f"run {shlex.quote(str(run_dir(figure, run, hcfg)))} {shlex.join(cmd)}")
	if run.chain and not gpu:
		raise SystemExit(f"{run.name}: chained runs on Setonix are GPU only")
	template = (ANALYSIS / "slurm" / ("setonix-gpu.sbatch" if gpu else "setonix-cpu.sbatch")).read_text()
	fields = {
		"PARTITION": run.partition,
		"ACCOUNT": hcfg["account"] + ("-gpu" if gpu else ""),
		"JOB_NAME": f"{figure}.{run.name}",
		"NODES": str(run.nodes),
		"NTASKS": str(run.nodes * 8 if gpu else run.ntasks or hcfg["ntasks"]),
		"TIME": run.time,
		"QUOKKA": hcfg["quokka"],
		"COMMAND": shlex.join(cmd),
		"CHAIN": str(run.chain),
	}
	for key, value in fields.items():
		template = template.replace(f"@{key}@", value)
	return template


def _sbatch_gcd(job_name, run, hcfg, runs, log="slurm-%j.out", time=None):
	"""Single-GCD job: `runs` is the body, one `run <dir> <command...>` line per run directory (executed in order)."""
	fields = {
		"PARTITION": run.partition,
		"ACCOUNT": hcfg["account"] + "-gpu",
		"JOB_NAME": job_name,
		"LOG": log,
		"TIME": time or run.time,
		"QUOKKA": hcfg["quokka"],
		"RUNS": runs,
	}
	template = (ANALYSIS / "slurm" / "setonix-gpu-gcd.sbatch").read_text()
	for key, value in fields.items():
		template = template.replace(f"@{key}@", value)
	return template


def submit_batch(figure, runs, time):
	"""Run the (prepared) `runs` one after another, skipping finished ones, in one single-GCD job on a Setonix gpu partition or,
	for partition "login", in one nohup process on this Setonix login node. The script goes to
	<runs>/<figure>/_batches/<timestamp>.{sbatch,sh}; its job id (or pid) is recorded in each run.json."""
	_, hcfg = load_config()
	if len({(r.partition, r.runs, r.env) for r in runs}) > 1:
		raise SystemExit("batched runs must share partition, runs root and environment")
	run0 = runs[0]
	lines = []
	for run in runs:
		meta = json.loads((run_dir(figure, run, hcfg) / "run.json").read_text())
		lines.append(f"run {shlex.quote(str(run_dir(figure, run, hcfg)))} {shlex.join(meta['command'])}")
	bdir = run_dir(figure, run0, hcfg).parent / "_batches"
	bdir.mkdir(parents=True, exist_ok=True)
	stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	record = {}
	if run0.partition == "login":
		if host() != "setonix":
			raise SystemExit("partition 'login' is a Setonix login node")
		script = bdir / f"{stamp}.sh"
		env = env_script(run0, hcfg)
		script.write_text(_fill("setonix-login.sh", {"ENV_SCRIPT": str(env) if env else "/dev/null", "RUNS": "\n".join(lines)}))
		with open(bdir / f"{stamp}.out", "w") as log:
			proc = subprocess.Popen(["nohup", "bash", str(script)], cwd=bdir, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
		record["pid"], record["login_node"] = proc.pid, socket.gethostname()
		jobid = f"pid {proc.pid} on {record['login_node']}"
	elif host() == "setonix":
		_check_gpu_dev("batch", run0.partition, 1, time)
		script = bdir / f"{stamp}.sbatch"
		script.write_text(_sbatch_gcd(f"{figure}.batch", run0, hcfg, "\n".join(lines), log=str(bdir / f"{stamp}-%j.out"), time=time))
		out = subprocess.run(["sbatch", "--parsable", str(script)], cwd=bdir, capture_output=True, text=True, check=True)
		jobid = record["jobid"] = out.stdout.strip().split(";")[0]
	else:
		raise SystemExit(f"--batch is for Setonix, not {host()}")
	for run in runs:
		path = run_dir(figure, run, hcfg) / "run.json"
		meta = json.loads(path.read_text())
		meta.pop("pid", None), meta.pop("jobid", None)
		meta.update(record, batch=str(script), launched=datetime.datetime.now().isoformat(timespec="seconds"))
		path.write_text(json.dumps(meta, indent=2) + "\n")
	return jobid, script


def _fill(template, fields):
	text = (ANALYSIS / "slurm" / template).read_text()
	for key, value in fields.items():
		text = text.replace(f"@{key}@", value)
	return text


def launch(rdir):
	"""Submit (Setonix) or start in the background (local). Records the job id / pid in run.json."""
	meta = json.loads((rdir / "run.json").read_text())
	if (rdir / "job.sbatch").exists():
		out = subprocess.run(["sbatch", "--parsable", "job.sbatch"], cwd=rdir, capture_output=True, text=True, check=True)
		meta["jobid"] = out.stdout.strip().split(";")[0]
	else:
		proc = subprocess.Popen(["nohup", "bash", "launch.sh"], cwd=rdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		meta["pid"] = proc.pid
	meta["launched"] = datetime.datetime.now().isoformat(timespec="seconds")
	(rdir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
	return meta


def state(rdir):
	"""Short status string for a run directory."""
	if not (rdir / "run.json").exists():
		return "not prepared"
	meta = json.loads((rdir / "run.json").read_text())
	if finished(rdir):
		return "finished"
	if "login_node" in meta:
		alive = subprocess.run(["ps", "-p", str(meta["pid"])], capture_output=True).returncode == 0 if socket.gethostname() == meta["login_node"] else None
		ran = "started" if (rdir / "run.log").exists() else "waiting"
		return f"pid {meta['pid']} on {meta['login_node']}: " + ({True: f"batch running, run {ran}", False: "batch ended, run unfinished"}.get(alive, f"run {ran} (check on that node)"))
	if "jobid" in meta and (rdir / "chain.log").exists():  # chained jobs log every start/exit/resubmission in chain.log
		chain = (rdir / "chain.log").read_text().strip().splitlines()
		jobs = [meta["jobid"], *(line.split()[-1] for line in chain if line.startswith("Submitted batch job"))]
		out = subprocess.run(["sacct", "-nX", "-j", jobs[-1], "-o", "State%20"], capture_output=True, text=True)
		return f"job {jobs[-1]} ({len(jobs)} in chain): {out.stdout.strip() or 'unknown'}"
	if "jobid" in meta:
		out = subprocess.run(["sacct", "-nX", "-j", meta["jobid"], "-o", "State%20"], capture_output=True, text=True)
		log = rdir / "run.log"
		run_state = "finished" if log.exists() and "\nFinished." in log.read_text(errors="replace") else "started" if log.exists() else "waiting"
		return f"job {meta['jobid']}: {out.stdout.strip() or 'unknown'}" + (f", run {run_state}" if "batch" in meta else "")
	if "pid" in meta:
		alive = subprocess.run(["kill", "-0", str(meta["pid"])], capture_output=True).returncode == 0
		return f"pid {meta['pid']}: {'running' if alive else 'finished'}"
	return "prepared, not launched"
