import datetime
import shlex
import subprocess
from pathlib import Path

from .config import ANALYSIS, load_config
from .source import ensure_worktree, worktree


def source_dir(run, hcfg):
	if run.commit:
		if run.quokka:
			raise SystemExit(f"{run.name}: give either `quokka` or `commit`, not both")
		return worktree(run, hcfg)
	return Path(run.quokka or hcfg["quokka"])


def build_dir(run, hcfg):
	src = source_dir(run, hcfg)
	base = Path(run.build) if run.build else src / hcfg["builds"][run.dim]
	if run.commit and not base.is_relative_to(src):  # a build root outside the source tree is shared by every commit
		base = base.with_name(f"{base.name}-{run.commit[:9]}")
	return base


def variant_dir(bdir, variant):
	"""Where a variant's executables are kept. Variants share their build dir (and its AMReX); the `build` stage reconfigures it
	with the variant's cmake_args, rebuilds only the targets those change and copies the executables here."""
	return bdir / "variants" / variant


def target(run):
	return run.exe or run.problem


def built_executable(run, bdir):
	return bdir / "src" / "problems" / run.problem / target(run)


def env_script(run, hcfg):
	"""The run's environment script (Run.env, relative to this repository) or the host's (relative to the host's quokka)."""
	if run.env:
		return ANALYSIS / run.env
	if env := hcfg.get("env_script"):
		return Path(hcfg["quokka"]) / env
	return None


def _shell(cmd, run, hcfg):
	"""Run `cmd` in bash, after sourcing the environment script (e.g. Setonix module profile)."""
	if env := env_script(run, hcfg):
		cmd = f"source {shlex.quote(str(env))} >/dev/null && {cmd}"
	print(f"$ {cmd}", flush=True)
	subprocess.run(["bash", "-c", cmd], check=True)


def _by_dir(runs, hcfg):
	"""Group runs by (build dir, variant); runs sharing one must agree on cmake_args."""
	groups = {}
	for run in runs:
		if run.cmake_args and not (run.variant or run.build):
			raise SystemExit(f"{run.name}: cmake_args need a variant or build dir, so the shared build dir's cache is not changed")
		groups.setdefault((build_dir(run, hcfg), run.variant), []).append(run)
	for (bdir, variant), group in groups.items():
		if len({tuple(r.cmake_args) for r in group}) > 1:
			raise SystemExit(f"runs {[r.name for r in group]} share {bdir} ({variant or 'no variant'}) but differ in cmake_args: give them distinct variants")
	for bdir in {bdir for bdir, _ in groups}:
		shared = [r for (b, _), group in groups.items() if b == bdir for r in group]
		if len({source_dir(r, hcfg) for r in shared}) > 1:
			raise SystemExit(f"runs {[r.name for r in shared]} share {bdir} but differ in quokka source tree")
		if len({r.env for r in shared}) > 1:
			raise SystemExit(f"runs {[r.name for r in shared]} share {bdir} but differ in environment script")
	return groups


def _configure_cmd(run, hcfg, bdir, extra=()):
	# cmake_args are used unquoted so that $VARS from the environment script expand
	args = " ".join([*hcfg.get("cmake_args", []), *extra])
	return f"cmake -S {shlex.quote(str(source_dir(run, hcfg)))} -B {shlex.quote(str(bdir))} {args} -DAMReX_SPACEDIM={run.dim.rstrip('d')}"


def configure(runs):
	"""Configure each build dir. A variant's cmake_args are not applied here: `build` applies them only while building it."""
	_, hcfg = load_config()
	for (bdir, variant), group in _by_dir(runs, hcfg).items():
		run = group[0]
		if run.commit:
			ensure_worktree(run, hcfg)
		_shell(_configure_cmd(run, hcfg, bdir, () if variant else run.cmake_args), run, hcfg)


def _build_cmd(bdir, variant, group, hcfg, jobs):
	"""Shell command building a group's targets. For a variant: configure with its cmake_args, build, copy the executables to
	variant_dir, then drop those cache entries again (cmake -U), so they fall back to their defaults for everything else."""
	run = group[0]
	targets = " ".join(sorted({target(r) for r in group}))
	build = f"cmake --build {shlex.quote(str(bdir))} -j {jobs} --target {targets}"
	if not variant:
		return build
	out = variant_dir(bdir, variant)
	exes = sorted({str(built_executable(r, bdir)) for r in group})
	names = sorted({a[2:].split("=")[0].split(":")[0] for a in run.cmake_args if a.startswith("-D")})
	unset = " ".join(f"-U{n}" for n in names)
	steps = [_configure_cmd(run, hcfg, bdir, run.cmake_args), build, f"mkdir -p {shlex.quote(str(out))}", f"cp {' '.join(map(shlex.quote, exes))} {shlex.quote(str(out))}/"]
	return f"( {' && '.join(steps)} ); rc=$?; {_configure_cmd(run, hcfg, bdir)} {unset} > /dev/null; exit $rc"


def build(runs):
	"""Build every run's target, configuring first any build dir without a CMakeCache (or with run-specific cmake_args, for a run
	with its own build dir). Variants are built one after another in their shared build dir. On a host with `build_queue` (Gadi:
	login-node processes are killed), configure here and compile in one batch job; returns its id."""
	_, hcfg = load_config()
	groups = _by_dir(runs, hcfg)
	for (bdir, variant), group in groups.items():
		if not (bdir / "CMakeCache.txt").exists() or (group[0].cmake_args and not variant):
			configure(group[:1])
	if "build_queue" in hcfg:
		return _submit_build(groups, hcfg)
	for (bdir, variant), group in groups.items():
		_shell(_build_cmd(bdir, variant, group, hcfg, hcfg.get("build_jobs", 8)), group[0], hcfg)
	return None


def _submit_build(groups, hcfg):
	from .runs import storage  # avoid a circular import

	queue = hcfg["build_queue"]
	lines = []
	for (bdir, variant), group in groups.items():
		env = env_script(group[0], hcfg)
		lines += [f"source {shlex.quote(str(env))} > /dev/null 2>&1" if env else "", f"bash -c {shlex.quote(_build_cmd(bdir, variant, group, hcfg, queue['ncpus']))} || exit 1"]
	bdirs = sorted({bdir for bdir, _ in groups})
	stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	jdir = Path(hcfg["runs"]) / "_builds"
	jdir.mkdir(parents=True, exist_ok=True)
	script = jdir / f"{stamp}.pbs"
	script.write_text(
		"\n".join(
			[
				"#!/bin/bash",
				f"#PBS -P {hcfg['project']}",
				f"#PBS -q {queue['queue']}",
				"#PBS -N quokka-build",
				f"#PBS -l ncpus={queue['ncpus']}",
				f"#PBS -l mem={queue['mem']}",
				f"#PBS -l walltime={queue['time']}",
				"#PBS -l jobfs=20GB",
				f"#PBS -l storage={storage([hcfg['quokka'], *bdirs, *(env_script(g[0], hcfg) or hcfg['quokka'] for g in groups.values())])}",
				"#PBS -j oe",
				f"#PBS -o {jdir / stamp}.log",
				"module purge",
				*lines,
				"",
			]
		)
	)
	out = subprocess.run(["qsub", str(script)], cwd=jdir, capture_output=True, text=True, check=True)
	jobid = out.stdout.strip()
	print(f"submitted build job {jobid}: {script} (log {jdir / stamp}.log)")
	return jobid
