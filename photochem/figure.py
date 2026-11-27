"""Command-line driver shared by every figure script.

    python figures/fig_x.py configure [RUN ...]
    python figures/fig_x.py build [RUN ...]
    python figures/fig_x.py run [RUN ...] [--launch] [--force]
    python figures/fig_x.py reduce [RUN ...]
    python figures/fig_x.py plot
    python figures/fig_x.py status
"""

import argparse
import shutil

from . import build as build_mod
from . import runs as runs_mod
from .config import RESULTS, host, load_config
from .source import check_commit, resolve


def _check_pins(runs, stage):
	"""Warn, at every stage, when a run's pinned commit is not the head of its branch."""
	_, hcfg = load_config()
	for branch, commit in sorted({(r.branch, r.commit) for r in runs if r.commit}):
		if not branch:
			raise SystemExit(f"runs pinned to {commit} must name their branch")
		check_commit(branch, commit, hcfg, fetch=stage != "plot")


def _done(figure, run, equivalent, hcfg):
	"""Message if the run already finished (it is then not run again), else None."""
	rdir = runs_mod.run_dir(figure, run)
	if not runs_mod.finished(rdir):
		return None
	commit, dirty = runs_mod.prepared_commit(rdir) or (None, None)
	if not run.commit:
		return "finished"
	pinned = resolve(hcfg["quokka"], run.commit) or run.commit
	if commit == pinned and not dirty:
		return f"finished at the pinned commit {pinned[:9]}"
	if commit in equivalent:
		return f"finished at {commit[:9]} ({equivalent[commit]})"
	return f"finished at {str(commit)[:9]}{' + uncommitted changes' if dirty else ''}, NOT the pinned {pinned[:9]}; rerun with --force"


def main(figure, runs, reduce, plot, setup=None, equivalent=None):
	"""`reduce(run, run_dir, out_dir)` writes small data into out_dir; `plot(results_dir)` makes the figure.
	Optional `setup(run, run_dir)` stages extra inputs after the run directory is prepared. `equivalent` maps commits (full
	hashes) whose finished runs are accepted for runs pinned to another commit to the reason (e.g. verified code-identical)."""
	equivalent = equivalent or {}
	parser = argparse.ArgumentParser(prog=figure)
	sub = parser.add_subparsers(dest="stage", required=True)
	sub.add_parser("configure", help="(re)run cmake for the runs' build dirs").add_argument("names", nargs="*")
	sub.add_parser("build", help="build the runs' targets (configures if needed)").add_argument("names", nargs="*")
	p_run = sub.add_parser("run", help="prepare run directories; --launch to submit/start them; finished runs are skipped")
	p_run.add_argument("names", nargs="*")
	p_run.add_argument("--launch", action="store_true")
	p_run.add_argument("--force", action="store_true", help="prepare (and run) again, even runs that already finished")
	p_run.add_argument(
		"--batch",
		metavar="TIME",
		help="run the selected runs one after another, one batch per partition: a single-GPU job of walltime TIME (Setonix, Gadi) "
		"or a nohup process on the Setonix login node (partition 'login'); already-prepared runs are reused (unless --force)",
	)
	p_red = sub.add_parser("reduce", help="reduce raw output into results/<figure>/<run>/")
	p_red.add_argument("names", nargs="*")
	sub.add_parser("plot", help="make the figure from results/<figure>/")
	sub.add_parser("status", help="show run and reduction state")
	args = parser.parse_args()

	by_name = {r.name: r for r in runs}
	selected = [by_name[n] for n in getattr(args, "names", [])] or runs
	here = [r for r in selected if not r.host or r.host == host()]
	_check_pins(selected if args.stage == "plot" else here, args.stage)  # a pinned commit may exist only where its runs run
	if args.stage in ("configure", "build", "run", "reduce"):
		elsewhere = [r.name for r in selected if r.host and r.host != host()]
		if elsewhere:
			print(f"skipping runs that belong to another host: {elsewhere}")
		selected = here
	results = RESULTS / figure

	if args.stage == "run" and not args.force:
		_, hcfg = load_config()
		todo = []
		for run in selected:
			if msg := _done(figure, run, equivalent, hcfg):
				print(f"{run.name}: {msg}; not run again")
			else:
				todo.append(run)
		selected = todo
		if not selected:
			print("nothing to run")
			return

	if args.stage == "configure":
		build_mod.configure(selected)
	elif args.stage == "build":
		build_mod.build(selected)
	elif args.stage == "run" and args.batch:
		for run in selected:
			rdir = runs_mod.run_dir(figure, run)
			if (rdir / "run.json").exists() and not args.force:
				print(f"reusing {rdir}")
				continue
			rdir = runs_mod.prepare(figure, run, force=args.force)
			if setup:
				setup(run, rdir)
			print(f"prepared {rdir}")
		if args.launch:
			groups = {}
			for run in selected:
				groups.setdefault(run.partition, []).append(run)
			for group in groups.values():
				jobid, script = runs_mod.submit_batch(figure, group, args.batch)
				print(f"submitted {script} as {jobid}: {[r.name for r in group]}")
	elif args.stage == "run":
		for run in selected:
			rdir = runs_mod.prepare(figure, run, force=args.force)
			if setup:
				setup(run, rdir)
			print(f"prepared {rdir}")
			if args.launch:
				meta = runs_mod.launch(rdir)
				print(f"  launched: {meta.get('jobid') or meta.get('pid')}")
	elif args.stage == "reduce":
		for run in selected:
			rdir = runs_mod.run_dir(figure, run)
			if not (rdir / "run.json").exists():
				print(f"skip {run.name}: not prepared")
				continue
			out = results / run.name
			out.mkdir(parents=True, exist_ok=True)
			reduce(run, rdir, out)
			shutil.copy(rdir / "run.json", out / "run.json")
			print(f"reduced {run.name} -> {out}")
	elif args.stage == "plot":
		plot(results)
	elif args.stage == "status":
		for run in runs:
			reduced = (results / run.name / "run.json").exists()
			where = runs_mod.state(runs_mod.run_dir(figure, run)) if not run.host or run.host == host() else f"(on {run.host})"
			print(f"{run.name:24s} {where:50s} reduced={'yes' if reduced else 'no'}")
