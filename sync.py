#!/usr/bin/env python3
"""Move data/commands between the local machine and a cluster (run locally). Setonix by default; `--host H` (before the
subcommand) for another `hosts` entry of config.yaml.

    python sync.py [--host H] pull [FIGURE]           # rsync reduced results/ from the cluster's repo
    python sync.py [--host H] raw FIGURE RUN [-n]     # rsync one raw run directory from the cluster
    python sync.py [--host H] remote -- CMD ...       # run CMD in the cluster's repo with the venv active
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from photochem.config import RESULTS, load_config  # noqa: E402

_, LOCAL = load_config("local")


def rsync(src, dst, dry_run=False):
	cmd = ["rsync", "-az", "-v", *(["-n"] if dry_run else []), src, dst]
	print(" ".join(cmd))
	subprocess.run(cmd, check=True)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--host", default="setonix", help="cluster entry in config.yaml (default: setonix)")
	sub = parser.add_subparsers(dest="cmd", required=True)
	p_pull = sub.add_parser("pull")
	p_pull.add_argument("figure", nargs="?", default="")
	p_raw = sub.add_parser("raw")
	p_raw.add_argument("figure")
	p_raw.add_argument("run")
	p_raw.add_argument("-n", "--dry-run", action="store_true")
	p_remote = sub.add_parser("remote")
	p_remote.add_argument("command", nargs=argparse.REMAINDER)
	args = parser.parse_args()
	_, remote = load_config(args.host)
	ssh = remote["ssh"]

	if args.cmd == "pull":
		sub_path = f"{args.figure}/" if args.figure else ""
		(RESULTS / sub_path).mkdir(parents=True, exist_ok=True)
		rsync(f"{ssh}:{remote['repo']}/results/{sub_path}", str(RESULTS / sub_path))
	elif args.cmd == "raw":
		src = f"{remote['runs']}/{args.figure}/{args.run}"
		subprocess.run(["ssh", ssh, f"du -sh {shlex.quote(src)}"], check=True)
		dst = Path(LOCAL["runs"]) / args.figure / args.run
		dst.mkdir(parents=True, exist_ok=True)
		rsync(f"{ssh}:{src}/", f"{dst}/", dry_run=args.dry_run)
	elif args.cmd == "remote":
		command = args.command[1:] if args.command[:1] == ["--"] else args.command
		script = (
			f"module load {remote['python_module']} >/dev/null 2>&1; "
			f"source {remote['venv']}/bin/activate && cd {remote['repo']} && {shlex.join(command)}"
		)
		sys.exit(subprocess.run(["ssh", "-A", ssh, script]).returncode)


if __name__ == "__main__":
	main()
