"""Quokka source trees pinned to a commit (Run.commit): one detached worktree per commit, quokka-<commit[:9]> next to the host's
Quokka checkout, with its submodules borrowing objects from that checkout's."""

import subprocess
from pathlib import Path


def _git(repo, *args, check=True, timeout=None):
	return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check, timeout=timeout)


def worktree(run, hcfg):
	return Path(hcfg["quokka"]).parent / f"quokka-{run.commit[:9]}"


def _fetch(repo, branch):
	"""Update the remote-tracking copies of `branch` from every remote that has it; skipped quietly when offline."""
	for remote in _git(repo, "remote").stdout.split():
		try:
			_git(repo, "fetch", "--quiet", remote, f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}", check=False, timeout=60)
		except subprocess.TimeoutExpired:
			pass


def resolve(repo, commit):
	out = _git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}", check=False)
	return out.stdout.strip() if out.returncode == 0 else None


def check_commit(branch, commit, hcfg, fetch=True):
	"""Warn when `commit` is not the head of `branch` (the local branch or any remote-tracking copy), so that code changed on the
	branch is not silently missing from the runs."""
	repo = Path(hcfg["quokka"])
	if not repo.exists():
		return
	if fetch:
		_fetch(repo, branch)
	full = resolve(repo, commit)
	if full is None:
		print(f"WARNING: commit {commit} is not in {repo} (fetch the branch {branch} there)")
		return
	refs = _git(repo, "for-each-ref", "--format=%(refname:short) %(objectname)", f"refs/heads/{branch}", f"refs/remotes/*/{branch}").stdout
	heads = dict(line.split() for line in refs.splitlines())
	if not heads:
		print(f"WARNING: branch {branch} not found in {repo}; cannot check that commit {full[:9]} is its head")
	by_sha = {}
	for ref, sha in heads.items():
		by_sha.setdefault(sha, []).append(ref)
	for sha, names in by_sha.items():
		if sha == full:
			continue
		ref = " and ".join(names)
		ahead = _git(repo, "rev-list", "--count", f"{full}..{sha}").stdout.strip()
		subject = _git(repo, "log", "-1", "--format=%s", sha).stdout.strip()
		on_branch = _git(repo, "merge-base", "--is-ancestor", full, sha, check=False).returncode == 0
		if _git(repo, "merge-base", "--is-ancestor", sha, full, check=False).returncode == 0:
			print(f"note: {ref} is at {sha[:9]}, behind the pinned {full[:9]} (a stale checkout; the runs are not affected)")
			continue
		print("=" * 100)
		print(f"WARNING: {ref} {'are' if len(names) > 1 else 'is'} at {sha[:9]} ({subject}), but the runs are pinned to {full[:9]}.")
		print(f"         {ahead} commit(s) on the branch are NOT in these runs" + ("." if on_branch else f", and {full[:9]} is not on the branch at all."))
		print("         Update the pinned commit in the figure script if the runs should include them.")
		print("=" * 100)


def ensure_worktree(run, hcfg):
	"""Create the run's worktree (detached at run.commit, submodules initialised) if missing; fail if it was moved off the commit."""
	repo = Path(hcfg["quokka"])
	path = worktree(run, hcfg)
	full = resolve(repo, run.commit)
	if full is None:
		_fetch(repo, run.branch)
		full = resolve(repo, run.commit)
	if full is None:
		raise SystemExit(f"commit {run.commit} not found in {repo}, even after fetching {run.branch}")
	if path.exists():
		head = _git(path, "rev-parse", "HEAD").stdout.strip()
		if head != full:
			raise SystemExit(f"{path} is at {head[:9]}, not the pinned {full[:9]}: it was changed by hand")
		return path
	print(f"creating worktree {path} at {full[:9]}", flush=True)
	_git(repo, "worktree", "add", "--detach", str(path), full)
	paths = _git(path, "config", "-f", ".gitmodules", "--get-regexp", r"submodule\..*\.path").stdout.split()[1::2]
	for sub in paths:
		ref = repo / sub
		args = ["--reference", str(ref)] if (ref / ".git").exists() else []
		subprocess.run(["git", "-C", str(path), "submodule", "update", "--init", *args, sub], check=True)
	return path


def dirty(path):
	return bool(_git(path, "status", "--porcelain", "--untracked-files=no").stdout.strip())
