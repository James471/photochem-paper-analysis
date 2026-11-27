#!/usr/bin/env python3
"""Stromgren sphere with recombination (StromgrenSphereRSLA) at c_hat = c/1000, c/100, c/10 and c, 128^3, one level, source at the
centre, to 2.5 t_rec.

    python figures/fig_stromgren_rsla.py build [RUN ...]       # one variant per c_hat (cmake -DSTROMGREN_RSLA_C_HAT_DIVISOR) in the shared build dir
    python figures/fig_stromgren_rsla.py run [RUN ...] --launch
    python figures/fig_stromgren_rsla.py reduce [RUN ...]
    python figures/fig_stromgren_rsla.py plot

Figures (each drawn from whichever runs have been reduced; time series cut at T_PLOT_TREC):
    stromgren_snapshot.pdf    x_HI in the z = dz/2 plane above the central source of SNAPSHOT_RUN (c, 128^3) at t_end (2.5 t_rec), with the ODE radius
    stromgren_ode.pdf         ODE_RUN (c, 128^3) effective radius vs front_ode at c_hat = c (the paper's ODE), error in cells
    stromgren_rsla_chat.pdf   the same for every run in CHAT_RUNS, all against the c_hat = c ODE (the physical solution)

The problem writes stromgren_sphere_radii.csv (time, r_effective, r_analytical, r_half) every step. Both radii are
sphere-equivalent (8x the octant): r_effective of the ionized volume int (1 - x_HI) dV, and r_half of the region with
x_HI < 1/2 (the figures use RADIUS). r_analytical is the ODE dR/dt = (Q - 4/3 pi R^3 alpha_B n^2) / (Q/c + 4 pi R^2 n),
integrated with the true c. Both radii exceed it at late times (c/100 at 4.2 t_rec: r_effective +4.1%, r_half +6.8%)
because the classical solution assumes a fully ionized interior, whereas here (n sigma r_S ~ 54) x_HI rises to 0.1-0.2
near the edge; alpha_B n^2 int x_HII^2 dV / Q is then 0.963, short of 1 because the volume is still growing.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from photochem import Run, main, read_inputs, style  # noqa: E402

FIGURE = "stromgren_rsla"
CSV = "stromgren_sphere_radii.csv"
SNAPSHOT = "x_HI.npz"
MAX_ROWS = 20000  # radius time series are subsampled to at most this many rows
RADIUS = "r_effective"  # or "r_half" (runs from before r_half existed only have r_effective)

ALPHA_B = 2.63e-13  # case-B coefficient of the network (temperature-independent) and of the problem's ODE
SIGMA_PI = 3.2385396673401957e-18  # photoionization cross-section of the network (photoionization_H_caseB_handwritten)
C_LIGHT = 2.99792458e10
GRID_128 = ["amr.n_cell=[128,128,128]", "amr.max_level=0"]
T_END_2P5TREC = 2.5 / 5.0 * 1.901141e16  # the toml's stromgen.tend is 5 t_rec
TO_2P5TREC = [f"stromgen.tend={T_END_2P5TREC:.7e}", f"plottime_interval={T_END_2P5TREC:.7e}", "skip_initial_plotfile=1"]  # one plotfile, at t_end
T_PLOT_TREC = 2.5  # the figures show t <= T_PLOT_TREC t_rec


def c_hat_divisor(run):
	return int(next(a for a in run.cmake_args if a.startswith("-DSTROMGREN_RSLA_C_HAT_DIVISOR=")).split("=")[1])


def front_ode(t, Q, n_H, c_hat):
	"""Front with a partially ionized interior in photoionization equilibrium, at the reduced speed of light:
	    du/dr = -4 pi r^2 alpha_B n^2 x^2, u(0) = Q;   (1 - x) sigma u / (4 pi r^2 alpha_B n) = x^2
	    dR/dt = u(R) / (u(R)/c_hat + 4 pi R^2 n x(R)),  R(0) = 0
	u(r) is the ionizing photon rate through radius r. Returns the front R(t) and the sphere-equivalent radius of the
	ionized volume int_0^R 4 pi r^2 x dr (comparable to the problem's r_effective)."""
	from scipy.integrate import solve_ivp

	def x_eq(u, r):
		gamma = SIGMA_PI * max(u, 0.0) / (4.0 * np.pi * r * r * ALPHA_B * n_H)
		return 2.0 / (1.0 + np.sqrt(1.0 + 4.0 / gamma)) if gamma > 0.0 else 0.0

	def structure(r, y):
		x = x_eq(y[0], r)
		return [-4.0 * np.pi * r * r * ALPHA_B * n_H**2 * x * x, 4.0 * np.pi * r * r * x]

	r_s = (3.0 * Q / (4.0 * np.pi * ALPHA_B * n_H**2)) ** (1.0 / 3.0)
	r0 = 1e-8 * r_s  # x = 1 to machine precision here; the volume and recombinations inside are negligible
	prof = solve_ivp(structure, (r0, 2.0 * r_s), [Q, 4.0 / 3.0 * np.pi * r0**3], dense_output=True, rtol=1e-11, atol=[1e-12 * Q, 1e-12 * r_s**3])

	def front(_, R):
		r = max(R[0], r0)
		u = prof.sol(r)[0]
		return [u / (u / c_hat + 4.0 * np.pi * r * r * n_H * x_eq(u, r))]

	sol = solve_ivp(front, (0.0, t[-1]), [0.0], t_eval=t, dense_output=False, rtol=1e-11, atol=1e-12 * r_s)
	R = sol.y[0]
	volume = prof.sol(np.maximum(R, r0))[1]
	return R, (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)


def rsla(divisor, name, extra=(), **kwargs):
	"""128^3 in 8 boxes of 64^3 (one Setonix node unless ngpus=1), one plotfile at t_end (2.5 t_rec). Every run writes a final checkpoint
	(Quokka does when checkpoint_interval > 0; chained runs get it from `prepare`)."""
	checkpoints = [] if kwargs.get("chain") else ["checkpoint_interval=1000000"]
	return Run(
		name=name,
		problem="StromgrenSphereRSLA",
		inputs="inputs/StromgrenSphereRSLA.toml",
		overrides=[*GRID_128, "plotfile_interval=-1", *checkpoints, "amr.max_grid_size=64", *TO_2P5TREC, *extra],
		cmake_args=[f"-DSTROMGREN_RSLA_C_HAT_DIVISOR={divisor}"],
		variant=f"chat{divisor}",
		host="setonix",
		**kwargs,
	)


# The Wendland C2 source at the centre of [-1.27 r_S, 1.27 r_S]^3 (inputs toml): dx = r_S/50, 2.9e6 steps to 2.5 t_rec at c
# (~3.5 h at ~4.3 ms/step, with one resubmission from the checkpoint as a safety net; the c/10 run takes ~20 min).
RUNS = [
	# one GCD (8 boxes of 64^3), which the gpu queue starts sooner: 15 s and 125 s on a whole node
	rsla(1000, "c1000_ctr", ngpus=1, partition="gpu", time="00:30:00"),
	rsla(100, "c100_ctr", ngpus=1, partition="gpu", time="00:30:00"),
	rsla(10, "c10_ctr", partition="gpu", time="01:30:00"),
	rsla(1, "c1_ctr", extra=["max_timesteps=10000000"], partition="gpu-dev", time="04:00:00", chain=1),
]
SNAPSHOT_RUN = "c1_ctr"  # stromgren_snapshot.pdf (paper Fig. stromgren_snapshot)
ODE_RUN = "c1_ctr"  # stromgren_ode.pdf (Fig. stromgren_profile)
CHAT_RUNS = ["c1000_ctr", "c100_ctr", "c10_ctr", "c1_ctr"]  # stromgren_rsla_chat.pdf (Fig. stromgren_rsla)

COLORS = {"c1000_ctr": "#CC78BC", "c100_ctr": "#DE8F05", "c10_ctr": "#029E73", "c1_ctr": "#0173B2"}
STYLES = {"c1000_ctr": "-.", "c100_ctr": "--", "c10_ctr": (0, (5, 1, 1, 1)), "c1_ctr": "-"}
REF = dict(color="k", ls=":", lw=1.2)


def subsample(n, m=MAX_ROWS):
	"""Row indices: all if n <= m, else a union of log- and linearly-spaced indices (keeps early times and the end)."""
	if n <= m:
		return np.arange(n)
	idx = np.concatenate([np.geomspace(1, n, m // 2), np.linspace(1, n, m // 2)]).astype(int) - 1
	return np.unique(np.append(idx, n - 1))


def final_plotfile(run_dir, prefix):
	plotfiles = [p for p in run_dir.glob(f"{prefix}*") if p.is_dir() and (p / "Header").exists()]
	if not plotfiles:
		raise SystemExit(f"no plotfile {prefix}* in {run_dir}")
	return max(plotfiles, key=lambda p: int(p.name[len(prefix) :]))


def drop_superseded(d):
	"""A restarted job appends rows from its checkpoint time on; drop earlier rows it supersedes (times >= a restart time)."""
	later_min = np.append(np.minimum.accumulate(d[::-1, 0])[::-1][1:], np.inf)  # min time over all later rows
	return d[d[:, 0] < later_min]


def reduce(run, run_dir, out_dir):
	shutil.copy(run_dir / "inputs.toml", out_dir / "inputs.toml")
	# run.log without the per-step lines and the problem's per-step "Test failed" blocks (5 lines each), which are counted instead
	per_step = ("Coarse STEP", "Test failed", "Analytical radius", "Effective radius", "(r_effective - r_analytical)", "Expected range")
	failed = 0
	with open(run_dir / "run.log") as src, open(out_dir / "run.log", "w") as dst:
		for line in src:
			failed += line.startswith("Test failed")
			if line.strip() and not line.startswith(per_step):
				dst.write(line)
		dst.write(f"[reduce] {failed} 'Test failed' blocks removed\n")
	columns = ["time", "r_effective", "r_analytical", "r_half"]
	has_header = (run_dir / CSV).open().readline().startswith("time")  # a run started from another run's checkpoint has none
	df = pd.read_csv(run_dir / CSV, header=0 if has_header else None, names=None if has_header else columns)
	header = ",".join(df.columns)
	d = drop_superseded(df.to_numpy())
	t_end = read_inputs(run_dir)["stromgen.tend"]
	if not np.isclose(d[-1, 0], t_end, rtol=1e-6):
		print(f"  WARNING {run.name}: last row t = {d[-1, 0]:.6e} s, stromgen.tend = {t_end:.6e} s (run unfinished?)")
	np.savetxt(out_dir / CSV, d[subsample(len(d))], delimiter=",", header=header, comments="", fmt="%.10e")

	if run.name == SNAPSHOT_RUN:
		import yt

		yt.set_log_level(40)
		plotfile = final_plotfile(run_dir, read_inputs(run_dir)["plotfile_prefix"])
		ds = yt.load(str(plotfile))
		if not np.isclose(float(ds.current_time.d), t_end, rtol=1e-6):
			print(f"  WARNING {run.name}: snapshot {plotfile.name} is at t = {float(ds.current_time.d):.6e} s, not stromgen.tend")
		grid =ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
		rho_H = grid[("boxlib", "scalar_0")].d  # species order H, H_p, e; AION(H) = AION(H_p), so mass and number fractions agree
		rho_Hp = grid[("boxlib", "scalar_1")].d
		x_HI = (rho_H / (rho_H + rho_Hp)).astype(np.float32)
		np.savez_compressed(
			out_dir / SNAPSHOT,
			x_HI=x_HI,
			time=float(ds.current_time.d),
			left_edge=ds.domain_left_edge.d,
			right_edge=ds.domain_right_edge.d,
			plotfile=plotfile.name,
		)
		print(f"  {plotfile.name}: t = {float(ds.current_time.d):.4e} s, x_HI {x_HI.shape} -> {SNAPSHOT}")


def load_run(results_dir, name):
	data = results_dir / name
	if not (data / CSV).exists():
		return None
	d = np.genfromtxt(data / CSV, delimiter=",", names=True)
	params = read_inputs(data)
	t_rec = scales(params)["t_rec"]
	d = d[(d["time"] > 0) & (d["time"] <= T_PLOT_TREC * t_rec * (1.0 + 1e-9))]
	if RADIUS not in d.dtype.names:
		raise SystemExit(f"{name}: no {RADIUS} column (the run predates it)")
	R_ode, r_eff_ode = ode(params, d["time"])
	return {"t": d["time"], "r": d[RADIUS], "r_ode": r_eff_ode, "R_ode": R_ode, "params": params}


def ode(params, t):
	"""The paper's front ODE (front_ode) at c_hat = c, for every run: the RSLA runs are judged against the physical solution."""
	return front_ode(np.asarray(t, dtype=float), params["stromgen.Q"], params["stromgen.n_HI_init"] + params["stromgen.n_HII_init"], C_LIGHT)


def scales(params):
	"""Stromgren radius, recombination time, cell size and source truncation radius of a run."""
	Q, n_H = params["stromgen.Q"], params["stromgen.n_HI_init"] + params["stromgen.n_HII_init"]
	box = params["geometry.prob_hi"][0] - params["geometry.prob_lo"][0]
	r_s = (3.0 * Q / (4.0 * np.pi * ALPHA_B * n_H**2)) ** (1.0 / 3.0)
	return {"r_s": r_s, "t_rec": 1.0 / (ALPHA_B * n_H), "dx": box / params["amr.n_cell"][0]}


def c_hat_label(name):
	divisor = c_hat_divisor(next(r for r in RUNS if r.name == name))
	return r"$\hat c = c$" if divisor == 1 else rf"$\hat c = c/{divisor}$"


def plot_ode(results_dir):
	run = load_run(results_dir, ODE_RUN)
	if run is None:
		print(f"skip stromgren_ode: {ODE_RUN} not reduced")
		return
	s = scales(run["params"])
	t = run["t"] / s["t_rec"]

	fig, (ax1, ax2) = style.figure("column", aspect=1.05, nrows=2, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
	ax1.plot(t, run["r"] / s["r_s"], color=COLORS[ODE_RUN], label="Quokka, " + c_hat_label(ODE_RUN))
	ax1.plot(t, run["r_ode"] / s["r_s"], **REF, label="ODE")
	ax1.set_ylabel(r"$R_{\rm I} / R_{\rm S}$")
	ax1.set_ylim(0, 1.05)
	ax1.legend(loc="lower right")

	ax2.axhline(0, **REF)
	ax2.plot(t, (run["r"] - run["r_ode"]) / s["dx"], color=COLORS[ODE_RUN])
	ax2.set_ylabel(r"$(R_{\rm I} - R_{\rm I}^{\rm ref}) / \Delta x$")
	ax2.set_xlabel(r"$t / t_{\rm S}$")
	ax2.set_xlim(0, t[-1])
	fig.tight_layout()
	style.save(fig, "stromgren_ode")


def plot_chat(results_dir):
	runs = {name: load_run(results_dir, name) for name in CHAT_RUNS}
	runs = {k: v for k, v in runs.items() if v is not None}
	if not runs:
		print("skip stromgren_rsla_chat: no runs reduced")
		return
	missing = [name for name in CHAT_RUNS if name not in runs]
	if missing:
		print(f"stromgren_rsla_chat: not reduced yet, left out: {missing}")
	ref = max(runs.values(), key=lambda r: len(r["t"]))  # the ODE (the same for every run) is drawn on the densest time grid
	s = scales(ref["params"])
	t_ref = ref["t"] / s["t_rec"]

	fig, (ax1, ax2) = style.figure("column", aspect=1.05, nrows=2, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
	for name, run in runs.items():
		t = run["t"] / s["t_rec"]
		ax1.plot(t, run["r"] / s["r_s"], color=COLORS[name], ls=STYLES[name], label=c_hat_label(name))
		ax2.plot(t, (run["r"] - run["r_ode"]) / s["dx"], color=COLORS[name], ls=STYLES[name])
	ax1.plot(t_ref, ref["r_ode"] / s["r_s"], **REF, label="ODE")
	ax1.set_ylabel(r"$R_{\rm I} / R_{\rm S}$")
	ax1.set_ylim(0, 1.05)
	ax1.legend(loc="lower right")
	ax2.axhline(0, **REF)
	ax2.set_ylabel(r"$(R_{\rm I} - R_{\rm I}^{\rm ref}) / \Delta x$")
	ax2.set_xlabel(r"$t / t_{\rm S}$")
	ax2.set_xlim(0, max(r["t"][-1] for r in runs.values()) / s["t_rec"])
	fig.tight_layout()
	style.save(fig, "stromgren_rsla_chat")


def plot_snapshot(results_dir):
	data = results_dir / SNAPSHOT_RUN
	run = load_run(results_dir, SNAPSHOT_RUN)
	if run is None or not (data / SNAPSHOT).exists():
		print(f"skip stromgren_snapshot: {SNAPSHOT_RUN} snapshot not reduced")
		return
	snap = np.load(data / SNAPSHOT)
	s = scales(run["params"])
	lo, hi = snap["left_edge"] / s["r_s"], snap["right_edge"] / s["r_s"]

	x_HI = snap["x_HI"]
	k = x_HI.shape[2] // 2  # the plane z = dz/2 just above the source at the centre

	fig, ax = style.figure("column", aspect=0.86)
	im = ax.imshow(x_HI[:, :, k].T, origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]], cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
	ax.set_xlim(lo[0], hi[0])
	ax.set_ylim(lo[1], hi[1])
	ax.set_aspect("equal")
	ax.set_xlabel(r"$x / R_{\rm S}$")
	ax.set_ylabel(r"$y / R_{\rm S}$")
	ax.text(
		0.97, 0.95, c_hat_label(SNAPSHOT_RUN) + "\n" + rf"$t = {snap['time'] / s['t_rec']:.2f}\,t_{{\rm S}}$",
		transform=ax.transAxes, ha="right", va="top", fontsize=7,
		bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
	)
	style.colorbar(fig, im, ax, r"$x_{\rm H^0}$")
	fig.tight_layout()
	style.save(fig, "stromgren_snapshot")


def plot(results_dir):
	plot_ode(results_dir)
	plot_chat(results_dir)
	plot_snapshot(results_dir)


if __name__ == "__main__":
	main(FIGURE, RUNS, reduce, plot)
