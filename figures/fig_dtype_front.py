#!/usr/bin/env python3
"""D-type ionization front to 5 t_s against the Hosokawa & Inutsuka (2006) solution r_s (1 + (7/4) sqrt(4/3) t / t_s)^(4/7).
The run (c128) is DTypeFront_JAFF_Centered: the source in the middle of [-6 r_s, 6 r_s]^3 at 128^3.

    python figures/fig_dtype_front.py build [RUN ...]
    python figures/fig_dtype_front.py run [RUN ...] --launch
    python figures/fig_dtype_front.py reduce [RUN ...]
    python figures/fig_dtype_front.py plot

Figures (each drawn from whichever runs have been reduced):
    dtype_front_radius.pdf    effective radius vs Hosokawa-Inutsuka, and their difference in cells
    dtype_front_snapshot.pdf  x_HI and T in the z = dz/2 plane at 5 t_s (finest reduced run)

The source sits at the origin, with reflecting boundaries. The problem writes dtype_front_radii.csv (time, r_effective,
r_analytical) every step: r_effective is the sphere-equivalent radius of the ionized volume int (1 - x_HI) dV,
and r_analytical is the Spitzer radius r_s (1 + 7 t / 4 t_s)^(4/7) (the problem is in the gas-pressure regime,
r_ch / r_s ~ 2e-4). r_s and t_s = r_s / c_i use the problem's ionized equilibrium temperature T_eq, which `spitzer_scales`
reproduces (same balance and bisection); `reduce` checks it against r_analytical and adds the Hosokawa-Inutsuka radius r_HI.
stop_time (5 t_s) and the box are set in the toml.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from photochem import Run, main, read_inputs, style  # noqa: E402

FIGURE = "dtype_front"
CSV = "dtype_front_radii.csv"
SNAPSHOT = "slice.npz"
MAX_ROWS = 20000

K_B, M_P, EV = 1.3806490000000002e-16, 1.67262192595e-24, 1.602176634e-12  # Microphysics fundamental_constants.H
BAND_EV = (13.6, 26.0)  # chem band of photoionization_H_caseB (jaffgen.toml, power_law_index = 0)
SPECIES_MASS = {"H": 1.673773e-24, "H_p": 1.6728620616289998e-24, "e": 9.10938371e-28}  # photoionization EOS _parameters
SCALARS = {"H": "scalar_0", "H_p": "scalar_1", "e": "scalar_2"}
Q, N_H = 4e46, 139.9  # stromgen.Q and stromgen.n_HI_init of inputs/DTypeFront_JAFF_Centered.toml
N_TS = 5.0  # stop_time in units of t_s, as set in the toml
BOX_RS = {"DTypeFront_JAFF_Centered": 12.0}  # box side in r_s, per problem

# The paper's tolerances (sec:performance:tolerance, as fig_tolerance_scaled): atol_i = RTOL * s_i, failure tolerances 5 atol
RTOL = 1e-2
SCALE = {"spec": 1e-3, "enuc": 1e8, "rad_num": 0.1}
TOLERANCES = [
	*(f"integrator.rtol_{v}={RTOL:g}" for v in SCALE),
	*(f"integrator.atol_{v}={RTOL * s:g}" for v, s in SCALE.items()),
	f"integrator.species_failure_tolerance={5 * RTOL * SCALE['spec']:g}",
	f"integrator.radiation_failure_tolerance={5 * RTOL * SCALE['rad_num']:g}",
	"integrator.ode_max_steps=500000",
]
PLOTTED = ("c128",)


def band_energy():
	lo, hi = BAND_EV
	return lo * hi * np.log(hi / lo) / (hi - lo) * EV


def cle_term(T):
	"""Frazer & Heitsch (2019), as cle_term() in testDTypeFront_JAFF.cpp."""
	edges = [1e2, 10**2.8, 10**3.6, 1e4, 10**4.5, 1e5, 1e6]
	fits = [(3.47e-29, 1.915), (2.34e-26, 0.5), (1.11e-24, -0.099), (1.08e-32, 2.127), (2.67e-30, 1.529), (1.74e-24, 0.237), (1.10e-21, -0.323), (7.49e-21, -0.462)]
	a, b = fits[next((i for i, e in enumerate(edges) if T < e), len(edges))]
	return a * T**b


def alpha_B(T):
	return 2.63e-13 * (T / 1e4) ** -0.7


def net_energy_ionized(T, n_e):
	recombination = 0.0 if T < 100.0 else 6.1e-10 * K_B * T * T**-0.89
	ion_ff = 1.3 * 1.427e-27 * np.sqrt(T) + cle_term(T)
	return n_e * n_e * (alpha_B(T) * max(band_energy() - 13.6 * EV, 0.0) - recombination - ion_ff)


def spitzer_scales(Q, n_H):
	"""T_eq, r_s, c_i, t_s exactly as computeAfterTimestep() in testDTypeFront_JAFF.cpp."""
	lo, hi = 1000.0, 1e5
	for _ in range(10000):
		mid = 0.5 * (lo + hi)
		if net_energy_ionized(mid, n_H) > 0.0:
			lo = mid
		else:
			hi = mid
		if hi - lo < 1.0:
			break
	T_eq = 0.5 * (lo + hi)
	r_s = (3.0 * Q / (4.0 * np.pi * alpha_B(T_eq) * n_H**2)) ** (1.0 / 3.0)
	c_i = np.sqrt(K_B * T_eq / (0.5 * M_P))
	return {"T_eq": T_eq, "r_s": r_s, "c_i": c_i, "t_s": r_s / c_i}


def r_spitzer(t, s):
	return s["r_s"] * (1.0 + 7.0 * t / (4.0 * s["t_s"])) ** (4.0 / 7.0)


def r_hosokawa_inutsuka(t, s):
	return s["r_s"] * (1.0 + 7.0 / 4.0 * np.sqrt(4.0 / 3.0) * t / s["t_s"]) ** (4.0 / 7.0)


def dtype(n, max_grid_size, extra=(), name=None, plotfile_interval=10, problem="DTypeFront_JAFF_Centered", inputs="inputs/DTypeFront_JAFF_Centered.toml",
	  **kwargs):
	return Run(
		name=name or f"n{n}",
		problem=problem,
		inputs=inputs,
		overrides=[
			f"amr.n_cell=[{n},{n},{n}]",
			f"amr.max_grid_size={max_grid_size}",
			f"amr.blocking_factor={min(32, max_grid_size)}",
			f"plotfile_interval={plotfile_interval}",
			*TOLERANCES,
			*extra,
		],
		**kwargs,
	)


RUNS = [
	# The source in the middle of [-6 r_s, 6 r_s]^3, 128^3 in 8 boxes of 64^3, one per GCD of one node
	dtype(128, 64, name="c128", host="setonix", partition="gpu", time="00:10:00"),  # ~7 min
]

COLORS = {"c128": "#0173B2"}
LABELS = {"c128": r"$128^3$"}
REF = dict(color="k", ls=":", lw=1.2)


def subsample(n, m=MAX_ROWS):
	if n <= m:
		return np.arange(n)
	idx = np.concatenate([np.geomspace(1, n, m // 2), np.linspace(1, n, m // 2)]).astype(int) - 1
	return np.unique(np.append(idx, n - 1))


def drop_superseded(d):
	"""A restarted job appends rows from its checkpoint time on; drop earlier rows it supersedes."""
	later_min = np.append(np.minimum.accumulate(d[::-1, 0])[::-1][1:], np.inf)
	return d[d[:, 0] < later_min]


def final_plotfile(run_dir, prefix):
	plotfiles = [p for p in run_dir.glob(f"{prefix}*") if p.is_dir() and (p / "Header").exists()]
	if not plotfiles:
		raise SystemExit(f"no plotfile {prefix}* in {run_dir}")
	return max(plotfiles, key=lambda p: int(p.name[len(prefix) :]))


def run_scales(params):
	s = spitzer_scales(params["stromgen.Q"], params["stromgen.n_HI_init"] + params["stromgen.n_HII_init"])
	s["dx"] = (params["geometry.prob_hi"][0] - params["geometry.prob_lo"][0]) / params["amr.n_cell"][0]
	return s


def reduce(run, run_dir, out_dir):
	for name in ("inputs.toml", "run.log"):
		shutil.copy(run_dir / name, out_dir / name)
	params = read_inputs(run_dir)
	s = run_scales(params)
	df = pd.read_csv(run_dir / CSV)
	d = drop_superseded(df.to_numpy())
	if not np.isclose(d[-1, 0], params["stop_time"], rtol=1e-5):
		print(f"  WARNING {run.name}: last row t = {d[-1, 0]:.6e} s, stop_time = {params['stop_time']:.6e} s (run unfinished?)")
	err = np.max(np.abs(d[:, 2] / r_spitzer(d[:, 0], s) - 1.0))
	print(f"  r_analytical vs spitzer_scales: max relative difference {err:.1e} (T_eq = {s['T_eq']:.1f} K, t_s = {s['t_s']:.6e} s)")
	if err > 1e-5:  # the problem writes 6 significant digits
		print(f"  WARNING {run.name}: spitzer_scales does not reproduce the problem's r_analytical")
	box = params["geometry.prob_hi"][0] - params["geometry.prob_lo"][0]
	if not (np.isclose(params["stop_time"], N_TS * s["t_s"], rtol=1e-6) and np.isclose(box, BOX_RS[run.problem] * s["r_s"], rtol=1e-6)):
		print(f"  WARNING {run.name}: stop_time = {params['stop_time'] / s['t_s']:.4f} t_s, box = {box / s['r_s']:.4f} r_s")
	d = np.column_stack([d, r_hosokawa_inutsuka(d[:, 0], s)])
	header = ",".join([*df.columns, "r_HI"])
	np.savetxt(out_dir / CSV, d[subsample(len(d))], delimiter=",", header=header, comments="", fmt="%.10e")

	import yt

	yt.set_log_level(40)
	plotfile = final_plotfile(run_dir, params["plotfile_prefix"])
	ds = yt.load(str(plotfile))
	t = float(ds.current_time.d)
	if not np.isclose(t, params["stop_time"], rtol=1e-6):
		print(f"  WARNING {run.name}: snapshot {plotfile.name} is at t = {t:.6e} s, not stop_time")
	nx, ny, nz = (int(d) for d in ds.domain_dimensions)
	if nx != ny:
		raise SystemExit(f"x = y plane needs nx = ny (got {nx}, {ny})")
	grid = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
	diag = np.arange(nx)
	k0 = round(-float(ds.domain_left_edge.d[2]) / float(ds.domain_width.d[2]) * nz)  # the cells just above z = 0 (the source)
	# z = dz/2 (key "wall": in the reflecting wall for the corner source), and x = y (cells (i, i, k))
	planes = {"wall": (slice(None), slice(None), k0), "diag": (diag, diag, slice(None))}
	out = {}
	for plane, idx in planes.items():
		n = {sp: grid[("boxlib", f)].d[idx] / SPECIES_MASS[sp] for sp, f in SCALARS.items()}
		eint = grid[("boxlib", "gasInternalEnergy")].d[idx]
		T = (2.0 / 3.0) * eint / (K_B * (n["H"] + n["H_p"] + n["e"]))  # every species has gamma = 5/3
		out |= {f"n_H_{plane}": (n["H"] + n["H_p"]).astype(np.float32), f"x_HI_{plane}": (n["H"] / (n["H"] + n["H_p"])).astype(np.float32)}
		out[f"T_{plane}"] = T.astype(np.float32)
	np.savez_compressed(
		out_dir / SNAPSHOT,
		**out,
		time=t,
		left_edge=ds.domain_left_edge.d,
		right_edge=ds.domain_right_edge.d,
		plotfile=plotfile.name,
	)
	print(f"  {plotfile.name}: t = {t / s['t_s']:.4f} t_s, planes z = dz/2 and x = y of {nx}x{ny}x{nz} -> {SNAPSHOT}")


def load_run(results_dir, name):
	data = results_dir / name
	if not (data / CSV).exists():
		return None
	d = np.genfromtxt(data / CSV, delimiter=",", names=True)
	params = read_inputs(data)
	return {"t": d["time"], "r": d["r_effective"], "r_HI": d["r_HI"], "s": run_scales(params), "params": params}


def reduced_runs(results_dir):
	runs = {r.name: load_run(results_dir, r.name) for r in RUNS if r.name in PLOTTED}
	return {k: v for k, v in runs.items() if v is not None}


def plot_radius(results_dir):
	runs = reduced_runs(results_dir)
	if not runs:
		print("skip dtype_front_radius: no runs reduced")
		return
	s = next(iter(runs.values()))["s"]
	t_ref = np.linspace(0.0, max(r["t"][-1] for r in runs.values()), 500)

	fig, (ax1, ax2) = style.figure("column", aspect=1.05, nrows=2, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
	for name, run in runs.items():
		t = run["t"] / s["t_s"]
		ax1.plot(t, run["r"] / s["r_s"], color=COLORS[name], label=LABELS[name])
		# the difference from the D-type solution only once R_I has reached r_s (R-type phase before, R_I - R_HI ~ -45 cells)
		d_type = run["r"] >= s["r_s"]
		d_type[np.argmax(d_type) :] = True
		ax2.plot(t[d_type], ((run["r"] - run["r_HI"]) / run["s"]["dx"])[d_type], color=COLORS[name])
	ax1.plot(t_ref / s["t_s"], r_hosokawa_inutsuka(t_ref, s) / s["r_s"], **REF, label=r"$R_{\rm S}\left(1 +\frac{7}{4}\sqrt{\frac{4}{3}}\,t/t_{\rm D}\right)^{4/7}$")
	ax1.set_ylabel(r"$R_{\rm I} / R_{\rm S}$")
	ax1.set_ylim(bottom=0)
	# mathtext puts the exponent of \right) at mid-height; LaTeX puts it on top
	ax1.legend(loc="upper left").get_texts()[-1].set_usetex(True)
	ax2.axhline(0, **REF)
	ax2.set_ylabel(r"$(R_{\rm I} - R_{\rm I}^{\rm ref}) / \Delta x$")
	ax2.set_xlabel(r"$t / t_{\rm D}$")
	ax2.set_xlim(0, t_ref[-1] / s["t_s"])
	fig.tight_layout()
	style.save(fig, "dtype_front_radius")


def plot_snapshot(results_dir):
	runs = reduced_runs(results_dir)
	name = max((k for k in runs if (results_dir / k / SNAPSHOT).exists()), key=lambda k: runs[k]["params"]["amr.n_cell"][0], default=None)
	if name is None:
		print("skip dtype_front_snapshot: no snapshot reduced")
		return
	snap = np.load(results_dir / name / SNAPSHOT)
	s = runs[name]["s"]
	lo, hi = snap["left_edge"] / s["r_s"], snap["right_edge"] / s["r_s"]
	extent = [lo[0], hi[0], lo[1], hi[1]]

	from matplotlib.colors import LogNorm, Normalize

	fig, axes = style.figure("column", aspect=1.75, nrows=2, sharex=True)
	panels = [
		(snap["x_HI_wall"], r"$x_{\rm H^0}$", "Blues", Normalize(vmin=0.0, vmax=1.0)),
		(snap["T_wall"], r"$T$ (K)", "inferno", LogNorm()),
	]
	for ax, (field, label, cmap, norm) in zip(axes, panels):
		im = ax.imshow(field.T, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
		ax.set_xlim(lo[0], hi[0])
		ax.set_ylim(lo[1], hi[1])
		ax.set_aspect("equal")
		ax.set_ylabel(r"$y / R_{\rm S}$")
		style.colorbar(fig, im, ax, label)
	axes[1].set_xlabel(r"$x / R_{\rm S}$")
	print(f"snapshot {name}: t = {snap['time'] / s['t_s']:.4f} t_s")
	fig.tight_layout()
	style.save(fig, "dtype_front_snapshot")


def plot(results_dir):
	plot_radius(results_dir)
	plot_snapshot(results_dir)


if __name__ == "__main__":
	main(FIGURE, RUNS, reduce, plot)
