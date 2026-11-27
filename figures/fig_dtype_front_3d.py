#!/usr/bin/env python3
"""Dusty D-type H II region with AMR (DTypeFront3D) to 1 Myr, against a thin-shell model of the front.

    python figures/fig_dtype_front_3d.py build [RUN ...]
    python figures/fig_dtype_front_3d.py run [RUN ...] --launch
    python figures/fig_dtype_front_3d.py reduce [RUN ...]
    python figures/fig_dtype_front_3d.py plot

Figures (each drawn from whichever runs have been reduced):
    dtype_front_3d_radius.pdf    effective radius vs the thin-shell model, and their difference in finest-level cells
    dtype_front_3d_snapshot.pdf  x_HI and T in the plane just above z = 0 at the last plotfile (finest reduced run), the model radius on x_HI

A point source with ionizing and optical bands sits at the centre of the box. The problem writes dtype_front_3d_radii.csv
(time, r_effective, r_shell, r_spitzer, r_ode, E_opt_tot, E_ir_tot) every coarse step. r_effective is the sphere-equivalent
radius of the ionized volume int (1 - x_HI) dV. r_ode is the problem's thin-shell ODE (integrate_front in testDTypeFront3D.cpp)
with gas pressure, radiation pressure and dust absorption of ionizing photons, started at the dusty Stromgren radius at t = 0.

`reduce` solves the paper's thin-shell model and adds it to the csv as r_thin_shell, which the figures use. It differs from r_ode:
  - the optical depth of the optical band includes the dust in the ionized cavity, kappa_opt rho_0 R [(1 - s)/3 + s], not
    only the shell column kappa_opt rho_0 R / 3 (s = n_i / n_0; the ODE already counts cavity dust for the ionizing band);
  - the ambient neutral gas pushes back with P_0 = n_0 k_B T_0, T_0 the neutral equilibrium temperature;
  - only the optical luminosity of the source pushes the shell. L_ion / c and the recombination emission L_rec / h are dropped
    (together they move R(1 Myr) by 0.085 finest cells at 64^3, max_level = 1).
It also solves the problem's ODE and checks it against r_ode, so the scales and luminosities match the problem's.
The dashed Hosokawa & Inutsuka (2006) curve is the dust-free, gas-pressure-only law, for comparison.
The cell width used for the radius error is that of the finest level the inputs allow (amr.max_level), as in the problem's check.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from photochem import Run, main, read_inputs, style  # noqa: E402

FIGURE = "dtype_front_3d"
CSV = "dtype_front_3d_radii.csv"
SNAPSHOT = "slice.npz"
MAX_ROWS = 20000

K_B, M_P, EV, C_LIGHT = 1.3806490000000002e-16, 1.67262192595e-24, 1.602176634e-12, 2.99792458e10  # Microphysics fundamental_constants.H
SPECIES_MASS = {"H": 1.673773e-24, "H_p": 1.6728620616289998e-24, "e": 9.10938371e-28}  # photoionization EOS _parameters
SCALARS = {"H": "scalar_0", "H_p": "scalar_1", "e": "scalar_2"}
PC, MYR = 3.085677581491367e18, 3.15576e13  # as in testDTypeFront3D.cpp
ION_BAND_EV = (13.6, 26.0)  # chem band of thermophotochemistry_handwritten (NetworkRegistry.cmake, power-law index 0)
OPTICAL_BAND_EV = (0.413567, 13.6)  # radBoundaries of testDTypeFront3D.cpp; a thermal band's quantum is its midpoint


def band_energies():
	"""Mean photon energies (erg) of the ionizing and optical bands, as GetChemBandQuanta / GetThermalBandQuanta."""
	lo, hi = ION_BAND_EV
	return lo * hi * np.log(hi / lo) / (hi - lo) * EV, 0.5 * sum(OPTICAL_BAND_EV) * EV


def cle_term(T):
	"""Frazer & Heitsch (2019), as get_cle_term() in testDTypeFront3D.cpp."""
	edges = [1e2, 10**2.8, 10**3.6, 1e4, 10**4.5, 1e5, 1e6]
	fits = [(3.47e-29, 1.915), (2.34e-26, 0.5), (1.11e-24, -0.099), (1.08e-32, 2.127), (2.67e-30, 1.529), (1.74e-24, 0.237), (1.10e-21, -0.323), (7.49e-21, -0.462)]
	a, b = fits[next((i for i, e in enumerate(edges) if T < e), len(edges))]
	return a * T**b


def alpha_B(T):
	return 2.63e-13 * (T / 1e4) ** -0.7  # get_recombination_coefficient() of thermophotochemistry_handwritten


def lambda_rec(T):
	return 0.0 if T < 100.0 else 6.1e-10 * K_B * T * T**-0.89


def ionized_temperature(eps_ion, n):
	"""compute_equilibrium_temperature_ionized() of testDTypeFront3D.cpp (same balance and bisection)."""

	def net(T):
		return n * n * (alpha_B(T) * max(eps_ion - 13.6 * EV, 0.0) - lambda_rec(T) - (1.3 * 1.427e-27 * np.sqrt(T) + cle_term(T)))

	lo, hi = 1000.0, 1e5
	for _ in range(10000):
		mid = 0.5 * (lo + hi)
		if net(mid) > 0.0:
			lo = mid
		else:
			hi = mid
		if hi - lo < 1.0:
			break
	return 0.5 * (lo + hi)


def neutral_temperature(n):
	"""compute_equilibrium_temperature_neutral() of testDTypeFront3D.cpp: KI heating n Gamma_0 = cooling n^2 Lambda_KI(T)."""

	def net(T):
		return n * 2e-26 - n * n * 2e-26 * (1e7 * np.exp(-118400.0 / (T + 1e3)) + 1.4e-2 * np.sqrt(T) * np.exp(-92.0 / T))

	lo, hi = 1.0, 1000.0
	for _ in range(10000):
		mid = 0.5 * (lo + hi)
		if net(mid) > 0.0:
			lo = mid
		else:
			hi = mid
		if hi - lo < 1e-2:
			break
	return 0.5 * (lo + hi)


def dust_attenuation(tau):
	"""h(tau) = 3 [e^tau (tau^2 - 2 tau + 2) - 2] / tau^3, with the series below tau = 1 (spherical_dust_attenuation_factor)."""
	if tau <= 0.0:
		return 1.0
	if tau < 1.0:
		from math import factorial

		return 3.0 * sum(tau**n / (factorial(n) * (n + 3)) for n in range(30))
	return 3.0 * (np.exp(tau) * (tau * tau - 2.0 * tau + 2.0) - 2.0) / tau**3


def stromgren_scales(params):
	"""T_i, the dust-free Stromgren radius R_s and the ionized sound speed c_i, as preCalculateInitialConditions()."""
	n0, Q_ion = params["photoionize.n_HI_init"], params["photoionize.flux_ion"]
	T_i = ionized_temperature(band_energies()[0], n0)
	R_s = (3.0 * Q_ion / (4.0 * np.pi * alpha_B(T_i) * n0**2)) ** (1.0 / 3.0)
	return T_i, R_s, np.sqrt(K_B * T_i / (0.5 * M_P))


def r_hosokawa_inutsuka(t, params):
	"""Dust-free, gas-pressure-only D-type front, R_s (1 + (7/4) sqrt(4/3) t / t_s)^(4/7) with t_s = R_s / c_i."""
	_, R_s, c_i = stromgren_scales(params)
	return R_s * (1.0 + 7.0 / 4.0 * np.sqrt(4.0 / 3.0) * t * c_i / R_s) ** (4.0 / 7.0)


def thin_shell(t, params, corrected=True):
	"""Front radius R(t) of the thin-shell model, d(M Rdot)/dt = 4 pi R^2 (P_i - P_0) + L_abs / c.
	corrected=True is the paper's model: L_abs = L_opt (1 - e^-tau_opt) with the cavity dust in tau_opt, and P_0 = n_0 k_B T_0 at
	the neutral equilibrium temperature. corrected=False reproduces the problem's r_ode (integrate_front() in testDTypeFront3D.cpp):
	L_abs = L_ion + (L_opt + L_rec / h) (1 - e^-tau) with the shell column only, and P_0 = 0."""
	from scipy.integrate import solve_ivp
	from scipy.optimize import brentq

	n0 = params["photoionize.n_HI_init"]
	rho0 = n0 * SPECIES_MASS["H"]
	Q_ion, Q_opt = params["photoionize.flux_ion"], params["photoionize.flux_optical"]
	kappa_opt, kappa_rho0 = params["photoionize.kappa_optical"], params["network.dust_kappa"] * rho0
	eps_ion, eps_opt = band_energies()
	T_i, R_s, c_i = stromgren_scales(params)
	R_d = brentq(lambda R: R**3 * dust_attenuation(kappa_rho0 * R) - R_s**3, 1e-6 * R_s, R_s, xtol=1e-12 * R_s)
	L_opt = Q_opt * eps_opt
	L_ion = 0.0 if corrected else Q_ion * eps_ion
	L_rec = 0.0 if corrected else Q_ion * (13.6 * EV + lambda_rec(T_i) / alpha_B(T_i))
	P_0 = n0 * K_B * neutral_temperature(n0) if corrected else 0.0

	def density_ratio(R):  # ionization balance with dust: (R_s / R)^3 = s^2 h(kappa rho_0 s R)
		target = (R_s / R) ** 3
		return brentq(lambda s: s * s * dust_attenuation(kappa_rho0 * s * R) - target, 0.0, np.sqrt(target), xtol=1e-14, rtol=1e-14)

	def rhs(_, y):
		R, v = max(y[0], R_d), y[1]
		area, mass = 4.0 * np.pi * R * R, 4.0 / 3.0 * np.pi * R**3 * rho0
		s = density_ratio(R)
		tau = kappa_opt * rho0 * R * ((1.0 - s) / 3.0 + s if corrected else 1.0 / 3.0)
		L_abs = L_ion + (L_opt + L_rec * s * s * (R / R_s) ** 3) * -np.expm1(-tau)
		return [v, (area * (s * rho0 * c_i**2 - P_0) + L_abs / C_LIGHT - area * rho0 * v * v) / mass]

	sol = solve_ivp(rhs, (0.0, t[-1]), [R_d, c_i], t_eval=t, rtol=1e-10, atol=[1e-8 * R_s, 1e-8 * c_i])
	return np.maximum(sol.y[0], R_d)


def dtype3d(n, extra=(), name=None, plotfile_interval=10, **kwargs):
	return Run(
		name=name or f"n{n}",
		problem="DTypeFront3D",
		inputs="inputs/DTypeFront3D.toml",
		overrides=[f"amr.n_cell=[{n},{n},{n}]", f"plotfile_interval={plotfile_interval}", *extra],
		**kwargs,
	)


RUNS = [
	# Production run: 64^3 + 2 levels (256^3 effective), refined on the x_HI gradient only. One node, 8 GCDs.
	dtype3d(64, name="n64_l2", extra=["amr.max_level=2"], host="setonix", time="02:00:00"),
]

COLORS = {"n64_l2": "#0173B2"}
LABELS = {"n64_l2": r"$64^3$, 2 levels"}
REF = dict(color="k", ls=":", lw=1.2)
HI_REF = dict(color="k", ls="--", lw=1.0)


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


def dx_finest(params):
	"""Cell width of level amr.max_level (Quokka's default refinement ratio is 2)."""
	ratio = params.get("amr.ref_ratio", 2)
	ratio = ratio if isinstance(ratio, list) else [ratio] * params["amr.max_level"]
	dx0 = (params["geometry.prob_hi"][0] - params["geometry.prob_lo"][0]) / params["amr.n_cell"][0]
	return dx0 / np.prod(ratio[: params["amr.max_level"]])


def reduce(run, run_dir, out_dir):
	for name in ("inputs.toml", "run.log"):
		shutil.copy(run_dir / name, out_dir / name)
	params = read_inputs(run_dir)
	df = pd.read_csv(run_dir / CSV)
	d = drop_superseded(df.to_numpy())
	if not np.isclose(d[-1, 0], params["stop_time"], rtol=1e-5):
		print(f"  WARNING {run.name}: last row t = {d[-1, 0]:.6e} s, stop_time = {params['stop_time']:.6e} s (run unfinished?)")
	d = d[subsample(len(d))]
	r_ode = d[:, list(df.columns).index("r_ode")]
	err = np.max(np.abs(thin_shell(d[:, 0], params, corrected=False) / r_ode - 1.0))
	print(f"  uncorrected thin-shell ODE vs the problem's r_ode: max relative difference {err:.1e}")
	if err > 2e-5:  # the problem writes 6 significant digits
		print(f"  WARNING {run.name}: thin_shell(corrected=False) does not reproduce the problem's r_ode")
	r_thin_shell = thin_shell(d[:, 0], params)
	print(f"  corrected thin-shell radius at t = {d[-1, 0] / MYR:.4f} Myr: {r_thin_shell[-1] / PC:.4f} pc (problem's r_ode {r_ode[-1] / PC:.4f} pc)")
	d = np.column_stack([d, r_thin_shell])
	np.savetxt(out_dir / CSV, d, delimiter=",", header=",".join([*df.columns, "r_thin_shell"]), comments="", fmt="%.10e")

	import yt

	yt.set_log_level(40)
	plotfile = final_plotfile(run_dir, params["plotfile_prefix"])
	ds = yt.load(str(plotfile))
	t = float(ds.current_time.d)
	if not np.isclose(t, params["stop_time"], rtol=1e-6):
		print(f"  WARNING {run.name}: snapshot {plotfile.name} is at t = {t:.6e} s, not stop_time")
	# The source sits at the box centre, on a cell corner, so z = 0 is a cell face: take the plane of finest-level cells just above it.
	# (yt rejects a one-cell-thick covering grid away from the domain edge, hence two planes.)
	level = ds.index.max_level
	dims = ds.domain_dimensions * ds.refine_by**level
	left = ds.arr([ds.domain_left_edge[0].d, ds.domain_left_edge[1].d, 0.0], "code_length")
	grid = ds.covering_grid(level=level, left_edge=left, dims=[int(dims[0]), int(dims[1]), 2])
	n = {sp: grid[("boxlib", f)].d[:, :, 0] / SPECIES_MASS[sp] for sp, f in SCALARS.items()}
	eint = grid[("boxlib", "gasInternalEnergy")].d[:, :, 0]
	T = (2.0 / 3.0) * eint / (K_B * (n["H"] + n["H_p"] + n["e"]))  # every species has gamma = 5/3
	np.savez_compressed(
		out_dir / SNAPSHOT,
		x_HI=(n["H"] / (n["H"] + n["H_p"])).astype(np.float32),
		T=T.astype(np.float32),
		time=t,
		level=level,
		left_edge=ds.domain_left_edge.d,
		right_edge=ds.domain_right_edge.d,
		plotfile=plotfile.name,
	)
	print(f"  {plotfile.name}: t = {t / MYR:.4f} Myr, level {level} slice {T.shape} -> {SNAPSHOT}")


def load_run(results_dir, name):
	data = results_dir / name
	if not (data / CSV).exists():
		return None
	d = np.genfromtxt(data / CSV, delimiter=",", names=True)
	params = read_inputs(data)
	return {"t": d["time"], "r": d["r_effective"], "r_model": d["r_thin_shell"], "dx": dx_finest(params), "params": params}


def reduced_runs(results_dir):
	runs = {r.name: load_run(results_dir, r.name) for r in RUNS}
	return {k: v for k, v in runs.items() if v is not None}


def plot_radius(results_dir):
	runs = reduced_runs(results_dir)
	if not runs:
		print("skip dtype_front_3d_radius: no runs reduced")
		return
	# the model depends only on the physical setup, so draw it once, from the longest run
	ref = max(runs.values(), key=lambda r: r["t"][-1])

	fig, (ax1, ax2) = style.figure("column", aspect=1.05, nrows=2, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
	for name, run in runs.items():
		t = run["t"] / MYR
		ax1.plot(t, run["r"] / PC, color=COLORS[name], label=LABELS[name])
		ax2.plot(t, (run["r"] - run["r_model"]) / run["dx"], color=COLORS[name])
	ax1.plot(ref["t"] / MYR, ref["r_model"] / PC, **REF, label="thin-shell model")
	ax1.plot(ref["t"] / MYR, r_hosokawa_inutsuka(ref["t"], ref["params"]) / PC, **HI_REF, label="Hosokawa & Inutsuka")
	ax1.set_ylabel(r"$R_{\rm I}$ (pc)")
	ax1.set_ylim(bottom=0)
	ax1.legend(loc="upper left")
	ax2.axhline(0, **REF)
	ax2.set_ylabel(r"$(R_{\rm I} - R_{\rm I}^{\rm ref}) / \Delta x_{\rm min}$")
	ax2.set_xlabel(r"$t$ (Myr)")
	ax2.set_xlim(0, ref["t"][-1] / MYR)
	fig.tight_layout()
	style.save(fig, "dtype_front_3d_radius")


def plot_snapshot(results_dir):
	runs = reduced_runs(results_dir)
	name = min((k for k in runs if (results_dir / k / SNAPSHOT).exists()), key=lambda k: runs[k]["dx"], default=None)
	if name is None:
		print("skip dtype_front_3d_snapshot: no snapshot reduced")
		return
	snap = np.load(results_dir / name / SNAPSHOT)
	run = runs[name]
	lo, hi = snap["left_edge"] / PC, snap["right_edge"] / PC
	extent = [lo[0], hi[0], lo[1], hi[1]]
	r_model = np.interp(float(snap["time"]), run["t"], run["r_model"]) / PC
	theta = np.linspace(0.0, 2.0 * np.pi, 400)

	from matplotlib.colors import LogNorm, Normalize

	fig, axes = style.figure("column", aspect=1.75, nrows=2, sharex=True)
	panels = [
		(snap["x_HI"], r"$x_{\rm H^0}$", "Blues", Normalize(vmin=0.0, vmax=1.0)),
		(snap["T"], r"$T$ (K)", "inferno", LogNorm()),
	]
	for ax, (field, label, cmap, norm) in zip(axes, panels):
		im = ax.imshow(field.T, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
		ax.set_xlim(lo[0], hi[0])
		ax.set_ylim(lo[1], hi[1])
		ax.set_aspect("equal")
		ax.set_ylabel(r"$y$ (pc)")
		style.colorbar(fig, im, ax, label)
	axes[1].set_xlabel(r"$x$ (pc)")
	axes[0].plot(r_model * np.cos(theta), r_model * np.sin(theta), **REF, label="thin-shell model")
	r_hi = r_hosokawa_inutsuka(float(snap["time"]), run["params"]) / PC
	axes[0].plot(r_hi * np.cos(theta), r_hi * np.sin(theta), **HI_REF, label="Hosokawa & Inutsuka")
	axes[0].legend(loc="upper right", facecolor="white", framealpha=0.8, frameon=True, edgecolor="none")
	print(f"snapshot {name}: t = {snap['time'] / MYR:.4f} Myr, level {int(snap['level'])}")
	fig.tight_layout()
	style.save(fig, "dtype_front_3d_snapshot")


def plot(results_dir):
	plot_radius(results_dir)
	plot_snapshot(results_dir)


if __name__ == "__main__":
	main(FIGURE, RUNS, reduce, plot)
