#!/usr/bin/env python3
"""Paper Fig. 3 (choice of integrator, paper/Figures/integrator_choice.pdf): wall time of one radiation subcycle per cell for the four
Rosenbrock tableaus on DTypeFront_JAFF_Centered (the source in the middle of [-6 r_s, 6 r_s]^3, one 128^3 box on one Setonix GCD,
c_hat = c/1000, paper tolerances), against the same subcycle without photochemistry (radiation transport only).

    python sync.py remote -- python figures/fig_integrator_choice.py build
    python sync.py remote -- python figures/fig_integrator_choice.py run --batch 00:30:00 --launch   # gpu partition
    python sync.py remote -- python figures/fig_integrator_choice.py reduce
    python sync.py pull integrator_choice
    python figures/fig_integrator_choice.py plot

Tolerances (from fig_tolerance_scaled): rtol = 1e-2 for all variables and atol_i = rtol * s_i with s_X = 1e-3 cm^-3,
s_e = 1e8 erg/g, s_gamma = 0.1 cm^-3 (atol 1e-5 cm^-3, 1e6 erg/g, 1e-3 cm^-3); failure tolerances 5 atol.

Timing. Restarting from a checkpoint does not work for this problem (it sets its parameters and calls eos_init() / network_init() in
preCalculateInitialConditions(), which Quokka only calls when starting from scratch), so every run starts from scratch, without
plotfiles, and prints per-step wall times (print_cycle_timing=1). The run stops at step MAX_STEPS; the cycle time printed with
"Coarse STEP k" is the wall time of step k-1. The timed steps are the WINDOW x N_WINDOWS steps from the first step that starts at
t >= t_s, in N_WINDOWS consecutive windows.

Metric. The hydro time step depends on the flow, so the cost per hydro step is not comparable between runs in general. Instead:
cost of one radiation subcycle per cell = (window time - WINDOW x hydro time per step) / (subcycles in the window) / 128^3, with the
hydro time per step the mean window time of the run without radiation (<tag>_c<divisor>_norad). Ratio = that cost / the cost without
photochemistry. Reported: mean and standard deviation over the windows.

Baselines: a build without photochemistry (<prefix>_nophotochem, -DDTypeFront_JAFF_NO_PHOTOCHEMISTRY=ON) and a build without
radiation, i.e. hydro with the species as passive scalars (<prefix>_norad). Their gas does not evolve, so they run with no stop
time, to MAX_STEPS.

Radiation subcycles per step, N = ceil(dt / (0.3 dx / c_hat)) (QuokkaSimulation::computeNumberOfRadiationSubsteps), from the step's
dt in the log. Steps with dt / dt_rad an integer to 1e-6 are ambiguous: in floating point Quokka's ceil gives n or n + 1. They are
resolved together by matching Quokka's whole-run average ("rad_subcycles").
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from photochem import main, style  # noqa: E402
from fig_integrator_study import C_HAT_DIVISOR, COLORS, LABELS, MARKERS, TABLEAU, dtype_run, load, ns_per_subcycle, reduce  # noqa: E402

FIGURE = "integrator_choice"
N = 128
RTOL = 1e-2
SCALE = {"spec": 1e-3, "enuc": 1e8, "rad_num": 0.1}
ATOL = {v: RTOL * s for v, s in SCALE.items()}
FAILURE = {"species": 5 * ATOL["spec"], "radiation": 5 * ATOL["rad_num"]}
T_S = 1.0215474958e12  # the problem's t_s (stop_time = 5 t_s)
C_LIGHT = 2.99792458e10
RAD_CFL = 0.3
WINDOW = 10
N_WINDOWS = 10
MAX_STEPS = 400  # must exceed the step at t_s plus WINDOW x N_WINDOWS
BOX = 2 * 6.958844100e18  # inputs/DTypeFront_JAFF_Centered.toml: [-6 r_s, 6 r_s]


@dataclasses.dataclass(frozen=True)
class Setup:
	"""A version of the D-type problem: where its results and PDF go, which problem and toml it runs, its box side (cm), and the run
	and variant name prefix (tag)."""

	figure: str
	problem: str
	inputs: str
	box: float
	tag: str

	def save(self, fig):
		style.save(fig, self.figure)

	def prefix(self, divisor):
		return f"{self.tag}_c{divisor}"

	@property
	def hydro_only(self):  # no radiation, so shared by every c_hat
		return f"{self.tag}_c{C_HAT_DIVISOR}_norad"


SETUP = Setup(FIGURE, "DTypeFront_JAFF_Centered", "inputs/DTypeFront_JAFF_Centered.toml", BOX, "cen")


def crun(name, integrator, photochem=True, radiation=True):
	extra = [f"max_timesteps={MAX_STEPS}", "print_cycle_timing=1"]
	if not (photochem and radiation):
		extra += ["stop_time=1e30"]  # nothing evolves: stop at MAX_STEPS
	return dtype_run(name, integrator, RTOL, radiation=radiation, photochemistry=photochem, tag=SETUP.tag, scale=ATOL, failure=FAILURE,
			 atol_eps=1.0, n=N, plotfiles=False, extra=extra, problem=SETUP.problem, inputs=SETUP.inputs, partition="gpu", time="00:30:00")


PREFIX = SETUP.prefix(C_HAT_DIVISOR)
RT_ONLY = f"{PREFIX}_nophotochem"
RUNS = [
	*(crun(f"{PREFIX}_{i}", i) for i in TABLEAU),
	crun(RT_ONLY, "ros2s", photochem=False),
	crun(SETUP.hydro_only, "ros2s", radiation=False),
]


def window(s, radiation=True):
	"""Wall time of each window, and the radiation subcycles of each timed step, or None if the run did not reach the last window."""
	c = np.asarray(s.get("cycle_times", []))  # c[i]: wall time of step i (c[0] is not a step)
	t = np.asarray(s.get("step_t", []))  # t[i]: time at which step i + 1 starts
	after = np.nonzero(t >= T_S)[0]
	if not len(after):
		return None
	first = after[0] + 1
	last = first + WINDOW * N_WINDOWS
	if len(c) < last or len(t) < last:
		return None
	if not radiation:
		sub = np.zeros(len(t) - 1)
	else:
		q = np.diff(t) / (RAD_CFL * (SETUP.box / N) / (C_LIGHT / C_HAT_DIVISOR))  # q[i]: dt / dt_rad of step i + 1
		ambiguous = np.abs(q - np.round(q)) < 1e-6
		sub = np.where(ambiguous, np.round(q), np.ceil(q))
		if abs((sub + ambiguous).mean() - s["rad_subcycles"]) < abs(sub.mean() - s["rad_subcycles"]):
			sub = sub + ambiguous
	return {
		"first": int(first),
		"times": c[first:last].reshape(N_WINDOWS, WINDOW).sum(axis=1),
		"subcycles": sub[first - 1 : last - 1].reshape(N_WINDOWS, WINDOW).sum(axis=1),
		"subcycles_run": sub.mean() if len(sub) else np.nan,
	}


def table(results_dir):
	hydro_only = SETUP.hydro_only
	wins = {}
	for run in RUNS:
		s = load(results_dir, run.name)
		if s is not None and s["finished"]:
			w = window(s, radiation=run.name != hydro_only)
			if w is not None:
				wins[run.name] = (w, s)
	if hydro_only not in wins:
		print(f"skip {FIGURE}: {hydro_only} (hydro time per step) not reduced")
		return pd.DataFrame()
	hydro = wins[hydro_only][0]["times"].mean() / WINDOW  # wall time of one hydro step
	rows = []
	for name, (w, s) in wins.items():
		row = {
			"run": name,
			"first_step": w["first"],
			"step_ms": w["times"].mean() / WINDOW * 1e3,
			"subcycles_per_step": w["subcycles"].mean() / WINDOW,
			"subcycles_run_reconstructed": w["subcycles_run"],
			"rad_subcycles_run": s.get("rad_subcycles", np.nan),
			"hydro_ms_profiler": s["hydro_s"] / s["step_ncalls"] * 1e3 if "hydro_s" in s else np.nan,
		}
		if name != hydro_only:
			cost = (w["times"] - WINDOW * hydro) / w["subcycles"] / N**3 * 1e9  # ns per cell per subcycle, per window
			row["ns_per_subcycle"], row["ns_per_subcycle_std"] = cost.mean(), cost.std(ddof=1)
			row["ns_per_subcycle_profiler"] = ns_per_subcycle(s)
			# the photochemistry burn alone (TinyProfiler, whole run)
			if "photochem_s" in s:
				row["ns_burn_profiler"] = s["photochem_s"] / (s["step_ncalls"] * s["rad_subcycles"]) / N**3 * 1e9
		rows.append(row)
	df = pd.DataFrame(rows).set_index("run")
	if RT_ONLY in df.index:
		base = df.loc[RT_ONLY, "ns_per_subcycle"]
		df["ratio"], df["ratio_std"] = df["ns_per_subcycle"] / base, df["ns_per_subcycle_std"] / base
		df["ratio_profiler"] = df["ns_per_subcycle_profiler"] / df.loc[RT_ONLY, "ns_per_subcycle_profiler"]
	df = df.reset_index()
	df.to_csv(results_dir / f"{FIGURE}.csv", index=False, float_format="%.6e")
	print(df.to_string(index=False))
	return df


def plot(results_dir):
	df = table(results_dir)
	ints = [i for i in TABLEAU if f"{PREFIX}_{i}" in set(df.get("run", []))]
	if not ints:
		print(f"skip {FIGURE}: nothing reduced")
		return
	by = df.set_index("run")
	fig, axes = style.figure("column", aspect=1.0, nrows=2, sharex=True)
	for k, i in enumerate(ints):
		r = f"{PREFIX}_{i}"
		axes[0].errorbar(k, by.loc[r, "ns_per_subcycle"], yerr=by.loc[r, "ns_per_subcycle_std"], fmt=MARKERS[i], color=COLORS[i], ms=5, capsize=2)
		if "ratio" in by:
			axes[1].bar(k, by.loc[r, "ratio"], yerr=by.loc[r, "ratio_std"], width=0.6, color=COLORS[i], capsize=2)
	top = max(by.loc[[f"{PREFIX}_{i}" for i in ints], "ns_per_subcycle"])
	if RT_ONLY in by.index:
		base = by.loc[RT_ONLY, "ns_per_subcycle"]
		axes[0].axhline(base, color="k", ls=":", lw=1.2)
		axes[0].text(len(ints) - 0.5, base - 0.03 * top, "Radiation transport only", ha="right", va="top")
		axes[1].axhline(1.0, color="k", ls=":", lw=1.2)
	axes[0].set_ylim(0.0, 1.15 * top)
	axes[0].set_ylabel(r"$\tau_{\rm sub}$ (ns)")  # wall time of one radiation subcycle per cell, defined in the caption
	axes[1].set_ylabel("Slowdown Factor")  # tau_sub / tau_RT
	axes[1].set_xticks(range(len(ints)), [LABELS[i] for i in ints])
	axes[1].set_xlim(-0.6, len(ints) - 0.4)
	fig.tight_layout()
	fig.align_ylabels(axes)
	SETUP.save(fig)


if __name__ == "__main__":
	main(FIGURE, RUNS, reduce, plot)
