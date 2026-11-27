#!/usr/bin/env python3
"""Paper Fig. 2 (choice of tolerance, paper/Figures/tolerance_scaled.pdf): ROS2S on DTypeFront_JAFF_Centered (the source in the
middle of [-6 r_s, 6 r_s]^3, one 128^3 box on one Setonix GCD, c_hat = c/1000, to 5 t_s) with the absolute tolerances tied to the
relative one, atol_i = rtol * s_i, swept over rtol.

    python sync.py remote -- python figures/fig_tolerance_scaled.py build
    python sync.py remote -- python figures/fig_tolerance_scaled.py run --batch 00:30:00 --launch   # gpu partition
    python sync.py remote -- python figures/fig_tolerance_scaled.py reduce
    python sync.py pull tolerance_scaled
    python figures/fig_tolerance_scaled.py plot

Scales (the paper's): s_X = 1e-3 cm^-3, s_e = 1e8 erg/g, s_gamma = 0.1 cm^-3 (stored photon density). The same rtol applies to all
variables; failure tolerances are 5 atol. Metrics at 5 t_s: |R - R_HI| / dx (fig_tolerance_choice.metrics), and the whole-run wall
time of one radiation subcycle per cell (ns_per_subcycle in fig_integrator_study: TinyProfiler time stepping minus hydro, over
steps x Quokka's average subcycles per step), tau_sub in the plot. Mupdates/s per hydro step is kept in the table.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from photochem import main, style  # noqa: E402
from fig_integrator_choice import BOX, Setup  # noqa: E402
from fig_integrator_study import C_HAT_DIVISOR, dtype_run, load, ns_per_subcycle, reduce  # noqa: E402
from fig_tolerance_choice import metrics  # noqa: E402

FIGURE = "tolerance_scaled"
N = 128
SCALE = {"spec": 1e-3, "enuc": 1e8, "rad_num": 0.1}
RTOLS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
SETUP = Setup(FIGURE, "DTypeFront_JAFF_Centered", "inputs/DTypeFront_JAFF_Centered.toml", BOX, "cen")


def srun(rtol):
	failure = {"species": 5 * rtol * SCALE["spec"], "radiation": 5 * rtol * SCALE["rad_num"]}
	return dtype_run(f"{SETUP.prefix(C_HAT_DIVISOR)}_n{N}_rtol_{rtol:g}", "ros2s", rtol, tag=SETUP.tag, scale=SCALE, failure=failure, n=N,
			 problem=SETUP.problem, inputs=SETUP.inputs, partition="gpu", time="00:30:00")


RUNS = [srun(r) for r in RTOLS]


def table(results_dir):
	rows = []
	for r, run in zip(RTOLS, RUNS):
		s = load(results_dir, run.name)
		if s is not None:
			rows.append({"rtol": r, **metrics(s), "ns_per_subcycle": ns_per_subcycle(s), "rad_subcycles": s.get("rad_subcycles", float("nan"))})
	df = pd.DataFrame(rows)
	if len(df):
		df.to_csv(results_dir / f"{FIGURE}.csv", index=False, float_format="%.6e")
		print(df.to_string(index=False))
	return df


def log_label(y, _):
	"""Plain-number tick label on a log axis, only for mantissas 1, 2, 3, 5, 6 and 8 (the others crowd)."""
	m = y / 10 ** math.floor(math.log10(y))
	return f"{y:g}" if min(abs(m - k) for k in (1, 2, 3, 5, 6, 8)) < 1e-6 else ""


def plot(results_dir):
	df = table(results_dir)
	rows = df[df["finished"] == True].sort_values("rtol") if len(df) else df  # noqa: E712
	if not len(rows):
		print(f"skip {FIGURE}: nothing reduced")
		return
	fig, axes = style.figure("column", aspect=1.0, nrows=2, sharex=True)
	for ax, col in zip(axes, ("front_err_cells", "ns_per_subcycle")):
		ax.plot(rows["rtol"], rows[col], "o-", color="#2a78d6", ms=3.5, lw=1.2)
	axes[0].set_ylim(0.0, 1.25 * rows["front_err_cells"].max())  # from zero, so that a small spread is not magnified
	axes[0].set_ylabel(r"$|R_{\rm I} - R_{\rm I}^{\rm ref}| / \Delta x$")
	axes[1].set_yscale("log")
	axes[1].margins(y=0.15)
	for axis in (axes[1].yaxis.set_major_formatter, axes[1].yaxis.set_minor_formatter):  # plain numbers, not 4x10^0
		axis(FuncFormatter(log_label))
	axes[1].set_ylabel(r"$\tau_{\rm sub}$ (ns)")  # wall time of one radiation subcycle per cell, defined in the caption
	axes[1].set_xscale("log")
	axes[1].set_xlabel(r"$\epsilon_{\rm rel}$ (with $\epsilon_{{\rm abs},i} = \epsilon_{\rm rel}\,s_i$)")
	fig.tight_layout()
	fig.align_ylabels(axes)
	SETUP.save(fig)


if __name__ == "__main__":
	main(FIGURE, RUNS, reduce, plot)
