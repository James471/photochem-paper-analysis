#!/usr/bin/env python3
"""Photon conservation across a moving ionization front (StromgrenSpherePhotonConservation, c_hat = c, no recombination).

    python figures/fig_stromgren_front_photons.py build           # on Setonix
    python figures/fig_stromgren_front_photons.py run --launch    # on Setonix
    python figures/fig_stromgren_front_photons.py reduce          # on Setonix
    python figures/fig_stromgren_front_photons.py plot            # stromgren_front_photons.pdf

Runs to t_end = 50 t_c, with R_c = sqrt(Q / (16 pi n c)) and t_c = R_c / c; the front then sits at 7.96 R_c = 0.89 L,
moving at c/17 with 16% of the injected photons still in flight. Panels: r_effective and the point-source radius; the
relative photon conservation error per cell.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from photochem import Run, main, read_inputs, style  # noqa: E402

FIGURE = "stromgren_front_photons"
CSV = "stromgren_sphere_photon_conservation.csv"

C_LIGHT = 2.99792458e10

# The Wendland C2 source at the centre of [-L, L]^3, one 128^3 box on one GCD (~1200 steps); the figure only needs the per-step CSV
RUNS = [
	Run(
		name="front128_ctr",
		problem="StromgrenSpherePhotonConservation",
		inputs="inputs/StromgrenSpherePhotonConservation.toml",
		overrides=["amr.n_cell=[128,128,128]", "amr.max_level=0", "amr.max_grid_size=128", "amr.blocking_factor=32", "plotfile_interval=500"],
		host="setonix",
		ngpus=1,
		partition="gpu",
		time="00:30:00",
	)
]
PLOT_RUN = "front128_ctr"

COL_NUM, COL_LOST, COL_GAINED = "#0173B2", "#D55E00", "#029E73"


def reduce(run, run_dir, out_dir):
	for name in (CSV, "inputs.toml", "run.log"):
		shutil.copy(run_dir / name, out_dir / name)


def plot(results_dir):
	data = results_dir / PLOT_RUN
	d = np.genfromtxt(data / CSV, delimiter=",", names=True)
	p = read_inputs(data)
	R_c = np.sqrt(p["stromgen.Q"] / (16.0 * np.pi * p["stromgen.n_HI_init"] * C_LIGHT))
	t = d["time"] * C_LIGHT / R_c
	n_cells = np.prod(p["amr.n_cell"])

	fig, (ax1, ax2) = style.figure("column", aspect=1.2, nrows=2, sharex=True)
	ax1.plot(t, d["r_effective"] / R_c, color=COL_NUM, label="simulation")
	ax1.plot(t, d["r_analytical"] / R_c, color="k", ls=":", lw=1.2, label="analytical")
	ax1.set_ylabel(r"$R_{\rm I} / R_{\rm c}$")
	ax1.legend(loc="lower right")

	# err > 0: fewer photons accounted for than injected (a loss); err < 0: a gain. Plotted separately, never folded together.
	err = d["rel_err"]
	# lost, gained = np.where(err > 0, err, np.nan), np.where(err < 0, -err, np.nan)
	ax2.plot(t[1:], np.abs(err[1:]), color="k", marker=".", ms=1.0, ls="none")  # the first step (|err| ~ 1e-15) only stretches the axis
	# ax2.plot(t, lost, color=COL_LOST, marker="o", ms=1.0, ls="none", label="photons lost")
	# ax2.plot(t, gained, color=COL_GAINED, marker="s", ms=1.0, ls="none", label="photons gained")
	ax2.set_yscale("log")
	ax2.set_ylabel("|Relative error|")
	ax2.set_xlabel(r"$t / t_{\rm c}$")
	# ax2.set_xlim(0, t[-1])
	# ax2.set_ylim(top=1e-19)
	# ax2.legend(loc="upper left", ncol=2, markerscale=3)
	fig.tight_layout()
	style.save(fig, FIGURE)

	print(f"max |relative error| = {np.max(np.abs(d['rel_err'])):.3e} over {len(err)} steps ({n_cells} cells); final = {d['rel_err'][-1]:.3e}")


if __name__ == "__main__":
	main(FIGURE, RUNS, reduce, plot)
