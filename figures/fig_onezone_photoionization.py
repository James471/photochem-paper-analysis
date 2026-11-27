#!/usr/bin/env python3
"""Zero-dimensional photoionization test (OneZonePhotoionizationWithEnergy) vs the Julia reference.

    python figures/fig_onezone_photoionization.py build          # 1D Release build (config.yaml builds.1d)
    python figures/fig_onezone_photoionization.py run --launch   # Julia reference + Quokka run (Mac, seconds)
    python figures/fig_onezone_photoionization.py reduce
    python figures/fig_onezone_photoionization.py plot           # onezone_species_photons.pdf, onezone_errors.pdf

The Quokka test reads the Julia CSV from ../extern/photoionization-julia/ relative to its working directory,
so `setup` writes a fresh Julia reference next to the run directories (the Quokka repo's copy is never touched).
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from photochem import Run, load_config, main, style  # noqa: E402
from photochem.build import source_dir  # noqa: E402

FIGURE = "onezone"
RUNS = [Run(name="fiducial", problem="OneZonePhotoionization", inputs="inputs/OneZonePhotoionizationWithEnergy.toml", dim="1d", ntasks=1)]

QUOKKA_CSV = "photoionization_quokka_output.csv"
JULIA_CSV = "with_energy.csv"
# n_photon n_e n_HI n_HII gamma_e gamma_HI gamma_HII T energy_switch tend dt abstol reltol
JULIA_ARGS = ["5e3", "1e-10", "1e2", "1e-10", "5/3", "5/3", "5/3", "1e3", "1", "10000", "50", "1e-14", "1e-14"]

COL_E, COL_HI, COL_HII, COL_NG, COL_T = "#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#D55E00"


def julia_csv(run_dir):
	return run_dir.parent / "extern" / "photoionization-julia" / JULIA_CSV


def setup(run, run_dir):
	script = source_dir(run, load_config()[1]) / "extern" / "photoionization-julia" / "fraction.jl"
	out = julia_csv(run_dir)
	out.parent.mkdir(parents=True, exist_ok=True)
	with open(run_dir.parent / "julia.log", "w") as log:
		subprocess.run(["julia", str(script), *JULIA_ARGS, str(out)], check=True, stdout=log, stderr=subprocess.STDOUT)
	print(f"  julia reference: {out}")


def reduce(run, run_dir, out_dir):
	shutil.copy(run_dir / QUOKKA_CSV, out_dir / QUOKKA_CSV)
	shutil.copy(julia_csv(run_dir), out_dir / JULIA_CSV)


def load(path):
	d = np.genfromtxt(path, delimiter=",", names=True)
	return {k: d[k] for k in d.dtype.names}


def plot(results_dir):
	data = results_dir / RUNS[0].name
	q, j = load(data / QUOKKA_CSV), load(data / JULIA_CSV)
	if len(q["time"]) != len(j["time"]) or not np.allclose(q["time"], j["time"], rtol=1e-10, atol=0.0):
		sys.exit(f"Quokka ({len(q['time'])} rows) and Julia ({len(j['time'])} rows) time grids differ")
	t = q["time"]

	species = [("n_HI", r"$n_{\rm H^0}$", COL_HI, "-"), ("n_HII", r"$n_{\rm H^+}$", COL_HII, "-"), ("n_e", r"$n_e$", COL_E, "--")]
	ref = dict(color="k", ls=":", lw=1.2)

	fig, (ax1, ax2, ax3) = style.figure("column", aspect=6.6 / 3.5, nrows=3, sharex=True)
	for key, lab, col, ls in species:
		ax1.plot(t, q[key], color=col, ls=ls, label=lab)
	for key, _, _, _ in species:
		ax1.plot(t, j[key], **ref, label="Julia" if key == "n_HI" else None)
	ax1.set_yscale("log")
	ax1.set_ylabel(r"Number density [cm$^{-3}$]")
	ax1.set_ylim(1e-1, 3e2)
	ax1.legend(loc="lower center", ncol=2)

	ax2.plot(t, q["n_gamma"], color=COL_NG, label=r"$n_\gamma$")
	ax2.plot(t, j["n_gamma"], **ref, label="Julia")
	ax2.set_ylabel(r"Photon number density [cm$^{-3}$]")
	ax2.legend(loc="best")

	ax3.plot(t, q["gas_temp"], color=COL_T, label=r"$T$")
	ax3.plot(t, j["gas_temp"], **ref, label="Julia")
	ax3.set_ylabel("Gas temperature [K]")
	ax3.legend(loc="best")
	ax3.set_xlabel("Time [s]")
	ax3.set_xlim(t[0], t[-1])
	fig.tight_layout()
	style.save(fig, "onezone_species_photons")

	def relerr(key):
		return np.maximum(np.abs(q[key] - j[key]) / np.abs(j[key]), 1e-17)

	fig, (ax1, ax2, ax3) = style.figure("column", aspect=6.6 / 3.5, nrows=3, sharex=True)
	for key, lab, col, ls in species:
		ax1.plot(t[1:], relerr(key)[1:], color=col, ls=ls, label=lab)
	ax1.set_ylabel("Relative error (species)")
	ax1.legend(loc="best")
	ax2.plot(t[1:], relerr("n_gamma")[1:], color=COL_NG, label=r"$n_\gamma$")
	ax2.set_ylabel("Relative error (photons)")
	ax2.legend(loc="best")
	ax3.plot(t[1:], relerr("gas_temp")[1:], color=COL_T, label=r"$T$")
	ax3.set_ylabel("Relative error (temperature)")
	ax3.legend(loc="best")
	for ax in (ax1, ax2, ax3):
		ax.set_yscale("log")
	ax3.set_xlabel("Time [s]")
	ax3.set_xlim(t[0], t[-1])
	fig.tight_layout()
	style.save(fig, "onezone_errors")

	for key in ("n_e", "n_HI", "n_HII", "n_gamma", "gas_temp"):
		print(f"L1 relative error {key}: {np.mean(relerr(key)):.3e}")


if __name__ == "__main__":
	main(FIGURE, RUNS, reduce, plot, setup=setup)
