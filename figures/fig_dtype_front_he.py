#!/usr/bin/env python3
"""D-type front with the H + He network (DTypeFrontHe, inputs/DTypeFrontHe.toml): ionization fractions and temperature in
the plane of cells just above the source at the final plotfile.

    python figures/fig_dtype_front_he.py build [RUN ...]
    python figures/fig_dtype_front_he.py run [RUN ...] --launch
    python figures/fig_dtype_front_he.py reduce [RUN ...]
    python figures/fig_dtype_front_he.py plot

Figure (from PLOTTED, the 128^3 production run):
    dtype_front_he_snapshot.pdf   x_HI, x_HII, x_HeI, x_HeII, x_HeIII and T (2 x 3)

The source sits at the centre of the box (reflecting boundaries), on a cell corner. x_HI = n_H / (n_H + n_H+),
x_HeI = n_He / (n_He + n_He+ + n_He++), etc.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from photochem import Run, main, read_inputs, style  # noqa: E402

FIGURE = "dtype_front_he"
CSV = "dtype_front_radii.csv"
SNAPSHOT = "slice.npz"

K_B = 1.3806490000000002e-16  # Microphysics fundamental_constants.H
PC = 3.085677581491367e18
MYR = 3.15576e13
# helium network: species order and masses from Helium::SpeciesMass() in src/networks/helium/network.hpp
SPECIES = {"H": 1.673773e-24, "H+": 1.6728620616289998e-24, "e-": 9.10938371e-28, "He": 6.646473e-24, "He+": 6.645562061629e-24, "He++": 6.644651123258e-24}


def he_run(n, name=None, extra=(), **kwargs):
	return Run(name=name or f"n{n}", problem="DTypeFrontHe", inputs="inputs/DTypeFrontHe.toml", overrides=[f"amr.n_cell=[{n},{n},{n}]", *extra], **kwargs)


RUNS = [
	# 128^3 in one box on one GCD, centred source in a 5 pc box (~420 steps). Plotfiles every 100 steps (toml).
	he_run(128, host="setonix", ngpus=1, partition="gpu", time="01:00:00"),
]
PLOTTED = "n128"

PANELS = [
	("x_HI", r"$x_{\rm H^0}$"),
	("x_HII", r"$x_{\rm H^+}$"),
	("x_HeI", r"$x_{\rm He^0}$"),
	("x_HeII", r"$x_{\rm He^+}$"),
	("x_HeIII", r"$x_{\rm He^{2+}}$"),
	("T", r"$T$ ($10^4$ K)"),
]
T_MIN = 1e4  # T panel: linear from here up, so the temperature gradient across the ionized gas is visible; colder gas saturates


def final_plotfile(run_dir, prefix):
	plotfiles = [p for p in run_dir.glob(f"{prefix}*") if p.is_dir() and (p / "Header").exists()]
	if not plotfiles:
		raise SystemExit(f"no plotfile {prefix}* in {run_dir}")
	return max(plotfiles, key=lambda p: int(p.name[len(prefix) :]))


def reduce(run, run_dir, out_dir):
	for name in ("inputs.toml", CSV):
		shutil.copy(run_dir / name, out_dir / name)
	lines = (run_dir / "run.log").read_text().splitlines()
	(out_dir / "run_tail.log").write_text("\n".join(lines[-200:]) + "\n")
	params = read_inputs(run_dir)

	import yt

	yt.set_log_level(40)
	plotfile = final_plotfile(run_dir, params["plotfile_prefix"])
	ds = yt.load(str(plotfile))
	t = float(ds.current_time.d)
	if not np.isclose(t, params["stop_time"], rtol=1e-6):
		print(f"  WARNING {run.name}: snapshot {plotfile.name} is at t = {t:.6e} s, not stop_time = {params['stop_time']:.6e} s")
	# The source sits on a cell corner at the box centre: take the plane of cells just above it.
	# (yt rejects a one-cell-thick covering grid away from the domain edge, hence two planes.)
	z_mid = 0.5 * (ds.domain_left_edge[2].d + ds.domain_right_edge[2].d)
	left = ds.arr([ds.domain_left_edge[0].d, ds.domain_left_edge[1].d, z_mid], "code_length")
	grid = ds.covering_grid(level=0, left_edge=left, dims=[*ds.domain_dimensions[:2], 2])
	n = {sp: grid[("boxlib", f"scalar_{i}")].d[:, :, 0] / m for i, (sp, m) in enumerate(SPECIES.items())}
	n_H = n["H"] + n["H+"]
	n_He = n["He"] + n["He+"] + n["He++"]
	eint = grid[("boxlib", "gasInternalEnergy")].d[:, :, 0]
	fields = {
		"x_HI": n["H"] / n_H,
		"x_HII": n["H+"] / n_H,
		"x_HeI": n["He"] / n_He,
		"x_HeII": n["He+"] / n_He,
		"x_HeIII": n["He++"] / n_He,
		"T": (2.0 / 3.0) * eint / (K_B * sum(n.values())),  # every species has gamma = 5/3
	}
	np.savez_compressed(
		out_dir / SNAPSHOT,
		**{k: v.astype(np.float32) for k, v in fields.items()},
		time=t,
		left_edge=ds.domain_left_edge.d,
		right_edge=ds.domain_right_edge.d,
		plotfile=plotfile.name,
	)
	print(f"  {plotfile.name}: t = {t / MYR:.4f} Myr, slice {eint.shape} -> {SNAPSHOT}")


def plot(results_dir):
	from matplotlib.colors import Normalize

	if not (results_dir / PLOTTED / SNAPSHOT).exists():
		print(f"skip {FIGURE}_snapshot: {PLOTTED} not reduced")
		return
	snap = np.load(results_dir / PLOTTED / SNAPSHOT)
	lo, hi = snap["left_edge"] / PC, snap["right_edge"] / PC
	extent = [lo[0], hi[0], lo[1], hi[1]]

	fig, axes = style.figure("page", aspect=0.74, nrows=2, ncols=3, sharex=True, sharey=True)
	for ax, (key, label) in zip(axes.flat, PANELS):
		data = snap[key]
		if key == "T":
			data = data / 1e4
			cmap, norm = "inferno", Normalize(vmin=T_MIN / 1e4, vmax=float(data.max()))
		else:
			cmap, norm = "viridis", Normalize(vmin=0.0, vmax=1.0)
		im = ax.imshow(data.T, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
		ax.set_xlim(lo[0], hi[0])
		ax.set_ylim(lo[1], hi[1])
		ax.set_aspect("equal")
		cbar = style.colorbar(fig, im, ax, label, location="top", **({"extend": "min"} if key == "T" else {}))
		if key != "T":
			cbar.set_ticks([0.0, 0.5, 1.0])
	for ax in axes[1]:
		ax.set_xlabel(r"$x$ (pc)")
	for ax in axes[:, 0]:
		ax.set_ylabel(r"$y$ (pc)")
	fig.subplots_adjust(hspace=0.4, wspace=0.12)  # not tight_layout: it ignores the inset colorbars above each panel
	style.save(fig, f"{FIGURE}_snapshot")
	print(f"snapshot {snap['plotfile']}: t = {float(snap['time']) / MYR:.4f} Myr")


if __name__ == "__main__":
	main(FIGURE, RUNS, reduce, plot)
