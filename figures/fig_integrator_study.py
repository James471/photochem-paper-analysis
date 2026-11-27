"""Shared by fig_tolerance_scaled and fig_integrator_choice (paper Figs. 2 and 3): the D-type run builder, `reduce` and the
log/timing helpers. Not a figure script.

Runs: DTypeFront_JAFF_Centered, one n^3 box (no AMR) on one Setonix GCD, at the paper's pinned commit (config.yaml `paper`), one
build variant per (tableau, c_hat). photochemistry=False builds with -DDTypeFront_JAFF_NO_PHOTOCHEMISTRY=ON (the chemical band is
transported and never absorbed); radiation=False with -DDTypeFront_JAFF_NO_RADIATION=ON (hydro with the species as passive scalars).
"""

import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from photochem import Run, read_inputs  # noqa: E402

ODE_MAX_STEPS = 500000
TABLEAU = ["ros2s", "rodas3p", "rodas4p", "rodas5p"]
LABELS = {"ros2s": "ROS2S", "rodas3p": "Rodas3P", "rodas4p": "Rodas4P", "rodas5p": "Rodas5P"}
COLORS = {"ros2s": "#2a78d6", "rodas3p": "#eb6834", "rodas4p": "#1baf7a", "rodas5p": "#eda100"}
MARKERS = {"ros2s": "o", "rodas3p": "s", "rodas4p": "^", "rodas5p": "D"}
C_HAT_DIVISOR = 1000


def tol_overrides(rtol, scale, failure, atol_eps=None):
	out = [f"integrator.rtol_{v}={rtol:g}" for v in ("spec", "enuc", "rad_num")]
	out += [f"integrator.atol_{v}={(atol_eps or rtol) * scale[v]:g}" for v in ("spec", "enuc", "rad_num")]
	return out + [
		f"integrator.species_failure_tolerance={failure['species']:g}",
		f"integrator.radiation_failure_tolerance={failure['radiation']:g}",
		f"integrator.ode_max_steps={ODE_MAX_STEPS}",
	]


def dtype_run(name, integrator, rtol, *, scale, failure, n, problem, inputs, tag, radiation=True, photochemistry=True, divisor=C_HAT_DIVISOR,
	      atol_eps=None, plotfiles=True, extra=(), **kwargs):
	photochemistry = photochemistry and radiation
	cmake_args = [
		"-DQUOKKA_PYTHON=OFF",
		f"-DDTypeFront_JAFF_TABLEAU={LABELS[integrator] if photochemistry else 'ROS2S'}",
		f"-DDTypeFront_JAFF_C_HAT_DIVISOR={divisor}",
		f"-DDTypeFront_JAFF_NO_RADIATION={'OFF' if radiation else 'ON'}",
		f"-DDTypeFront_JAFF_NO_PHOTOCHEMISTRY={'OFF' if photochemistry else 'ON'}",
	]
	if not radiation:
		variant = f"{tag}-norad"
	else:
		variant = f"{tag}-c{divisor}-{integrator if photochemistry else 'nophotochem'}"
	overrides = [
		f"amr.n_cell=[{n},{n},{n}]",
		f"amr.max_grid_size={n}",
		"amr.blocking_factor=32",
		"amr.max_level=0",
		f"plotfile_interval={10 if plotfiles else -1}",
		*tol_overrides(rtol, scale, failure, atol_eps),
		*extra,
	]
	return Run(name=name, problem=problem, inputs=inputs, host="setonix", ngpus=1, variant=variant, cmake_args=cmake_args, overrides=overrides,
		   **kwargs)


def parse_tinyprofiler(log):
	"""{name: (ncalls, excl_avg, incl_avg)} from the first exclusive and inclusive TinyProfiler time tables."""
	tables, current = [], None
	for line in log.splitlines():
		if line.startswith("Name") and ("Excl. Avg" in line or "Incl. Avg" in line) and len(tables) < 2:
			current = {}
			tables.append(current)
			continue
		if current is None:
			continue
		if line.startswith("---"):
			if current:
				current = None
			continue
		parts = line.split()
		if len(parts) >= 6 and parts[-1].endswith("%"):
			current[" ".join(parts[:-5])] = (int(parts[-5]), float(parts[-3]))
	excl, incl = tables
	return {k: (incl[k][0], excl.get(k, (0, 0.0))[1], incl[k][1]) for k in incl}


def parse_log(run_dir):
	log = (run_dir / "run.log").read_text(errors="replace")
	out = {"finished": "\nFinished." in log}
	if m := re.search(r"elapsed time: ([0-9.eE+-]+) seconds", log):
		out["elapsed_s"] = float(m.group(1))
	if m := re.search(r"figure-of-merit: ([0-9.eE+-]+) μs/zone-update", log):
		out["us_per_update"] = float(m.group(1))
	if m := re.search(r"Zone-updates on level 0: (\d+)", log):
		out["zone_updates"] = int(m.group(1))
	prof = parse_tinyprofiler(log) if "TinyProfiler" in log else {}
	for key, name in [
		("photochem", "PhotoChemistry::computePhotoChemistry()"),
		("evolve", "AMRSimulation::evolve()"),
		("plotfile", "AMRSimulation::WritePlotFile()"),
		("step", "QuokkaSimulation::advanceSingleTimestepAtLevel()"),
		("hydro", "REG::HydroSolver"),
	]:
		if name in prof:
			out[f"{key}_ncalls"], _, out[f"{key}_s"] = prof[name]
	out["n_steps"] = len(re.findall(r"^Coarse STEP \d+ at", log, flags=re.M))
	if m := re.search(r"avg\. num\. of radiation subcycles = ([0-9.eE+-]+)", log):
		out["rad_subcycles"] = float(m.group(1))
	return out


def reduce(run, run_dir, out_dir):
	for name in ("inputs.toml", "dtype_front_radii.csv"):
		if (run_dir / name).exists():
			shutil.copy(run_dir / name, out_dir / name)
	params = read_inputs(run_dir)
	log = parse_log(run_dir)
	text = (run_dir / "run.log").read_text(errors="replace")
	log["aborted"] = "Burn failed" if "Burn failed" in text else ("" if log["finished"] else "unfinished")
	t_s = params["stop_time"] / 5.0
	n = params["amr.n_cell"][0]
	summary = {"run": run.name, **log, "t_s": t_s, "dx": (params["geometry.prob_hi"][0] - params["geometry.prob_lo"][0]) / n, "n_cell": n}
	# with print_cycle_timing=1, "Coarse STEP k ... (cycle time: X s)" is the wall time of step k-1
	summary["cycle_times"] = [float(x) for x in re.findall(r"\(cycle time: ([0-9.eE+-]+) s\)", text)]
	# "Coarse STEP k at t = X" is the time at which step k starts
	summary["step_t"] = [float(x) for x in re.findall(r"^Coarse STEP \d+ at t = ([0-9.eE+-]+)", text, flags=re.M)]
	(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
	print(f"  {run.name}: finished={log['finished']} {log['aborted']} evolve {log.get('evolve_s', np.nan):.1f} s, photochem {log.get('photochem_s', np.nan):.1f} s")


def load(results_dir, name):
	d =results_dir / name
	if not (d / "summary.json").exists():
		return None
	s = json.loads((d / "summary.json").read_text())
	s["run"] = name
	s["params"] = read_inputs(d)
	if (d / "dtype_front_radii.csv").exists() and (d / "dtype_front_radii.csv").stat().st_size > 0:  # empty for runs killed early
		s["radii"] = pd.read_csv(d / "dtype_front_radii.csv")
	return s


def ns_per_subcycle(s):
	"""Whole-run wall time of one radiation subcycle per cell, in ns, from the TinyProfiler: (time stepping - hydro) / (steps x
	Quokka's average subcycles per step) / cells. Includes the photochemistry burn when it is on."""
	if not all(k in s for k in ("step_s", "hydro_s", "step_ncalls", "rad_subcycles")) or not s["rad_subcycles"]:
		return np.nan
	return (s["step_s"] - s["hydro_s"]) / (s["step_ncalls"] * s["rad_subcycles"]) / s["n_cell"] ** 3 * 1e9
