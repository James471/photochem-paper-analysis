"""Front-error and speed metrics of a D-type run (used by fig_tolerance_scaled, paper Fig. 2). Not a figure script.

Front error at 5 t_s: |R - R_HI| / dx (R = volume-equivalent ionized radius; R_HI the Hosokawa-Inutsuka radius, derived from the
Spitzer radius that the problem's dtype_front_radii.csv writes as r_analytical). Mupdates/s = zone updates / (evolve time - plotfile
time).
"""

import numpy as np


def metrics(s):
	out = {"run": s["run"], "finished": s["finished"], "aborted": s.get("aborted", "")}
	evolve, plot = s.get("evolve_s", np.nan), s.get("plotfile_s", 0.0)
	out["mupdates_per_s"] = s.get("zone_updates", np.nan) / (evolve - plot) / 1e6
	if s["finished"] and "radii" in s:
		last = s["radii"].iloc[-1]
		out["front_err_cells"] = abs(last["r_effective"] - r_hosokawa_inutsuka(last["r_analytical"], last["time"], s["t_s"])) / s["dx"]
	return out


def r_hosokawa_inutsuka(r_spitzer, t, t_s):
	"""Hosokawa & Inutsuka (2006) radius R_S (1 + 7/4 sqrt(4/3) t/t_S)^(4/7). The problem's r_analytical column is the Spitzer
	radius R_S (1 + 7/4 t/t_S)^(4/7) (no sqrt(4/3)), from which R_S follows."""
	r_s = r_spitzer / (1.0 + 7.0 * t / (4.0 * t_s)) ** (4.0 / 7.0)
	return r_s * (1.0 + 7.0 / 4.0 * np.sqrt(4.0 / 3.0) * t / t_s) ** (4.0 / 7.0)
