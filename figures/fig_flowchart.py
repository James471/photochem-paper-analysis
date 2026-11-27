#!/usr/bin/env python3
"""Flowchart of one level timestep (hydro + radiation subcycle). No simulations: the figure is the TikZ
source fig_flowchart.tex next to this script.

    python figures/fig_flowchart.py plot   -> <figures>/fig_flowchart.pdf
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from photochem import main  # noqa: E402
from photochem.config import FIGURES  # noqa: E402

FIGURE = "fig_flowchart"
SOURCE = Path(__file__).with_suffix(".tex")

RUNS = []


def reduce(run, run_dir, out_dir):
	pass


def plot(results_dir):
	results_dir.mkdir(parents=True, exist_ok=True)
	subprocess.run(
		["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={results_dir}", str(SOURCE)],
		check=True,
		stdout=subprocess.DEVNULL,
	)
	FIGURES.mkdir(parents=True, exist_ok=True)
	out = FIGURES / f"{FIGURE}.pdf"
	shutil.copy(results_dir / f"{FIGURE}.pdf", out)
	print(f"wrote {out}")


if __name__ == "__main__":
	main(FIGURE, RUNS, reduce, plot)
