import os
import re
import socket
from pathlib import Path

import yaml

ANALYSIS = Path(__file__).resolve().parents[1]  # the repository root
CONFIG = ANALYSIS / "config.yaml"  # this machine's settings (not in git): copy config.example.yaml
RESULTS = ANALYSIS / "results"
TEST_FIGURES = ANALYSIS / "test_figures"  # figures of small tests that are not in the paper


def _read_config():
	if not CONFIG.exists():
		raise SystemExit(f"{CONFIG} not found: copy config.example.yaml to config.yaml and set your paths")
	return yaml.safe_load(CONFIG.read_text())


def host():
	if name := os.environ.get("PHOTOCHEM_HOST"):
		return name
	if os.environ.get("PAWSEY_CLUSTER") == "setonix" or re.match(r"(setonix|nid)\d*", socket.gethostname()):
		return "setonix"
	if socket.gethostname().startswith("gadi-"):
		return "gadi"
	return "local"


def load_config(name=None):
	"""Return (full config, config of host `name` or of this machine). The host's `runs` root is that of the pinned commit."""
	cfg = _read_config()
	hcfg = dict(cfg["hosts"][name or host()])
	hcfg["runs"] = str(Path(hcfg["runs"]) / cfg["paper"]["commit"][:9])
	return cfg, hcfg


_CFG = _read_config()
PAPER_BRANCH, PAPER_COMMIT = _CFG["paper"]["branch"], _CFG["paper"]["commit"]
FIGURES = ANALYSIS / _CFG.get("figures", "figures")  # where `plot` writes the PDFs (relative to the repository, or absolute)
