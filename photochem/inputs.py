import json
import tomllib
from pathlib import Path


def _flatten(d, prefix=""):
	out = {}
	for key, value in d.items():
		if isinstance(value, dict):
			out.update(_flatten(value, f"{prefix}{key}."))
		else:
			out[f"{prefix}{key}"] = value
	return out


def read_inputs(directory):
	"""Flat {"amr.n_cell": [...], ...} from `inputs.toml` in `directory`, with the runtime overrides in its run.json applied."""
	directory = Path(directory)
	params = _flatten(tomllib.loads((directory / "inputs.toml").read_text()))
	for override in json.loads((directory / "run.json").read_text())["run"]["overrides"]:
		key, value = override.split("=", 1)
		try:
			params[key.strip()] = tomllib.loads(f"v = {value}")["v"]
		except tomllib.TOMLDecodeError:  # an unquoted string, e.g. restartfile=../run/chk0001000
			params[key.strip()] = value
	return params
