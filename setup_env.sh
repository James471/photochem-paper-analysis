#!/bin/bash
# Create/update the Python venv for the figure scripts: ./setup_env.sh [VENV]  (default: .venv in this repo).
# On Setonix, load a Python module first (`module load cray-python/3.11.7`).
set -euo pipefail
cd "$(dirname "$0")"
VENV=${1:-.venv}

[[ -d "$VENV" ]] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip -q
"$VENV/bin/python" -m pip install -r requirements.txt -q
echo "venv ready: $VENV"
