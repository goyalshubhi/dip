#!/usr/bin/env bash
# Reproduce every result from a fresh clone: environment, data, tables, figures, tests.
# Usage: bash scripts/run_all.sh        (set PYTHON=/path/to/python3.11 to choose the interpreter)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

if [ ! -d .venv ]; then
    "$PYTHON" -m venv .venv
fi
if [ -x .venv/bin/python ]; then VPY=.venv/bin/python; else VPY=.venv/Scripts/python.exe; fi
"$VPY" -m pip install --quiet -r requirements.txt

start=$(date +%s)
bash scripts/download_data.sh
"$VPY" -m src.loader                                   # cleaning, splits, dataset stats, GT overlays
"$VPY" scripts/run_segmentation_eval.py                # design checks, radius, tuning (train/val), test, sensitivity
"$VPY" scripts/run_classification.py                   # features, classifiers (both settings), interpretability
"$VPY" scripts/leakage_checks.py                       # sanity checks behind the >99% accuracy
"$VPY" scripts/make_figures.py                         # all remaining figures
"$VPY" -m pytest -q tests
echo "run_all.sh finished in $(( $(date +%s) - start )) s"
