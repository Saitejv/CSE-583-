#!/bin/bash
# Run from repository root. New timings are development-only, not a gate.
set -euo pipefail
export PYTHONPATH=src:.
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 JAX_ENABLE_COMPILATION_CACHE=false
PYTHON=.venv/bin/python
$PYTHON -m benchmarks.corpus.static_experiment --check-pairs
$PYTHON -m benchmarks.corpus.static_matched --matrix primary --out results/static_features/matched_1023_raw.csv
$PYTHON -m benchmarks.corpus.static_matched --matrix divisible --out results/static_features/matched_1024_raw.csv
$PYTHON - <<'PY'
from pathlib import Path
import pandas as pd
out = Path('results/static_features')
pd.concat([pd.read_csv(out / f'matched_{t}_raw.csv') for t in (1023,1024)],ignore_index=True).to_csv(out/'matched_raw.csv',index=False)
PY
$PYTHON -m benchmarks.corpus.static_experiment
$PYTHON -m pytest -q
make quick DATE=2026-11-17_static_features
make corpus-pilot-report
make recovery-verify
make verify-week3
