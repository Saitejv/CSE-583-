#!/usr/bin/env bash
# From the repository root, with its pinned .venv installed.
set -euo pipefail
export PYTHONPATH=src:.
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 JAX_ENABLE_COMPILATION_CACHE=false
python_bin=.venv/bin/python
out=results/fori_dev
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

shasum -a 256 -c "$out/prescore.sha256"
"$python_bin" -m benchmarks.corpus.fori_dev --baselines-only --out "$scratch"
"$python_bin" -m benchmarks.corpus.fori_audit
"$python_bin" -m benchmarks.corpus.fori_dev --out "$scratch"
for file in baselines.json report.json warm_ratios.csv predictions.csv policy_pairs.csv; do
    cmp "$scratch/$file" "$out/$file"
done
shasum -a 256 -c "$out/outputs.sha256"

"$python_bin" -m pytest -q
# A unique suffix preserves earlier quick smoke measurements.
make quick DATE="fori_dev_$(basename "$scratch")"
# Check both pilot outputs before running the normal in-place Make target.
"$python_bin" -m benchmarks.corpus.pilot --out "$scratch/pilot"
cmp "$scratch/pilot/report.json" results/corpus_pilot/report.json
cmp "$scratch/pilot/warm_ratios.csv" results/corpus_pilot/warm_ratios.csv
make corpus-pilot-report
make recovery-verify
make verify-week3
