# Compilation-Budget-Aware Loop Unroll Selection for JAX

CSE 583 Advanced Compilers course project, Group 10 (Saitej Velaga, Maanas Pavuluri, Jack McClure).

We automatically choose a partial unroll factor for fixed-trip-count `lax.scan` / `lax.fori_loop` loops using static Jaxpr features and a lightweight cost model, instead of benchmarking every candidate factor at decision time. The selector minimizes predicted `C(u) + R * T(u)` subject to a compilation budget.

## Source of truth

- `CSE583_Cost_Aware_JAX_Loop_Unrolling_Proposal.docx` -- proposal, scope, algorithm, evaluation plan
- `group10_slides.pdf` / `CSE 583 Presentation - Google Slides.pdf` -- presentation (identical content)

## Environment

Pinned in `requirements.txt`: Python 3.13, JAX/jaxlib 0.12.0.1 (CPU backend on Apple M2), NumPy, pandas, scikit-learn, matplotlib, pytest. A `.venv` is already created.

```bash
source .venv/bin/activate
python -m jax_unroll
```

To recreate from scratch:

```bash
uv venv .venv
uv pip install -r requirements.txt
```

`requirements.txt` installs the project in editable mode (`-e .`). The Makefile
and `conftest.py` also add `src/` and the repo root to the path.

## Quickstart

```bash
source .venv/bin/activate

make test           # pytest suite
make list           # print the workload registry as JSON
make quick          # small smoke sweep (8 measurements)
make sweep          # primary-shape factor sweep, 3 process repeats
make sweep-shapes   # secondary shape variants at T=256,1000, 3 repeats
make oracle         # aggregate raw rows into oracle table + risk-gate report
make dataset        # join static features with sweep labels (Phase 2)
```

`make sweep` and `make sweep-shapes` write to `results/raw/` (gitignored);
`make oracle` writes `results/oracle_<backend>_<date>.csv/.md` and
`results/risk_gate_<backend>_<date>.md`; `make dataset` writes
`data/features_<backend>_<date>.csv` and a `.meta.json` sidecar. See
[docs/data_schema.md](docs/data_schema.md) for every column, the measurement
protocol, and the aggregation/gate rules,
[docs/week1_results.md](docs/week1_results.md) for the Week 1 results,
[docs/features.md](docs/features.md) for the feature definitions and cost,
[docs/splits.md](docs/splits.md) for the frozen held-out families, and
[docs/week2_results.md](docs/week2_results.md) for the Phase 2 summary.

## Layout

```
src/jax_unroll/         selector package (features, model, policy, auto_* wrappers)
src/jax_unroll/features.py  static Jaxpr feature extractor (Phase 2)
src/jax_unroll/baselines.py factor-1/fixed/LLVM-size baseline policies
src/jax_unroll/splits.py    frozen dev/held-out families and LOFO folds
src/jax_unroll/schema.py shared SWEEP/ORACLE/DATASET column contract and paths
benchmarks/             workload definitions, factor-sweep harness, oracle
benchmarks/dataset.py   feature/label dataset builder (Phase 2)
tests/                  pytest suite (Waves 1-2)
Makefile                test/list/sweep/oracle/dataset targets and output paths
docs/data_schema.md     data pipeline, column definitions, measurement protocol
docs/features.md        feature definitions, captured limits, extraction cost
docs/splits.md          frozen held-out families, LOFO folds, change control
docs/hardware.md        target-hardware decision (CPU-only)
docs/week1_results.md   Week 1 sweep, oracle, and risk-gate results
docs/week2_results.md   Phase 2 feature table, baselines, and split summary
conftest.py             adds repo root and src/ to sys.path for pytest
data/features_*.csv     Phase 2 feature/label dataset
data/features_*.meta.json  dataset versions, hashes, and split manifest
results/raw/            generated measurements (gitignored)
results/                oracle and risk-gate outputs
TASKS.md                actionable project plan and definition of done
```

## Status

- Phase 0: complete. Pinned environment with editable package install, package
  skeleton, shared schema, and Makefile targets.
- Phase 1: complete. Six loop families, fresh-process factor sweep with
  correctness checks, measured oracle, and Week 1 risk gate. The 2026-11-17 CPU
  run collected 672 measurements over 56 configurations (3 process repeats,
  0 errors); 660/672 rows are correct and 12 rows are excluded because the
  `dense_matvec`/`n32`/`T=1000` workload diverges to `NaN`. The risk gate
  **passes** with useful distinct factors 1, 2, and 8. See
  [docs/week1_results.md](docs/week1_results.md).
- Phase 2: complete. Static Jaxpr features (mean 0.75 ms, median 0.56 ms per
  configuration), factor-1/fixed/LLVM-size baselines, a 220-row feature/label
  dataset over 55 configurations, and frozen dev/held-out families. See
  [docs/week2_results.md](docs/week2_results.md) and
  [docs/features.md](docs/features.md).
- Phase 3-4: stubs. `model.py`, `selector.py`, and `auto_loop.py` raise
  `NotImplementedError`.
- Tests: `pytest -q` passes (197 tests on 2026-11-17); it covers workload
  correctness across factors, sweep enumeration/schema, measurement-row units,
  oracle aggregation, the risk gate, feature values/parity/no-compile, baseline
  policies, dataset aggregation/join/metadata, and frozen splits.
- Target hardware: CPU-only on Apple M2; see
  [docs/hardware.md](docs/hardware.md) for the decision, deadline, and
  reproduction steps.
