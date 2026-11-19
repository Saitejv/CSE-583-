# Compilation-Budget-Aware Loop Unroll Selection for JAX

CSE 583 Advanced Compilers course project, Group 10 (Saitej Velaga, Maanas Pavuluri, Jack McClure).

The experimental selector chooses a partial unroll factor for fixed-trip-count loops using static Jaxpr features and a lightweight cost model. It minimizes predicted `C(u) + R * T(u) / 1000` (cold ms and warm µs) under a cold budget. Project-owned static loops can opt in through `auto_fori_loop`.

Two opt-in compile-once reductions improved warm full-caller latency on Apple M2. Native JAX is the general default, and learned selection remains experimental. [Final evaluation and reproduction guide](docs/final_report.md) · [Final slides](docs/phase5_slides.pptx)

## Source of truth

- `CSE583_Cost_Aware_JAX_Loop_Unrolling_Proposal.docx` -- proposal, scope, algorithm, evaluation plan
- `group10_slides.pdf` / `CSE 583 Presentation - Google Slides.pdf` -- original proposal presentation (identical content)

## Environment

Use Python 3.13 and the pinned dependencies in `requirements.txt`. Published measurements use JAX/jaxlib 0.12.0.1 on Apple M2 CPU.

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m jax_unroll
```

`requirements.txt` installs the project in editable mode (`-e .`). The Makefile
and `conftest.py` also add `src/` and the repo root to the path.

## Quickstart

```bash
source .venv/bin/activate

make test           # pytest suite
make list           # print the workload registry as JSON
make quick          # small smoke sweep (8 measurements)
make recovery-train  # reproduce the recovery join, dev LOFO and frozen model
make recovery-verify # verify fresh confirmation from recorded factors, no new gate
make verify-week3    # rebuild and replay the archived data and both old gates
make phase5-demo     # run the reduction comparisons and learned fori_loop example
make phase5-replay   # regenerate the headline tables from archived process measurements
make phase4-evaluate # run fresh isolated CPU timings under results/raw/phase4/
```

`make sweep` and `make sweep-shapes` write to `results/raw/` (gitignored
apart from the two exact Week 3 input sweeps retained for verification);
`make oracle` writes `results/oracle_<backend>_<date>.csv/.md` and
`results/risk_gate_<backend>_<date>.md`; `make dataset` writes
`data/features_<backend>_<date>.csv` and a `.meta.json` sidecar. See
[docs/data_schema.md](docs/data_schema.md) for every column, the measurement
protocol, and the aggregation/gate rules,
[docs/week1_results.md](docs/week1_results.md) for the Week 1 results,
[docs/features.md](docs/features.md) for the feature definitions and cost,
[docs/splits.md](docs/splits.md) for the frozen held-out families, and
[docs/week2_results.md](docs/week2_results.md) for the Phase 2 summary.

Use scratch paths for new sweeps, joins, and legacy training (for example,
`make train MODEL_DIR=/tmp/jax-unroll-legacy`); do not overwrite the frozen
inputs, models, or gate records. The two original holdout evaluations are
linked from [Week 3 results](docs/week3_results.md). The separate [recovery
protocol](docs/recovery_protocol.md) fixes the expanded dev and fresh
confirmation workloads. Recovery measurements and joins are under
`data/recovery/`, the opt-in model is `models/cost_cpu_recovery_v1.json`,
and paired evidence and the one-shot ledger are under `results/recovery/`.

## Layout

```
src/jax_unroll/         selector package (features, model, policy, auto_* wrappers)
src/jax_unroll/features.py  static Jaxpr feature extractor (Phase 2)
src/jax_unroll/baselines.py factor-1/fixed/LLVM-size baseline policies
src/jax_unroll/splits.py    frozen dev/held-out families and LOFO folds
src/jax_unroll/schema.py shared SWEEP/ORACLE/DATASET column contract and paths
benchmarks/             workload definitions, factor-sweep harness, oracle
benchmarks/dataset.py   feature/label dataset builder (Phase 2)
benchmarks/evaluate.py  offline CPU policy comparison and frozen holdout gate
benchmarks/phase5.py   final live demo and replay of archived full-caller timings
models/                 versioned CPU cost-model JSON
tests/                  pytest suite
Makefile                test/list/sweep/oracle/dataset targets and output paths
docs/data_schema.md     data pipeline, column definitions, measurement protocol
docs/features.md        feature definitions, captured limits, extraction cost
docs/splits.md          frozen held-out families, LOFO folds, change control
docs/hardware.md        target-hardware decision (CPU-only)
docs/week1_results.md   Week 1 sweep, oracle, and risk-gate results
docs/week2_results.md   Phase 2 feature table, baselines, and split summary
docs/week3_results.md   dev LOFO, CPU decision overhead, and holdout gates
docs/phase4_handoff.md  recovery evidence, supported domain, and integration API
docs/phase4_report.md   compile-once deployment, learned evaluation, and compiler audit
docs/final_report.md    final evaluation and reproduction guide
docs/phase5_slides.pptx final presentation
conftest.py             adds repo root and src/ to sys.path for pytest
data/features_*.csv     Phase 2 feature/label dataset
data/features_*.meta.json  dataset versions, hashes, and split manifest
results/raw/            generated measurements and two archived Week 3 inputs
results/                oracle and risk-gate outputs
TASKS.md                actionable project plan and definition of done
```

## Status

Two [compile-once reduction specializations](docs/compile_once_reductions.md) beat native JAX on measured Apple M2 shapes. For the original twelve library configurations, use native JAX by default. The [library deployment benchmark](docs/library_deployment.md) covers actual public functions, held-out inputs, and calibration costs.

For static `jax.lax.fori_loop`, use [`auto_fori_loop` with the expansion-trained CPU predictor](docs/fori_predictor.md) to select an unroll factor for the existing loop body. On 192 configurations the live wrapper measured 1.12× pooled warm speedup with a median 4.35 ms trace-time decision. The [paired-trip predictor evaluation](docs/targeted_predictor.md) compares eight model designs and four fixed factors.

The [prediction follow-up](docs/fori_prediction_update.md) adds 19 generated workloads and evaluates a static refit and a shorter-trip profile correction.

The proposal-aligned CPU evaluation covers new loops with repeatable unrolling gains and retains cases that favor factor 1. It measures the wrapper at multiple call counts and includes a NumPy-only export of the incumbent to reduce cold startup. [Coverage, end-to-end results and reproduction](docs/fori_proposal_evaluation.md)

The [workload-specific experiment](docs/fori_specialization.md) benchmarks manually restructured versions of six loops through a separate opt-in builder.

The [66-workload CPU model](docs/all_workload_prediction.md) selects unroll factors from static features and runs through the decision engine.

An opt-in CPU profile predictor is available for recurring workloads. See the [usage example and fresh validation results](docs/profile_prediction.md).

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
- Phase 3 implementation is complete. The two frozen holdout gates **failed**.
  Both models chose factor 1 on the entire holdout grid. The report includes
  dev LOFO calibration and CPU decision overhead.
  [Week 3 results](docs/week3_results.md)
- Phase 3 recovery is complete. The separate fresh confirmation gate
  **passed** for a subset, not for overall speed versus factor 1.
  [Phase 4 handoff](docs/phase4_handoff.md)
- Phase 4 integrates the compile-once M2 reduction path and `auto_fori_loop`.
  The learned path remains experimental. [Phase 4 report](docs/phase4_report.md)
- Phase 5 provides the final demo, exact replay of the published full-caller
  tables, and the course presentation. [Final report](docs/final_report.md)
- The pytest suite covers wrapper correctness, JIT reuse, fitting and holdout
  exclusion, policy and cache behavior, evaluation, and earlier phases.
- Target hardware: CPU-only on Apple M2; see
  [docs/hardware.md](docs/hardware.md) for the decision, deadline, and
  reproduction steps.
