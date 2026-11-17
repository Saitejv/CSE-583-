# Project Tasks: Compilation-Budget-Aware Loop Unroll Selection for JAX

Status: Phase 0, Phase 1, and Phase 2 complete (2026-11-17); Phase 3 next. Target: 4 weeks (Nov 17 - Dec 15, 2026; confirm course deadline).

Sources of truth: `CSE583_Cost_Aware_JAX_Loop_Unrolling_Proposal.docx` and `group10_slides.pdf`. If this file conflicts with them, they win.

## Goal and success criteria

Automatically choose a partial unroll factor `u` for fixed-trip-count `lax.scan` / `lax.fori_loop` loops from static Jaxpr features, without compiling/benchmarking all candidates at decision time. For each candidate, predict `C(u)` (compile cost) and `T(u)` (warm latency), then pick the feasible factor minimizing `C(u) + R * T(u)` under budget `B`.

Project is done when:

- An opt-in `auto_scan` / `auto_fori_loop` wrapper selects `u` from features and passes it to JAX's `unroll`.
- On held-out loop families, the selector beats factor 1 and the LLVM-style size heuristic on total cost (or warm latency when `R` is large) for a useful subset of loops.
- We report median latency, end-to-end time over `R` calls, budget-violation rate, factor agreement, and regret vs the measured oracle, with CPU/GPU results separated.
- Deliverables exist: sweep suite, feature extractor, learned selector, opt-in interface, correctness tests, evaluation report, demo.

Non-goals: new XLA/LLVM pass, dynamic trip counts, arbitrary nested control flow, upstream JAX integration (optional stretch only).

## Ground rules

- Never benchmark candidate factors during a decision. Offline sweeps only.
- Hold out entire loop families from training and model selection.
- Pin JAX/jaxlib and record device/backend with every row.
- Correctness vs `u=1` is mandatory before reporting speedups.
- Inspect HLO before claiming a win: XLA may unroll or fuse independently.
- Time with warmup + `block_until_ready()`; medians and spread, not single runs.
- Do not merge CPU and GPU results.

## Done

- [x] Read proposal and slides; extracted scope, baselines, metrics, 4-week plan.
- [x] Create Python 3.13 `.venv` with pinned `requirements.txt` (JAX/jaxlib 0.12.0.1, NumPy, pandas, scikit-learn, matplotlib, pytest).
- [x] Verify CPU backend and `lax.scan(unroll=1/2/4/8)` execution on Apple M2.
- [x] Create project directories (`src/jax_unroll`, `benchmarks`, `tests`, `data`, `results`) and `.gitignore`.
- [x] Phase 2 complete (2026-11-17): static Jaxpr features, baselines, feature/label dataset, and frozen splits. See `docs/week2_results.md`.

## Phase 0 - Scaffolding and target hardware

- [x] Owner: TBD. Confirm primary target device. Ideal: one documented GPU. If none by end of week 1, commit to CPU-only results and say so in the report. Decision: CPU-only Apple M2 for now; GPU checkpoint 2026-11-23 (`docs/hardware.md`).
- [x] `pip install -r requirements.txt` reproduces the environment on the target machine; pin CUDA/jaxlib variant there if GPU. `uv pip install -r requirements.txt` installs the editable package and `python -m jax_unroll` works.
- [x] Add package skeleton:
  - `src/jax_unroll/features.py` - Jaxpr feature extraction
  - `src/jax_unroll/model.py` - cost-model training/prediction
  - `src/jax_unroll/selector.py` - budget-aware decision policy
  - `src/jax_unroll/auto_loop.py` - `auto_scan`, `auto_fori_loop`
  - `benchmarks/workloads.py` - loop-family registry
  - `benchmarks/sweep.py` - measurement harness
  - `tests/` - pytest suite
- [x] Agree on data schema (below) and file layout before collecting data. See `docs/data_schema.md`.
- [x] Add a `Makefile` or scripts for: `sweep`, `train`, `evaluate`, `test`.

Acceptance: fresh checkout + `uv pip install -r requirements.txt` runs the smoke test; schema documented. **Met 2026-11-17.**

## Phase 1 - Workloads and factor sweeps (Week 1)

- [x] T1.1 Implement the workload registry with static shapes and fixed trip counts:
  - scalar recurrence (confirmed 2.05x signal at `u=8`)
  - elementwise sine (confirmed regression at `u=8`)
  - reduction + vector
  - dense matrix-vector update
  - indexed tensor update (gather/scatter-style)
  - one application-style iterative workload (e.g. GRU-like scan or iterative solver)
- [x] T1.2 Vary trip count (`16, 64, 256, 1024`; add `100, 1000` for remainder behavior), array shape/size, and factor `{1, 2, 4, 8}`. Record whether `u` divides trip count.
- [x] T1.3 Measurement protocol:
  - one fresh process per configuration (or `jax.clear_caches()` plus separate timing where valid)
  - randomized configuration order, >=3 process repeats, warmups
  - separate tracing + lowering time (`.lower()`), backend compile time (`.compile()`), and warm execution time (`block_until_ready()`)
  - cold compile = trace + lower + compile; also record end-to-end time for `R` calls
  - IR size from StableHLO text (`.lower().as_text()`); optimized HLO via `compiled.as_text()` if supported, else XLA dump flags
- [x] T1.4 Correctness: compare every factor against `u=1`; assert within tolerance and record pass/fail in the dataset.
- [x] T1.5 Emit `results/raw/sweep_<backend>_<date>.csv` and a human-readable oracle table (best measured factor per configuration). Produced `results/raw/sweep_cpu_2026-11-17.csv` (432 rows) and `..._shapes.csv` (240 rows); oracle in `results/oracle_cpu_2026-11-17.csv/.md`.
- [x] T1.6 Risk gate: verify at least two workloads show distinct useful choices with non-trivial gains. If not, broaden workloads or narrow the research question before building the model. **PASS**: useful dominant warm factors 1, 2, and 8; evidence in `results/risk_gate_cpu_2026-11-17.md` and `docs/week1_results.md`.

Suggested columns:
```
workload, family, backend, device, trip_count, shape, dtype, factor, divides_trip,
process_repeat, trace_ms, lower_ms, compile_ms, cold_total_ms, warm_median_us,
warm_p95_us, end_to_end_ms_R, stablehlo_bytes, hlo_bytes, correct, notes
```

Acceptance: reproducible sweep command; >=6 families; all factors valid/correct; oracle table reproduced from raw data in one script; gate decision documented. **Met 2026-11-17**: 660/672 rows correct (12 excluded for a divergent `NaN` configuration), 0 errors, 55 valid configurations, 3 process repeats; see `docs/week1_results.md`.

### Week 1 outcome (2026-11-17)

- Data: 672 measurements over 56 configurations (six families, primary and
  secondary shapes, trip counts 16/64/100/256/1000/1024, factors 1/2/4/8),
  three fresh-process repeats. 660 rows are correct; the 12 rows for
  `dense_matvec`/`n32`/`T=1000` are excluded because the `float32` state
  overflows to `NaN`, leaving 55 valid configurations.
- Gate: PASS. Scalar recurrence favors `u=8` (1.32-1.35x warm at large trip
  counts), indexed update favors `u=2` (1.07x median), sine/Jacobi/reduction
  favor `u=1`; dense matvec has no useful signal.
- Total-cost caveat: at `R=100` and `R=1000` compilation dominates and the
  total-cost oracle chooses `u=1` everywhere; at `R=10000` scalar and indexed
  update show 1.05x/1.04x median total gains. Phase 3/4 must sweep `R` and `B`.
- Full report: `docs/week1_results.md`.

## Phase 2 - Jaxpr features and heuristics (Week 2)

- [x] T2.1 Feature extractor from the loop body via `jax.make_jaxpr`:
  - trip structure: trip count, candidate factor, remainder, remainder ratio, divide-evenly flag
  - operation mix: total equations; counts of elementwise, dot/matmul, reduce, gather/scatter/index, convert, other
  - tensor footprint: input/output/carry shapes and dtypes; estimated carry bytes, intermediate bytes, max tensor bytes
  - dependences: loop-carried state count; approximate longest def-use path; count of body outputs that feed later equations
  - arithmetic intensity proxy and per-iteration equation count
  - context: backend/device, dtype bits, `B`, `R`
- [x] T2.2 Handle both `scan` and `fori_loop` bodies; document what is and is not captured. Keep extraction cheap (microseconds/milliseconds) and measured.
- [x] T2.3 Tests: known-loop feature values; `scan` vs equivalent `fori_loop` parity where possible; assert feature extraction does not compile candidates.
- [x] T2.4 Implement baselines:
  - factor 1 (JAX default)
  - fixed factors 2/4/8
  - LLVM-style size heuristic (rolled vs unrolled size estimate, threshold and max factor)
- [x] T2.5 Dataset builder: join features and sweep labels by configuration key; stable schema; deduplicate; version data and config.
- [x] T2.6 Define leave-one-family-out splits now; freeze the held-out families before model work.

Acceptance: feature table for all swept configs; tests pass; baseline heuristic implemented and documented; held-out families frozen and recorded. **Met 2026-11-17**: 220-row dataset over 55 configurations (`data/features_cpu_2026-11-17.csv`), 197 tests pass, `llvm_size` baseline documented, split version 1 frozen. See `docs/week2_results.md`.

### Week 2 outcome (2026-11-17)

- Features: `src/jax_unroll/features.py` extracts trip/remainder structure, operation mix, tensor footprint, dependence proxies, and work proxies from `jax.make_jaxpr` for `scan` and static-bound `fori_loop` (which traces to `scan` in JAX 0.12; the counter is excluded from state features). It also locates loops through `jit`/`pjit` wrappers and supports bare bodies with an explicit `body_num_xs`. No candidate is lowered or compiled. Median extraction cost 0.56 ms (p95 1.13 ms) over 220 configurations.
- Dataset: `benchmarks/dataset.py` joins features with the Week 1 sweep medians into `data/features_cpu_2026-11-17.csv` (220 rows x 78 columns, 55 configurations, 672 raw rows, 12 dropped as non-finite) plus a versioned metadata sidecar (dataset version 2) with raw-file hashes and the split manifest. `B`/`R` context columns are empty until the Phase 3 R/B grid.
- Baselines: `src/jax_unroll/baselines.py` implements factor 1, fixed 2/4/8, and an LLVM `UnrollThreshold`-style size heuristic. Warm-only sanity check over 55 configurations: factor 1 is the best median warm baseline (1.000x), while `llvm_size` over-unrolls small scan bodies (0.842x median, 45/55 choices u=8). The measured warm oracle uses u=1 31/55, u=2 9/55, u=4 7/55, u=8 8/55; this gap is Phase 3 work.
- Splits: split version 1 frozen before model work. Dev = recurrence, elementwise, reduction, dense_linear_algebra; held out = indexed_memory, iterative_solver. LOFO folds over dev families only; see `docs/splits.md`.
- Full report: `docs/week2_results.md`; feature definitions: `docs/features.md`.

## Phase 3 - Cost model, selector, and overhead (Week 3)

- [ ] T3.1 Train small models on dev families only:
  - nearest-neighbor / shallow tree regressors for `log C(u)` and `log T(u)`, or normalized cost
  - optional direct factor classifier as a comparison
  - no deep learning; prioritize interpretability and tiny inference cost
- [ ] T3.2 Validate with leave-one-family-out CV. Report MAE of `C`/`T`, factor agreement, and regret.
- [ ] T3.3 Implement the selector:
  - candidate `u in {1,2,4,8}`
  - `score(u) = C_hat(u) + R * T_hat(u)`
  - feasibility: `C_hat(u) <= B`
  - throughput-only mode: minimize `T_hat(u)`
  - fallback to `u=1` when predictions are uncertain, infeasible, or feature extraction fails
  - deterministic tie-breaking (prefer smaller `u`)
- [ ] T3.4 Measure analysis + inference overhead and include it in end-to-end cost.
- [ ] T3.5 Cache decisions by loop signature (shapes, dtypes, body hash, backend, `B`, `R`); invalidate on version/backend change.
- [ ] T3.6 Report prediction calibration and failure cases; avoid overfitting to one shape.

Acceptance: selector beats factor 1 and the size heuristic on at least a subset of held-out families for some `R`/`B`; regret and CIs reported; overhead measured.

## Phase 4 - JAX integration and held-out evaluation (Week 4)

- [ ] T4.1 Implement opt-in `auto_scan(f, init, xs, *, B, R, ...)` and `auto_fori_loop(lower, upper, body_fun, init_val, ...)`:
  - inspect body/features at trace time
  - select `u` and call existing `lax.scan` / `lax.fori_loop` with `unroll=u`
  - preserve JAX semantics and pytree handling; static trip count only
  - expose `force_factor` / `disable` escape hatches; clear errors for unsupported cases
- [ ] T4.2 End-to-end benchmark: cold compile + `R` invocations for `R in {1, 10, 100, 1000}`; include decision overhead; measure budget-violation rate.
- [ ] T4.3 Held-out evaluation only: no tuning on these families. Primary table on the documented target device; CPU table separate.
- [ ] T4.4 HLO audit: confirm whether XLA unrolls/fuses beyond our setting; report gains over the actual baseline, not an assumed one.
- [ ] T4.5 Failure analysis: no-gain cases explained by backend optimization, memory behavior, dependence chains, or prediction error.
- [ ] T4.6 Full pytest suite: correctness across factors, fallback paths, budget edge cases, cache behavior, unsupported-loop errors.

Acceptance: demo script shows a selected factor and measured end-to-end improvement; held-out results table with baselines, oracle regret, CIs, and budget violations; HLO evidence attached.

## Phase 5 - Deliverables and demo

- [ ] D1 Reproducible factor-sweep suite (`benchmarks/sweep.py` + documented commands).
- [ ] D2 Jaxpr feature extractor with documented feature definitions.
- [ ] D3 Learned selector + versioned model artifacts + training script.
- [ ] D4 Opt-in automatic JAX loop interface.
- [ ] D5 Correctness/test suite and CI-style `pytest -q` instructions.
- [ ] D6 Evaluation report with tradeoffs, failure cases, CPU/GPU separation, and threats to validity.
- [ ] D7 Final demo (notebook or script) and updated slides.
- [ ] D8 Stretch, only if time: Jaxpr scan-parameter rewrite demonstration; upstream JAX issue/discussion.

Acceptance: another student can clone, install, run the sweep and demo, and reproduce headline numbers.

## Milestones and gates

| End of | Must have | Gate |
|---|---|---|
| Week 1 (Nov 23) | Sweep harness + oracle over 6 families + target hardware decision | At least 2 families with distinct useful factors, else adjust scope |
| Week 2 (Nov 30) | Features + baselines + frozen splits | Held-out families frozen; feature extraction cheap and tested |
| Week 3 (Dec 7) | Trained selector + policy + overhead | Beats both baselines on some held-out settings |
| Week 4 (Dec 14) | Wrapper + held-out evaluation + report/demo | Reproducible headline result; limitations stated |

## Risks and contingencies

- Weak optimization signal: broaden workload set or narrow the claim; report negative results honestly.
- XLA already unrolls: audit optimized HLO; compare against actual JAX/XLA baseline.
- Overfitting / small dataset: leave-one-family-out, simple models, CIs, fixed baselines.
- No GPU: finish CPU results, state the limitation, avoid GPU speedup claims.
- Feature extraction too slow: cache per loop signature; simplify longest-path proxy.
- Schedule slip: drop the stretch Jaxpr rewrite and the application workload before dropping evaluation or correctness.

## Open questions

- Which GPU will be available, and by when? (Owner: TBD)
- Which application-style iterative workload best fits the 4-week budget?
- Preferred `R` default and budget `B` values for the demo?
- Predict `C(u)`/`T(u)` vs directly classify the best factor? (Start with regression; compare.)
- Where will model artifacts and aggregated (non-raw) results be committed?

## Commands

```bash
source .venv/bin/activate
python -m pytest -q
# run today:
python -m benchmarks.dataset --raw results/raw/sweep_cpu_2026-11-17.csv results/raw/sweep_cpu_2026-11-17_shapes.csv --out data/features_cpu_2026-11-17.csv
# planned:
# python benchmarks/sweep.py --backend cpu --out results/raw/sweep.csv
# python -m jax_unroll.train --data results/raw/sweep.csv --out models/
# python benchmarks/evaluate.py --config configs/default.yaml
```
