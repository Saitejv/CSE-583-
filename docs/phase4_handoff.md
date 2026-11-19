# Phase 4 handoff: CPU unroll selection

## Current entry decision (2026-11-18)

The Phase 4 entry decision selected the opt-in static `fori_loop`
specializations for measured reduction shapes on Apple M2. Phase 4 added
the compile-once demo, full-caller first-call and break-even measurements,
and an experimental engine path for project-owned loops. Native JAX is the
general default. [Phase 4 deployment report](phase4_report.md)

[Current plan](../TASKS.md#phase-4---jax-integration-and-held-out-evaluation-week-4) ·
[Compile-once results](compile_once_reductions.md) ·
[Library evaluation](library_deployment.md)

## Historical Phase 3 recovery handoff

**Use factor 1 by default.** The learned policy is ready for opt-in CPU
integration and a scoped demo, not for general automatic selection. The
fresh confirmation gate passed on six configurations, but the typical
feasible long-horizon decision was still slower than factor 1.
At the time of the recovery handoff, both wrappers were still stubs.

## What changed and why

Both historical Week 3 models selected factor 1 on all 20 original holdout
configurations. At R=100,000 with unlimited budget, the measured oracle chose
above 1 for 8 of 10 indexed-memory configurations; the revised model's
median indexed-memory regret was 6.39%. Its 87.4% overall LOFO agreement
included forced fallbacks. The original holdout was never used for recovery
training, selection, or the fresh gate.

The [protocol](recovery_protocol.md) was committed at `97aea34`. Recovery
dev included 35 archived dev and 36 new configurations (six distinct memory
bodies, 432 correct new sweep rows). Seven whole-family LOFO folds kept each
workload's variants, trips, and factors together. The useful headline is the
subset where a measured larger factor can beat factor 1, not the grid average.

| New-dev LOFO diagnostic | Frozen Week 3 | Selected refit |
|---|---:|---:|
| Warm log-ratio MAE vs measured factor 1 | 0.761 | 0.115 |
| Warm log-ratio bias / correct direction | +0.754 / 52.8% | +0.018 / 50.0% |
| Absolute cold / warm log MAE | 0.682 / 4.752 | 0.645 / 0.636 |
| All settings: agreement / mean positive regret | 92.6% / 0.50% | 91.1% / 0.62% |
| Feasible long (144): agreement / mean positive regret | 55.6% / 2.99% | 48.6% / 3.73% |
| Oracle improves, long (64): agreement / mean positive regret | 0% / 6.72% | 25.0% / 3.94% |
| Oracle improves, long: median paired speedup vs 1 / LLVM size | 1.000× / 1.060× | 1.000× / 1.089× |

The positive Week 3 warm-ratio bias on the new bodies is consistent with
over-penalizing unrolling on unfamiliar memory workloads. The refit improves
that calibration and cuts worst-new-family useful regret from 7.73% to
5.15%, but the zero-effect predictor still has lower relative MAE (0.100).
In the 864-setting new-dev grid, 432 settings had no measured feasible
candidate; both models went over the measured budget in those 432 and in
none of the remaining settings. Dev results are before decision overhead.
The four predeclared indexed/traffic variants all missed the warm-MAE
shortlist (0.121–0.158), so the same four small ridge equations were refit
to expanded dev data. The [dev report](../results/recovery/dev_report.json)
includes absolute/relative calibration, all policies and folds; in its
legacy-dev group the historical frozen model is in-sample, not a LOFO model.

## One-shot confirmation: pass, with regressions

Model `cpu-recovery-v1-607aa4f6077e111b` was frozen at `ad45dc9` before the
first confirmation sweep. SHA-256:
`e1f88ef91875f4712ea848f31394000a78cbe079a0c4a0ed32cd3982615b98eb`.
The [ledger](../results/recovery/confirmation_gate.jsonl) records the raw
data hash, freeze commit, model/protocol hashes, and evidence hashes. The
separate [manifest](../data/recovery/confirmation.meta.json) records workload
membership and sweep provenance. All 288 CPU measurement rows passed;
24 configurations from four distinct confirmation bodies reached the gate.

Across 96 feasible long-horizon settings, 92 had all three policies within
the measured cold budget. **12 of those 92 were strict paired wins**, ten
by at least 1% against both baselines. All were at R=100,000 (six each at
100 ms and unlimited B), with factor 2 on five `bucket_max` configurations
and one `strided_mix` configuration. No wins occurred outside feasible long
horizons. Savings on the wins were **0.30–9.99%** vs factor 1 (median 3.31%)
and **4.81–26.51%** vs `llvm_size` (median 7.66%), after charging the learned
policy one uncached decision per session and charging neither baseline.

| Matched feasible subset | Median oracle regret [95% CI] | Speedup vs 1 [95% CI] | Speedup vs LLVM size [95% CI] |
|---|---:|---:|---:|
| All feasible long (92 settings) | 1.70% [1.39%, 2.43%] | 0.990× [0.983, 0.994] | 1.150× [1.078, 1.346] |
| Oracle improves, long (32 settings) | 1.78% [0.32%, 5.70%] | 1.003× [0.998, 1.021] | 1.054× [0.984, 1.093] |

The intervals use 2,000 configuration-bootstrap draws; repeated settings
within each configuration are reduced to a median first. In the 92 long
settings the selector regressed against factor 1 in 78 (43 by at least 1%),
and against LLVM size in 10 (6 by at least 1%). Worst regressions were
41.70% vs factor 1 (`masked_rows/n64/T=1023`, erroneous factor 2) and 7.05%
vs LLVM size (`ring_feedback/n64/T=1023`, missed factor 2), both at R=100,000
with unlimited B. Overall factor agreement was 93.4%, falling to 60.4% on
feasible long settings and 31.3% where the oracle improved. Measured budget
violations were 288/576, all where no candidate was feasible. Warm log-ratio
MAE was 0.120, but absolute warm estimates were a median 2.25× measured;
absolute cost error can misplace a cold/warm crossover. The full [report](../results/recovery/confirmation_report.json)
and [paired rows](../results/recovery/confirmation_pairs.csv) include the
short-horizon regressions and per-setting intervals. These are CPU replay
results, not confidence in transfer to new hardware or workload families.

[Optimized HLO](../results/recovery/confirmation_hlo.json) still shows one
outer `while` per compared program. For `bucket_max/n64/T=256`, factors
1/2/8 have 1/2/8 scatter ops and 5/8/26 fusion ops; for
`strided_mix/n512/T=256`, factors 1/2/4 have 2/4/8 gather ops. The factors
produce distinct optimized IR, not identical code labeled as a speedup.
The counts do not establish which compiler optimization caused the gains.

## Integration contract

- Empirical domain: Apple M2 CPU, JAX/jaxlib 0.12.0.1, static `scan` or
  `fori_loop` represented by one top-level `scan` Jaxpr, observed trips
  16–1,024 and 32-bit element width. Runtime checks are not a guarantee on
  unseen bodies. No GPU or dynamic/nested-loop performance claims.
- Load with `CostModel.load(Path("models/cost_cpu_recovery_v1.json"))`.
  `predict_candidates(features)` yields `LogCosts` for `{1, 2, 4, 8}`;
  `.linear()` returns cold ms and warm µs. The training regressions never
  see B or R. `select_factor(predictions, B=..., R=...)` minimizes
  `C_hat + R*T_hat/1000` under the predicted cold budget; `throughput_only`
  minimizes predicted warm time under that same budget. Ties prefer the
  smaller factor. Unsupported, invalid, missing or infeasible predictions
  fall back to 1; this does not guarantee factor 1 meets a tight budget.
- `DecisionEngine(model, cache).decide(fn, args, trip_count=..., loop_kind=...,
  B=..., R=...)` expects a callable containing the factor-1 loop, not the
  body function alone. Apply the returned `.factor` using JAX's native
  `unroll=` argument. Decision work is one `jax.make_jaxpr` plus static
  inference; candidate lowering, compilation, execution and timing are offline.
- Reuse an engine/cache across calls. A trusted returned `LoopSignature`
  for the same body and abstract arguments lets the engine skip tracing on
  a cache hit. Cache keys include shape/dtype/body identity, trip, budget,
  horizon and mode; model/feature version and backend invalidate the cache.
  An optional `DecisionCache(Path(...))` writes an atomic JSON snapshot;
  persistent I/O was not timed. Confirmation medians (p95) in ms: cold
  decision 1.257 (1.627), trace-then-hit 0.494 (0.725), known-signature hit
  0.0044 (0.0046). Model load and argument construction are not charged.

## Reproduce without reopening the gate

```bash
make recovery-train    # rebuild dev join, LOFO outputs and model; compare bytes
make recovery-verify   # rebuild confirmation join and summarize recorded pairs, no scoring
make verify-week3      # rebuild historical join, replay both reports into a temp directory
make test
make quick
```

The archived Week 3 raw sweeps are retained byte for byte. Rebuilding the
220 historical dataset rows reproduces all fields except fresh extraction
timing; both historical gate CSV/Markdown pairs replay byte for byte. The
fresh gate CLI refuses a second score. Verification checks the frozen
inputs and evidence without requiring later code to match the freeze.
Use recorded measurements, not newly timed sweeps, to reproduce reports.
