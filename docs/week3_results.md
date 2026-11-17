# Week 3 results — cost model, selector, and decision overhead

Phase 3 implementation is complete on Apple M2 CPU. **The Week 3 acceptance
gate failed in both recorded evaluations.** The frozen selector beat the LLVM
size heuristic on median held-out cost, but it chose factor 1 for every
held-out configuration across the grid. Its decisions therefore added
overhead to the factor-1 baseline. The result is an offline replay of the
correctness-checked Week 1 measurements.

## Model and policy

The 2026-11-17 dataset has 55 configurations, four factor rows per
configuration, and 78 columns. Training and model selection used the 140 dev
rows (35 configurations). LOFO withheld one entire dev family per fold.
`indexed_memory` and `iterative_solver` supplied the 80 frozen holdout rows
(20 configurations) for the two prediction-only gates.

`CostModel` fits four small ridge regressions. Two predict factor-1
`log_cold_ms` and `log_warm_us` from static trip count, equation counts, work
and footprint proxies. Two predict per-factor log differences from factor 1.
The cold regression uses nonnegative weights on log factor, body size,
constant footprint, reductions, and tail presence. This gives nondecreasing
cold predictions for the supported powers-of-two factors. The final warm
regression combines chunk-count relief (`1 − 1/u`) and log factor with a
saturating trip-count term, per-iteration work, and bytes per unit of
estimated work. Models predict absolute natural-log costs for all four
factors. Inference evaluates the fitted weights directly from JSON.

The factor-dependent dataset columns are `factor`, `full_chunks`,
`remainder`, `remainder_ratio`, `divides_evenly`, `has_tail`,
`factor_over_trip`, and `unrolled_equations`. The decision path traces once
at factor 1 and derives these eight fields for each candidate. The regressions
use `factor` and `has_tail` from that group. Body structure and tensor
features are factor-independent. The audit found zero variance in
`max_out_degree`, so it was excluded. The dataset's `B` and `R` cells are
empty. Both values enter the policy as context.

The policy minimizes `C_hat + R × T_hat / 1000` over factors with
`C_hat <= B`, with cold cost in ms and warm latency in µs. Throughput mode
minimizes `T_hat` under the same budget. Ties favor smaller factors. Failed
extraction, missing or invalid forecasts, unsupported loops or runtime
versions, and an empty feasible set return factor 1. Static support checks
cover the backend, loop primitive, observed trip-count range (16–1,024),
and dtype width. The checks allow new operation mixes, so a supported case
can still have large prediction error.

Model JSON under [`models/`](../models) includes the version, hyperparameters,
feature list, dev-only fingerprint, runtime and training versions, train date,
and CV metrics. The evaluator uses JAX/jaxlib 0.12.0.1 labels from dataset
version 2 with feature version 1 and split version 1. Training used Python
3.13.2, NumPy 2.5.5, and scikit-learn 1.9.1. The divergent
`dense_matvec/n32/T=1000` configuration remains excluded.

## Dev leave-one-family-out

The first model minimized pooled mean positive regret over the predeclared
grid. Dev diagnostics showed it selected factor 1 throughout the recurrence
fold and missed low-work, long-trip gains. The retry tested a bytes-per-work
term and weaker shrinkage, with model selection based on the worst dev fold
at the two longest horizons (10,000 and 100,000 calls). Worst-fold mean
positive regret there fell from **7.10% to 4.63%**. Pooled grid agreement
fell from 88.8% to 87.4%, and pooled mean positive regret increased from
1.07% to 1.14%. Both variants and the selection criteria are recorded with
their artifacts.

| Dev family held out | Configs | C log MAE | T log MAE | C relative MAE | T relative MAE | Agreement | Mean positive regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| recurrence | 8 | 0.471 | 0.834 | 61.7% | 59.2% | 83.3% | 1.62% |
| elementwise | 10 | 0.204 | 1.691 | 22.4% | 76.7% | 96.7% | 1.24% |
| reduction | 8 | 0.414 | 1.028 | 29.8% | 185.2% | 97.9% | 0.37% |
| dense_linear_algebra | 9 | 0.316 | 0.417 | 25.4% | 53.8% | 71.3% | 1.28% |
| Pooled | 35 | 0.342 | 1.016 | 33.8% | 91.6% | 87.4% | 1.14% |

Relative MAE is the mean of `abs(predicted / measured − 1)` over factor
rows. Agreement and regret are pooled across the R/B grid before overhead.
Positive regret floors per-case signed regret at zero. The median signed
regret is zero in every fold, in part because both policies choose factor 1
when the budget is infeasible. At 1 and 10 ms every measured factor is
infeasible for all dev configurations.

### Calibration and failure cases

| Dev LOFO diagnostic | Cold C | Warm T |
|---|---:|---:|
| Median predicted / measured | 1.163× | 1.181× |
| Predictions within ±25% | 42.9% | 6.4% |
| p90 absolute natural-log error | 0.588 | 2.040 |
| MAE of predicted log ratios relative to factor 1 | 0.235 | 0.205 |

The warm regression's calibration varies by family. Its median prediction
was 0.195× the measured warm cost on the elementwise fold and 2.832× on
the reduction fold. Factor-relative errors are smaller, but they still
change decisions. With unlimited budget, the model missed the factor-2
choice for `elementwise_sine/n1024/T=100` at R=100 (26.26% regret).
At R=100,000 it missed factor 8 on `scalar_recurrence/vector8/T=1000`
(43.22% regret), and chose factor 2 on `reduction_vector/n256/T=1000` where
factor 1 was best (16.75%). These regret values exclude selector overhead.

## Decision overhead and cache

Fifty samples per loop followed a discarded warmup. The benchmark covered
scalar recurrence, elementwise sine, vector reduction, and dense matvec.
A fresh workload function and empty memory cache were used for each
miss. Argument creation and model loading were outside the timed regions.

| CPU path | Median (ms) | p95 (ms) |
|---|---:|---:|
| Jaxpr extraction and feature computation | 0.4823 | 0.5138 |
| Four forecasts and policy | 0.0279 | 0.0315 |
| Complete decision on a cache miss | 0.5091 | 0.5791 |
| Trace again, then hit the decision cache | 0.2218 | 0.2413 |
| Reuse a known signature and hit the cache | 0.0046 | 0.0050 |

The cache key includes the structural body hash, input, output, carry and
constant shapes, dtypes, trip count, loop kind and primitive, budget, horizon,
and mode. Backend, feature version, and model version namespace the entries.
An in-memory hit can bypass feature extraction when the caller supplies the
signature returned for the same loop and abstract inputs. The optional JSON
store is an atomic snapshot of the in-process decisions. Changing the
backend or a version invalidates prior entries.

Evaluation adds the pooled median miss once to the selected factor's measured
`C + R × T / 1000` per distinct signature and context. This is approximately
`0.5091 / R` ms per call and is outside the cold budget test. Further calls
using the existing decision incur no new miss, and a later explicit
known-signature lookup costs the hit time shown above. Tables assign zero
decision overhead to both comparison baselines and the oracle. The
[overhead artifact](../results/selector_cpu_2026-11-17_overhead.json) contains
the per-loop timings.

## Frozen holdout gate

Both evaluations used R = {1, 10, 100, 1,000, 10,000, 100,000} and
B = {unlimited, 1, 10, 100 ms}. The gate requires a strict paired
end-to-end win against both factor 1 and LLVM size, including selector
overhead. At a finite budget all three choices must meet it.

| Gate | Frozen model | Unique configs | R/B settings per config | Paired wins | Result |
|---|---|---:|---:|---:|---|
| 1 | [`cpu-ridge-v1-7864d652dd6de258`](../models/cost_cpu_gate1.json) | 20 | 24 | 0 | **Fail** |
| 2 | [`cpu-ridge-v1-7d1bfd3b3d66fd5b`](../models/cost_cpu.json) | 20 | 24 | 0 | **Fail** |

The [gate log](../results/selector_cpu_holdout_gates.jsonl) records the two
evaluations, model and dataset hashes, and status. The retry used dev LOFO
evidence. Each frozen model scored holdout configurations once. The
[R/B and policy tables for gate 1](../results/selector_cpu_2026-11-17.md)
and [gate 2](../results/selector_cpu_2026-11-17_gate2.md) have companion
[gate 1 CSV](../results/selector_cpu_2026-11-17.csv) and
[gate 2 CSV](../results/selector_cpu_2026-11-17_gate2.csv) with all seven
policies and family-level rows.

| Gate 2 at R=100,000, unlimited B | Selector median total | Median regret [95% CI] | Oracle agreement | Median paired speedup vs LLVM size |
|---|---:|---:|---:|---:|
| All holdout | 10,864.520 ms | 0.007% [0.002%, 4.78%] | 60% | 1.523× |
| Indexed memory | 6,293.370 ms | 6.39% [1.98%, 10.31%] | 20% | 1.208× |
| Iterative solver | 23,901.559 ms | 0.002% [0.001%, 0.006%] | 100% | 1.537× |

The median pairs pool ratios across configurations. Each confidence interval
uses 2,000 configuration bootstrap resamples (seed 583), preserving pairing
between policies. The selector chose factor 1 for both families at every
setting. At R=100,000 the measured oracle chose a larger factor for eight
indexed-memory configurations. The frozen model's median warm forecast on
that family was 23.76× the measurement. On iterative solver the median cold
forecast was 0.084× the measurement. With B=100 ms, six of the 20
configurations had no measured feasible factor and overall measured violation
was 30%. At B=1 and 10 ms every held-out configuration violated the budget
even with factor 1. The oracle also returns factor 1 when no measured choice
meets a budget.

## Limits and reproduction

The four dev families contain one workload apiece and cover a limited set of
trip counts and shapes. LOFO tests transfer to an unseen dev workload, and its
metrics also guided model choices. The absolute forecasts extrapolate to
holdout operation mixes. Budget feasibility uses those forecasts, with
measured violations reported separately. The replay combines measured cold
and warm medians and a representative CPU overhead charge. It does not
measure integration or attribute differences to optimized HLO.

```bash
source .venv/bin/activate
python -m jax_unroll.train --data data/features_cpu_2026-11-17.csv --out models/
make evaluate                   # dev LOFO comparison using the overhead artifact
make decision-overhead          # take a new timing sample
python -m pytest -q
make quick
```

The held-out results are published above. The evaluator's explicit
`--holdout-gate` flag records one evaluation per frozen model version and
rejects a version already in the gate log.
