# Static-feature experiment: failed transfer criterion

**Do not launch the full training sweep.** The predeclared static extension made held-out warm predictions worse. Keep factor 1 as default and recovery v1 unchanged. This is development evidence, not a confirmation gate.

## Fixed experiment

The [protocol](static_feature_protocol.md) and input hashes were written before new measurements. Three agents separately audited structure/carry, index provenance, and matched-input grouping; an independent fourth reviewer checked the merged implementation and actual outputs. No feature or hyperparameter selection followed the results; no nested selection was needed. Original holdout and scored confirmation labels were excluded from this experiment. The required historical verification commands were run separately without tuning or a new score.

Apple M2 CPU, JAX/jaxlib 0.12.0.1, x64 and persistent compilation cache disabled. The new matrix comprises three bodies, two runtime index inputs each, T=1023, plus T=1024 for gather/add only, four factors, three fresh-process repeats: **120/120 correct, no dropped bodies**. Existing pilot data adds 564 measurements. All 20 declared trip/factor pairs have identical complete Jaxprs and deterministic v1/v2 features across their two inputs.

## Held-out warm log-ratio error

Original 47-body pilot is the headline. Eight family-component folds keep all related bodies together; matched variants/trips join the indexed component. Lower is better; differences are paired candidate-minus-zero absolute error with 2,000 body-bootstrap draws (seed 583).

| Predictor | MAE | Difference vs zero [95% CI] | Folds beating zero |
|---|---:|---:|---:|
| zero | 0.359 | +0.000 [+0.000, +0.000] | 0/8 |
| factor_median | 0.385 | +0.026 [+0.015, +0.036] | 1/8 |
| frozen | 0.337 | -0.022 [-0.043, +0.001] | 6/8 |
| warm_refit | 0.460 | +0.101 [+0.019, +0.198] | 3/8 |
| structure | 0.663 | +0.304 [+0.181, +0.442] | 1/8 |
| indices | 0.508 | +0.149 [+0.050, +0.266] | 3/8 |
| all_static | 0.686 | +0.327 [+0.203, +0.472] | 0/8 |

**Primary fails all three criteria:** it loses to both simple baselines, wins zero folds, and its interval shows worse error. Structure and index ablations also fail; they are diagnostics, not replacement winners. Paired comparisons against the median and same-design refit are retained in the JSON.

| Held-out component | Zero | Same design | Structure | Indices | Combined |
|---|---:|---:|---:|---:|---:|
| control_arithmetic | 0.185 | 0.544 | 0.164 | 0.907 | 0.367 |
| dependent_indexing | 0.211 | 0.226 | 0.342 | 0.272 | 0.303 |
| indirect_gather+indirect_scatter+locality_access | 0.179 | 0.393 | 0.585 | 0.394 | 0.677 |
| rolling_window | 0.496 | 0.398 | 1.298 | 0.405 | 1.233 |
| segmented_scan | 0.107 | 0.081 | 0.313 | 0.105 | 0.274 |
| stateful_recurrence | 0.188 | 0.258 | 0.303 | 0.325 | 0.273 |
| stencil | 1.952 | 1.874 | 2.102 | 1.795 | 1.980 |
| streaming_reduction | 0.138 | 0.224 | 0.437 | 0.329 | 0.466 |

Augmented sensitivity analysis (50 bodies, 57 configurations): zero 0.346, median 0.388, same-design 0.459, combined 0.813; 0/8 wins. Its body-weighted paired delta is +0.377 [+0.243, +0.521]. This differs from the row-pooled MAE subtraction because variants/trips are averaged inside body before bootstrap. Neither analysis supports scaling.

Pilot direction stability reproduces 133/141; augmented stability is 163/171. Median repeat log-ratio range is 0.018 pilot / 0.017 augmented. These counts do not filter rows.

## What the static blocks revealed

The extension describes mechanisms that v1 misses: stencil carry slice counts are 4/8/24/12 for 1D/five-point/nine-point/anisotropic, versus two for the rolling-dot body. It separates scatter assignment/add/min and propagates constant, counter, scanned-input and carry origins through supported jit wrappers. Carry update bytes and path depths describe SSA structure, not physical traffic. Loop-invariant runtime arrays count as constant origins; a carry dependency does not prove data-dependent values. Scalar dynamic_slice is not counted as gather. Unsupported nested regions are marked opaque.

These distinctions did not transfer as a fixed ridge warm-effect design. The index-only block helps the rolling, stencil and segmented folds relative to zero, but loses overall; structure helps only the control fold. Combined errors are large in rolling and indexed workloads as well as stencil. Slice counts alone do not forecast compiler transformations or code growth. With only eight independent components and correlated structural proxies, held-out extrapolation remains poor. This experiment cannot establish that all static representations fail; it rejects this predeclared one.

## Runtime inputs remain invisible

Each row below compares log(T(u)/T(1)) under input B minus input A. Every pair has identical static features. Signs agree in all three repeats for **13/15** comparisons, including all six gather comparisons. Reported deltas are descriptive; three repeats do not establish statistical significance. No timing outlier was removed.

| Body | T | A → B | u2 median delta | u4 median delta | u8 median delta | Stable comparisons |
|---|---:|---|---:|---:|---:|---:|
| matched_gather | 1023 | clustered → dispersed | +0.117 | +0.089 | +0.102 | 3/3 |
| matched_gather | 1024 | clustered → dispersed | +0.101 | +0.104 | +0.123 | 3/3 |
| matched_scatter_add | 1023 | colliding → unique | +0.011 | +0.042 | +0.084 | 1/3 |
| matched_scatter_add | 1024 | colliding → unique | -0.011 | +0.015 | +0.072 | 3/3 |
| matched_scatter_min | 1023 | colliding → unique | +0.024 | +0.045 | +0.108 | 3/3 |

Gather factor-relative differences persist at divisible T=1024. Scatter-add u2/u8 at T=1023 change difference signs across repeats; the T=1024 subset is consistent but does not isolate a causal tail effect. Raw repeat deltas and ranges are retained. Static Jaxpr dependence cannot recover locality or collisions encoded only in runtime index values.

## Diagnostic policy replay: no decision overhead

These are **not end-to-end results**. The existing budget/horizon grid uses frozen cold forecasts and factor-1 warm forecasts; only warm effects change. The 47-body pilot has 180 feasible long settings and 89 where the measured oracle beats factor 1. Regret includes the measured cost of the chosen factor even when it exceeds budget.

| Predictor/policy | Feasible long mean positive regret | Useful long median / mean positive regret | Budget violations all / feasible-choice cases | Long / useful violations |
|---|---:|---:|---:|---:|
| factor_1 | 13.46% | 16.24% / 27.23% | 564 / 0 | 0 / 0 |
| llvm_size | 55.26% | 0.00% / 5.24% | 648 / 84 | 20 / 4 |
| frozen | 10.91% | 9.12% / 17.08% | 569 / 5 | 0 / 0 |
| warm_refit | 13.12% | 2.94% / 16.86% | 564 / 0 | 0 / 0 |
| all_static | 32.48% | 4.33% / 23.78% | 583 / 19 | 14 / 5 |

All policies face 564 grid settings with no measured feasible factor. Combined features add 19 violations where a measured feasible factor exists (frozen adds five). The augmented replay is also worse: 39.49% feasible-long mean positive regret, 35 additional violations across the full grid. Full stratum counts, speedups, agreements, ablations, calibration and paired setting rows are retained. Improved median regret on a useful subset does not compensate for poor pooled prediction and worse average regret.

## Reproduction and verification

Run [commands.sh](../results/static_features/commands.sh) from the repository root to reacquire the 120-row matrix and rebuild reports. Report-only: `JAX_PLATFORMS=cpu PYTHONPATH=src:. .venv/bin/python -m benchmarks.corpus.static_experiment`. Hash receipts, environment, acquisition logs, separate trip raw CSVs, merged raw CSV, features, folds, ratios, replay pairs, and [report JSON](../results/static_features/report.json) are retained under `results/static_features/`.

Independent review recomputed every MAE, solved each fold ridge equation independently (predictions agree within 1e-9), checked grouping, and recounted every replay violation. Frozen v1 feature/model/selector files and model artifact remain unchanged; auto_scan/auto_fori_loop remain stubs. No commits were made, preserving the preexisting uncommitted pilot and user edits.

Validation status is recorded in [checks](../results/static_features/checks.md). **A larger training sweep is not justified by this design. Stop before any new confirmation gate.**
