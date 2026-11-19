# Five iterations on compile-once JAX loops

The five cycles use the compact warm selector from the proposal-aligned benchmark as the reference. The previous benchmark's 32 fresh configurations became development data for this round. Six new families were frozen before timing and excluded from model fitting and payoff calibration. All comparisons use unchanged `jax.lax.fori_loop` bodies.

## What changed in each cycle

| Cycle | Experiment and result | Disposition |
|---:|---|---|
| 1 | Export the same tree arrays without ZIP compression. Seven fresh processes per format put median model load at **3.29 ms vs 6.27 ms**. All 41 previous forecast rows matched exactly. The artifact grew from 1.70 MB to 2.69 MB. | Offer the uncompressed incumbent as a fast-loading option. |
| 2 | Reuse the rolled JAX trace when selecting and applying the factor. On 48 paired development configurations the reproduced median savings were **0.08 ms** for tracing and lowering and **0.10 ms** through the first call. Jaxprs, choices, and outputs matched. | Keep the IR rewrite in the benchmark. |
| 3 | Add the prior mixed suite to family-grouped training. Development pooled warm speedup improved from **1.018× to 1.039×**, and workload-weighted factor-effect log MAE fell from **0.253 to 0.245**. Warm slowdowns above 3% increased from 45 to 68 of 581 configurations. | Retain the refit for comparison and keep the incumbent as the baseline. |
| 4 | Fit a margin on the error in predicted per-call savings. Replaying grouped development forecasts at 20,000 calls gave **1.003×** pooled speedup with 55 larger factors, versus **0.994×** and zero larger factors for the previous payoff rule. At 5,000 calls the two rules were effectively tied in replay. | Offer the calibrated rule as an opt-in. |
| 5 | Defer model loading until selection needs a forecast. A known single-call loop took **11.97 ms vs 13.83 ms** in paired development measurements, with zero model loads on the deferred path. A budget below decision cost also bypasses prediction. | Keep deferred loading as an opt-in for either payoff rule. |

Cycle 3 compares six ExtraTrees fits using a split that keeps related development cases together. Cycle 4 selects among 12 development configurations of the calibration procedure. The selected rule uses the incumbent and takes a one-sided residual quantile in each predicted latency band. It subtracts a margin from the predicted difference in warm times, then checks the modeled cost of additional compilation, the decision, and the shared model load. The live budget was 125 ms. [Cycle measurements and model files](../results/fori_iterations/)

## Independent confirmation

The matrix has six new families with two settings each. Each native factor was compiled and tested in three independent CPU processes. Eight of the 12 settings had a material warm opportunity, defined as at least 1.05× median paired gain with a warm win in all three repeats. The native sweep oracle gave **1.160× pooled warm speedup**. The incumbent reached **1.123×**, and the refit reached **1.116×**. Factor-effect log MAE improved from **0.232 to 0.167** with the refit, while warm slowdowns above 3% went from zero to one. [Frozen matrix](../results/fori_iterations/confirmation_matrix.json) · [Static confirmation](../results/fori_iterations/confirmation_static.json)

### Direct runtime

Pooled seconds per process across all 12 configurations. Totals include setup and every synchronized call. Each specialization compiles once. Loaded models are charged once per process and horizon. The warm oracle uses measured factors from the separate native sweep, with the search excluded from runtime.

| Calls per configuration | Native factor 1 | Compact incumbent | Compact refit | Old payoff | New payoff + deferred load | Warm oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.304 | 0.356 | 0.436 | 0.299 | 0.290 | 0.428 |
| 300 | 0.426 | 0.425 | 0.513 | 0.407 | 0.400 | 0.511 |
| 5,000 | 1.738 | 1.604 | 1.688 | 1.684 | 1.782 | 1.633 |
| 20,000 | 5.821 | 5.311 | 5.486 | 6.011 | 5.829 | 5.196 |

The warm incumbent reduced pooled total time by **7.7% at 5,000 calls** and **8.8% at 20,000 calls** versus native. At 20,000 it won all three pooled comparisons. Six configurations were at least 1.05× faster, with zero slowdowns above 3%. In the eight opportunity cases it saved **23.9%** pooled at 20,000 calls. The estimated pooled crossover was about 1,800 calls using the first-call and measured warm timing at 20,000 calls and holding the decisions fixed.

The refit predicted the factor effects more accurately and ran slower overall than the incumbent at both long horizons. At 20,000 calls it had three slowdowns above 3%, with a worst case of 10.7%. The new payoff rule chose larger factors in two configurations at 20,000 calls, up from zero for the old rule. It cut 3.0% off the old policy's pooled total and was effectively tied with native factor 1. Both payoff arms chose factor 1 at 1, 300, and 5,000 calls. The apparent timing differences from native at those horizons include run-to-run variation on identical compiled bodies. Cycle 5 isolates the setup effect by comparing deferred and eager loading under the same rule.

Keep the incumbent for the warm-throughput path. Use an uncompressed artifact when startup cost matters, and defer loading for a caller that uses a payoff rule and may bypass prediction. The savings-rule calibration is experimental. [Live timings, factor counts, and crossover estimates](../results/fori_iterations/confirmation_summary.json)

Export uncompressed arrays with `CompactForiPredictor.export(source, destination, compressed=False)`. Load them with `CompactForiPredictor.load(destination)`. For deferred loading, pass `model_loader=lambda: CompactForiPredictor.load(destination)` to `DecisionEngine`, along with the desired payoff rule.

## Reproduction

```sh
export PYTHONPATH=src:.
export JAX_PLATFORMS=cpu
.venv/bin/python -m benchmarks.corpus.fori_iterations.confirmation freeze
.venv/bin/python -m benchmarks.corpus.fori_iterations.storage
.venv/bin/python -m benchmarks.corpus.fori_iterations.trace
.venv/bin/python -m benchmarks.corpus.fori_iterations.refit
.venv/bin/python -m benchmarks.corpus.fori_iterations.payoff
.venv/bin/python -m benchmarks.corpus.fori_iterations.confirmation sweep
.venv/bin/python -m benchmarks.corpus.fori_iterations.defer
.venv/bin/python -m benchmarks.corpus.fori_iterations.validate
.venv/bin/python -m pytest -q
```

Run these on the pinned Apple M2 CPU environment with JAX and jaxlib 0.12.0.1. Setup and warm samples are retained per process. The timed wrapper variants verify output equality within workload tolerances and Jaxpr equality after changing only the unroll parameter.
