# Learning from costly fori_loop predictions

The three families in this experiment serve as regression controls in the broader proposal-aligned evaluation, which also covers bodies with repeatable gains at factors 2, 4 and 8. [Proposal-aligned results](fori_proposal_evaluation.md)

## Presentation summary

The regret-weighted refit made modestly better factor-effect predictions and avoided two expensive warm selections. Its live total was 6.7% lower than the incumbent at 5,000 calls and higher at 1 and 300 calls. The payoff rule cut the 5,000-call total by about 9% versus the incumbent by keeping factor 1. Native factor 1 was fastest overall at every measured horizon. Keep the incumbent as the general static predictor and use native factor 1 for these held-out workloads.

## Prediction and warm runtime

The historical development ranking found 177.2 and 173.9 µs of missed savings per call on covariance and Gram accumulation. Harmful unrolling added 25.8 µs on a coupled oscillator and 12.7 µs on a patch-write case. The new development drafts reproduced both types of mistake. The incumbent missed up to 56.9 µs on the negative-binomial bank and cost 11.9 µs above the best factor on the packet scoreboard. [Development ranking](../results/fori_regret/development_regret.csv) · [Expanded training ranking](../results/fori_regret/training_regret.csv)

The new sweep recorded 672 first-call measurements and 6,048 batches of synchronized warm timings over 56 configurations. The fit used 549 development configurations from 141 bodies. Five-fold validation kept related families together and weighted each collection, then each workload, equally. The training multiplier increased with the square root of absolute regret through 30 µs per call for harmful unrolling and missed speedups. The selected model blends 25% of the new extra-trees log forecasts with 75% of the incumbent. The collection-weighted development effect log MAE went from 0.263 to 0.259. The best unweighted fit reached 0.258 in the same folds, so weighting itself has no demonstrated advantage. [Validation grid](../results/fori_regret/validation.csv) · [Training keys](../results/fori_regret/training_keys.csv)

| Held-out model | Factor-effect log MAE ↓ | Workload-weighted speedup ↑ | Pooled speedup ↑ | Slowdowns >3% | Median / worst slowdown |
|---|---:|---:|---:|---:|---:|
| Incumbent | 0.414 | 0.930× | 0.869× | 6/24 | 1.339× / 1.712× |
| Regret-weighted refit | 0.409 | 0.969× | 0.981× | 4/24 | 1.037× / 1.712× |
| Same refit, historical data only | 0.406 | 0.949× | 0.921× | 5/24 | 1.037× / 1.712× |

Factor-effect log MAE compares measured and predicted `log(warm_u / warm_1)` for factors 2, 4 and 8. Speedups use native factor 1. Workload weighting gives each body equal weight in log space. One call per configuration summed to 869 µs for native factor 1, 1,000 µs for the incumbent, and 886 µs for the refit. The median slowdown uses cases above 3%. Counting all slowdowns, the incumbent had 12 and the refit had 10. The refit improved absolute factor-effect error by 0.0049, with a three-family bootstrap interval of −0.0046 to +0.0161. Mean measured regret fell from 5.45 to 0.70 µs per call, mostly from choosing factor 1 on two spectral equalizer configurations. The held-out data test avoidance of regressions and offer limited evidence about finding speedups in new families. [Holdout metrics](../results/fori_regret/holdout.json)

## Payoff rule

For each larger factor, multiply a lower estimate of warm savings by `R - 1`. The cold forecast includes the first synchronized call. Subtract the nonnegative estimated additional cold time, trace-time decision cost, and an optional share of model loading. Unroll only when this conservative surplus is positive and the candidate's estimated first-call cost, including decision time, fits `B`. The selector uses factor 1 on uncertain forecasts and skips feature extraction for horizons of zero or one call.

Calibration uses one-sided 80th-percentile residuals from family-grouped development predictions that forecast a warm win. The warm ratio receives an upward correction and baseline warm time a downward correction. Separate cold residuals correct the incremental cost and budget check. Development wrapper traces supplied the decision cost. The experiment budgets 125 ms per specialization and estimates the model-load share across six specializations. The empirical margins are specific to the measured CPU environment. [Calibrated margins](../results/fori_regret/payoff.json)

## Direct wrapper measurements

Pooled seconds per process across all 24 frozen configurations. Each row uses the stated number of synchronized calls per configuration.

| Calls | Native factor 1 | Incumbent | Refit | Incumbent + payoff | Refit + payoff |
|---:|---:|---:|---:|---:|---:|
| 1 | **0.939** | 2.224 | 2.279 | 2.196 | 2.085 |
| 300 | **1.187** | 2.528 | 2.575 | 2.484 | 2.441 |
| 5,000 | **5.252** | 7.204 | 6.722 | 6.563 | 6.530 |

Both payoff variants chose factor 1 in all 24 configurations at every horizon. At 5,000 calls, the incumbent and refit took 4.53 ms and 7.09 ms respectively for the median trace-time decision. The payoff variants' median decisions at one call took about 0.005 ms, with feature extraction skipped. Median model import and load in the full run ranged from 1.11 to 1.21 s per process. Both guarded variants produced the same factors, so their small timing differences provide no evidence that one model selects better under this policy.

Every wrapper arm was over 3% slower than native in 24/24 configurations at each horizon, except the warm-only incumbent at one call, which exceeded 3% in 23/24. At 5,000 calls, the median and worst slowdown ratios were 1.54× and 2.10× for the incumbent, 1.47× and 2.12× for the refit, 1.43× and 1.99× for the incumbent with the payoff rule, and 1.45× and 1.93× for the refit with the payoff rule. The initial six-configuration matrix had identical warm-only factors across both models. The full matrix includes the two spectral equalizer cases where the refit switched to factor 1. [Full wrapper summary](../results/fori_regret/live_all_summary.json) · [Initial six-configuration summary](../results/fori_regret/live_summary.json)

## Extrapolated totals and break-even

Keeping the 5,000-call decisions fixed, the estimated pooled speedups versus native factor 1 at 20,000 calls were 0.823× for the incumbent, 0.914× for the refit, and 0.933× and 0.934× for the two payoff variants. At 100,000 calls they were 0.858×, 0.966×, 0.985× and 0.986×. These extrapolations append measured warm rates to the direct 5,000-call totals and use native's warm rate for identical factor-1 code.

No wrapper arm has a pooled break-even versus native from the full-matrix measurements. Factor 1 was the warm oracle for all 24 configurations, and the payoff rule incurs startup cost on the same factor-1 execution. One per-configuration refit estimate extrapolated a small trie timing difference to roughly 2.2 million calls. The native sweep found factor 1 faster for that case. [Extrapolations and break-even estimates](../results/fori_regret/live_all_summary.json)

## Frozen split and protocol

The incumbent is `results/fori_predictor/model.joblib` with the documented warm-throughput selector. Rank historical development configurations by selected median warm time minus the lowest measured factor median, in microseconds per call. Use the ranking to target the development drafts. Reserve all bodies from the three holdout families before native timing or policy calibration.

| Partition | Body | Mechanism and overlap review |
|---|---|---|
| Development | Branch tag cache, packet reorder scoreboard | Carry-dependent indexed writes. Keep these together in the associative state writes family. The tag cache reads and conditionally replaces a way. The scoreboard sets a bit and retires a pending bit by inspecting a carry-selected word. Neither performs the existing hot block gather, patch blend, histogram increment, or fixed-stride delay. |
| Development | Negative-binomial bank | Dense likelihood and carry feedback over a bank of models. No outer-product or Gram updates. |
| Development | Discounted path soft minimum | Fused exponential and mean reduction across per-step hypotheses, with bounded feedback. Different carry and reduction from the covariance and norm-clipping cases. |
| Holdout | Spectral equalizer | Nonlinear per-band gain feedback and weighted residual reduction. The existing overlap-add and spectral detector have different state and operations. |
| Holdout | Packed prefix rank | Bit-parallel arithmetic popcount over carry-selected prefixes. It has no table updates or queue and shares no draft lineage with the scoreboard. |
| Holdout | Compressed trie walk | An integer state selects the next row with miss-link feedback. The existing indirect B-tree gather uses external indices, then accumulates the gathered values. |

Gemini 3.8 Flash via Anti-Gravity drafted the likelihood bank, scoreboard, equalizer, packed prefix rank, and trie cases. OpenCode drafted the branch tags and soft minimum with a local GPT-OSS 120B provider backed by the Anti-Gravity CLI, after the configured OpenCode providers failed to return an eligible draft. Review corrected the OpenCode draft's carry structure, deterministic inputs, sizing, and dense reduction. The heap, ring-buffer encoder, butterfly, ray clipping, and frontier drafts were dropped for overlap with the existing heap, zero-run decoder, FFT, ray-box and graph frontier cases. The draft frontier also saturates quickly.

Rebuild each accepted body at factors 1, 2, 4 and 8 with identical arguments. Compare Jaxprs and outputs against native factor 1. Sweep two working-set sizes and trips 255, 256, 1023 and 1024. Use three fresh CPU processes, randomized factor order, and nine interleaved synchronized warm samples. Save first-call and raw warm samples. Freeze the fit and payoff calibration before collecting holdout measurements.

The initial wrapper benchmark uses every held-out body at small/256 and large/1023. The follow-up covers all 24 frozen configurations with the same models and policy. Both execute 1, 300 and 5,000 synchronized calls per configuration in three fresh processes per arm. Each direct total sums measured model import and load, JIT tracing and lowering, compilation, and all calls in that horizon. Decision time is measured inside the trace interval. Model load is charged once per process per horizon. The JAX backend is warmed and inputs are resident before timing. Larger extrapolations keep the 5,000-call decision fixed.

## Reproduce

```sh
export PYTHONPATH=src:.
export JAX_PLATFORMS=cpu
.venv/bin/python -m benchmarks.corpus.fori_regret.development
.venv/bin/python -m benchmarks.corpus.fori_regret.freeze
.venv/bin/python -m benchmarks.corpus.fori_regret.acquire train
.venv/bin/python -m benchmarks.corpus.fori_regret.train fit
.venv/bin/python -m benchmarks.corpus.fori_regret.calibrate
.venv/bin/python -m benchmarks.corpus.fori_regret.acquire holdout
.venv/bin/python -m benchmarks.corpus.fori_regret.train score
.venv/bin/python -m benchmarks.corpus.fori_regret.live
.venv/bin/python -m benchmarks.corpus.fori_regret.live --all-configs
.venv/bin/python -m pytest -q
```

The measurements use an Apple M2 CPU with JAX and jaxlib 0.12.0.1. The freeze command verifies source digests against the recorded split. Raw native timings, the frozen models, calibration, and individual wrapper measurements are retained with the summaries. [Benchmark artifacts](../results/fori_regret/)

To try the rule, construct `PayoffPolicy(**entry["policy"])` using the entry for your saved model in `payoff.json`. Pass it to `DecisionEngine(model, payoff=policy)`, then call `auto_fori_loop` with that engine, the expected call count `R`, and an optional millisecond budget `B`.
