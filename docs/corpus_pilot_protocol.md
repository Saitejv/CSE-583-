# Expanded corpus: development pilot v1

This is a development experiment, not a third gate. The corpus implementation
and membership come from `a6efbbb`. Neither historical holdout nor scored
recovery confirmation measurements are inputs, including for fitting,
diagnostics, or deciding what to measure. The previous recovery model is
read-only. Do not use the earlier one-repeat smoke timings in this experiment.

## Acquisition

Use Apple M2 CPU and pinned JAX/jaxlib 0.12.0.1. Measure **all 47 corpus
bodies**, one primary variant each, at the long nondivisible trip count
1023, with factors 1/2/4/8 and three fresh-process repeats (564 measurements).
The order is the existing harness's seed-583 shuffle by process repeat.
Use existing sweep defaults and finite/close-to-factor-1 correctness checks.
Disable persistent compilation caching and x64. Require a complete correct
four-factor matrix and three valid repeats; report and drop an entire body
if any timing row is missing or invalid. Do not filter by performance.
The protocol and corpus manifest hashes are recorded before the sweep.

This matrix prioritizes independent bodies over shape replicates. It cannot
measure sensitivity to short trips, tails vs no tail, secondary shapes, or
other CPUs. The full 4,560-row tier is out of scope for this first diagnostic.

## Analysis (fixed before timing)

Aggregate medians with the original sweep/dataset builders. Study
`log(T(u)/T(1))` for factors 2/4/8. Compare:

1. Zero warm effect (same warm latency for every factor).
2. The frozen recovery model's warm ratios.
3. A per-factor median ratio learned on other corpus groups, without features.
4. A warm-effect-only refit using the same eight static design terms and
   ridge alpha 0.025 as recovery. Train only on other corpus groups; no other
   fit, feature search, or model selection. Keep cold predictions and
   factor-1 warm predictions frozen for the policy replay.

Leave out one family component per fold: connect all families that share
a manifest `related_group` (transitively); never separate related bodies
or split rows for a body. Exclude historical dev rows from fitting as well,
to avoid leaking close structural relatives. Report pooled and per-component
warm log-ratio MAE, bias, direction agreement, and paired mean absolute-error
differences vs zero with 2,000 body-bootstrap draws (seed 583, average the
three errors within each body first). Separately count how often the direction
of the measured factor effect is consistent across all three repeat blocks;
do not use that count to exclude observations. Report absolute cold and
warm calibration for the frozen and refit forecasts.

Replay the existing R/B grid without decision overhead: compare factor 1,
`llvm_size`, measured oracle, frozen recovery and warm-only refit. Report
all, feasible long-horizon, and oracle-improves-long strata separately,
including measured budget violations, factor agreement, paired regret and
speedup. These are diagnostic replays, not end-to-end wins. The signal is
promising enough to justify scaling timing **for model training** if the
warm-effect refit beats both simple warm baselines in pooled MAE, beats
zero in a majority of independent folds, and its paired MAE-difference
interval against zero lies below zero. Otherwise prioritize feature/label
diagnostics over a full sweep. Either way, factor 1 remains the default:
this pilot cannot promote or freeze a replacement policy.
