# Static-bound fori_loop development protocol

Frozen design, 2026-11-17. This retrospective experiment asks whether a smaller warm-effect forecast transfers across the measured public `jax.lax.fori_loop` families. The pilot labels and earlier mixed-corpus results already exist. Only the two new candidates below are prospective to this analysis.

## Development population and folds

Use the retained expanded-corpus pilot on Apple M2 CPU, JAX/jaxlib 0.12.0.1, primary shape, integer bounds 0 and 1,023, factors 1/2/4/8 and three process repeats. Identify the public loop API from the corpus specification. These static loops trace to JAX's internal `scan` primitive. Apply the pilot complete-matrix, version, correctness and positive-time checks. A missing or invalid matrix excludes the whole body and must be reported. A missing test component prevents a four-fold result.

Construct the transitive family components from the **whole** corpus manifest before intersecting with eligible public bodies. Hold out each entire component once and train on all other eligible components. Use body identity as the minimum unit for splitting. Keep every shape, trip, factor, process repeat and runtime variant of a body together. Freeze the following test memberships.

| Held-out component | Eligible bodies | Training bodies |
|---|---|---:|
| dependent_indexing | dep_bucket_stateful, dep_linked_list, dep_pointer_chase, dep_two_level_indirection | 7 |
| indirect_gather+indirect_scatter+locality_access | locality_block_reuse, scatter_min_update | 9 |
| stateful_recurrence | stateful_coupled_oscillator | 10 |
| stencil | stencil_anisotropic_2d, stencil_diffusion_1d_fori, stencil_five_point_2d, stencil_nine_point_2d | 7 |

This gives 11 of the 47 pilot bodies, four of eight components, 132 raw timing rows, 44 absolute cost pairs and 33 nontrivial warm effects. The historical recovery development join has 284 public scan rows. The matched-input acquisition has 120 public scan rows. Neither supplies rows to the present fits or evaluation. The matched scan experiment cannot quantify runtime-input effects in public fori_loop. Use the pilot development measurements only. Historical holdout and scored confirmation labels are outside this analysis.

## Predictions fixed before scoring the two candidates

For each body and candidate factor `u`, set the target to the natural log of the ratio of the three-repeat median warm time at `u` to that at 1. Predict from the factor-1 static trace and the candidate integer factor. Use the frozen recovery cold forecasts at every factor and its factor-1 warm forecast for every method. Add each predicted log effect to the latter to obtain absolute log warm time. The forecast for factor 1 has exactly zero effect. No measured factor-1 cost is available at inference. Runtime array values, names, hashes, family membership, compiler outputs, budgets and horizons are excluded from model inputs.

Establish four baselines first.

1. Predict zero warm effect.
2. For each factor independently, take the median measured log effect of training bodies in the outer fold.
3. Use the frozen recovery v1 artifact as shipped. Its training data predates this pilot.
4. Refit its warm-effect equation with the original ridge penalty 0.025, no intercept. Use the existing eight `memory_work` columns without changes to scaling. These are the products of each factor basis (`1-1/u`, `ln(u)`) with each of 1, `T/(T+128)`, `log1p(flops_proxy)/8`, and `log1p(bytes_moved_proxy/(1+flops_proxy))/8`. Fit on all four factors within the training fold. The factor-1 design row and target are zero.

| Candidate | Fixed equation | Reason |
|---|---|---|
| Primary, half frozen effect | Multiply the frozen warm log effect at every factor by exactly 0.5. | The frozen predictor transferred better than earlier refits in the mixed pilot, but its advantage over predicting zero was uncertain. The midpoint tests whether retaining its direction at lower magnitude helps without estimating any coefficient from four groups. |
| Secondary, ridge 1 | Repeat baseline 4 with ridge penalty exactly 1.0, the existing absolute-base penalty. Keep its eight columns, units, no intercept and ordinary unweighted sum-of-squares objective. | At fixed trip, the trip-weight column is proportional to the constant within each factor basis. Training on 7–10 bodies offers little independent variation. Stronger shrinkage can reduce unstable extrapolation at the cost of bias. |

The primary remains the headline under all outcomes. Both candidate equations, penalties and the selector are fixed. No inner search, cutoff fitting, model selection, recalibration or feature elimination takes place. Any later data-driven choice would require its own nested component validation and sufficient independent groups. Four outer components, including a singleton, support a development diagnostic with limited evidence for generalization.

## Error, repeat stability and policy value

Report pooled and per-component mean absolute warm log-ratio error across factors 2, 4 and 8 for all six predictors. For paired error intervals, subtract baseline absolute error from method absolute error on each matched body/factor. Average factors within each body, sample bodies with replacement 2,000 times (`default_rng(583)`, lexicographic body order) and take the 2.5th and 97.5th percentiles of mean differences. Report at least the comparison with zero. The interval reflects the measured body mix and does not estimate uncertainty over future families. Separately count effects with the same sign in all three repeat-indexed warm log ratios and give their median within-effect range. Never filter by effect size or stability. Report absolute cold and warm log MAE and median predicted/measured over all four factors.

Use the existing constrained total-cost selector and record every choice and matching measured cost. Replay `B` of 1, 10, 100 ms and unlimited across `R` of 1, 10, 100, 1,000, 10,000 and 100,000 calls. Use cold milliseconds plus `R * warm_microseconds / 1000`. Compare all forecast methods with factor 1, LLVM-size (128 equations) and the measured feasible oracle. Preserve existing tie and fallback behavior. The diagnostics charge zero decision overhead.

Report all settings, feasible long horizons (`R >= 10000` with at least two measured feasible factors), and the subset of feasible long horizons where factor 1 is feasible and the measured oracle strictly beats it. Report median regret, mean positive regret, median paired speedup vs factor 1 and LLVM-size, and measured budget violations in each stratum. Regret is measured chosen total / measured oracle total minus 1. Mean positive regret clips each regret at zero before averaging. Compute timing ratios even for violating choices, and count violations separately, including how many occurred when at least one factor met budget. Keep the per-setting rows so invalid choices can be identified. Repeated settings for one body do not increase the number of independent bodies.

## Decision and stopping

Call the fixed primary promising only if its pooled held-out MAE beats **both** zero effect and training-fold median, it beats zero in at least three of the four components, and its paired body-bootstrap upper endpoint versus zero is strictly below zero. Assess practical value separately. Require strictly lower useful-long mean positive regret than **both** factor 1 and frozen recovery, with no more all-grid violations in settings with a feasible measured choice than either comparator. If no useful settings exist, practical value is unassessable.

Publish every baseline and both candidates, including failures. End the experiment after this comparison. A favorable development result can motivate independent data collection. Integration or promotion would need separate evidence across families, shapes, trip regimes and actual decision overhead. Factor 1 remains the default.
