# Can we predict unrolling for static-bound fori_loop?

**Keep factor 1 as the default. This experiment does not support opt-in integration.** Halving the frozen recovery model's warm effect helped in three of four held-out components, but its paired interval includes zero improvement. It made exactly the same choices as frozen recovery in all 264 settings. Stronger ridge regularization also failed to beat zero effect.

## What remains after restricting the public API

Eleven of 47 pilot bodies qualify. They span five named families and **four independent family components** after accounting for related bodies. The folds have 4, 2, 1 and 4 bodies. Each body contributes one primary shape, one trip count (1,023), four factors and three fresh-process repeats on the pinned Apple M2 CPU. All 132 timing rows passed, yielding 44 absolute cost pairs and 33 non-unit factor effects. The three matched runtime-index workloads use the public scan API. No eligible fori_loop has a matched runtime-input pair. [Protocol and fold membership](fori_dev_protocol.md)

## Warm-effect prediction

Warm log-ratio MAE measures error in the predicted change from factor 1. Smaller is better. Columns are pooled and held-out errors. The interval is the method-minus-zero difference in absolute error, paired within body and bootstrapped over 11 bodies. It describes this body mix. Four components, one a singleton, give little precision about a future family.

| Method | Pooled | Dependent indexing | Indexed / locality | Stateful | Stencil | Difference vs zero [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| Zero effect | 0.833 | 0.166 | 0.105 | 0.478 | 1.952 | 0 |
| Training-fold factor median | 1.414 | 1.497 | 0.481 | 0.419 | 2.048 | +0.582 [+0.251, +0.926] |
| Frozen recovery | 0.812 | 0.236 | 0.185 | 0.336 | 1.821 | −0.021 [−0.081, +0.042] |
| Same-design refit | 0.993 | 0.454 | 0.827 | 0.212 | 1.811 | +0.160 [−0.058, +0.384] |
| **Half frozen effect (primary)** | **0.813** | **0.196** | **0.102** | **0.407** | **1.887** | **−0.020 [−0.049, +0.009]** |
| Ridge penalty 1 (secondary) | 0.983 | 0.603 | 0.176 | 0.241 | 1.951 | +0.150 [−0.004, +0.340] |

The primary cleared the pooled and fold-count criteria and missed the interval criterion. For 29 of 33 measured effects the direction agreed in all three repeat blocks. The median repeat range was 0.012 log units and the median absolute effect was 0.251. Large errors persist on repeat-stable outcomes. At factor 8, the anisotropic stencil was 29.7× slower warm than factor 1. The primary forecast 1.16× slower.

All six methods share frozen cold and factor-1 warm forecasts. Across the 44 body/factor pairs, absolute cold log MAE was 0.935, with median predicted/measured cold time of 0.484×. For the primary, absolute warm log MAE was 0.890 and the median predicted/measured warm time was 0.595×. For frozen recovery these were 0.846 and 0.607×. Cold underprediction matters for budget checks, and errors in absolute warm time change the horizon at which an estimated saving repays compile cost. [All methods' calibration and intervals](../results/fori_dev/report.json)

## Factor selection in diagnostic replay

**Feasible long horizons.** There were 40 settings on 11 bodies with at least two measured feasible factors and at least 10,000 calls. The primary's mean positive regret was 15.08%, compared with 9.61% at factor 1. Both had 1.000× median speedup versus factor 1. At 100,000 calls, choosing factor 2 gave 1.46× speedup on the stateful bucket-index workload and 0.39× on the 1D diffusion stencil. Each occurred at the 100 ms and unlimited budgets.

**Where the oracle beat factor 1.** The 16 useful long settings covered just four bodies across two components. The primary's mean positive regret fell from factor 1's 24.02% to 18.11%, tying frozen recovery. It took the faster-than-default choice in two settings on the same bucket workload and chose factor 1 in the other 14. The oracle achieved 1.194× median speedup. Decision overhead was zero throughout this replay, so the figures describe measured choices before integration costs.

| Policy | Feasible long mean positive regret | Useful long median / mean positive regret | Useful long median speedup vs 1 / LLVM-size | Violations, all / where any factor feasible |
|---|---:|---:|---:|---:|
| Factor 1 | 9.61% | 19.41% / 24.02% | 1.000× / 0.837× | 132 / 0 |
| LLVM-size | 169.42% | 0.00% / 0.79% | 1.194× / 1.000× | 162 / 30 |
| Measured oracle | 0.00% | 0.00% / 0.00% | 1.194× / 1.000× | 132 / 0 |
| Zero effect | 9.61% | 19.41% / 24.02% | 1.000× / 0.837× | 132 / 0 |
| Training-fold median | 199.04% | 19.41% / 24.02% | 1.000× / 0.837× | 142 / 10 |
| Frozen recovery | 15.08% | 17.24% / 18.11% | 1.000× / 0.853× | 132 / 0 |
| Same-design refit | 71.28% | 14.81% / 17.68% | 1.000× / 0.871× | 137 / 5 |
| **Half frozen effect** | **15.08%** | **17.24% / 18.11%** | **1.000× / 0.853×** | **132 / 0** |
| Ridge penalty 1 | 73.75% | 19.41% / 24.02% | 1.000× / 0.837× | 136 / 4 |

At least one choice met budget in 132 of 264 settings. In the other 132, every policy necessarily violated budget. The primary met the feasible-case violation criterion, but tied frozen recovery on useful-case regret, failing the predeclared strict improvement criterion. [Per-setting choices, measured costs and violations](../results/fori_dev/policy_pairs.csv)

## What evidence would change the decision?

The fixed-trip warm design has rank 6 across eight columns. Nine of eleven bodies have at least one body-level warm feature outside the training components' observed range. The retained data gives particularly weak coverage of irregular indexing, stateful loops and stencil compiler behavior. Existing mixed-API locality results motivate a controlled test in fori_loop. They cannot quantify how much runtime index values matter here. [Feature and label audit](fori_dev_audit.md)

A targeted **development** acquisition is justified. Prioritize independently authored public fori_loop bodies across new family components, with multiple bodies per component. Select by mechanism before timing, including counter-driven table reads, colliding updates, loop-carried recurrences and neighborhood stencils. For these bodies, cross a short trip and adjacent 1,023/1,024 trips with two working-set sizes. Include matched index-locality and collision variants of the same fori_loop with identical abstract arguments. Time all four factors in repeated fresh processes, retaining cold and warm costs. This would separate cross-family transfer, trip/tail effects, size effects and runtime-only variation. More repetitions of the existing 11 configurations cannot answer those questions. Any later integration claim also needs measured decision overhead. The bounded experiment ends here.

Verification passed all 941 tests, the eight-row quick smoke check, the corpus pilot report, recovery verification and Week 3 verification. A second run reproduced the development outputs byte for byte. The independent reviewer recomputed all 264 absolute forecasts and 1,584 policy rows.

[Reproduction commands](../results/fori_dev/commands.sh) · [raw development measurements](../results/corpus_pilot/primary_1023_raw.csv) · [held-out forecasts](../results/fori_dev/predictions.csv) · [repeat and warm-ratio rows](../results/fori_dev/warm_ratios.csv) · [independent review](fori_dev_review.md)
