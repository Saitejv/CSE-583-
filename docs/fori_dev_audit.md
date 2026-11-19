# Static-bound fori_loop development audit

The retained primary pilot contains 11 eligible public loop bodies in four independent family components. Each uses one primary shape, one static trip count (1023) and one input pattern. Factors 2, 4 and 8 all leave a tail. All bodies use Python integer bounds in the corpus builders and trace to a scan primitive on the pinned CPU backend. The three prior runtime-matched bodies call public scan, leaving zero matched runtime comparisons in this audit. No new timing was acquired.

## What the measurements identify

The audit reconciled 132 correct fresh-process rows to 44 aggregate cold/warm pairs and 33 saved warm effects. Cold is the harness's trace plus adjusted lowering plus compile time in milliseconds. Adjusted lowering clamps lower wall time minus separately measured trace time at zero. Warm is the process median of synchronized calls in microseconds, after compilation and warmup. Each label takes the median across the three processes. The target takes the natural log of the ratio of those warm medians for the candidate and its own factor 1. Negative values mean faster warm execution. Policy replay converts warm microseconds to milliseconds before combining them with cold cost.

Pairing factor and baseline within each repeat block gives a separate noise diagnostic. The median absolute difference between the aggregate target and the median paired-repeat log ratio was 0.00058. In 29/33 comparisons all three repeats agreed on direction. The median repeat log-ratio range was 0.0125 against a median absolute aggregate effect of 0.251. The warm-fastest factor was the same in all three repeats for 10/11 bodies. Median within-candidate log ranges were 0.0417 for cold and 0.00635 for warm. Three blocks on one machine give limited evidence about repeatability across days or deployments.

## Semantics, rank and support

Tracing only factor 1 reproduced every retained deterministic v1 and v2 feature at all four factors, using static arithmetic to derive the candidate fields. The leading counter is present in raw carry count and excluded from data-carry count and bytes. The v1 operation bucket combines slices with indexed operations and omits scatter-min/add. For example, scatter_min_update has two v1 bucket operations and one v2 scatter-min. Scalar dynamic slices in the dependent-indexing bodies give zero v2 gather counts. Input and intermediate byte sizes are array-shape proxies. In particular, a full loop-invariant table enters input bytes even if a step touches one element. Neither those fields nor index provenance measure dynamic locality or collisions. The known v1 taxonomy issue does not enter the retained memory_work warm-effect regression directly. The label and feature checks found no reconciliation error.

At a single trip, trip/(trip+128) is constant. The two factor bases, 1−1/u and log(u), each acquire a duplicate proportional column. Across bodies the remaining two varying coordinates, log(1+work)/8 and log(1+bytes/(1+work))/8, correlate −0.821. The eight-column design has numerical rank six. Its column norms range from 2.61 to 8.60, and the ratio of largest to smallest nonzero singular value is 243. Ridge penalizes coefficients on these fixed scales. The prior combined extension has rank 22/42, falling to 14–20 in training folds. Some columns are zero and its nonzero singular-value ratio is about 63,059. Additional terms can thus increase coefficient sensitivity with little independent support.

| Held-out component | Bodies outside the training range in either varying coordinate | Total |
|---|---:|---:|
| Dependent indexing | 3 | 4 |
| Gather, scatter and locality | 2 | 2 |
| Stateful recurrence | 0 | 1 |
| Stencil | 4 | 4 |

Ranges use each fold's training bodies. All four stencils exceed its training maximum for work and fall below its training minimum for bytes per unit work. This marginal check can detect extrapolation even without estimating a joint feature distribution.

As an offline illustration, factor 8 grew optimized HLO text 24.1× for dep_bucket_stateful and 22.0× for stateful_coupled_oscillator. Warm time moved to 0.87× and 2.98× factor 1, respectively. Text size alone cannot explain the different responses. Compiler measurements are excluded from inference.

The present design confounds mechanism with coverage. Four stencils form a single held-out group and all slow down warm at larger factors. Several dependent-indexing bodies benefit at intermediate factors. A transfer experiment would need additional independent mechanisms on overlapping work/traffic ranges, a second shape and both divisible/nondivisible trips within body. Matched runtime-index inputs for public fori_loop would test static indistinguishability directly. The prior scan evidence motivates this test but supplies no fori_loop estimate.

## Reproduction

From the repository root run `JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 JAX_ENABLE_COMPILATION_CACHE=false .venv/bin/python -m benchmarks.corpus.fori_audit`. The audit records input digests, per-body repeat vectors, offline HLO ratios, training ranges, design column norms and ranks. [Audit JSON](../results/fori_dev/audit.json)
