# Independent review of the static fori_loop development experiment

I reproduced the results from the retained development measurements and frozen coefficient arrays using a separate Python calculation. The calculation used the earlier factor-1 feature table, the manifest JSON and NumPy linear algebra. It did not import the new evaluator or its fit, selector, aggregation or scoring helpers.

| Check | Result |
|---|---|
| Scope and split | The corpus builders use static integer bounds on all 11 included bodies. Closing the full manifest's related-family links before restricting the public loop kind gives the reported 4, 2, 1 and 4 held-out bodies. Each test component is disjoint from every body in its training partition. All 132 eligible CPU measurement rows pass correctness. |
| Predictions | Reconstructed 44 cold/warm median pairs, all fold-specific per-factor medians, both ridge solutions using the normal equations, and all four frozen regressions from their JSON weights. All 264 method/body/factor forecasts match. The maximum relative difference is below 2.4 × 10⁻¹⁴. |
| Error and calibration | All 33 measured warm effects and their repeat-stability records agree. Each method's pooled and component MAEs, body-bootstrap intervals and absolute cold/warm log MAEs and median calibration ratios reproduce. |
| Policy | Independently minimized the constrained predicted and measured total for every body and grid setting, including factor-1 fallbacks. All 1,584 method/setting rows agree on selector, factor 1, LLVM-size and oracle choices, measured totals, budget violations and long/useful membership. The reported stratum regret and speedups also reproduce. |
| Feature audit | The original eight warm-effect columns have numerical rank six. The earlier 42-column extension has rank 22. SVD condition numbers on the nonzero subspace, per-fold ranks, counts of out-of-range bodies, cold/warm repeat ranges and factor-8 optimized HLO text-size ratios reproduce. |
| Compatibility and scope | The protocol and other pre-score receipt hashes verify. The v1 extractor, model, selector, wrapper stubs and frozen artifact match their tracked versions. Training uses only the other eligible pilot components. Budget and horizon enter the policy replay after prediction. |

The primary's paired MAE difference from zero is −0.0200, with a 95% body-bootstrap interval of [−0.0488, +0.0095]. The 16 useful long settings come from four bodies. These diagnostics support the reported stopping decision. Generalization across new families and overhead-inclusive performance remain unverified.

I also checked the results narrative against the paired rows. The primary and frozen methods choose the same factor in all 264 settings. Each chooses factor 2 at 100,000 calls on exactly two bodies under two budgets. The measured speedups are 1.463× on the bucket workload and 0.390× on the 1D stencil. These examples and the stated calibration values match the retained measurements.
