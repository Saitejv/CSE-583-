# End-to-end application measurements on Apple M2

The packaged application engine reduced combined runtime from **5.287 s to 3.184 s** at 5,000 calls on a prospective test of three streamed matrix updates and a recursive regression control. The two repeatable gains came from historical daily signal replay (**1.861 s saved**) and class-balanced feature covariance (**0.288 s saved**). A separately frozen, broader four-application benchmark saved **0.119 s** and missed its predeclared 10% target.

## What shipped

`jax_unroll.compact_fori.application_engine()` creates a shared engine with deferred loading of the bundled CPU model. Callers can use it from an installed wheel. The revised policy admits a second indexing operation in its matrix rule. The previous rule allowed one. A qualifying loop with at least 1,000 expected calls, at least 2,048 carry bytes, a rank-two carry, no reductions or dot products, and sufficient predicted cold budget can use factor 2. The existing short-call bypass and general warm selector handle the other cases. The tested deployment budget was 125 ms. `R` counts calls to the compiled specialization.

Calibration compared 36 matrix-rule configurations and the prior rule on 626 development cases, using 51 validation groups. The selected configuration won in 48 of 51 leave-group-out fits. Held-out development replay across groups lost 0.020 s against the previous rule at 5,000 calls and gained 0.825 s at 20,000. The 24 reserved agent configurations and the real-data applications were excluded from fitting. [Fit and fold selections](../results/fori_application/fit.json)

## Direct CPU results

Seconds sum all configurations in a cohort and average three independent process repeats. Each arm starts a fresh process at each horizon with JAX persistent compilation caching disabled. Totals include engine creation, model loading when used, tracing and decision, compilation, and every synchronized invocation. Raw data loading, host input preparation and device staging precede the timer. For threshold curves the inputs are classifier scores, so Gaussian NB fitting is outside the timer. The direct oracle executes the horizon-specific factor selected from a separate native sweep. Its search cost is excluded. [Primary manifest](../results/fori_application/manifest.json) · [Follow-up manifest and locked policy digest](../results/fori_application/transfer_manifest.json)

| Cohort | Calls per configuration | Native 1 | Compact warm | Previous horizon policy | New application engine | Direct oracle |
|---|---:|---:|---:|---:|---:|---:|
| Original four applications | 300 | .219 | .232 | .221 | .217 | .217 |
| Original four applications | 5,000 | 1.514 | 1.541 | 1.407 | **1.395** | 1.353 |
| Targeted four applications | 300 | .684 | .686 | .684 | .684 | .296 |
| Targeted four applications | 5,000 | 5.287 | 5.275 | 5.305 | **3.184** | 2.960 |

**Original four.** The classifier-threshold and pixel-envelope jobs use the scikit-learn handwritten digits dataset. The envelope executes the installed JAX segmented maximum body under the benchmark's enclosing JIT. A morphological gradient uses a crop of scikit-learn's flower photo and normalized LMS fits its diabetes regression dataset. At 5,000 calls the candidate beat native in all three pooled repeats and recovered 74% of direct oracle headroom. Morphology saved 0.127 s. Segmented maximum lost 0.012 s and was 9.0% slower by median ratio. The candidate saved just 0.012 s versus the previous policy on this cohort. Keep native for segmented maximum. [Every configuration](../results/fori_application/applications_cases.csv) · [Full and opportunity totals](../results/fori_application/applications_totals.csv)

**Targeted follow-up.** After the original benchmark, four more applications and their input hashes were frozen with the same fitted policy. This measures transfer within the proposed class. Pooled savings at 5,000 calls were **39.8% versus native** and **40.0% versus the previous policy**, with wins in all three repeats against both and 90.4% of the directly measured oracle headroom recovered. At 300 calls every deployed configuration bypassed prediction. The oracle's .296 s shows that this conservative cutoff leaves short-horizon opportunities.

| Follow-up application, 5,000 calls | Chosen factor | Candidate seconds | Mean seconds saved vs native | Repeats faster than native |
|---|---:|---:|---:|---:|
| Historical entry/exit signal grid on adjusted daily closes | 2 | 1.682 | +1.861 | 3/3 |
| Weighted covariance on Wisconsin diagnostic feature data | 2 | .248 | +.288 | 3/3 |
| Nearest-sample interpolation on the Jacksboro elevation grid | 2 | 1.071 | −.043 | 2/3 |
| Recursive least squares on diabetes regression data | 1 | .183 | −.002 | 1/3 |

Daily returns and elevation data come from Matplotlib's bundled samples. The covariance and grid interpolation have development analogues in the earlier synthetic covariance and Voronoi benchmarks. The signal grid accounts for most of the gain. The terrain candidate took .858, .852, and 1.502 s versus native's 1.023, 1.021, and 1.040 s. Keep the existing choice there pending stronger evidence. [Every configuration](../results/fori_application/transfer_cases.csv) · [Full and opportunity totals](../results/fori_application/transfer_totals.csv)

## Held-out families

At 5,000 calls the engine saved **2.155 s (2.0%)** across all 24 held-out configurations, winning all three process repeats against both native and the previous selector. Spectral unmixing alone consumed 97.074 s of native's 108.216 s. The four configurations per family give full-cohort pooled and equal-workload totals identical weights.

| Direct cohort | Calls | Native 1 | Compact warm | Previous | Application engine | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| All 24 | 300 | 7.461 | 7.591 | 7.541 | 7.471 | 6.950 |
| All 24 | 5,000 | 108.216 | 108.340 | 107.415 | 106.061 | 102.739 |
| 16 warm opportunities, pooled | 5,000 | 9.660 | 9.123 | 8.191 | 6.983 | 4.057 |
| 16 warm opportunities, equal-workload | 5,000 | 7.991 | 7.546 | 6.801 | 5.835 | 3.440 |

Inventory lost-sales simulation saved 1.453 s and flood duration saved 1.185 s relative to native. The revised rule changed two decisions relative to the previous policy, both on large inventories. Eight configurations achieved at least 1.05× speedup, and four slowed by over 3%. The worst median slowdown was 35.2% on the large 255-step warranty ladder, which lost 0.108 s. Keep native for this configuration.

The engine recovered 39.3% of native-to-oracle headroom on all 24 and 47.8% on the opportunity subset. Two cases explain 1.948 s of the remaining 3.322 s. The large 1,024-step flood simulation took 1.093 s longer with factor 2 than the directly timed oracle at factor 8. The small 1,024-step inventory simulation took 0.855 s longer with factor 1 than the oracle at factor 4. [All configurations](../results/fori_application/holdout_cases.csv) · [Both subsets and weighting schemes](../results/fori_application/holdout_totals.csv) · [Raw process measurements](../results/fori_application/holdout_live.json)

## Long-horizon extension

After seeing the 5,000-call application results, we specified a 20,000-call extension on every configuration in both four-application cohorts. It uses the same inputs, calibrated policy, five arms, three repeats, and cold budget. The oracle selects the minimum of the four existing native timing curves at 20,000 calls. The extension has separate direct result files.

## Reproduction

From the CSE583 repository root with the pinned dependencies:

```sh
export PYTHONPATH=src:. JAX_PLATFORMS=cpu
py=.venv/bin/python
$py -m benchmarks.corpus.fori_application.protocol freeze
$py -m benchmarks.corpus.fori_application.calibrate
$py -m benchmarks.corpus.fori_application.protocol sweep
$py -m benchmarks.corpus.fori_application.live
$py -m benchmarks.corpus.fori_application.transfer freeze
$py -m benchmarks.corpus.fori_application.transfer sweep
$py -m benchmarks.corpus.fori_application.live --cohort transfer
for cohort in applications holdout transfer; do
  $py -m benchmarks.corpus.fori_application.report "$cohort"
done
for cohort in applications transfer; do
  $py -m benchmarks.corpus.fori_application.live --cohort "$cohort" --long
  $py -m benchmarks.corpus.fori_application.report "$cohort" --long
done
$py -m pytest -q
```

The live command resumes completed process blocks if their host and artifact hashes match. For a completely new acquisition use a separate checkout with empty live result files. The report commands independently check completeness, chosen factors across repeats, freeze hashes, and native/oracle factor consistency. The raw direct measurements are separate from the saved per-factor cold and warm sweep timings.
