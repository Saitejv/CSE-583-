# CPU unroll prediction with a prior workload profile

Use the opt-in profile predictor when the same workload and input variant will run again at a nearby trip count. On the Apple M2, profiles measured at 1,023 iterations predicted the best factor within 1.85% of measured cost across all 96 feasible long-horizon settings at 775 and 1,015 iterations. The benchmark compared factors 1, 2, 4 and 8. Each factor had three fresh-process measurements per configuration. The 96 settings reuse 24 configurations across budgets and horizons.

| Trip count | Exact best, previous → profile | Mean excess cost, previous → profile | Worst excess cost, profile | Mean excess including one live decision |
|---:|---:|---:|---:|---:|
| 775 | 22/48 → 44/48 | 6.02% → 0.04% | 0.58% | 0.35% |
| 1,015 | 18/48 → 36/48 | 7.58% → 0.19% | 1.84% | 0.46% |

Excess cost compares the selected factor with the fastest measured factor that meets the cold-cost budget at the same call horizon. Feasible long settings have at least two feasible factors and at least 10,000 calls. Across the full budget grid, the profile policy had zero violations on settings where a feasible factor existed. The overhead estimate adds one timed selection per configuration after feature extraction code has warmed up. The cost of collecting the original profile is excluded.

## Using a profile

Load the prior factor medians once. Scope the profile file to the same machine and compiler settings. Give each workload and input distribution a stable identifier. The predictor also checks device, JAX and jaxlib versions, body IR hash, body shapes and types. It reuses a profile at its original trip count or up to 25% shorter when the remainder modulo 8 matches. This preserves the tail pattern for every candidate factor. It holds cold time constant and scales warm time with iterations. On a profile miss, the opt-in selector returns factor 1.

```python
import csv

from benchmarks.corpus.registry import build_workload
from jax_unroll.profile_model import ProfileCostModel

with open("results/static_features/pilot_features.csv") as file:
    model = ProfileCostModel(csv.DictReader(file))

loop = build_workload("rolling_mean_shift", 775, 1, "w8")
factor = model.choose(loop.fn, loop.args, workload="rolling_mean_shift",
                      variant="w8", trip_count=775, loop_kind=loop.loop_kind,
                      B=100, R=100_000)
```

The validation covers 12 existing smoke-tier workloads and two shorter trip counts. Upward transfer, different input distributions and other devices need independent measurements. A changed data distribution can alter the best factor even when the body IR is identical.

## Prediction on unfamiliar families

Five alternatives were compared with the recovery model on 47 bodies at 1,023 iterations. Each outer fold withheld a complete component of related loop families. The estimators used only static input features. Fitting excluded the held-out component. The hybrid used kernel ridge for cold time and the existing recovery model for warm time. The residual model used kernel ridge to correct the existing forecasts. Grouped evaluation follows the [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

| Model | Cold log MAE | Warm log MAE | Warm factor-effect log MAE | Mean feasible-long excess cost |
|---|---:|---:|---:|---:|
| Existing recovery model | 0.651 | 0.671 | 0.337 | 10.91% |
| Extra trees | 0.458 | 0.743 | 0.488 | 17.76% |
| Random forest | 0.443 | 0.736 | 0.441 | 16.83% |
| RBF kernel ridge | 0.415 | 0.845 | 0.392 | 24.10% |
| Hybrid | 0.415 | 0.671 | 0.337 | 10.91% |
| Residual model | 0.540 | 0.635 | 0.385 | 21.92% |

The hybrid improved cold calibration and avoided one additional budget violation. Its mean decision cost was effectively tied with the existing model. The current evidence supports keeping the existing general-purpose policy and using profiles for recurring workloads.

## Reproduction

Run from the repository root with the pinned environment. The first two commands acquire fresh measurements. Use fresh output paths when retaining a second run. The final two commands replay the retained datasets.

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m benchmarks.corpus.sweep --tier smoke --trip-counts 1015 --repeats 3 --out results/profile_transfer/validation_raw.csv
JAX_PLATFORMS=cpu .venv/bin/python -m benchmarks.corpus.sweep --tier smoke --trip-counts 775 --repeats 3 --out results/profile_transfer/transfer_775_raw.csv
.venv/bin/python -m benchmarks.corpus.profile_transfer
.venv/bin/python -m benchmarks.corpus.nonlinear_cost
```
