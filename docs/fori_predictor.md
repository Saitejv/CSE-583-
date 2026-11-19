# Warm-runtime prediction for static JAX fori_loop

The expansion-trained extra-trees model chooses among unroll factors 1, 2, 4 and 8. Configure a shared engine once, then call the wrapper with the same bounds, loop body and initial carry you would pass to `jax.lax.fori_loop`. The supported warm-throughput measurements use the project's pinned Apple M2 CPU environment.

```python
from functools import partial

import jax

from jax_unroll.auto_loop import auto_fori_loop
from jax_unroll.fori_model import ForiCostPredictor
from jax_unroll.selector import DecisionEngine

model = ForiCostPredictor.load("results/fori_predictor/model.joblib")
engine = DecisionEngine(model)
auto_fori = partial(auto_fori_loop, engine=engine, throughput_only=True)

@jax.jit
def run(values, initial):
    def body(i, carry):
        return 0.95 * carry + values[i]

    return auto_fori(0, values.shape[0], body, initial)
```

The model runs during tracing. JAX reuses the chosen factor when the caller hits the JIT cache. Dynamic bounds fall back to native JAX. The wrapper also accepts `B` as a predicted cold budget in milliseconds. The default policy with a supplied engine ranks feasible factors by predicted total cost for `R` calls.

## Live wrapper benchmark on unchanged loop bodies

The live comparison covered all 192 targeted evaluation configurations from 24 workloads. All 576 paired runs produced outputs within the corpus tolerances. For every pair, the Jaxpr was identical after substituting the selected unroll factor. The selected factors were consistent across runs. The model chose 1 for 110 configurations, 2 for 63 and 4 for 19.

| Measurement | Native factor 1 time / wrapper time |
|---|---:|
| Pooled warm time | **1.120×** |
| Warm time, geometric mean across configurations and processes | **1.044×** |
| Pooled total at 1 call per configuration | 0.708× |
| Pooled total at 100 calls per configuration | 0.796× |
| Pooled total at 1,000 calls per configuration | 1.010× |
| Pooled total at 10,000 calls per configuration | 1.106× |

First-call timing includes tracing, lowering, compilation and one synchronized call. Total time estimates use that measurement plus the remaining warm calls and charge model loading once per process. Median trace-time selection took 4.35 ms. The waveguide accounted for 92% of the pooled warm time saved. The worst median warm slowdown across the three repeats was 1.38× on large Levinson–Durbin at 255 iterations. [Per-process measurements](../results/targeted_predictor/auto_fori_live.json)

```sh
PYTHONPATH=src:. .venv/bin/python -m benchmarks.corpus.fori_targeted.auto_fori
```

## Model and selection

The model uses static features from tracing factor 1, including the trip count, remainder, body operation counts and memory footprint. It learns the baseline cold and warm times and the log cost ratios for the other factors. Throughput-only selection minimizes predicted warm time. No measured timing, workload name, family ID, partition tag or prior profile is an inference input.

Training used the expansion's 104 training bodies and 253 configurations. Five-fold validation withheld whole families, joining any families linked by the split's related-group identifiers. Four designs compared a joint extra-trees cost model with separate extra-trees, random forest and ridge regressors for the warm ratios. Each used four minimum predicted gain margins. The criterion was geometric-mean warm speedup versus factor 1, weighting workloads equally. Joint extra trees with 128 trees and zero margin won with a 1.0166× workload-weighted validation speedup. Its worst validation slowdown was 2.02×.

The prior mixed-loop model was loaded from its saved artifact. Both models used the same warm-only policy for evaluation. All expanded tables contained complete four-factor groups, and no factor-1 body hash crossed the training/evaluation boundary. The 1,208 newly measured factor-level cold and warm medians also reproduced from the 3,624 retained raw rows.

## Held-out results

| Cohort | Model | Warm speedup vs factor 1¹ | Exact fastest factor² | Worst time / factor-1 time | Median absolute relative warm prediction error |
|---|---|---:|---:|---:|---:|
| Fresh, 18 bodies / 40 configurations | Previous | 0.940× | 40.0% | 3.785× | 89.0% |
| Fresh | Expanded | 1.005× | 52.5% | 1.068× | 32.2% |
| Historical, 14 bodies / 44 configurations | Previous | 1.020× | 68.2% | 1.081× | 104.2% |
| Historical | Expanded | 1.022× | 72.7% | 1.000× | 48.3% |

¹ Geometric mean over configurations using measured median warm execution time. Compilation, tracing and model inference costs are excluded. When each configuration executes once and times are summed, the expanded model yields 1.008× on fresh and 1.001× on historical workloads. Choosing the measured fastest factor for each configuration would yield geometric-mean speedups of 1.021× and 1.031× respectively.

² Any factor tied for the minimum measured median counts as exact. On fresh configurations, the expanded model was faster than factor 1 in 14, equal in 20 and slower in six. It was within 10% of the measured best on 39/40; factor 1 alone was also within 10% on 39/40.

Fresh evaluation workloads are new bodies from ten broad families also present in training. Weighting the 18 bodies equally, the speedup was 1.004× with a workload-bootstrap 95% interval of 0.996–1.012×. This supports an accuracy improvement over the previous predictor while leaving the typical throughput advantage over factor 1 uncertain. Historical body-weighted speedup was 1.024× (1.011–1.039×). The current evaluation covers warm throughput and does not establish an end-to-end benefit on short runs or validate cold-budget selection.

Run `.venv/bin/python -m benchmarks.corpus.train_fori` to reproduce training and replay. The [model, cross-validation scores, per-configuration decisions and evaluation summary](../results/fori_predictor/) are retained together.
