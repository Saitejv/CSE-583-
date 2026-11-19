# Warm-throughput prediction on paired trip counts

The targeted benchmark measures static JAX fori_loop bodies at 255, 256, 1,023 and 1,024 iterations on the Apple M2. Each workload has two working-set sizes. Three fresh processes measure each of factors 1, 2, 4 and 8. One invocation at each configuration defines the pooled-runtime comparison. The workload-weighted comparison gives each body equal influence. Neither comparison charges compilation or a model decision to warm execution.

## Held-out result and model choice

Use the previous expansion-trained ForiCostPredictor for opt-in warm-throughput selection. The resumed sweep supplied the remaining 4,473 fresh-process rows. All 4,608 rows were correct, covered unique planned keys, and yielded 192 four-factor development configurations and 192 four-factor sealed configurations. Combining both development collections yielded 445 configurations from 128 named workloads. Twelve names in the targeted design are adaptations of historical algorithms.

| Policy on 24 sealed workloads | Workload-weighted speedup vs factor 1 | Speedup when summing warm times | Within 10% of best | Worst time / factor-1 time |
|---|---:|---:|---:|---:|
| Previous expansion-trained extra trees | 1.043× | 1.130× | 88.5% | 1.37× |
| New development winner | 1.015× | 1.130× | 84.9% | 2.28× |
| Always factor 1 | 1.000× | 1.000× | 72.9% | 1.00× |
| Always factor 2 | 0.966× | 1.119× | 71.9% | 2.50× |
| Fastest measured factor per configuration | 1.081× | 1.174× | 100% | 1.00× |

The development winner used Huber gradient boosting with a shared regression across factors and zero margin. The previous predictor beat it on the workload-weighted metric in all three repeats. The paired speedup of the new candidate relative to the previous one was 0.974×, with a 95% workload-bootstrap interval of 0.955–0.991×. The previous predictor's speedup over factor 1 had a corresponding interval of 1.014–1.080×. These intervals describe variation among the benchmark workloads.

At one warm invocation per configuration, total measured time fell from 31.21 ms at factor 1 to 27.62 ms with the previous predictor. The previous predictor was faster in 70 configurations, tied in 110, and slower in 12. It performed particularly well at 1,023 and 1,024 iterations (1.069× and 1.065× geometric-mean speedup). Its worst slowdown occurred on Levinson–Durbin. The new candidate won more individual comparisons but incurred 42 slowdowns, including a 2.28× runtime on the TCP congestion controller. The difference in pooled runtime between the two trained policies was under 0.04%.

Results apply to the specified mix of static loops, sizes, trips and hardware. The sealed bodies are distinct from the training bodies; similar application mechanisms can recur. The recommended model and usage are in the [fori_loop predictor guide](fori_predictor.md).

## Development and selection

Training combines the original expansion's development partition and the targeted development partition. The frozen incumbent is the previous expansion-trained extra-trees predictor. Candidates include joint extra trees and random forests, a forest trained on arithmetic warm-time ratios, a blend of the ratio and joint forecasts, regularized radial-basis kernel regression, Huber gradient boosting, a shared Huber model that learns how the remainder and body size interact with factor, and a conservative upper-quartile forecast across trees. Each design uses four minimum-gain margins. A positive margin makes the selector require a larger predicted improvement before unrolling.

Five-fold development validation groups whole families, also tying together declared related groups, identical body hashes and the twelve targeted adaptations of historical algorithms. For each policy the validation utility uses log warm speedup relative to factor 1, charging twice for negative gains. It averages within workload, then gives the two development collections equal weight. The highest utility determines the fitted artifact before sealed scoring.

## Reproduction

Run in the pinned environment from the repository root, after completing the serial sweep and exporting its two feature tables:

```sh
.venv/bin/python -m benchmarks.corpus.train_targeted fit
.venv/bin/python -m benchmarks.corpus.train_targeted score
```

The fit stage reads only development timing labels. The score stage checks the acquisition keys and recomputes cold and warm medians from the three process repeats, checks for body-hash overlap, verifies batch choices against single-loop selection and replays both trained policies, all four fixed factors and the oracle. The per-repeat replay keeps the trained choices fixed. The paired interval for candidate versus incumbent bootstraps whole workloads to keep the eight measurements on each workload together.

Artifacts are in [results/targeted_predictor](../results/targeted_predictor/). The candidate can be loaded as follows:

```python
from jax_unroll.selector import DecisionEngine
from jax_unroll.targeted_model import TargetedCostPredictor

model = TargetedCostPredictor.load("results/targeted_predictor/model.joblib")
engine = DecisionEngine(model)
decision = engine.decide(fn, args, trip_count=trip_count,
                         loop_kind="fori_loop", throughput_only=True)
```
