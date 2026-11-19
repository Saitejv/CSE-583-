# Static CPU unroll selection across all 66 workloads

The fitted extra-trees model selects among factors 1, 2, 4 and 8 from factor-1 static features. A prior profile of the target workload is optional for the project and unnecessary for this model. On the held-out workloads, 216 of 220 feasible long-horizon choices were within 10% of the best measured feasible cost. The worst choice exceeded that cost by 11.53%.

## Data and split

The combined matrix contains 220 distinct workload/variant/trip configurations and 880 per-factor cold and warm timing pairs. It combines the original feature table, recovery development and confirmation tables, the augmented static-feature corpus, the smoke sweep and both profile-transfer sweeps. Historical development rows present twice are deduplicated by configuration and factor. The experiment uses measured configurations. The registry also defines a larger unmeasured grid.

| Partition | Workloads | Configurations | Factor timing pairs |
|---|---:|---:|---:|
| Training and grouped model selection | 52 | 160 | 640 |
| Held-out evaluation | 14 | 60 | 240 |

The test partition consists of the two original holdout workloads, all four recovery confirmation workloads and the eight stencil and segmented-scan bodies. All variants, trips, factors and repetitions of each workload stay on one side. The evaluator checks that factor-1 body hashes are disjoint. These workloads had measurements and results in earlier project experiments. This evaluation tests a new fit on previously collected data.

Five-fold training validation grouped the expanded corpus by connected families and related bodies, placing matched-index workloads in the indexed component. The other training workloads were grouped by family. Ridge, radial-basis kernel ridge, random forest and extra trees each used four candidate margins for selecting a larger factor. All models forecast log cold and warm times for factor 1 and log changes for factors 2, 4 and 8. The validation loss sums mean positive excess cost, the fraction outside the 10% tolerance or over budget, and the avoidable budget violation rate. Extra trees with zero margin had the lowest loss. Selection and fitting use only training rows.

## Results

The stated acceptance target was at least 90% within 10% of best, at most 3% mean positive excess cost and at most 1% avoidable budget violations. The first two metrics use calls of 10,000 or 100,000 with at least two measured feasible factors. The violation rate uses all settings with at least one feasible factor.

| Method and partition | Exact best | Within 10% and feasible | Mean excess cost | Avoidable violations |
|---|---:|---:|---:|---:|
| Extra trees, grouped training validation | 49.5% | 87.9% | 4.75% | 0.16% |
| Always factor 1, test | 63.6% | 89.1% | 2.62% | 0% |
| Existing recovery model, test | 60.0% | 95.5% | 3.69% | 0% |
| Extra trees, test | 62.7% | 98.2% | 1.52% | 0% |

Extra trees chose the exact best factor in 138 of 220 test settings and had zero avoidable violations across 684 feasible settings. A timed uncached DecisionEngine call per test configuration raises mean excess to 2.47% in the retained replay. Model loading and process startup are excluded. Single-loop forest inference advances the trees in parallel with NumPy. All 1,440 batch choices match the selector applied to the runtime forecasts, and live traces were checked on all 60 configurations.

The held-out test meets the decision target. Grouped training validation missed two thresholds, so different workload mixes can perform worse. The 220 test decisions share 60 configurations and 14 workloads. Absolute timing remains less reliable. Test median absolute relative error was 22.6% for cold time and 70.2% for warm time, compared with 26.7% and 51.2% for the recovery model. Use the new model for opt-in factor selection on the measured Apple M2 CPU environment.

## Using and reproducing the result

From the project root, load the trusted fitted artifact once and use the standard decision engine.

```python
from jax_unroll.corpus_model import CorpusPredictor
from jax_unroll.selector import DecisionEngine

model = CorpusPredictor.load("results/all_workloads/model.joblib")
engine = DecisionEngine(model)
decision = engine.decide(fn, args, trip_count=trip_count, loop_kind="scan", B=100, R=100_000)
factor = decision.factor
```

Run `.venv/bin/python -m benchmarks.corpus.all_workloads` to reproduce the split, refit every candidate, save the winner, and evaluate it. The split, validation scores, fitted model, test decisions and summary are in [results/all_workloads](../results/all_workloads).
