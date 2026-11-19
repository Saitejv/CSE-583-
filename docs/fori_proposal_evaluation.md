# When native JAX loop unrolling pays off

Half of the 32 fresh configurations had a repeatable warm gain from factors 2, 4 or 8. Across the 24 fresh configurations in the direct wrapper comparison, native factor 1 had lower pooled total time than the compact warm selector at each measured horizon. In a separate 20,000-call follow-up on all 12 live cases with a verified warm opportunity, the compact wrapper saved **7.7%** of total time. The measured per-call saving and the number of calls determine whether setup pays back.

## Evaluation design

The original proposal's Apple M2 example showed a 2.05× scalar recurrence speedup at factor 8, a large slowdown for sine, and near-ties for reduction and dense matrix work. The first regret holdout had three bodies where factor 1 took every warm win. The follow-up adds the missing inexpensive loop bodies where large factors can help, alongside cases where arithmetic, memory, or setup can dominate. [Proposal slides, pages 4–5](../group10_slides.pdf) · [Original regression holdout](fori_regret.md)

The matrix was frozen before measuring. It contains 32 fresh configurations from eight new bodies, with two sizes and 255 or 1,024 loop iterations. Each configuration runs the same body and inputs with factors 1, 2, 4 and 8. The fresh families cover scalar control, compensated integration, lane-wise arithmetic, integer word mixing, dense matrix-vector updates, indexed deposition, streamed cross-products, and a one-dimensional acoustic simulation. These are purposefully chosen mechanisms on one CPU. The count of warm opportunities measures coverage within this matrix.

Four proposal-style controls use the repository's native `fori_loop` versions of the scalar recurrence, sine, reduction and dense matvec. They use the settings in the manifest. Two historical controls retest the previously favorable waveguide and Gram accumulation workloads. Three regression controls keep one case from each original holdout body. Controls are reported separately from fresh results. The audit checks fresh workload names, families, and current and archived body hashes against development data. The incumbent and its forecast error margins were fixed throughout. [Frozen matrix](../results/fori_proposal/matrix.json) · [Native decisions](../results/fori_proposal/native_decisions.csv)

## Warm execution

A warm opportunity requires at least 1.05× median paired speedup at a larger factor and a warm win in all three independent processes. Sixteen of the 32 fresh configurations met that gate. The repeat-supported oracle chose factor **1 in 16 cases, 2 in four, 4 in two, and 8 in ten**.

| Example | Factor 1 | Best repeat-supported factor | Warm speedup |
|---|---:|---:|---:|
| Fresh lane-wise moments, large, 1,024 iterations | 26.5 µs | 4 at 10.9 µs | 2.44× |
| Fresh compensated integration, large, 1,024 iterations | 17.6 µs | 8 at 11.9 µs | 1.49× |
| Fresh indexed deposition, small, 1,024 iterations | 72.6 µs | 2 at 66.2 µs | 1.10× |
| Fresh matrix-vector state update, small, 255 iterations | 10.3 µs | 1 | 1.00× |
| Proposal scalar recurrence control | 8.0 µs | 8 at 6.0 µs | 1.33× |
| Historical waveguide control | 931.7 µs | 4 at 596.1 µs | 1.56× |
| Historical Gram control | 234.7 µs | 2 at 59.4 µs | 3.95× |

Factor 8 took about 2.11× as long as factor 1 on the small matrix-vector case and 2.08× as long on the sine control. Across all fresh settings, the warm oracle achieved **1.124× geometric-mean and 1.025× pooled speedup**. The incumbent achieved **1.051× and 1.005×**, respectively, with 12 selections at least 1.05× faster by median and one selection over 3% slower. The size heuristic achieved **1.037× and 0.988×**, with 15 warm gains and eight regressions by the same median thresholds. The lower pooled gains reflect higher-time cases where the selected factor has little effect. [Warm summary](../results/fori_proposal/native_summary.json)

## Cold startup and wrapper comparison

The NumPy export contains the incumbent's tree arrays, normalizer and metadata. It uses the same forest traversal and predicts the same costs for factors 1, 2, 4 and 8. It can load from a cold process without importing scikit-learn, pandas, or joblib. Use it with the existing wrapper and a shared engine.

```python
from jax_unroll.auto_loop import auto_fori_loop
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.selector import DecisionEngine

model = CompactForiPredictor.load("results/fori_proposal/incumbent.npz")
engine = DecisionEngine(model)
# Inside a jitted caller with static bounds
result = auto_fori_loop(lower, upper, body, initial, engine=engine, throughput_only=True)
```

The wrapper benchmark executes 1, 300 and 5,000 synchronized calls on 24 predeclared fresh configurations and nine controls. It records tracing and lowering, decision time within tracing, compilation and every invocation. Each program compiles once for each timed configuration and horizon. Each arm pays one measured load per process and horizon, divided across all 33 cases for subgroup accounting. The device and arguments are ready before timing. The native oracle arm uses factors from the separate sweep and incurs no search cost in its timed total. It shows attainable runtime given the measured choice. The size heuristic traces and selects through the same wrapper. The compact payoff arm keeps the previous margins, remeasures runtime overhead on development cases, and uses a 125 ms predicted cold budget.

Median timed model startup fell from **1,466 ms** with the joblib incumbent to **7.2 ms** with the NumPy export. The forecasts and selected factors were identical. [Runtime summary](../results/fori_proposal/live_summary.json)

### Direct totals on the 24 fresh configurations

Pooled seconds per process, including each arm's share of model startup.

| Calls per configuration | Native factor 1 | Incumbent | Compact incumbent | Compact + payoff | Size heuristic | Warm oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.540 | 1.717 | 0.683 | 0.544 | 1.243 | 0.733 |
| 300 | 0.849 | 2.017 | 0.987 | 0.872 | 1.571 | 1.032 |
| 5,000 | 5.645 | 6.748 | 5.795 | 5.579 | 6.539 | 5.608 |

At 5,000 calls the compact warm selector beat native by at least 1.05× in four fresh configurations and ran over 3% slower in nine. Lane-wise moments at large/255 took 0.72× native's total time. Small compensated integration at 1,024 iterations took 0.90×. Across all fresh configurations the pooled total was still 2.7% higher. The payoff arm picked factor 1 at every horizon. Its near-native totals measure primarily timing variation and its small added setup on identical compiled loops.

Across the full 33-case matrix with controls, the compact warm selector took **12.573 s vs 13.297 s for native** at 5,000 calls. The historical waveguide drove that pooled advantage. The selector chose factor 2 and took about 0.69× native's total time on that control. The historical Gram case had a 3.95× warm opportunity, and both incumbent formats chose factor 1. The learned policy still misses strong gains.

The estimated pooled crossover on the 12 fresh opportunities in the live matrix was about 4,900 calls for the compact warm selector, using first-call and warm timing from the 5,000-call run and holding the selected factors fixed. The directly measured 5,000-call totals for those cases were effectively tied, 1.177 s native and 1.175 s compact. [Direct results and break-even estimates](../results/fori_proposal/live_summary.json)

### Direct 20,000-call confirmation of warm opportunities

This follow-up includes every fresh case in the live matrix that met the native sweep's repeatability gate. The static model and its selection rule were the same as in the first wrapper comparison. Model load is charged once across the 12 cases.

| Arm | Pooled total | Speedup vs native | Cases at least 1.05× faster | Cases over 3% slower |
|---|---:|---:|---:|---:|
| Native factor 1 | 4.144 s | 1.000× | 0/12 | 0/12 |
| Compact incumbent | 3.823 s | 1.084× | 6/12 | 2/12 |
| Warm oracle | 3.714 s | 1.116× | 6/12 | 3/12 |

The compact selector won all three pooled process comparisons, reducing total time by 2.5%, 8.9%, and 11.5%. It saved 31.5% on large lane-wise moments at 255 iterations and 17.7% on small compensated integration at 1,024 iterations. Two short-iteration configurations still regressed, with a worst slowdown of 12.1%. A warm win justifies unrolling once its accumulated savings exceed setup. [20,000-call measurements and estimates](../results/fori_proposal/opportunities_summary.json)

## Reproduction

```sh
export PYTHONPATH=src:.
export JAX_PLATFORMS=cpu
.venv/bin/python -m benchmarks.corpus.fori_proposal.protocol freeze
.venv/bin/python -m benchmarks.corpus.fori_proposal.protocol sweep
.venv/bin/python -m benchmarks.corpus.fori_proposal.compact
.venv/bin/python -m benchmarks.corpus.fori_proposal.score
.venv/bin/python -m benchmarks.corpus.fori_proposal.live
.venv/bin/python -m benchmarks.corpus.fori_proposal.live --repeat-opportunities
.venv/bin/python -m pytest -q
```

The Apple M2 runs used JAX and jaxlib 0.12.0.1. Three independent processes per arm retained every cold measurement and the native sweep's repeated warm samples. The sweep and live runs compare outputs against factor 1 and verify that only the unroll parameter changes in the Jaxpr. [Raw runs and model export](../results/fori_proposal/)
