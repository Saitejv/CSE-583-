# Choosing an unroll factor for a known call count

A compact horizon policy saved **2.764 s versus the warm incumbent** at 20,000 calls across three held-out matrix-update families. On the original mixed cohort its long-horizon choices matched the incumbent. The short-call crossover and the targeted matrix action address different costs.

## Where a better decision can pay

For each of the four native factors, replay charges its own median first-call milliseconds plus `(calls - 1) × warm_us / 1000`. Warm latency is the median of the per-process sample medians on repeated sweeps. The oracle minimizes this total separately at each horizon. Replay excludes wrapper and model overhead. The direct measurements below charge both. The archive contains overlapping historical controls, so the cohort labels are retained in the [626-configuration ranking at each horizon](../results/fori_horizon/development_regret.csv).

| Previous 12-case confirmation, replay | Native | Warm incumbent | Horizon oracle | Three largest incumbent misses, milliseconds |
|---:|---:|---:|---:|---|
| 300 calls | 0.390 s | 0.415 s | 0.366 s | Expiry counters large 12.8, expiry small 12.8, tracker large 10.2 |
| 5,000 calls | 1.679 s | 1.563 s | 1.537 s | Expiry small 10.3, tracker large 7.4, phase large 4.7 |
| 20,000 calls | 5.794 s | 5.227 s | 5.140 s | Phase large 29.9, tracker large 20.1, envelope large 17.0 |

At 300 calls, seven of twelve had factor 1 as the horizon oracle. Five had a warm gain that still could not repay setup. At 5,000 calls, all three leading misses came from paying for unrolling too early. At 20,000 calls, the three leading misses came from selecting 2 when 4, 8, and 8 were faster end to end. The incumbent recovered 87% of the previous cohort's long-horizon factor-choice headroom. Its largest outright missed unroll at that horizon was just 5.9 ms. The previous savings rule vetoed profitable incumbent choices in three, six, and five configurations respectively, relinquishing 22, 139, and 485 ms. Small spike recovery alone accounted for 328 ms of the last number.

On the older targeted suite the long-horizon incumbent missed 24.349 s. **10.115 s came from Gram accumulation and 5.569 s from covariance updates**, across sizes and trip counts. The three most expensive single decisions missed 3.549, 3.474, and 3.474 s. The same two bodies accounted for 3.907 s of the suite's 6.152 s miss at 5,000 calls. These results justified testing a targeted factor-2 decision while retaining the incumbent's general warm ranking.

## Candidates and measurement

The simple crossover bypasses tracing for selection and loading below 1,800 expected calls, the approximate crossover in the previous mixed study. Above it, the compact incumbent ranks feasible factors by warm time. The more specialized candidate can choose factor 2 starting at 1,000 calls for a matrix carry with at least 2,048 total carry bytes, at most one indexed operation, and zero dot products or reductions. A cheap carry-shape check can bypass the model before tracing in that window. Both policies reserve the measured decision allowance within a 125 ms predicted cold budget. Native factor 1 is the fallback. The warm incumbent arm uses its original unconstrained throughput rule. All learned arms use the identical uncompressed incumbent artifact.

Calibration used 581 development configurations and existing out-of-family forecasts. It scored actions by pooled runtime savings and equal-workload relative gains with a double penalty on slowdowns. In five family-grouped folds, the dense-reduction family was held out together. Without that family the training split chose no override. The grouped candidate and crossover tied at 5,000 and 20,000 calls, and the candidate lost 0.181 s at 1,000. This is a localized hypothesis. [Calibration and fold choices](../results/fori_horizon/policy.json)

Six new families with two configurations each were frozen before calibration. After historical diagnosis, two additional matrix-transfer probes and one historical Gram control were frozen before fitting or timing. Every native sweep used identical inputs at factors 1, 2, 4, and 8, checked numerical outputs and checked that the Jaxpr changed only in its unroll parameter. A warm opportunity requires at least 1.05× median paired speedup and a win in every sweep repeat. All frozen configurations enter the direct comparison, including those with no opportunity. [Mixed manifest](../results/fori_horizon/matrix.json) · [Transfer and control manifest](../results/fori_horizon/transfer_matrix.json)

Each arm, horizon and repeat starts a fresh Apple M2 CPU process with persistent compilation caching disabled. The benchmark stages identical inputs, then charges engine setup, model loading, tracing and decision, compilation, and every synchronized call. A shared load is counted once and allocated equally within that process for subgroup totals. The direct oracle runs the independently chosen horizon-specific factor with search excluded. Pooled seconds are sums per process averaged over three repeats. Workload-weighted seconds give each body equal weight, scaled to the same configuration count. Per-case comparisons use median paired time ratios.

## Direct runtime on the original frozen cohorts

All 14 fresh configurations, including the two predeclared transfer probes. Each entry is **pooled seconds / workload-weighted seconds**. The historical Gram control is excluded.

| Calls | Native 1 | Compact warm | Simple crossover | Matrix candidate | Horizon oracle |
|---:|---:|---:|---:|---:|---:|
| 1 | .272 / .275 | .372 / .363 | .271 / .274 | .273 / .277 | .264 / .262 |
| 300 | .372 / .393 | .469 / .469 | .377 / .398 | .388 / .410 | .346 / .348 |
| 5,000 | 1.932 / 2.254 | 1.782 / 1.969 | 1.776 / 1.964 | 1.772 / 1.959 | 1.546 / 1.599 |
| 20,000 | 6.905 / 8.173 | 6.085 / 6.855 | 6.045 / 6.772 | 6.089 / 6.861 | 5.128 / 5.391 |

| Cohort and horizon | Native 1 | Compact warm | Crossover | Candidate | Horizon oracle |
|---|---:|---:|---:|---:|---:|
| Original mixed 12, 5,000 | 1.289 | 1.313 | 1.306 | 1.304 | 1.265 |
| Original mixed 12, 20,000 | 4.469 | 4.336 | 4.350 | 4.337 | 4.095 |
| All 10 fresh warm opportunities, 5,000 | 1.224 | 1.043 | 1.040 | 1.039 | .827 |
| All 10 fresh warm opportunities, 20,000 | 4.331 | 3.466 | 3.427 | 3.460 | 2.577 |
| Historical Gram, 5,000 | 1.179 | 1.178 | 1.183 | **.310** | .308 |
| Historical Gram, 20,000 | 4.651 | 4.646 | 4.669 | **1.180** | 1.233 |

The short crossover saved 0.101 s and 0.092 s versus the incumbent on all 14 fresh configurations at 1 and 300 calls. At both long horizons, all three learned arms chose exactly the same factors on the 14 fresh configurations. Their small timing differences cannot establish a better long-horizon selection. At 5,000 and 20,000 the candidate recovered 41.5% and 45.9% of the directly measured native-to-oracle savings, with four and seven configurations at least 1.05× faster than native. Three and two configurations ran over 3% slower, respectively. The worst slowdowns were 16.5% on large polynomial evaluation and 11.0% on small union labels. The corresponding factor-only replay recoveries were 43.0% and 48.5%.

Within the original mixed 12, native won every pooled repeat at 5,000. At 20,000 the incumbent won every pooled repeat versus native. Among the full 14, synaptic eligibility supplied 0.686 s of the candidate's 0.815 s gain over native. The largest remaining opportunity was Manhattan Voronoi, where all three learned arms chose 1. The directly measured oracle at factor 4 saved another 0.716 s on that body. Both transfer probes have two indexing operations. Neither activated the matrix override. The historical Gram control did activate it, saving 3.466 s versus the incumbent in the long run and winning all repeats. [All totals, opportunity subsets, and recovery](../results/fori_horizon/totals.csv) · [Per-configuration seconds, gains, slowdowns, and factors](../results/fori_horizon/cases.csv) · [Raw process timings](../results/fori_horizon/live.json)

## Prospective follow-up on matrix updates

The initial transfer probes exposed a coverage gap in the structural guard. Three additional families were frozen after that sweep and measured with the same locked policy. They cover a lagged empirical CDF, a saturating photodetector array, and drawdown on a streamed panel of returns. Their inclusion was conditional on the shape and mechanism under investigation. This follow-up measures transfer within that class. All three qualified as warm opportunities, so the full and opportunity cohorts coincide. [Follow-up manifest and policy digest](../results/fori_horizon/coverage/matrix.json)

| Calls | Native 1 | Compact warm | Simple crossover | Matrix candidate | Horizon oracle | Saved vs native / incumbent |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | .362 | .377 | .364 | **.232** | .193 | .130 / .145 s |
| 5,000 | 1.489 | 1.500 | 1.498 | **.809** | .638 | .679 / .690 s |
| 20,000 | 5.700 | 5.713 | 5.704 | **2.949** | 2.291 | 2.751 / 2.764 s |

Each family has one configuration, so workload-weighted and pooled seconds coincide. The matrix candidate chose 2 on all three. The incumbent and simple crossover chose 1. Every configuration beat native in every repeat at every measured horizon. At 20,000 calls the candidate saved **1.340 s on the empirical CDF, 0.989 s on the detector, and 0.423 s on drawdown** versus native. Its least favorable median paired runtime ratio was 0.791 at 20,000 calls and 0.939 across all three horizons. It recovered 76.8%, 79.9%, and 80.7% of the directly measured oracle headroom. The corresponding replay fractions were 78.6%, 79.6%, and 79.9%. Most remaining headroom was in drawdown, where factor 8 won. [Follow-up totals](../results/fori_horizon/coverage/totals.csv) · [Every configuration](../results/fori_horizon/coverage/cases.csv) · [Raw timings](../results/fori_horizon/coverage/live.json)

## Using the result

Use the short bypass for low call counts. Keep the existing warm ranking for general long-running loops and enable the matrix candidate for validated matrix-update workloads. On the mixed twelve at 5,000 calls, native had the lowest deployable total. The earlier twelve-case control had only 87 ms of long-horizon factor-choice regret, while the present three matrix updates offered seconds. Adoption depends on which bodies actually run.

```python
import json
from pathlib import Path
from jax_unroll.auto_loop import auto_fori_loop
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.selector import DecisionEngine, HorizonPolicy

rules = json.loads(Path("results/fori_horizon/policy.json").read_text())
engine = DecisionEngine(
    model_loader=lambda: CompactForiPredictor.load("results/fori_iterations/incumbent_uncompressed.npz"),
    payoff=HorizonPolicy(**rules["candidate"]),  # use "crossover" for the simple policy
)
# Inside an enclosing jit, with unchanged body and static bounds
result = auto_fori_loop(lower, upper, body, initial, R=20_000, B=125, engine=engine)
```

## Reproduction

Run from the CSE583 repository root on the pinned CPU environment. The manifests protect the workload sources and the coverage follow-up also checks the locked policy digest. The final report commands verify the expected factor for every live row and can run independently on the saved data.

```sh
export PYTHONPATH=src:. JAX_PLATFORMS=cpu
py=.venv/bin/python
$py -m benchmarks.corpus.fori_horizon.cohort freeze
$py -m benchmarks.corpus.fori_horizon.replay
$py -m benchmarks.corpus.fori_horizon.transfer freeze
$py -m benchmarks.corpus.fori_horizon.calibrate
$py -m benchmarks.corpus.fori_horizon.transfer sweep
$py -m benchmarks.corpus.fori_horizon.live
$py -m benchmarks.corpus.fori_horizon.report
$py -c 'from benchmarks.corpus.fori_horizon.coverage import freeze; freeze(create=True)'
$py -m benchmarks.corpus.fori_horizon.live --coverage --sweep
$py -m benchmarks.corpus.fori_horizon.live --coverage
$py -m benchmarks.corpus.fori_horizon.report --coverage
$py -m pytest -q
```
