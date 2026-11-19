# Agent-generated native fori_loop cases

The corpus adds **28 training and 24 held-out configurations**. Seven training bodies cover six family groups. Six evaluation bodies cover five different groups. Each body runs at small and large sizes with 255 and 1,024 iterations. These include cheap recurrences, streamed matrix updates, indexed writes, and a dense spectral-unmixing control. [Frozen matrix and source hashes](../results/fori_agent_cases/split.json)

On training data, **20 of 28 configurations** had a larger factor that beat factor 1 in all three processes with at least 1.05× median paired warm speedup. The largest such gain was **2.09×** on large fog compositing at 1,024 iterations. These are native warm timings. The evaluation labels were collected and verified separately for subsequent policy evaluation.

## Author assignments

Five independent CLI sessions received partition-specific briefs. Each brief required a deterministic NumPy input factory, a single unchanged native JAX loop, identical inputs at factors 1, 2, 4, and 8, shape-stable carries, and nontrivial outputs through 1,024 iterations.

| CLI and model | Assigned partition | Retained cases |
|---|---|---|
| OpenCode · DeepSeek V4.1 Flash Free | Training | Order-book depth, pulse-density modulation, fog compositing, tournament ratings |
| agy · Gemini 3.8 Flash Medium | Training | Error-diffusion tile, online soft k-means, cyclic traffic-signal scheduling |
| OpenCode · Nemotron 3.5 Lightning Free | Evaluation | Lindley waiting times, age-cohort warranty survival, rolling bit-window motion |
| OpenCode · MiMo V2.6 Flash Free | Evaluation | Adaptive nonnegative spectral unmixing |
| agy · Gemini 3.8 Flash Medium | Evaluation | Inventory with lost sales, flood excursion durations |

Review excluded the allpass filter and Givens rotation because both mechanisms already appear in the corpus. The two error-feedback workloads share one training family. Inventory and waiting-time workloads share one evaluation family because both update a nonnegative balance. The freeze checks archived body hashes, collisions between new bodies, and family separation. [Workload registry](../benchmarks/corpus/fori_agent_cases/workloads.py)

The retained sources include corrections to the drafts. The Nemotron cases use factor-independent host inputs, replenishment varies by step, and motion evidence reads a new sample each iteration. Review also corrected the matrix dimensions in spectral reconstruction, the error-feedback and fog-compositing equations, and the traffic arrival rate to avoid saturated queues at the larger size.

## Load the data

```python
from benchmarks.corpus.fori_agent_cases.dataset import load

features, costs = load("train")
groups = features["family"]  # keep all configurations of a family in one fold
```

Each row of `features` describes the rolled loop. The axes of `costs` are configuration, factor in the order `(1, 2, 4, 8)`, and metric in the order `(first_call_ms, warm_us)`. The first call includes tracing, compilation and one synchronized execution. Warm labels take the median across measurement blocks within a process, then the median across three fresh processes. A horizon of `R >= 1` calls costs `first_call_ms + (R - 1) * warm_us / 1000` in native-factor replay. Wrapper comparisons require separate end-to-end measurements.

Use `load("eval")` for the frozen family holdout after choosing a policy. Data collection records native labels in separate files, compares outputs against factor 1, checks inputs for equality across factors, and checks that the traced programs differ only in their unroll parameter. [Training measurements](../results/fori_agent_cases/train_runs.json) · [Held-out measurements](../results/fori_agent_cases/eval_runs.json)

## Reproduce

```sh
export PYTHONPATH=src:. JAX_PLATFORMS=cpu
py=.venv/bin/python
$py -m benchmarks.corpus.fori_agent_cases.dataset freeze
$py -m benchmarks.corpus.fori_agent_cases.dataset train
$py -m benchmarks.corpus.fori_agent_cases.dataset eval
$py -m benchmarks.corpus.fori_agent_cases.dataset verify
$py -m pytest -q tests/test_fori_agent_cases.py
```
