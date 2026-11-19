# Compilation-budget-aware loop unrolling for JAX

CSE 583, Group 10 · Saitej Velaga, Maanas Pavuluri, and Jack McClure

Two opt-in static loop specializations improved warm full-caller latency on Apple M2. At 1,024 rows, the residual caller achieved **1.250×** median warm speedup and the segmented-energy caller achieved **1.655×**. Both paid higher first-call costs. The experimental learned selector found wins on some held-out loops and underpredicted cold costs on new full callers. Native JAX is the default.

## Reproduce the results

From the repository root, create an environment with Python 3.13 and install the pinned dependencies.

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m jax_unroll
python -m pytest -q
make quick
make phase5-demo
make phase5-replay
```

`phase5-demo` checks exact parity and one trace per reduction caller and signature, then exercises learned selection through `auto_fori_loop` on an integer recurrence. The demo measures live first-call and warm latency in one process. `phase5-replay` rebuilds the headline tables from 133 recorded process measurements and compares the computed analysis against the archived JSON. Run live timing commands sequentially on an idle machine. Use the replay for the exact published values and fresh measurements for replication.

To regenerate the original factor-sweep grid and oracle with fresh timings, use a scratch directory.

```bash
scratch=$(mktemp -d)
make sweep-all oracle BACKEND=cpu \
  SWEEP_OUT="$scratch/primary.csv" SHAPE_OUT="$scratch/secondary.csv" \
  ORACLE_OUT="$scratch/oracle.csv" ORACLE_MD="$scratch/oracle.md" \
  GATE_OUT="$scratch/risk.md"
```

The sweep covers six workload families, two shape tiers, factors 1/2/4/8, trip counts up to 1,024, and three fresh process repeats. It records tracing, lowering, backend compilation, synchronized warm median and p95, and correctness against factor 1. The archived sweep retained 660 of 672 rows. The twelve omitted rows all belong to one dense-matvec configuration whose float32 recurrence overflowed. [Sweep protocol and columns](data_schema.md) · [Week 1 results](week1_results.md)

To collect fresh isolated full-caller timings and summarize them:

```bash
JAX_PLATFORMS=cpu python -m benchmarks.phase4 --out "$scratch/phase4"
python -m benchmarks.phase5 replay --results "$scratch/phase4"
```

`make verify-week3` replays the historical holdout gates. `make recovery-train` rebuilds the frozen dev-only model and leave-one-family-out results. `make recovery-verify` checks the recorded recovery confirmation without rescoring the one-shot gate. The [archived full-caller measurements](../results/phase4/runs.json), [freeze record](../results/phase4/freeze.json), and [analysis with HLO counts](../results/phase4/analysis.json) support the final CPU results.

## Selection and integration

The offline training pipeline fits lightweight ridge models to estimate cold compile cost and warm latency for each factor. The static extractor reads trip and remainder structure, operation mix, tensor sizes and dependency proxies from the Jaxpr. The experimental tracing-time policy minimizes predicted `C(u) + R × T(u) / 1000` among factors that fit budget `B`, with cold cost in milliseconds and warm latency in microseconds. The decision uses one rolled Jaxpr and model inference. An enclosing JIT reuses its executable for the same signature. [Feature definitions](features.md) · [Frozen splits](splits.md) · [Training module](../src/jax_unroll/train.py) · [Versioned recovery model](../models/cost_cpu_recovery_v1.json)

Use `auto_fori_loop` with an explicit factor for a project-owned static loop, or pass a shared `DecisionEngine` to opt into the learned policy. The two deployed reductions use fixed factors from prior confirmation on specific signatures. Unsupported signatures use the native reduction. The proposal included both loop types. The final integration covers static-bound `fori_loop`, with `auto_scan` deferred. The sweep and feature extractor still cover `scan`. [API examples and full-caller protocol](phase4_report.md) · [Supported reduction signatures](compile_once_reductions.md)

## CPU evaluation

Apple M2, JAX/jaxlib 0.12.0.1. Each reduction comparison uses seven fresh-process pairs with exact output checks and synchronized timing. Speedups are paired before aggregation. Break-even combines each first call with that repeat's measured warm rate. The reported count is the median among repeats with a durable crossover.

| Full caller | Factor | First native / deployed (ms) | Warm native / deployed (µs) | Paired warm speedup [min, max] | Break-even calls |
|---|---:|---:|---:|---:|---:|
| Residual, 128×8 | 8 | 40.82 / 53.70 | 4.73 / 4.67 | 1.003× [0.985, 1.034] | 455,704, with 2/7 lacking crossover |
| Residual, 1,024×8 | 8 | 41.48 / 53.08 | 21.37 / 17.03 | 1.250× [1.223, 1.347] | 2,581 |
| Energy, 256×1, 4 segments | 4 | 47.73 / 54.26 | 5.76 / 5.20 | 1.108× [1.105, 1.131] | 11,806 |
| Energy, 1,024×1, 16 segments | 4 | 47.99 / 54.58 | 20.22 / 12.13 | 1.655× [1.566, 1.733] | 858 |
| Residual, 513×8 native fallback | 1 | 41.80 / 42.07 | 15.11 / 15.21 | 0.984× [0.974, 1.019] | 1 in 2/7, with 5/7 lacking crossover |

The end-to-end estimate combines each repeat's measured first call and remaining warm calls. Speedups are medians of per-repeat ratios.

| 1,024-row caller | 1,000 calls, native / deployed (ms) | Paired speedup | 10,000 calls, native / deployed (ms) | Paired speedup |
|---|---:|---:|---:|---:|
| Residual | 62.83 / 69.99 | 0.903× | 255.54 / 222.66 | 1.141× |
| Energy | 68.15 / 66.44 | 1.017× | 250.15 / 176.44 | 1.412× |

The 128-row residual lost end to end in six of seven repeats even at 100,000 calls. In the energy win, optimized HLO retained one outer loop and grew from one to four dynamic slices and from 17.6 to 29.2 kB. The 128-row residual grew from one to 15 dynamic slices with little change in warm latency. [Full horizon table](phase4_report.md#reduction-deployment-results) · [Compiler audit](phase4_report.md#compiler-audit-and-rollout-limits)

### Learned policy and held-out loops

The original two frozen models each recorded zero strict wins over both factor 1 and the LLVM-style size baseline across 20 held-out configurations. A separate recovery model scored 12 strict paired wins across six configurations among 92 long-horizon settings where all three policies met the measured budget. Median speedup across those settings was 0.990× vs factor 1 and 1.150× vs LLVM size. All wins occurred at 100,000 calls. [Original gates](week3_results.md#frozen-holdout-gate) · [Recovery confirmation](phase4_handoff.md#one-shot-confirmation-pass-with-regressions)

The full-caller experiment evaluated the frozen recovery model on three new loop bodies. At 100,000 calls and a 100 ms budget, median end-to-end speedup was 0.997× vs native and 0.831× vs LLVM size. The model matched the measured oracle in 0/9 sessions and median regret was 20.27%. At 25 ms the model predicted zero violations and all nine measured first uses exceeded the budget. No measured candidate met that budget. Total time charges the learned policy a median 3.18 ms tracing-time decision within first use and a separate median 0.14 ms model load. [Budget and oracle results](phase4_report.md#experimental-learned-selection)

### GPU evaluation

The project collected CPU measurements only. GPU compile costs, factor rankings, and speedups remain unmeasured. [Hardware record](hardware.md)

## Threats to validity

- **Hardware and compiler.** Results cover one Apple M2 CPU and one pinned JAX/jaxlib pair. Optimized HLO confirms that factors can produce different code. Opcode counts alone cannot establish the cause of a timing difference.
- **Workloads and splits.** Whole-family holdouts and new full callers reduce leakage. The deployed signature set is small and reflects earlier confirmation. New routines, shapes, dtypes, and models need independent evidence.
- **Timing and invocation count.** Warm measurements use device-resident inputs and synchronized calls. Fresh first-use timings exclude interpreter startup, backend initialization, and input transfer. Break-even extrapolates from warm medians and depends on stable shapes, repeated calls, and machine load.
- **Prediction and budgets.** Static Jaxpr proxies omit backend optimizations. Cold estimates can understate actual first-use cost, and a tight budget can be infeasible for every factor.

The [correctness suite](../tests/) covers reference parity, pytrees, differentiation, overrides, fallback behavior, and JIT reuse. [Final presentation](phase5_slides.pptx)
