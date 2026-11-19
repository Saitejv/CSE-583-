# Phase 4 CPU deployment report

Phase 4 ships opt-in axis-zero reductions and a wrapper for project-owned static `lax.fori_loop` loops. The release policy uses factor 8 for five confirmed float32 subtraction shapes and factor 4 for five confirmed single-column float32 segmented-addition signatures with int32 starts. Other signatures use the native reduction. The policy checks the default backend and pinned JAX/jaxlib versions. Performance evidence is specific to Apple M2. [Supported signatures and stand-alone reduction timings](compile_once_reductions.md)

## Run the demo and evaluation

```bash
make phase4-demo
make phase4-evaluate
make test
```

The demo calls the native and opt-in reductions from equivalent jitted callers, with per-column scaling and absolute residuals for subtraction and square-root energy per segment for addition. It reports the factor, synchronized first-call and warm latency, exact output equality, and one trace per caller and signature. Repeated calls use different arrays with the same signatures. Its first-call timings share a process. Use the isolated evaluation for deployment costs. Fresh evaluation writes under `results/raw/phase4/` by default.

For a project-owned loop, run the wrapper inside a stable enclosing JIT to reuse the compiled executable.

```python
import jax

from jax_unroll.auto_loop import auto_fori_loop

@jax.jit
def weighted_sum(values):
    return auto_fori_loop(
        0, values.shape[0], lambda i, total: total + values[i], 0.0,
        force_factor=4,
    )
```

Omit `force_factor` for native defaults or use `disable=True` to bypass specialization. Forcing a factor applies to nonempty static loops. Dynamic bounds keep native handling. The native `unroll=` option remains available with JAX's own validation. For experimental selection, create a shared `DecisionEngine(CostModel.load(Path("models/cost_cpu_recovery_v1.json")))` outside JIT and pass it as `engine=engine` along with `B` in milliseconds and the expected invocation count `R`. The engine runs during the enclosing JIT trace and can reuse a cached decision on retraces. `auto_scan` is deferred under the Phase 4 static `fori_loop` scope. [Selector and decision cache contract](phase4_handoff.md#integration-contract)

## Evaluation protocol

The harness records source and frozen-model SHA-256 hashes before generating inputs or recording timings, then checks the hashes again before writing results. Factors and signatures were fixed from the earlier confirmation. The full-caller run used new seeds, four device-resident input sets per repeat, and sorted, unsorted and repeated segment starts. Four cases used supported signatures and one additional shape exercised native fallback. Each reduction policy and repeat had its own fresh process and its first synchronized call included tracing, lowering, compilation and execution. The timer starts after inputs are ready. It excludes interpreter startup, backend initialization, and host-to-device input transfer. JAX's persistent compilation cache was disabled. Warm comparisons ran in the native process after both variants compiled, with 15 interleaved blocks per repeat and synchronization on every call. Results use seven fresh pairs per reduction.

The experimental arm used the frozen recovery model on three new project-owned loop bodies and three fresh repeats per body. All four candidate factors and each budgeted learned path had separate fresh processes for first-call timings. Each learned process also measured paired synchronized warm latencies of its actual wrapper and all candidates. Its floating-point outputs were checked with relative tolerance 2×10⁻⁶ and absolute tolerance 2×10⁻⁷, and its integer outputs were checked exactly. A preflight run found a 1.49×10⁻⁸ absolute difference in the weighted recurrence across factors. The reduction deployment kept exact checks. The native and LLVM-size comparisons include their measured first call. The learned comparison charges its measured decision inside the first call and adds one separate model load. The oracle minimizes measured total time over feasible factors 1, 2, 4 and 8, with no selection charge. Oracle metrics exclude pairs where no measured candidate met the budget. Predictions cover the model's cold-cost estimate. The measured learned budget also includes decision overhead and model loading. [Freeze record](../results/phase4/freeze.json) · [Raw process measurements](../results/phase4/runs.json) · [Paired results and optimized HLO audit](../results/phase4/analysis.json)

## Reduction deployment results

Apple M2 CPU, JAX/jaxlib 0.12.0.1, 2026-11-18. Each latency column shows the median of seven repeats. Speedups are paired within each repeat before aggregation. Break-even is the first call count at which deployed elapsed time stays strictly below native, using the measured first call plus the remaining calls at the measured warm rate.

| Full caller | Factor | First native / deployed (ms) | Warm native / deployed (µs) | Warm speedup median [min, max] | Warm wins | Break-even calls median [min, max] |
|---|---:|---:|---:|---:|---:|---:|
| Residual, 128×8 | 8 | 40.82 / 53.70 | 4.73 / 4.67 | 1.003× [0.985, 1.034] | 5/7 | 455,704 [74,033, 6,896,559], 2 without crossover |
| Residual, 1024×8 | 8 | 41.48 / 53.08 | 21.37 / 17.03 | 1.250× [1.223, 1.347] | 7/7 | 2,581 [1,510, 2,995] |
| Energy, 256×1, 4 segments | 4 | 47.73 / 54.26 | 5.76 / 5.20 | 1.108× [1.105, 1.131] | 7/7 | 11,806 [10,548, 75,612] |
| Energy, 1024×1, 16 segments | 4 | 47.99 / 54.58 | 20.22 / 12.13 | 1.655× [1.566, 1.733] | 7/7 | 858 [620, 1,456] |
| Residual, 513×8 fallback | 1 | 41.80 / 42.07 | 15.11 / 15.21 | 0.984× [0.974, 1.019] | 2/7 | 1 in 2 repeats, 5 without crossover |

The 128-row caller is effectively near parity warm and lost end-to-end in six of seven repeats at 100,000 calls. The 256-row energy caller had a 91.08 ms deployed cold outlier, which pushed one break-even estimate to 75,612 calls. The 513-row control used native unrolling and had identical optimized HLO opcode counts for the inspected instructions.

| Median paired end-to-end speedup | 1 call | 10 calls | 100 calls | 1,000 calls | 10,000 calls | 100,000 calls |
|---|---:|---:|---:|---:|---:|---:|
| Residual, 128×8 | 0.759× | 0.760× | 0.762× | 0.780× | 0.878× | 0.980× |
| Residual, 1024×8 | 0.790× | 0.791× | 0.804× | 0.903× | 1.141× | 1.235× |
| Energy, 256×1, 4 segments | 0.869× | 0.869× | 0.872× | 0.891× | 0.991× | 1.087× |
| Energy, 1024×1, 16 segments | 0.875× | 0.876× | 0.890× | 1.017× | 1.412× | 1.623× |
| Residual, 513×8 fallback | 1.019× | 1.019× | 1.019× | 1.016× | 0.993× | 0.985× |

## Experimental learned selection

The selector picked factor 1 for weighted blend and clipped EMA and factor 2 for the debounced counter in every repeat and budget. The LLVM-style size baseline picked factors 8, 4, and 8 respectively. Across the 27 learned sessions, median selection overhead was 3.18 ms and model loading added 0.14 ms. At the 25 ms budget the model forecast a median 14.91 ms and the actual learned first-use median was 34.80 ms. The native and LLVM baselines also exceeded 25 ms in every repeat. In the feasible settings the learned selector never matched the oracle. The oracle picked factor 4 for the debounced counter, factor 8 for clipped EMA, and factor 4 in five of six weighted-blend comparisons, with factor 8 in the sixth.

| Cold budget at 100,000 calls | Learned violations predicted / measured | Median vs native [min, max] | Median vs LLVM [min, max] | Oracle factor agreement | Median oracle regret |
|---|---:|---:|---:|---:|---:|
| 25 ms | 0/9 / 9/9 | 1.007× [0.986, 1.033] | 0.828× [0.716, 1.017] | No feasible candidates | N/A |
| 100 ms | 0/9 / 0/9 | 0.997× [0.992, 1.020] | 0.831× [0.712, 1.005] | 0/9 | 20.27% |
| Unlimited | N/A | 1.000× [0.984, 1.019] | 0.835× [0.717, 1.009] | 0/9 | 20.26% |

At the two feasible settings, four of 18 sessions beat both baselines. All four were debounced-counter runs. The largest feasible loss against LLVM was on clipped EMA. The learned path remains experimental and native JAX is the general default.

## Compiler audit and rollout limits

The full-caller Jaxpr check normalizes the nested JIT's function label and checks that only the expected `unroll` parameter differs. The fallback has the same normalized Jaxpr as native. All inspected optimized HLO variants retained one outer `while`. The 1,024-row energy win grew from one to four dynamic slices and from 17.6 to 29.2 kB of optimized HLO, with seven fusions in both variants. The 128-row residual near-loss grew from one to 15 dynamic slices and from 10.1 to 30.0 kB, with six fusions in both. For clipped EMA, the model chose the rolled graph with one slice, while the faster LLVM factor 4 had four slices and the oracle factor 8 had eight. [Per-variant optimized HLO counts](../results/phase4/analysis.json)

The full pytest suite passed all 1,177 tests. The wrapper tests cover pytrees, reverse-mode differentiation, explicit overrides, dynamic and empty bounds, decision reuse, and outer JIT trace reuse.

The full-caller evidence covers two small pipelines and four confirmed signatures on one machine. The stand-alone confirmation supports the other six shipped signatures for the public reduction calls. Wider deployment needs exact or workload-appropriate output checks and fresh-process first-call and warm measurements on additional signatures and full callers. General learned selection also needs separate application holdouts and better cold-cost calibration across feasible and infeasible budgets.
