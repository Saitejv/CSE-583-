# Unrolled fori workloads on Apple M2

Across 48 configurations, one warm invocation per configuration took **7.24 ms** in total, down from **25.55 ms** at native factor 1 and **20.79 ms** at the fastest measured native factor. These are **3.53×** and **2.87×** pooled speedups on Apple M2 with JAX and jaxlib 0.12.0.1. Each specialized kernel was faster than native factor 1 in every repeat, and all outputs passed the workload tolerances on the original and alternate inputs.

For the 32 configurations covered by the previous report, pooled specialized time fell from **9.58 ms to 6.42 ms** across the two measurements. A separate comparison interleaved the previous and current kernels on the same inputs in three fresh processes. Waveguide warm time improved **1.85×** and CT backprojection improved **1.04×** when summing times across their respective configurations. [Current measurements](../results/fori_specialization/blocked_runs.json) · [Archived measurements](../results/fori_specialization/runs.json) · [Interleaved previous and current timings](../results/fori_specialization/previous_comparison.json)

| Workload | Warm speedup over native factor 1 | Over fastest native factor 1, 2, 4 or 8 | Over specialization at factor 1 |
|---|---:|---:|---:|
| CT backprojection | 2.72× | 2.47× | 1.17× |
| Attention block | 2.11× | 2.03× | 1.02× |
| Multitap delay ring | 5.37× | 1.52× | 5.36× |
| Waveguide mesh | 2.63× | 1.95× | 2.63× |
| Matrix Horner | 5.80× | 5.75× | 5.80× |
| Particle-in-cell charge deposition | 4.63× | 4.63× | 4.63× |

The table uses geometric means of paired speedups over both sizes, four trip counts and three repeats. The last column compares with the same opt-in API at factor 1. CT and attention still batch projections at factor 1. The waveguide, Horner, and charge deposition use their native loops at factor 1, so their speedups include the block reformulation.

| Large shape, 1,024 iterations | Native factor 1 warm | Specialized warm | Native / specialized first call | Estimated break-even vs factor 1 |
|---|---:|---:|---:|---:|
| CT backprojection | 2,308 µs | 590 µs | 20.3 / 36.9 ms | 11 calls |
| Attention block | 300 µs | 118 µs | 83.9 / 26.5 ms | 1 call |
| Multitap delay ring | 212 µs | 16.0 µs | 26.0 / 60.6 ms | 179 calls |
| Waveguide mesh | 3,640 µs | 1,244 µs | 133.9 / 157.5 ms | 10 calls |
| Matrix Horner | 1,363 µs | 135 µs | 32.4 / 54.3 ms | 19 calls |
| Particle-in-cell charge deposition | 509 µs | 27.6 µs | 46.8 / 45.5 ms | 1 call |

First-call and warm times are medians across three processes. Break-even is the median of each process's crossover using its measured first-call and warm times.

## Implementation

- **CT backprojection and attention.** `precompute_fori_loop` batches projections with `jax.vmap` and runs the carry updates in ascending order through an unrolled scan. CT projects and accumulates its partial block separately from the full blocks. It uses factor 8, with factor 4 for the short trips on large images. Attention uses factor 4.
- **Multitap delay ring.** Each aligned block reads four contiguous tap windows and commits up to 32 writes with one slice update. The nearest tap is at least 32 entries behind the write head, so a block can read all of its inputs before writing. The final block uses the exact remainder width.
- **Waveguide mesh.** A pair of steps cancels each direction's forward and reverse shifts. The kernel advances two pressure grids and accumulates two pressure corrections, then reconstructs the directional wave buffers once. Factor 4 unrolls two pairs at a time. An odd trip count ends with a single directional step.
- **Matrix Horner.** A block of 32 steps reduces to one multiplication by a power of the matrix plus a weighted sum of its lower powers. The matrix powers are computed once, and the block offsets are computed together. The native loop handles noncontractive matrices.
- **Particle-in-cell charge deposition.** Each block damps the grid once and weights every particle contribution by the remaining damping in its block. One scatter-add combines up to 256 particles and accumulates collisions. The partial block uses its own damping weights.

Waveguide, Horner, and charge deposition reassociate floating-point operations. Their comparisons use the corpus tolerances of `atol=1e-4` and `rtol=1e-5` for the tested float32 inputs.

## Use and reproduce

```python
import jax
from benchmarks.corpus.fori_targeted.registry import build_specialized_workload

workload = build_specialized_workload("eval_dense_horner_matrix_poly", 1023, "large")
run = jax.jit(workload.fn)
polynomial = run(*workload.args)
```

For a custom static loop, supply a pure, batchable `prepare(index)` and a pure `step(carry, prepared)` to `jax_unroll.auto_loop.precompute_fori_loop`. The default byte limit is 8 MiB of prepared arrays. If their estimated size exceeds the limit, the helper evaluates `prepare` inside the native loop. Peak memory also depends on compiler temporaries.

```sh
PYTHONPATH=src:. .venv/bin/python -m benchmarks.corpus.fori_targeted.specialization \
    --out results/raw/fori_specialization.json
```

The benchmark runs three fresh CPU processes per workload, with both sizes and trip counts of 255, 256, 1,023 and 1,024. It randomizes compilation order and checks each variant on original and alternate inputs. Nine interleaved blocks measure synchronized calls on the alternate inputs. First-call latency includes tracing, lowering, compilation, and one execution with a warmed backend.
