# Deployment choice for installed JAX on the Apple M2

Use native JAX for the four library routines tested here. The previously recommended Extra Trees model selected factor 1 in ten of twelve routine/shape combinations. It selected factor 2 in the other two, both spherical harmonics computations. Their median synchronized execution times increased from 221 to 399 µs and from 466 to 873 µs. This changes the deployment recommendation for library code despite the positive result on synthetic loops.

| Pinned public routine | Tested sizes | Model choices | Held-out warm result |
|---|---|---|---|
| jnp.add.reduceat | 128, 512, 1,024 rows of width 8; segment starts every 8 rows | 1, 1, 1 | Native execution |
| jnp.lexsort | 256, 512, 1,024 integer keys over 48 records, default batch size 16 | 1, 1, 1 | Native execution |
| jax.scipy.special.sph_harm_y | Maximum degree 18, 24, 32 at 24 points | 1, 2, 2 | The two larger cases took 1.80× and 1.87× as long as native |
| jax.scipy.linalg.rsf2csf | Real Schur pairs of order 20, 32, 48 | 1, 1, 1 | Existing model fell back on unsupported features |

A calibration policy tried all four factors and required a 3% advantage in at least five of seven interleaved rounds. It retained native JAX in all 36 trials. Its median additional setup cost across the twelve signatures was 595 ms, with no subsequent warm saving to amortize it. Across the independent timing inputs, none of the twelve signatures had a candidate factor more than 3% faster than native when comparing median times across processes.

| Policy | Warm speedup, equal weight per signature | Warm speedup, one invocation per signature | Estimated speedup including setup at 1,000 calls per signature, pooled |
|---|---:|---:|---:|
| Native JAX | 1.000× | 1.000× | 1.000× |
| Synthetic-trained Extra Trees | 0.903× | 0.822× | 0.817× |
| Calibrate on this machine, reuse the chosen kernel | 1.000× | 1.000× | 0.330× |

## What was measured

Three fresh processes per routine and size gave 36 independent trials in shuffled order. Each trial selected on one input and verified the whole public function on another seed with the same signature. Every factor was checked against native output, including both outputs of the Schur conversion. Eleven interleaved timing blocks measured synchronized calls on the second input. The baseline was the unmodified public function under jit. Selecting factor 1 reuses that same callable; the summary therefore assigns it native warm time. The raw CSV also records independent timings of the identical callables.

The library APIs do not expose these internal unroll factors. The benchmark specializes the installed function's traced program at its single source-identified static fori_loop, keeping the surrounding operations and output structure. A factor-1 rewrite control differed from native warm time by at most 2.3% across the twelve signature medians. No installed library files were edited.

Setup measurements rotate policy order across processes and clear JAX's in-memory caches between policies; the persistent compilation cache is disabled. Setup includes first execution and, where applicable, model loading and selection or calibration. Python startup, module imports and preparation of device-resident inputs are excluded. Finite-horizon results use measured setup plus the remaining call count times measured warm latency. They are estimates for repeated synchronous calls. A larger enclosing jit, different data distribution, or different hardware is a separate deployment case.

## Reproduction

Run `.venv/bin/python -m benchmarks.library_throughput` from the project root with the pinned environment. See [per-process measurements](../results/library_throughput/runs.csv) and [horizon comparison](../results/library_throughput/summary.csv).

For a frequently reused custom loop whose factor you control, an optional offline calibration helper is available. Pass a pure, non-donating factory that returns an equivalent full application kernel for each factor and representative device-resident inputs. Reuse the returned jitted callable, or record the chosen factor with the signature and environment to avoid paying the tuning cost on startup.

```python
from jax_unroll.autotune import tune_fori_kernel

result = tune_fori_kernel(build_kernel, example_args)
output = result.kernel(*next_args)
```
