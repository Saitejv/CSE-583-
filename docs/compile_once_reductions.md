# Compile-once reductions on Apple M2

Use the opt-in kernels below for repeated axis-zero float32 reductions matching the measured workloads. The factor is chosen while tracing from the operation and shape. Each entry point is jitted and compiles one executable per input signature. Steady-state calls reuse it. There is no model lookup, runtime timing, or compilation of alternative candidates in deployment.

```python
from jax_unroll.library_kernels import add_reduceat_rows, subtract_reduce_rows

# x has shape (1024, 8), float32
differences = subtract_reduce_rows(x)

# values has shape (1024, 1), float32; starts has shape (16,), int32
sums = add_reduceat_rows(values, starts)
```

Subtraction starts with row zero and subtracts subsequent rows in order. Segmented addition uses JAX's initial gather and its original per-index masked update, including its handling of repeated or nonmonotone starts. The deployed CPU path uses factor 8 for the five subtraction signatures below and factor 4 for the five segmented addition signatures. It requires float32 values and int32 segment starts on JAX/jaxlib 0.12.0.1 with CPU as the default backend. Other signatures, backends and JAX versions use the native axis-zero operation. Performance was measured on Apple M2. [Full-caller deployment timings](phase4_report.md)

## Warm execution against native JAX

| Operation | Input rows × columns | Segments | Native µs | Specialized µs | Median paired speedup |
|---|---:|---:|---:|---:|---:|
| add.reduceat | 256 × 1 | 4 | 5.60 | 5.05 | 1.11× |
| add.reduceat | 513 × 1 | 7 | 7.21 | 6.55 | 1.10× |
| add.reduceat | 769 × 1 | 13 | 10.51 | 9.74 | 1.07× |
| add.reduceat | 1024 × 1 | 1 | 8.94 | 6.74 | 1.32× |
| add.reduceat | 1024 × 1 | 16 | 9.53 | 6.81 | 1.40× |
| subtract.reduce | 128 × 8 | — | 4.69 | 4.35 | 1.08× |
| subtract.reduce | 257 × 1 | — | 5.05 | 4.46 | 1.14× |
| subtract.reduce | 512 × 8 | — | 6.40 | 4.92 | 1.31× |
| subtract.reduce | 769 × 16 | — | 7.77 | 5.20 | 1.49× |
| subtract.reduce | 1024 × 8 | — | 8.77 | 5.65 | 1.55× |

Each entry won in seven of seven independent processes. For one warm invocation at each of these ten shapes, the sum of the median latencies fell by 20.2%. That percentage depends on this mix. The earlier four-routine benchmark used eight-column segmented additions with many more segments; those stay native. The subtraction routine is a newly evaluated public JAX library operation. On the original 1,024-key sorting case, factor 4 won in five of seven processes and had a median speedup of only 1.008×. Native remains the recommendation on the original suite.

## Verification

The development sweep tried additional partial and full unroll settings across the original library suite, then varied segment counts and row widths. The confirmation run fixed the factor policy before evaluating fresh data, including intermediate sizes. Each process rotated among four independently generated device-resident inputs per signature. Segmented cases included sorted, unsorted and repeated starts. Every implementation's output matched the native array exactly. Fifteen interleaved timing blocks synchronized each call. The reported speedup is the median of seven paired process speedups.

The shipped entry points trace the public JAX function and specialize its unroll parameter. Regression tests compare the complete printed Jaxpr with the native trace and require exactly that one parameter change. An independent control changed only the unroll field in the original library Jaxpr. Its speedups matched the deployed kernels closely, including 1.55× for 1,024-row subtraction and 1.40× for 1,024-row addition with 16 segments. This attributes the benefit to loop unrolling while preserving the surrounding computation. The warm results exclude initial compilation, host-to-device input transfer and Python startup. They measure synchronized public calls, not execution fused inside a larger application.

Run `.venv/bin/python -m benchmarks.library_confirmation` to reproduce the confirmation. [Process measurements](../results/library_specialization/confirmation.csv) and [exploratory timings](../results/library_specialization/development.csv) are retained separately.
