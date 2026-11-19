# Four native fori_loop scenarios from the 583 scan benchmark

These external controls use the four step functions from `583/benchmark_unroll.py` at their original dimensions. The historical scan timings were available before this evaluation. The shipped application policy and compact warm selector were frozen throughout. These are synthetic kernels in families related to earlier project benchmarks. [Input fingerprints, source digest and protocol](../results/fori_application/scan_ports_manifest.json)

The native port carries the original state through an indexed loop. For the scalar scan it reads the original input at index `i`. The other three steps ignore the index. The float32 inputs use the source's seed (20261103) and draw order. We checked the inputs directly against the supplied script and compared all 16 native configurations against its factor-1 scan. All inputs were identical, the maximum absolute output difference was zero, and each native factor changed only the unroll parameter in the port's Jaxpr.

## Native factor sweep

Warm microseconds per synchronized call on Apple M2. The external scan column uses the source's pooled median. The four native columns use the median of three process medians from the new fori_loop sweep.

| Scenario | Dimensions | Source scan at 1 | Native 1 | Native 2 | Native 4 | Native 8 |
|---|---|---:|---:|---:|---:|---:|
| Scalar recurrence | 8,192 scalar updates | 32.88 | 32.53 | 23.11 | 18.49 | 16.14 |
| Elementwise sine | 128 steps, 4,096 elements | 1,245.94 | 1,195.62 | 1,520.60 | 1,667.45 | 2,086.42 |
| Mean plus vector update | 128 steps, 8,192 elements | 642.71 | 627.96 | 629.62 | 628.56 | 628.92 |
| Dense matrix recurrence | 64 steps, 128 × 128 matrices | 3,425.10 | 3,316.92 | 3,319.69 | 3,315.94 | 3,316.76 |

The scalar recurrence was the only repeatable warm opportunity at the 1.05× threshold. Factor 8 was 2.02× faster there. The elementwise loop took 75% longer with factor 8. All four factor medians on each of the two remaining loops lay within 0.3% of factor 1. [Original scan summary](../results/fori_application/scan_source_summary.json) · [Native first-call and warm samples](../results/fori_application/scan_ports_runs.json)

The saved native runs also contain per-body static features, body hashes, input dimensions, and per-factor costs in the same schema as the development sweeps. A later fit can use them as development data. The existing model's supported trip counts run from 16 to 1,024; the external scalar case deliberately exercises its out-of-range fallback.

## Horizon-specific replay

For each native factor, replay adds its median first-call milliseconds to its median warm microseconds times `(calls - 1) / 1000`. At 5,000 calls the best per-case choices saved an estimated 0.079 s across the four kernels (25.976 s to 25.897 s). The scalar loop accounted for 0.076 s. At 20,000 calls the estimated opportunity was 0.340 s out of 103.572 s, including 0.321 s from the scalar loop. Even perfect factor selection would save only about 0.3% of the replayed cohort time. These estimates exclude wrapper overhead.

## Direct wrapper timings

Each cell sums the four scenarios and averages three independent CPU processes. The totals include shared engine setup, lazy or eager model loading, tracing and decision, compilation, and every synchronized call. Inputs and the CPU backend were staged before timing. The oracle runs the best factor from the separate cold-plus-warm sweep and excludes the cost of that search. Each scenario has one configuration, so pooled and equal-workload seconds coincide.

| Calls per scenario | Native 1 | Compact warm | Previous policy | Shipped engine | Direct oracle |
|---:|---:|---:|---:|---:|---:|
| 300 | 1.709 | 1.721 | 1.698 | 1.707 | 1.703 |
| 5,000 | 26.687 | 26.697 | 26.665 | 26.632 | 26.517 |
| 20,000 | 107.387 | 107.099 | 106.455 | 106.842 | 106.492 |

At 300 calls the shipped engine bypassed selection for all four. At 5,000 it chose 1 on all four. Even though the engine's pooled mean was 0.055 s below native, their identical factors give no evidence of a selection gain. On the single opportunity, the scalar recurrence, native took 0.177 s, the engine took 0.180 s and the factor-8 oracle took 0.101 s. This was also the engine's worst median per-configuration slowdown (2.2%). [All configurations](../results/fori_application/scan_ports_cases.csv) · [Full cohort and opportunity totals](../results/fori_application/scan_ports_totals.csv)

At 20,000 calls the engine again selected 1 for every loop and was faster than native in only one of the three pooled repeats. The 0.545 s lower pooled mean includes a 0.441 s difference on the matrix case running the same factor. The raw direct recovery ratio (60.9%) is sensitive to this variation; the engine captured none of the scalar factor-choice opportunity. On that single-case subset, factor 8 took 0.350 s versus native's 0.672 s, saving **0.321 s** and winning all three repeats. The wrapper took 0.694 s, losing 0.022 s versus native. The 8,192-step case gives a concrete target for evaluating a future long-scalar policy. [Long-horizon configurations](../results/fori_application/scan_ports_long_cases.csv) · [Long-horizon totals and opportunity subset](../results/fori_application/scan_ports_long_totals.csv)

## Reproduction

From the repository root with the pinned environment:

```sh
export PYTHONPATH=src:. JAX_PLATFORMS=cpu
py=.venv/bin/python
module=benchmarks.corpus.fori_application
$py -m "$module.scan_ports" freeze
$py -m "$module.scan_ports" sweep
$py -m "$module.live" --cohort scan_ports
$py -m "$module.report" scan_ports
$py -m "$module.live" --cohort scan_ports --long
$py -m "$module.report" scan_ports --long
$py -m pytest -q tests/test_fori_scan_ports.py
```

The live commands resume completed process blocks. For new timings, use a fresh checkout and move the two saved scan_ports live JSON files aside. The imported source summary is historical. The sweep and direct results use the repository's pinned CPU environment and are independent of the external folder.
