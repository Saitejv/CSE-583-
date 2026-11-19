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

## Horizon-specific replay

For each native factor, replay adds its median first-call milliseconds to its median warm microseconds times `(calls - 1) / 1000`. At 5,000 calls the best per-case choices saved an estimated 0.079 s across the four kernels (25.976 s to 25.897 s). The scalar loop accounted for 0.076 s. At 20,000 calls the estimated opportunity was 0.340 s out of 103.572 s, including 0.321 s from the scalar loop. Even perfect factor selection would save only about 0.3% of the replayed cohort time. These estimates exclude wrapper overhead.

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
