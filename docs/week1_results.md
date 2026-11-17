# Week 1 results: factor sweeps and risk gate

Date: 2026-11-17. Scope: `TASKS.md` Phase 0 and Phase 1 (T1.1-T1.6).

Status: **Phase 0 and Phase 1 complete. The Week 1 risk gate passes on the CPU
backend.**

## Environment

| Item | Value |
|---|---|
| Machine | Apple M2, macOS 26.3.1 |
| Python | 3.13.2 |
| JAX / jaxlib | 0.12.0.1 / 0.12.0.1 |
| Backend / device | `cpu` / `cpu:0` |
| Sweep date | 2026-11-17 |

No GPU is available on this machine; these are CPU-only results. See
[hardware.md](hardware.md) for the target-hardware decision and the 2026-11-23
GPU checkpoint.

## Collected data

Commands:

```bash
make sweep          # primary shape variants, 36 configs x 4 factors x 3 repeats
make sweep-shapes   # 10 non-primary shape variants x T=256,1000 x 4 factors x 3 repeats
make oracle
```

| Artifact | Value |
|---|---|
| Primary raw CSV | `results/raw/sweep_cpu_2026-11-17.csv` (432 rows) |
| Secondary-shape raw CSV | `results/raw/sweep_cpu_2026-11-17_shapes.csv` (240 rows) |
| Unique configurations | 56 (36 primary + 20 shape) |
| Total measurements | 672 rows, 3 fresh-process repeats each |
| Correctness | 672/672 rows `correct=True`; 0 error rows |
| Oracle | `results/oracle_cpu_2026-11-17.csv` / `.md` |
| Risk gate | `results/risk_gate_cpu_2026-11-17.md` |
| Total-cost views | `results/oracle_cpu_2026-11-17_R{1000,10000}.csv` |

Each row measures tracing, lowering, backend compilation, warm latency with
`block_until_ready()` (adaptive 10-200 samples after warmup), end-to-end time
for `R=100` calls, StableHLO/optimized-HLO text sizes, and numerical
correctness against `u=1`. Columns and aggregation rules are in
[data_schema.md](data_schema.md).

## Risk gate (T1.6): PASS

Criterion: at least two workloads with distinct dominant warm factors, each
dominant for at least 50% of configurations and with a median warm gain of at
least 1.05x over factor 1. For a dominant factor of 1, the gain is
`worst-other / u=1`, i.e. the over-unrolling regression avoided.

| Workload | Dominant warm u | Dominance | Median warm gain vs default | Median gain vs best competitor | Dominant total u at R=100 | Useful? |
|---|---|---|---|---|---|---|
| `scalar_recurrence` | 8 | 0.625 | 1.319x | 1.039x | 1 | yes |
| `indexed_update` | 2 | 0.600 | 1.066x | 1.028x | 1 | yes |
| `elementwise_sine` | 1 | 1.000 | 2.010x | 1.028x | 1 | yes |
| `jacobi_solver` | 1 | 1.000 | 1.580x | 1.388x | 1 | yes |
| `reduction_vector` | 1 | 0.875 | 1.107x | 1.010x | 1 | yes |
| `dense_matvec` | 4 | 0.300 | 1.011x | 1.003x | 1 | no |

The useful workloads select three distinct factors (`1`, `2`, `8`), so the gate
passes. The dense matrix-vector family shows no usable signal, which is itself
an expected negative case: duplicating a dense matvec body does not help on
this CPU.

## Representative measurements

Medians of three fresh-process repeats. `warm` is the synchronized steady-state
latency; `compile` is backend compilation; `HLO` is optimized-HLO text length.

| Workload (variant, T) | u | warm (us) | compile (ms) | cold total (ms) | HLO bytes | Result |
|---|---:|---:|---:|---:|---:|---|
| scalar_recurrence (scalar, 1024) | 1 | 14.50 | 12.07 | 16.20 | 6,190 | baseline |
| scalar_recurrence (scalar, 1024) | 8 | 10.75 | 14.74 | 20.68 | 9,914 | **1.35x warm** |
| elementwise_sine (n1024, 1024) | 1 | 2590.23 | 14.36 | 18.61 | 6,668 | baseline |
| elementwise_sine (n1024, 1024) | 8 | 5276.67 | 18.61 | 24.57 | 10,459 | 2.04x slower |
| indexed_update (n1024, 1024) | 1 | 140.25 | 28.03 | 33.40 | 7,474 | baseline |
| indexed_update (n1024, 1024) | 2 | 127.08 | 29.81 | 36.56 | 10,656 | **1.10x warm** |
| indexed_update (n256, 1000) | 8 | 66.13 | 52.67 | 60.48 | 28,430 | **1.26x warm** vs u=1 (83.25 us) |
| jacobi_solver (n32, 1024) | 1 | 950.67 | 316.23 | 322.00 | 10,258 | baseline |
| jacobi_solver (n32, 1024) | 8 | 1503.67 | 2565.60 | 2571.74 | 84,962 | 1.58x slower |
| reduction_vector (n1024, 100) | 1 | 57.27 | 22.88 | 27.20 | 8,607 | baseline |
| reduction_vector (n1024, 100) | 8 | 74.98 | 126.14 | 134.23 | 39,657 | 1.31x slower |
| dense_matvec (n64, 1024) | 1 | 283.10 | 22.30 | 26.48 | 6,828 | baseline |
| dense_matvec (n64, 1024) | 8 | 286.79 | 55.43 | 61.84 | 18,337 | no gain |

Observations:

- The scalar recurrence benefit grows with trip count: roughly neutral for
  `T <= 256`, then 1.32-1.35x for `T=1000/1024`. Our measured signal is smaller
  than the proposal's preliminary 2.05x; the baseline loop here is already
  faster (14.5 us vs 32.9 us at `u=1`), so the headroom differs.
- Elementwise sine consistently regresses with unrolling; `u=1` is the correct
  choice for every sine configuration at `T >= 64`.
- Indexed gather/scatter prefers `u=2` at the primary shape and `u=4/8` for the
  smaller shape at large trip counts, showing a genuine shape-dependent choice.
- Jacobi (`fori_loop`) prefers `u=1`; larger factors multiply compile time by
  up to 8x and also slow execution.
- Reduction is flat at small trip counts and penalized by `u=8`; dense matvec
  is flat or slightly worse.
- StableHLO and optimized-HLO text grow with the factor, so the JAX-level
  `unroll` setting reaches backend IR. A full XLA-unrolling audit remains Phase
  4 work.

## Total-cost caveat

The project objective is `C(u) + R * T(u)`. At `R=100` (the sweep default) and
even `R=1000`, compilation dominates on this CPU: the oracle's total-cost choice
is `u=1` for every workload. At `R=10000`, the median total gain becomes 1.052x
for scalar recurrence (dominant total factor 4) and 1.044x for indexed update
(dominant factor 2); other workloads remain at 1.0.

This is a real tradeoff, not a contradiction: warm-only choices are useful, but
the selector must incorporate `R` and the compile budget `B`. Phase 3/4 should
evaluate a grid of `R` and `B` rather than only `R=100`.

## Reproduction

```bash
source .venv/bin/activate
make test
make sweep
make sweep-shapes
make oracle
```

`make oracle` reads the day's primary and shapes CSVs and writes the oracle
table plus the risk-gate report. Raw CSVs under `results/raw/` are gitignored;
the oracle and gate artifacts under `results/` are kept.

## Limitations

- CPU-only, single laptop-class device; no GPU claims.
- Three fresh-process repeats per configuration; adaptive warm samples reduce
  but do not eliminate timing noise.
- Correctness compares only the final loop output, within `atol=1e-4`,
  `rtol=1e-5`.
- The shape sweep covers two trip counts (`256`, `1000`), not the full grid.
- The gate is warm-latency based; the total-cost view is reported separately
  and is compilation-dominated at the default `R`.
- Phase 2 has not started: no Jaxpr features, learned model, baselines, or
  held-out split are implemented yet.
