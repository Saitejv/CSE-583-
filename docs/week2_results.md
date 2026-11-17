# Week 2 results: features, baselines, dataset, frozen splits

Status: Phase 2 complete on 2026-11-17. This wave adds the static feature
extractor, baseline policies, the feature/label dataset, and frozen
leave-one-family-out splits. No cost model or selector has been trained yet;
`model.py`, `selector.py`, and `auto_loop.py` remain Phase 3/4 stubs.

## Deliverables

| Task | Deliverable | Location |
|---|---|---|
| T2.1 | Jaxpr feature extractor with trip, operation-mix, footprint, dependence, and work features | `src/jax_unroll/features.py`, `FEATURE_COLUMNS` in `src/jax_unroll/schema.py` |
| T2.2 | `scan`/`fori_loop` handling, captured/not-captured documentation, measured extraction cost | [`docs/features.md`](features.md) |
| T2.3 | Known-value, scan/fori parity, determinism, and no-compile tests | `tests/test_features.py` |
| T2.4 | Factor-1, fixed 2/4/8, and LLVM-style size baselines | `src/jax_unroll/baselines.py`, `tests/test_baselines.py` |
| T2.5 | Feature/label dataset builder with stable schema, dedupe, and versioned metadata | `benchmarks/dataset.py`, `tests/test_dataset.py`, `data/features_cpu_2026-11-17.csv` |
| T2.6 | Frozen held-out families and LOFO folds | `src/jax_unroll/splits.py`, [`docs/splits.md`](splits.md), `tests/test_splits.py` |

## Dataset

Built from the Week 1 CPU sweeps (no new measurements):

```bash
make dataset
# python -m benchmarks.dataset \
#   --raw results/raw/sweep_cpu_2026-11-17.csv \
#         results/raw/sweep_cpu_2026-11-17_shapes.csv \
#   --out data/features_cpu_2026-11-17.csv
```

| Property | Value |
|---|---|
| Raw rows | 672 (0 dropped, 672 correct) |
| Dataset rows (configuration x factor) | 224 |
| Unique configurations (excluding factor) | 56 |
| Columns | 78 (`DATASET_COLUMNS`) |
| Families | 6 |
| Dev rows / holdout rows | 144 / 80 |
| Backend / device | cpu / cpu:0 |
| JAX / jaxlib | 0.12.0.1 / 0.12.0.1 |
| Labels | Median `C` (`cold_median_ms`), `T` (`warm_median_us`), end-to-end at `R=100`; natural-log labels included |

Metadata is in `data/features_cpu_2026-11-17.meta.json`: dataset/feature/split
versions, git commit, raw file SHA-256 hashes, row counts, device/backend,
derived build config (workloads, families, variants, trip counts, factors,
`R`), extraction timing, and the full frozen split manifest. The CSV has a `split`
column (`dev`/`holdout`) so Phase 3 can filter without re-deriving splits.

`B` and `R` feature columns are empty in this dataset because the Week 1 sweeps
used no budget and a single `R=100` horizon. Phase 3/4 will rebuild candidate
tables for the `R`/`B` grid; the feature extractor already accepts both.

## Feature examples

Primary shape at `T=256`, `u=1` (one body iteration). `jnp.roll` helper calls in
the Jacobi body are `jit` wrappers and are inlined into the operation counts.

| Workload | Loop | consts/carry/xs | eqns | elem/dot/red/gather/convert/other | carry bytes | max tensor | longest path | flops proxy |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `scalar_recurrence` | scan | 0/1/1 | 2 | 2/0/0/0/0/0 | 4 | 4 | 2 | 2 |
| `elementwise_sine` | scan | 0/1/1 | 2 | 2/0/0/0/0/0 | 4,096 | 4,096 | 2 | 2,048 |
| `reduction_vector` | scan | 1/1/1 | 3 | 2/0/1/0/0/0 | 4 | 4,096 | 3 | 2,049 |
| `dense_matvec` | scan | 1/1/1 | 2 | 1/1/0/0/0/0 | 256 | 16,384 | 2 | 8,256 |
| `indexed_update` | scan | 1/1/1 | 12 | 8/0/0/2/0/2 | 4,096 | 4,096 | 7 | 512 |
| `jacobi_solver` | fori_loop | 1/2/0 | 18 | 6/0/0/8/0/4 | 4,096 | 4,096 | 6 | 5,121 |

The Jacobi row shows the `fori_loop` treatment: raw `num_carry = 2` includes the
integer counter, while `num_loop_carried = 1` and `carry_bytes = 4,096` describe
only the 32x32 grid state. `docs/features.md` defines every column.

## Extraction cost

Over all 224 dataset rows (`extraction_ms`, one `jax.make_jaxpr` trace plus
feature computation per row, Apple M2 CPU):

| Mean | Median | p95 |
|---:|---:|---:|
| 1.33 ms | 1.02 ms | 2.07 ms |

Extraction never lowers, compiles, or times a factor; `tests/test_features.py`
asserts this by making `jax.jit` raise during extraction. These numbers are
decision overhead only (T3.4 will add model inference and include it in
end-to-end cost).

## Baseline sanity check

`src/jax_unroll/baselines.py` implements:

- `factor_1`: JAX default.
- `factor_2`, `factor_4`, `factor_8`: fixed factors.
- `llvm_size`: LLVM `UnrollThreshold`-style static size heuristic. Body size is
  `num_equations`; choose the largest factor with `u <= trip_count`,
  `u <= max_factor` (default 8), and `u * body_size <= threshold`
  (default 128 equations), else 1. It ignores remainder, memory, and
  dependences.

On all 56 measured configurations (warm latency only; Phase 2 has no learned
selector, so this is a baseline probe, not the final evaluation):

| Policy | Median warm gain vs `u=1` | Mean warm gain | Factor agreement with warm oracle | Choices |
|---|---:|---:|---:|---|
| `factor_1` | 1.000x | 1.000x | 55.4% | 56 x `u=1` |
| `factor_2` | 0.994x | 0.952x | 16.1% | 56 x `u=2` |
| `factor_4` | 0.975x | 0.885x | 12.5% | 56 x `u=4` |
| `factor_8` | 0.848x | 0.820x | 16.1% | 56 x `u=8` |
| `llvm_size` | 0.848x | 0.824x | 16.1% | 46 x `u=8`, 10 x `u=4` |

The measured warm oracle chooses `u=1` for 31/56 configurations, `u=2` for 9,
`u=4` for 7, and `u=8` for 9. The size heuristic over-unrolls small scan bodies
(most have only 2-3 equations) and therefore loses to `u=1` on median warm
latency, while the scalar-recurrence and indexed-memory families contain the
real gains. At `R=100`, compilation dominates and the total-cost oracle chooses
`u=1` for 55/56 configurations; Phase 3 must sweep `R` and `B` rather than
inherit this horizon. These are exactly the cases the learned selector has to
separate using static features (dependences, operation mix, footprint, trip
structure).

## Frozen splits

Frozen on 2026-11-17, split version 1:

- Dev: `recurrence`, `elementwise`, `reduction`, `dense_linear_algebra`.
- Held out: `indexed_memory`, `iterative_solver`.

LOFO cross-validation over the four dev families is the only development-time
validation. Held-out families are reserved for Phase 4. See
[`splits.md`](splits.md) for the fold table, rationale, metadata contract, and
change control.

## Tests and reproduction

```bash
source .venv/bin/activate
make test          # 186 passed on 2026-11-17
make dataset       # rebuild data/features_cpu_2026-11-17.csv + metadata
```

The suite covers feature values and parity, no-compile extraction, baseline
policies, dataset aggregation/join/dedupe/metadata, frozen splits and LOFO
folds, plus the Week 1 sweep/oracle/workload tests. Feature extraction and the
dataset builder are deterministic for a fixed JAX/device.

## Caveats

- CPU-only Apple M2 results; no GPU claims.
- Labels come from the Week 1 `R=100` sweeps, already aggregated over three
  fresh-process repeats per configuration.
- `B`/`R` context columns are unpopulated in this dataset; Phase 3 will build
  the `R`/`B` grid from the same sweep harness.
- `body_hash` is structural and factor-independent; shapes, constants, and trip
  count remain separate fields. The Phase 3 decision cache must include them.
- The `llvm_size` baseline is a deliberately simple static proxy, not a
  reproduction of LLVM's full cost model.
