# Data schema and measurement protocol

This document defines the on-disk columns, the measurement protocol, and the
aggregation rules for the Phase 1 factor sweep. The authoritative column list is
`SWEEP_COLUMNS` in `src/jax_unroll/schema.py`; the harness is
`benchmarks/sweep.py` and the aggregator is `benchmarks/oracle.py`. If this
document and the code disagree, the code wins.

No candidate factor is compiled or timed at decision time. Sweeps are offline
data collection only.

## Pipeline

```
benchmarks/workloads.py     registry: 6 workloads / 6 families, static shapes, fixed trips
        |
        v
benchmarks/sweep.py         parent enumerates configurations, randomizes order,
        |                   launches one child process per measurement
        v
python -m benchmarks.sweep --single-json '<config>'   (fresh OS process)
        |
        |  warm backend -> trace -> lower -> compile -> warm timings -> R calls -> correctness
        v
results/raw/sweep_<backend>_<date>[_shapes|_quick].csv
        |
        v
benchmarks/oracle.py        drop invalid rows -> medians over repeats -> oracle
        |
        +--> results/oracle_<backend>_<date>.csv
        +--> results/oracle_<backend>_<date>.md
        +--> results/risk_gate_<backend>_<date>.md
```

## Row identity and file layout

One raw row is one `(workload, variant, trip_count, factor, process_repeat)`
measurement. The workload registry currently contains:

| workload | family | loop | primary variant | other variants |
|---|---|---|---|---|
| `scalar_recurrence` | `recurrence` | `scan` | `scalar` | `vector8` |
| `elementwise_sine` | `elementwise` | `scan` | `n1024` | `n256`, `n4096` |
| `reduction_vector` | `reduction` | `scan` | `n1024` | `n256` |
| `dense_matvec` | `dense_linear_algebra` | `scan` | `n64` | `n32`, `n128` |
| `indexed_update` | `indexed_memory` | `scan` | `n1024` | `n256`, `n4096` |
| `jacobi_solver` | `iterative_solver` | `fori_loop` | `n32` | `n16`, `n64` |

Factors are `{1, 2, 4, 8}`; the default trip counts are
`{16, 64, 100, 256, 1000, 1024}`.

| Path | Produced by | Contents |
|---|---|---|
| `results/raw/sweep_<backend>_<date>.csv` | `make sweep` | Primary shape variant of every workload, 3 process repeats (432 rows with the current registry) |
| `results/raw/sweep_<backend>_<date>_shapes.csv` | `make sweep-shapes` | Non-primary shape variants only, trips `256,1000`, 3 repeats (240 rows with the current registry) |
| `results/raw/sweep_<backend>_<date>_quick.csv` | `make quick` | Harness smoke test: `recurrence,elementwise`, trips `16,100`, factors `1,8`, 1 repeat, 8 rows with the current registry. Not for analysis |
| `results/oracle_<backend>_<date>.csv` | `make oracle` | One aggregated oracle row per configuration key |
| `results/oracle_<backend>_<date>.md` | `make oracle` | Human-readable oracle table |
| `results/risk_gate_<backend>_<date>.md` | `make oracle` | Week 1 usefulness gate and evidence table |

`<backend>` comes from `--backend` (default `cpu`, or `JAX_PLATFORMS`).
`<date>` is an ISO `YYYY-MM-DD` date in both the Makefile and the
`benchmarks/sweep.py` defaults.

`.gitignore` ignores `results/raw/` and `data/raw/`. The oracle and risk-gate
files under `results/` are not ignored. `data/raw/` is currently unused.

## `SWEEP_COLUMNS`

Types are the logical types after CSV parsing. The `csv` module writes Python
booleans as `True`/`False`. Missing numeric fields (for example `hlo_bytes` when
the optimized-HLO text is unavailable) are empty strings / `NaN`. Timings are
rounded at write time (`6` decimals for milliseconds, `3` for microseconds,
`10` for `max_abs_diff`); derived fields use the unrounded intermediate values.

| Column | Type | Unit | Definition |
|---|---|---|---|
| `workload` | str | - | Registry key, e.g. `scalar_recurrence`. |
| `family` | str | - | Family key used for leave-one-family-out splits, e.g. `recurrence`. |
| `variant` | str | - | Shape variant key, e.g. `scalar`, `n1024`. |
| `loop_kind` | str | - | `scan` or `fori_loop`. |
| `backend` | str | - | Child process `jax.default_backend()`, normally `cpu`. |
| `device` | str | - | `str(jax.devices()[0])`, e.g. `cpu:0`. First enumerated device, not an explicit placement. |
| `jax_version` | str | - | `jax.__version__` in the child. |
| `jaxlib_version` | str | - | `jaxlib.__version__` in the child. |
| `trip_count` | int | iterations | Static loop trip count `T`. |
| `shape` | str | - | Compact JSON object of named shapes, keys sorted, e.g. `{"carry":[],"xs":[16]}`. Includes the matrix / indices / rhs where applicable. |
| `dtype` | str | - | Declared primary data dtype of the workload, currently `float32` for all workloads. Index arrays may still be integer. |
| `factor` | int | - | Unroll factor `u` passed to `lax.scan(..., unroll=u)` or `lax.fori_loop(..., unroll=u)`. |
| `divides_trip` | bool | - | `trip_count % factor == 0`. |
| `process_repeat` | int | - | Zero-based fresh-process repeat index. Repeat `r` shuffles the configuration list with `random.Random(seed + r)`. |
| `trace_ms` | float | ms | Wall time of `jax.make_jaxpr(fn)(*args)`. Host-side tracing only; no lowering or execution. |
| `lower_ms` | float | ms | `max(wall(jit(fn).lower(*args)) - trace_ms, 0)`. Lowering time derived by subtraction because `.lower()` traces again; the value therefore still contains a second trace and is not pure lowering time. |
| `compile_ms` | float | ms | Wall time of `lowered.compile()` (backend compilation). |
| `cold_total_ms` | float | ms | `trace_ms + lower_ms + compile_ms`. Excludes the untimed warmup call, the correctness reference compile, and process startup/import. |
| `warm_median_us` | float | us | Median per-call latency over the adaptive warm samples, each synchronized with `block_until_ready`. |
| `warm_p95_us` | float | us | 95th percentile of the same samples (`numpy.percentile` default linear interpolation). |
| `warm_min_us` | float | us | Minimum of the same samples. |
| `warm_std_us` | float | us | Population standard deviation (`statistics.pstdev`) of the samples; `0` when only one sample. |
| `warm_samples` | int | samples | Number of timed samples actually used; also recorded in `notes` as `adaptive_samples`. |
| `end_to_end_ms_R` | float | ms | `cold_total_ms + wall time of R synchronized calls` measured after the warm sampling. Default `R = 100`. |
| `r_calls` | int | calls | `R` used for the end-to-end measurement (`--r-calls`). |
| `stablehlo_bytes` | int | bytes | UTF-8 text length of `lowered.as_text()` (StableHLO text), not a binary artifact size. |
| `hlo_bytes` | int or empty | bytes | UTF-8 text length of `compiled.as_text()` (optimized HLO text), not a binary artifact size. Empty when the text is unavailable; a note records the exception type. |
| `correct` | bool | - | `allclose` of the final output against a freshly built `u=1` reference, and finite outputs; for `factor == 1`, finiteness of the final output (there is no comparison). |
| `max_abs_diff` | float | absolute | Maximum absolute difference over all output leaves, computed in `float64`; `0.0` for `factor == 1`. May be `NaN` when the measured output is non-finite. |
| `error` | str | - | Empty on success. Child exception as `Type: message`, or a harness-failure message including the child exit code and stderr tail. |
| `notes` | str | - | Semicolon-separated diagnostics: `pilot_call_us=<float>`, `adaptive_samples=<int>`, optional `optimized HLO text unavailable: <Exception>`, non-finite-output markers (`non-finite output at u=1; configuration excluded`, `non-finite output at u=<u>`), or `measurement failed in child process` / `harness failure`. |

## Measurement protocol

### Process isolation and ordering

1. The parent (`benchmarks/sweep.py` with no `--single-json`) enumerates every
   selected `(workload, variant, trip_count, factor)`.
2. For each `process_repeat` in `0..repeats-1`, it copies the base list and
   shuffles it with `random.Random(seed + repeat)` (`--seed`, default `583`),
   then measures the whole batch. Repeats are sequential batches, not
   interleaved by configuration.
3. Each measurement launches a fresh child:

   ```bash
   <sys.executable> -m benchmarks.sweep --single-json '<config json>' \
       --warmup 5 --sample-budget-ms 250 --min-samples 10 --max-samples 200 \
       --r-calls 100
   ```

   The child runs with `cwd` set to the project root and `PYTHONPATH`
   containing the project root and `src/`. For `--backend cpu|gpu|tpu` the
   parent also sets `JAX_PLATFORMS`. The child timeout is 600 s by default.
4. The child prints one `ROW_JSON:<json>` line; the parent parses the last such
   line and writes/flushes one CSV row. If no row is parsed, an error row is
   written instead.

Process startup and JAX import time are not included in any column.

### Cold timing

Before tracing the measured workload, the child compiles and runs a trivial
`jit(lambda x: x + 1)`, synchronizes it with `block_until_ready()`, and calls
`jax.clear_caches()` to initialize XLA without leaving the trivial compilation
cached.

Then, for the measured workload:

1. `trace_ms`: wall time of `jax.make_jaxpr(fn)(*args)`.
2. `lower_ms`: wall time of `jax.jit(fn).lower(*args)` minus `trace_ms`,
   clamped at zero. `.lower()` traces the function again, so this is a
   subtraction-derived lower+lowering value, not a pure lowering timer.
3. `stablehlo_bytes`: UTF-8 byte length of `lowered.as_text()`.
4. `compile_ms`: wall time of `lowered.compile()`.
5. `hlo_bytes`: UTF-8 byte length of `compiled.as_text()`, guarded by
   `try/except`.
6. `cold_total_ms = trace_ms + lower_ms + compile_ms`.

### Warm timing

After compilation the harness runs the compiled callable once untimed, then
`--warmup` untimed calls (default 5). With the default this is one initial call
plus five warmup calls, i.e. six executions before sampling.

Sample count is adaptive:

1. Run 3 pilot calls, each synchronized, and take the median pilot time.
2. `count = int(sample_budget_s / pilot_median)`, clamped to
   `[--min-samples, --max-samples]` (default `[10, 200]`), with
   `--sample-budget-ms 250`. If the pilot median is non-positive, use
   `max_samples`.
3. Run `count` timed calls. Every call is followed by
   `jax.block_until_ready(output)`; Python dispatch overhead is included.
4. Record median, p95, minimum, population standard deviation, and the sample
   count.

### End-to-end

Immediately after warm sampling, time `R` synchronized calls
(`--r-calls`, default 100) and set
`end_to_end_ms_R = cold_total_ms + wall_time_of_R_calls`. The compile cost is
the `cold_total_ms` from the same row; process startup is excluded.

### Correctness

For every row, first require every output leaf to be finite
(`numpy.isfinite`). A non-finite output means the workload itself diverged (for
example, a dense recurrence whose `float32` state overflows at a large trip
count), so it is not a valid correctness reference or performance label.

For `factor == 1`, `correct` is the finiteness of the final output and
`max_abs_diff` is `0.0` (there is no comparison). For every other factor, after
all timing is finished:

1. Rebuild the same workload at `factor=1`.
2. `jit(...).lower(...).compile()` and run it once, synchronized.
3. Require both the measured row's final output and the reference output to be
   finite.
4. Compare them with `numpy.allclose(..., atol, rtol, equal_nan=False)`.
5. Use the workload spec's `atol` / `rtol` (currently `1e-4` / `1e-5` for all
   workloads).
6. Compute `max_abs_diff` by casting every paired leaf to `float64`,
   subtracting, taking the per-leaf maximum, then the maximum over leaves.

Only the final loop output is compared, not intermediate iterations. The
reference compilation happens after all reported timings, so it does not affect
them. A row whose output is non-finite, or whose result is not close to a finite
reference, is marked `correct=False` with an explanatory `notes` marker; the
oracle and dataset builder drop it.

The original Week 1 sweep used `equal_nan=True` and marked `factor == 1` correct
unconditionally, so a divergent configuration (`dense_matvec`/`n32`/`T=1000`)
was incorrectly accepted. The harness was fixed on 2026-11-17, those 12 raw rows
were re-measured, and the dataset was rebuilt (dataset version 2).

### Failure rows

A child exception produces a row with the parent-supplied identity fields
(`workload`, `family`, `variant`, `loop_kind`, `trip_count`, `factor`,
`process_repeat`) filled in, `error` set, `notes` set to `measurement failed in
child process`, and the remaining fields empty. A child that exits without
printing a `ROW_JSON` line produces `notes=harness failure` and an error message
with the exit code and stderr tail. A subprocess timeout writes an error row with
`error=child timeout after Ns` and the sweep continues. The sweep exits non-zero
if any written row has a non-empty `error`. All error rows are dropped during
aggregation.

## Aggregation in `benchmarks/oracle.py`

### Loading and filtering

1. Concatenate the `--raw` files (or the most recently modified
   `results/raw/sweep_*.csv` if `--raw` is omitted).
2. Require the grouping columns plus `correct`, `error`, `cold_total_ms`, and
   `warm_median_us`.
3. Keep only rows where `error` is empty after stripping, and `correct` parses
   as one of `true`, `1`, `1.0` (case-insensitive). If no valid row remains,
   aggregation aborts.

The configuration key is:

```
workload, family, variant, loop_kind, backend, device,
trip_count, shape, dtype
```

The aggregation key adds `factor`. `divides_trip` is intentionally *not* part of
either key because it is a property of the candidate factor, not of the
configuration. It is recomputed per aggregated row as
`trip_count % factor == 0`, so a non-dividing factor such as `T=100, u=8`
competes normally with every other factor. In the oracle output, `divides_trip`
refers to the oracle warm factor.

### Per-factor medians

For each `(configuration key, factor)` group, compute the median over all valid
process repeats of:

`cold_total_ms`, `trace_ms`, `lower_ms`, `compile_ms`, `warm_median_us`,
`warm_p95_us`, `warm_min_us`, `warm_std_us`, `end_to_end_ms_R`,
`stablehlo_bytes`, and `hlo_bytes` (NaN values skipped; NaN if all missing).

`repeats` is the number of valid rows in the group. This produces one aggregated
row per configuration and factor. `divides_trip` is recomputed as
`trip_count % factor == 0` per aggregated row. (`warm_samples`, `r_calls`, and
`max_abs_diff` are coerced to numeric but are not aggregated.)

### Oracle row

For each configuration key that has a `factor == 1` row, the oracle selects:

- **Oracle warm factor**: factor with the smallest median `warm_median_us`;
  ties go to the smaller factor.
- **Oracle total factor for R**: factor minimizing
  `cold_median_ms + R * warm_median_us / 1000`; ties go to the smaller factor.
  `R` is the oracle's `--r-calls` argument (default 100), not the `r_calls`
  recorded in the raw file.
- **Warm gain vs u=1**: `factor_u1_warm_us / oracle_warm_us` (NaN if the oracle
  warm latency is zero). Gains greater than 1 mean the chosen factor is faster.
- **Total gain vs u=1**: `u1_total_ms_R / oracle_total_ms_R`.
- **Factor agreement**: `1` if the total-optimal factor equals the warm-optimal
  factor, else `0`.
- **divides_trip**: `trip_count % oracle_warm_factor == 0`; the flag is not part
  of the configuration key because it depends on the candidate factor.

Configurations without a valid `u=1` row are skipped. Note that the
`end_to_end_median_ms` aggregate is stored internally but not used for the
oracle total-cost columns; the oracle recomputes the linear cost from medians so
that a different `R` can be evaluated without re-running the sweep.

Output columns (`ORACLE_COLUMNS`): the configuration key, `repeats` (copied
from the `u=1` group), `factor_u1_cold_ms`, `factor_u1_warm_us`,
`oracle_warm_factor`,
`oracle_warm_us`, `warm_gain_vs_u1`, `oracle_total_factor_R`,
`oracle_total_ms_R`, `u1_total_ms_R`, `total_gain_vs_u1`, and
`factor_agreement_best_warm`. Rows are sorted by
`workload, variant, trip_count` (stable).

### Risk gate

The gate (`gate_evidence`) is evaluated per `workload`, pooling all of that
workload's variants and trip counts. For each workload:

1. Count the oracle warm factors. The **dominant warm factor** is the most
   frequent; **dominance** is its count divided by the number of configuration
   rows. Ties are resolved deterministically in favor of the smaller factor.
2. For each configuration whose oracle warm factor equals the dominant factor:
   - If the dominant factor is not 1, gain =
     `u1.warm_median_us / chosen.warm_median_us`.
   - If the dominant factor is 1, gain =
     `max(warm of every non-1 factor in the same aggregated group) /
     u1.warm_median_us`. This is the regression avoided by not over-unrolling.
   - Skip the configuration if the required row is missing; for a non-1
     dominant factor, also skip if the chosen latency is zero (the dominant-1
     branch does not guard against a zero `u=1` latency).
3. The workload's **median warm gain** is the median of those per-configuration
   gains, or `1.0` if none were computable.
4. `useful_choice = dominance >= 0.5 and median_warm_gain >= 1.0 + threshold`,
   with `--threshold 0.05` by default, i.e. a median gain of at least **1.05x**
   over factor 1.
5. The gate **PASSES** only if at least **2 workloads** are useful **and** those
   useful workloads have at least **2 distinct dominant warm factors**. The
   criterion is distinct factors, not distinct families; for example a useful
   `u=1` workload plus a useful `u=4` workload passes.
6. The total-cost view (dominant total factor, dominance, median total gain
   vs `u=1`, median gain vs the best competitor) is computed with the same
   rules on `cold_median_ms + R * warm_median_us / 1000` and written to the
   gate report. It is **secondary** and does not change the PASS decision.

The gate report also states the limitations: one CPU device, medians over
repeated fresh-process measurements, optimized-HLO text size as a proxy, and
the requirement to audit HLO before attributing gains to JAX-level unrolling.

## Reproducible commands

The Makefile exports `PYTHONPATH=<repo>/src:<repo>` and prefers
`.venv/bin/python`. Set `BACKEND` to label outputs (`make BACKEND=cpu ...`).
Do not merge CPU and GPU files.

```bash
source .venv/bin/activate

make test           # pytest entry point; see the caveat below
make list           # print the workload registry as JSON
make quick          # 8-measurement harness smoke sweep
make sweep          # primary-shape sweep, 3 process repeats
make sweep-shapes   # secondary shape variants at T=256,1000, 3 repeats
make sweep-all      # sweep + sweep-shapes
make oracle         # aggregate the day's primary and shapes CSVs, write oracle + gate
make help           # list targets
```

Equivalent raw Python commands:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD"
DATE="$(date +%Y%m%d)"

python -m benchmarks.sweep --list

python -m benchmarks.sweep --backend cpu \
    --out "results/raw/sweep_cpu_${DATE}.csv" --repeats 3

python -m benchmarks.sweep --backend cpu \
    --out "results/raw/sweep_cpu_${DATE}_shapes.csv" \
    --shapes secondary --trip-counts 256,1000 --repeats 3

python -m benchmarks.oracle \
    --raw "results/raw/sweep_cpu_${DATE}.csv" \
          "results/raw/sweep_cpu_${DATE}_shapes.csv" \
    --r-calls 100 \
    --out "results/oracle_cpu_${DATE}.csv" \
    --md-out "results/oracle_cpu_${DATE}.md" \
    --gate-out "results/risk_gate_cpu_${DATE}.md"
```

Quick smoke sweep, equivalent to `make quick`:

```bash
python -m benchmarks.sweep --backend cpu \
    --out "results/raw/sweep_cpu_${DATE}_quick.csv" \
    --families recurrence,elementwise --trip-counts 16,100 \
    --factors 1,8 --repeats 1 --limit 12 \
    --sample-budget-ms 50 --r-calls 5
```

Useful sweep flags: `--families`, `--shapes {primary,secondary,all}`,
`--trip-counts`, `--factors`, `--repeats`, `--seed`, `--warmup`,
`--sample-budget-ms`, `--min-samples`, `--max-samples`, `--r-calls`,
`--timeout-s`, `--limit`, `--append`, `--list`.

Caveats:

- `uv pip install -r requirements.txt` installs the project in editable mode
  (`-e .`), so `python -m jax_unroll` works without `PYTHONPATH`. The Makefile
  additionally exports `PYTHONPATH=<repo>/src:<repo>` for direct module runs.
- A sweep overwrites its target CSV unless `--append` is passed.
- `make oracle` reads only the day's `SWEEP_OUT` and, if present, `SHAPE_OUT`.
  It does not read the `_quick.csv` file. Invoking `benchmarks.oracle` without
  `--raw` picks the most recently modified primary sweep and ignores
  `*_quick.csv` and `*_shapes.csv`. Week 1 results are summarized in
  [docs/week1_results.md](week1_results.md).
- With the default trips and factors `{1,2,4,8}`, only `T=100` with `u=8`
  produces `divides_trip=False`; `T=1000` is divisible by all candidate
  factors. Both still compete in the oracle and gate.

## Phase 2 feature/label dataset

Phase 2 joins static features with the aggregated sweep labels. The feature
definitions are in [features.md](features.md), the frozen splits are in
[splits.md](splits.md), and the current table is summarized in
[week2_results.md](week2_results.md).

```
results/raw/sweep_<backend>_<date>[_shapes].csv
        |
        +--> benchmarks/oracle.py         aggregate_raw: drop bad rows, median repeats
        |
        +--> src/jax_unroll/features.py   static Jaxpr features per (workload, variant, T, u)
        |
        v
benchmarks/dataset.py                     join on the configuration key, add split + log labels
        |
        +--> data/features_<backend>_<date>.csv
        +--> data/features_<backend>_<date>.meta.json
```

`DATASET_COLUMNS` in `src/jax_unroll/schema.py` is the authoritative output
order. It contains:

- `DATASET_KEY_COLUMNS`: `workload, family, variant, loop_kind, backend,
  device, trip_count, dtype, factor`.
- `shape`: the raw compact JSON shape string.
- `split`: `dev` or `holdout` from `jax_unroll.splits.split_for_family`.
- `FEATURE_COLUMNS` not already present as key columns: trip/remainder context,
  operation mix, footprint, dependence and work proxies, `body_hash`, and
  `extraction_ms`.
- `DATASET_LABEL_COLUMNS`: aggregated medians over process repeats
  (`trace/lower/compile/cold/warm/p95/min/std/stablehlo/hlo/end_to_end`),
  `repeats`, `correct_all`, `r_calls`, and natural-log labels `log_cold_ms`,
  `log_warm_us`.

Rows are one per `(configuration, factor)`. Duplicate raw rows across input
files aggregate by median rather than producing duplicate dataset rows.
Incorrect or errored raw rows are dropped. The builder refuses to combine
multiple backends or mixed JAX/jaxlib versions in one table.

The metadata sidecar records `dataset_version`, `feature_version`,
`split_version`, the full `split_manifest`, UTC build time, git commit,
backend/device, JAX/jaxlib versions, SHA-256 hashes and row counts for each raw
file, raw/valid/dropped/unique-config counts, family lists, feature/label
columns, `r_calls`, mean/median/p95 `extraction_ms`, and a `config` object
recording the builder, backend, raw paths, workloads, families, variants, trip
counts, factors, and `r_calls`.

Build and inspect it with:

```bash
make dataset
# equivalent:
python -m benchmarks.dataset \
    --raw results/raw/sweep_cpu_2026-11-17.csv \
          results/raw/sweep_cpu_2026-11-17_shapes.csv \
    --out data/features_cpu_2026-11-17.csv
```

`--backend` selects a backend when raw rows span several; `--meta-out` overrides
the default metadata path (`<out stem>.meta.json`). `make dataset` globs
`results/raw/sweep_<backend>_*.csv` and excludes `*_quick.csv`.

Caveats:

- Feature extraction never compiles or benchmarks a candidate factor; only
  `jax.make_jaxpr` tracing is used, and the corresponding test makes `jax.jit`
  fail during extraction.
- `B` and `R` feature columns are empty in the Week 1 dataset (no budget and a
  single `R=100` sweep). Phase 3/4 rebuild the table for each `R`/`B` setting.
- Labels are CPU-only and come from the Week 1 fresh-process sweep; do not mix
  them with GPU rows or different split versions.
- Dataset version 2 (2026-11-17 follow-up) drops the 12 raw rows for
  `dense_matvec`/`n32`/`T=1000` because that workload diverges to `NaN`. The
  `correct` gate now requires finite outputs; see the correctness section.
