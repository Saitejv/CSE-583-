# Static Jaxpr features (Phase 2)

`src/jax_unroll/features.py` implements the Phase 2 static feature extractor.
It traces a loop function with `jax.make_jaxpr`, locates the loop primitive,
and describes the loop body without lowering, compiling, or benchmarking any
candidate unroll factor. The authoritative column order is
`FEATURE_COLUMNS` in `src/jax_unroll/schema.py`; tests live in
`tests/test_features.py`.

## API

```python
from benchmarks.workloads import build_workload
from jax_unroll.features import (
    extract_features,
    extract_features_from_workload,
    extract_loop_body,
)

workload = build_workload("scalar_recurrence", trip_count=100, factor=4)
features = extract_features_from_workload(workload, B=250.0, R=1000)

# Or extract from any function containing one top-level scan/while loop:
# (plain or wrapped in jax.jit; jit/pjit wrappers are descended into)
features = extract_features(
    fn, (init, xs), loop_kind="scan", trip_count=100, factor=4
)

# Or from a bare loop body whose call signature is (carry, x) / (i, carry):
# body_num_consts counts leading closed-over constants; body_num_xs counts
# trailing scanned values. The remaining inputs are loop-carried state.
features = extract_features(
    body_fn=step, body_args=(init, x0), loop_kind="scan",
    trip_count=100, factor=4, body_num_consts=0, body_num_xs=1,
)

loop = extract_loop_body(fn, (init, xs), loop_kind="scan")
```

`loop_kind` is the requested kind (`scan`, `fori_loop`, or `while`); it is part
of the dataset join key. With `loop_kind=None`, the traced primitive is reported
verbatim (`scan` or `while`) and the carry is counted as-is. `loop_primitive`
records what tracing actually found: `scan` (including static-bound
`fori_loop` in JAX 0.12), `while` (dynamic bounds), or `body` (bare-body
extraction). Passing `loop_kind="fori_loop"` marks a loop whose carry starts
with the `fori_loop` counter; only then is that counter stripped from the
loop-carried state and output features. Note that a static-bound `fori_loop`
requested as `loop_kind="scan"` also keeps the counter in the carry.

Extraction raises `FeatureExtractionError` when no loop is found, when the
traced function contains more than one `scan`/`while` loop, when the requested
`loop_kind` does not match the loop that was found (for example `scan`
requested for a dynamic-bound `while`), and for unsupported loop containers. `ValueError` is raised for invalid trip counts, factors, budgets,
unknown loop kinds, or inconsistent `body_num_consts`/`body_num_xs`. It does not
silently fall back; the Phase 3 selector owns fallback behavior.

## Feature groups

All features are scalar or compact JSON strings so a row can be written to CSV.
Units and definitions:

### Trip structure and context

| Column | Type | Definition |
|---|---|---|
| `loop_kind` | str | Requested loop kind (`scan`, `fori_loop`, or `while`); with `loop_kind=None` this is the traced primitive (`scan` or `while`). |
| `loop_primitive` | str | Traced container: `scan`, `while`, or `body`. |
| `trip_count` | int | Static trip count `T`. |
| `factor` | int | Candidate unroll factor `u`. |
| `full_chunks` | int | `T // u`. |
| `remainder` | int | `T % u`. |
| `remainder_ratio` | float | `remainder / T`. |
| `divides_evenly` | bool | `remainder == 0`. |
| `has_tail` | bool | `remainder > 0`. |
| `factor_over_trip` | float | `u / T`. |
| `backend` | str | `jax.default_backend()` unless overridden. |
| `device` | str | `str(jax.devices()[0])` unless overridden. |
| `dtype_bits` | int | Largest dtype width in the body, in bits. |
| `B` | float or empty | Optional compilation budget supplied by the caller. |
| `R` | int or empty | Optional call-count horizon supplied by the caller. |

### Loop structure and operation mix

| Column | Type | Definition |
|---|---|---|
| `num_consts` | int | Closed-over constant leaves passed to the body (for example a matrix or weight vector). |
| `num_carry` | int | Raw scan carry leaves. For `fori_loop` this includes the integer counter. |
| `num_loop_carried` | int | Loop-carried data state leaves, excluding a `fori_loop` counter. |
| `num_xs` | int | Scanned `xs` leaves per iteration. |
| `num_invars`, `num_outvars` | int | Body Jaxpr input/output arity. |
| `num_equations` | int | Leaf equations per body iteration, with `jit` wrappers inlined. |
| `unrolled_equations` | int | `num_equations * u`, a static code-size proxy. |
| `eq_elementwise` | int | Elementwise arithmetic/comparison/select equations. |
| `eq_dot` | int | `dot` / `dot_general` equations. |
| `eq_reduce` | int | `reduce_*` and `argmin`/`argmax` equations. |
| `eq_gather_scatter` | int | `gather`, `scatter`, `slice`, dynamic slice/update equations. |
| `eq_convert` | int | `convert_element_type` / `bitcast_convert_type` equations. |
| `eq_other` | int | Everything else, including `broadcast_in_dim`, reshape, concatenate, and nested control flow. |
| `equations_per_iteration` | int | Same as `num_equations`; kept for model-readable naming. |

### Tensor footprint

| Column | Type | Definition |
|---|---|---|
| `carry_bytes`, `carry_elements` | int | Bytes/elements of loop-carried data state (excludes a `fori_loop` counter). |
| `xs_bytes` | int | Bytes of one `xs` slice per iteration. |
| `const_bytes` | int | Bytes of closed-over constants. |
| `input_bytes`, `input_elements` | int | All body inputs (constants + carry + `xs` slice). |
| `output_bytes`, `output_elements` | int | Body data outputs (excludes a `fori_loop` counter output). |
| `intermediate_bytes` | int | Sum of intermediate tensor bytes produced by top-level equations and not returned as body outputs. |
| `max_tensor_bytes` | int | Largest tensor seen in the body, including inlined `jit` bodies. |
| `bytes_moved_proxy` | int | `input_bytes + output_bytes + intermediate_bytes`; a lower-bound traffic proxy. |
| `max_rank` | int | Maximum tensor rank in the body. |
| `carry_shapes`, `const_shapes`, `input_shapes`, `output_shapes` | str | Compact JSON lists of shapes, in body order. |
| `dtypes` | str | Compact JSON list of unique dtype names. |

### Dependences and work

| Column | Type | Definition |
|---|---|---|
| `longest_path` | int | Longest def-use path over top-level body equations, in equations. |
| `max_out_degree` | int | Maximum number of later uses of one equation result. |
| `num_feed_forward_values` | int | Distinct equation results consumed by at least one later equation. |
| `dependence_density` | float | Def-use edges divided by top-level equation count. |
| `flops_proxy` | int | Crude FLOP estimate: output elements for elementwise, `2*B*M*N*K` for `dot_general`, input elements for reductions. |
| `arithmetic_intensity_proxy` | float | `flops_proxy / bytes_moved_proxy` (0 when no bytes). |
| `body_hash` | str | First 16 hex characters of SHA-256 over the canonical body Jaxpr text; structural, factor-independent. |
| `extraction_ms` | float | Wall time of tracing plus feature computation for this call. |

## What is and is not captured

Captured:

- Static body structure for `lax.scan` and static-bound `lax.fori_loop`.
- Loops behind `jit`/`pjit` wrappers: tracing descends into the wrapper to find
  the `scan`/`while` equation.
- In JAX 0.12, `fori_loop` with Python-integer bounds traces to `scan`; the
  counter is identified by loop kind and removed from carry/output state
  features. It remains in raw `num_carry`, `num_invars`, and `num_outvars`.
- Closed-over constants, scanned slices, carry state, and intermediate bytes.
- Operation mix after inlining `jit`/`pjit` wrappers, so nested helper functions
  in a body (for example `jnp.roll` slices in the Jacobi workload) are counted.
- A top-level def-use path, fan-out, and feed-forward count.

Not captured:

- Runtime values, aliasing, memory layouts, sharding, register pressure,
  occupancy, or actual backend instruction selection.
- XLA fusion, re-rolling, or additional unrolling after StableHLO. HLO text is
  measured by the sweep, not by the feature extractor.
- Nested control flow contents: a nested `scan`/`while`/`cond` body is counted
  once as `eq_other` and not descended into. Only `jit`/`pjit` wrappers are
  inlined.
- Dynamic trip counts and dynamic-bound `fori_loop`: these trace to `while`.
  They are recognized (`loop_primitive == "while"`) but the while body also
  carries bound/counter values. Without a requested kind the raw carry is
  counted as-is, so `num_loop_carried` is an upper bound; with an explicitly
  requested `loop_kind="fori_loop"` the leading counter is stripped, though any
  remaining bound scalars still leave `num_loop_carried` as an upper bound, and
  tail/remainder features do not describe XLA's runtime behavior. This project
  is scoped to fixed trip counts.
- Functions with more than one top-level `scan`/`while` loop: extraction
  raises instead of guessing.
- Autodiff/transpose-specific effects and arbitrary nested user control flow.
- Cost-model targets: `C(u)` and `T(u)` are measurement labels joined later by
  `benchmarks/dataset.py`, not features.

## Cost and leakage guard

Extraction only calls `jax.make_jaxpr` and Python feature code. It never calls
`.lower()`, `.compile()`, `jax.jit`, or any timing loop. `tests/test_features.py`
monkeypatches `jax.jit` to fail during extraction to guard this property.

Measured over the 220 configurations in
`data/features_cpu_2026-11-17.csv` (Apple M2 CPU, JAX/jaxlib 0.12.0.1,
2026-11-17, dataset version 2), `extraction_ms` was:

| Statistic | Value |
|---|---:|
| Mean | 0.75 ms |
| Median | 0.56 ms |
| p95 | 1.13 ms |

The metadata sidecar for every dataset records these statistics. Traced
functions are small and use static shapes, so cost is milliseconds, not
seconds. Extraction cost is not included in the sweep's reported compile or
warm timings; Phase 3 will add it to end-to-end decision overhead (T3.4).

## Dataset join

`benchmarks/dataset.py` rebuilds each swept workload, calls
`extract_features_from_workload`, and joins the features with aggregated sweep
labels on `(workload, family, variant, loop_kind, backend, device, trip_count,
dtype, shape, factor)`. The output is `data/features_<backend>_<date>.csv` with
`DATASET_COLUMNS` plus a `split` column from
[`splits.md`](splits.md). See [`week2_results.md`](week2_results.md) for the
current table and baseline sanity numbers.
