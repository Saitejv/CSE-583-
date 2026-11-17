# Frozen loop-family splits

Status: **frozen** on 2026-11-17 (split version 1). This split was fixed before
any Phase 2 feature or cost-model work. The authoritative implementation is
`src/jax_unroll/splits.py`; `tests/test_splits.py` pins the current values and
must be updated intentionally if the split ever changes.

## Frozen split

| Field | Value |
|---|---|
| Split version | `1` (`SPLIT_VERSION`) |
| Frozen date | `2026-11-17` (`SPLIT_FROZEN_DATE`) |
| Source module | `src/jax_unroll/splits.py` |
| Development families | `recurrence`, `elementwise`, `reduction`, `dense_linear_algebra` |
| Held-out families | `indexed_memory`, `iterative_solver` |

`ALL_FAMILIES` is the six-family union. `DEV_FAMILIES` and
`HELD_OUT_FAMILIES` are disjoint and partition it.

| Family | Split |
|---|---|
| `recurrence` | dev |
| `elementwise` | dev |
| `reduction` | dev |
| `dense_linear_algebra` | dev |
| `indexed_memory` | holdout |
| `iterative_solver` | holdout |

A row's split is determined by its family, not by workload, shape, trip count,
or factor. Workloads in `benchmarks/workloads.py` map to families as follows
(primary shape variant marked):

| Family | Workload | Loop kind | Shape variants |
|---|---|---|---|
| `recurrence` | `scalar_recurrence` | `scan` | `scalar` (primary), `vector8` |
| `elementwise` | `elementwise_sine` | `scan` | `n256`, `n1024` (primary), `n4096` |
| `reduction` | `reduction_vector` | `scan` | `n256`, `n1024` (primary) |
| `dense_linear_algebra` | `dense_matvec` | `scan` | `n32`, `n64` (primary), `n128` |
| `indexed_memory` | `indexed_update` | `scan` | `n256`, `n1024` (primary), `n4096` |
| `iterative_solver` | `jacobi_solver` | `fori_loop` | `n16`, `n32` (primary), `n64` |

## Rationale for the held-out families

`indexed_memory` is the only gather/scatter family. `indexed_update` reads
`carry[indices]` and writes back with `carry.at[indices].set(...)` at a fixed set
of unique indices, so its dependence and memory behavior differ from the
elementwise, reduction, and dense-arithmetic dev families. Week 1 CPU results
showed a distinct choice in this family: the measured oracle favored `u=2` for
60% of configurations with a 1.066x median warm gain over `u=1` (1.10x at the
primary shape, `T=1024`), while elementwise and reduction favored `u=1`.
Holding it out tests transfer to gather/scatter code without letting that
memory-driven behavior influence model or threshold choices.

`iterative_solver` is the only `fori_loop` family and represents an
application-style kernel. `jacobi_solver` carries a 2D grid, receives the
`fori_loop` counter as a body argument (unused in the current body), and builds
a four-neighbor (up/down/left/right) stencil from four `jnp.roll` slices every
iteration. This exercises
static `fori_loop` lowering and slice/shift code that the scan-based dev
families do not cover. Week 1 CPU results favored `u=1` and showed sharply
rising cold cost at `u=8` (2.57 s cold total versus 0.32 s at `u=1`, driven by
compile time) and 1.58x slower warm latency, so the family also probes
compile-budget behavior. Holding it out keeps `fori_loop` and stencil patterns
out of fitting and selection.

These are design rationales plus Week 1 measurements only; no Phase 2 model
results exist yet.

## Split rule

- All Phase 3 model fitting, model selection, feature and threshold choices, and
  LOFO cross-validation use **only** `DEV_FAMILIES`.
- `HELD_OUT_FAMILIES` are used **only** for the final Phase 4 evaluation.
- Never tune, threshold, calibrate, or select features on held-out rows.
- Use the LOFO folds below for development-time validation instead of touching
  the held-out families.
- Call `assert_holdout_excluded(families)` at fit or selection entry points
  rather than relying on row filtering alone.

## LOFO folds

`lofo_folds()` defaults to `DEV_FAMILIES` and returns one fold per development
family, named `lofo_<test family>`. Each fold trains on the other three dev
families and tests on the named family.

| Fold | Train | Test |
|---|---|---|
| `lofo_recurrence` | `elementwise`, `reduction`, `dense_linear_algebra` | `recurrence` |
| `lofo_elementwise` | `recurrence`, `reduction`, `dense_linear_algebra` | `elementwise` |
| `lofo_reduction` | `recurrence`, `elementwise`, `dense_linear_algebra` | `reduction` |
| `lofo_dense_linear_algebra` | `recurrence`, `elementwise`, `reduction` | `dense_linear_algebra` |

`lofo_folds(families)` can be given an iterable of development families to
restrict the folds. It raises `ValueError` for unknown families, for any
held-out family, for an empty list, and when fewer than two development
families remain (no training family would be left).

## Dataset tagging and metadata

`DATASET_COLUMNS` in `src/jax_unroll/schema.py` places a `split` column after
`shape` and before the feature columns. The T2.5 builder in
`benchmarks/dataset.py` implements this contract for every
`data/features_*.csv`:

- Set each row's `split` to `split_for_family(row["family"])`, yielding exactly
  `"dev"` or `"holdout"`.
- Fail the build on an unknown family instead of defaulting it to `dev`.
  `split_for_family` raises `ValueError` listing the known families.
- Emit the full `manifest()` output as JSON in the dataset metadata sidecar
  (`data/features_<backend>_<date>.meta.json`) under the key `"split_manifest"`,
  together with `DATASET_VERSION` and `FEATURE_VERSION`. Do not hand-copy
  individual split fields.

```python
import json

from jax_unroll.schema import DATASET_VERSION, FEATURE_VERSION
from jax_unroll.splits import manifest

metadata = {
    "dataset_version": DATASET_VERSION,
    "feature_version": FEATURE_VERSION,
    "split_manifest": manifest(),
}
json.dumps(metadata)
```

The builder's metadata sidecar also records `split_version`, `dev_families`,
and `held_out_families` at top level, with the full `manifest()` output under
`split_manifest`. The manifest records `split_version`, `frozen_date`,
`all_families`, `dev_families`, `held_out_families`, `final_dev`,
`final_holdout`, and the four LOFO folds. Rows with `split == "holdout"` are
reserved for Phase 4 evaluation and must not enter Phase 3 fitting or selection,
including when they are physically present in the same CSV.

The builder fails on an unknown family because `split_for_family` raises; it
does not silently default rows to `dev`.

## Change control

The split is frozen: do not edit the family lists for modeling convenience. If
a change is unavoidable, all of the following are required:

1. Bump `SPLIT_VERSION` and set `SPLIT_FROZEN_DATE` to the new freeze date.
2. Update this document, including both split tables and the LOFO fold table.
3. Update the values pinned in `tests/test_splits.py` so the change is
   explicit.
4. Record the change in the changelog below.
5. Rebuild the feature dataset and rerun training; do not compare or mix results
   produced under different split versions.

| Version | Date | Change |
|---|---|---|
| 1 | 2026-11-17 | Initial freeze: `recurrence`, `elementwise`, `reduction`, and `dense_linear_algebra` dev; `indexed_memory` and `iterative_solver` held out. |

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_splits.py
```

The test module checks the frozen constants, the family partition, consistency
with `benchmarks.workloads.FAMILIES`, split lookup and errors, LOFO fold
coverage and determinism, holdout exclusion, and the JSON-serializable
manifest.
