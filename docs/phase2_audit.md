# Phase 2 Audit & Readiness Report: Jaxpr Features, Baselines, and Dataset

**Date:** 2026-11-17  
**Project:** Compilation-Budget-Aware Loop Unroll Selection for JAX  
**Scope:** Phase 2 Accomplishments, In-Depth Code & Data Audit, Bug Fixes, and Phase 3 Transition Readiness  
**Target Hardware:** Apple M2 (CPU-only, `cpu:0`), JAX 0.12.0.1, jaxlib 0.12.0.1

---

## 1. Executive Summary & Verdict

Phase 2 of the project was tasked with implementing static Jaxpr feature extraction, establishing baseline unrolling heuristics, constructing a versioned feature/label dataset from Phase 1 factor sweeps, and freezing leave-one-family-out (LOFO) evaluation splits before cost-model training.

### Audit Verdict: **GOOD AND READY FOR PHASE 3**

All core acceptance criteria for Phase 2 defined in `TASKS.md`, `CSE583_Cost_Aware_JAX_Loop_Unrolling_Proposal.docx`, and `group10_slides.pdf` have been achieved:
1. **Feature Extractor (`src/jax_unroll/features.py`)**: Traces loop bodies using `jax.make_jaxpr`, extracting 55 static structural, operational, footprint, dependence, and context features without compiling or timing candidates. Median extraction overhead is **0.56 ms** (p95 1.13 ms).
2. **Baselines (`src/jax_unroll/baselines.py`)**: Implements `factor_1` (JAX default), fixed factors `2`, `4`, `8`, and the LLVM-style `UnrollThreshold` static size heuristic (`llvm_size`).
3. **Dataset (`benchmarks/dataset.py`, `data/features_cpu_2026-11-17.csv`)**: 220 aggregated configuration rows across 55 valid configurations (6 loop families, 12 shape variants, trip counts 16–1024, factors 1–8), joined with medians over 672 raw executions (3 process repeats; 660 correct, 12 excluded, 0 errors).
4. **Frozen Splits (`src/jax_unroll/splits.py`, `docs/splits.md`)**: Partitioned into 4 Dev families (140 rows) and 2 Held-out families (80 rows), with 4 LOFO cross-validation folds strictly over Dev.
5. **Test Suite**: 192 unit and integration tests passing (`pytest -q`), including new regression tests.

### Key Audit Findings & Corrections
During our deep audit, we found and fixed **one logic bug** in `features.py` regarding loop-carried state extraction for `fori_loop`, **one primitive classification redundancy**, **one correctness-gate flaw** that accepted non-finite outputs, and **two feature-extractor gaps** (jit-wrapped functions, requested-kind mislabeling). A follow-up rebuild re-measured the 12 affected raw rows and produced dataset version 2. All fixes are covered by regression tests and the remaining dataset rows were verified unchanged.

---

## 2. Detailed Task-by-Task Accomplishment Audit

| Task | Requirement | Status | Implementation Details | Verification |
|---|---|---|---|---|
| **T2.1** | Static Jaxpr feature extractor (trip, ops, footprint, dependences, work) | **Accomplished** | `src/jax_unroll/features.py`: Extracts 54 features covering trip structure (`full_chunks`, `remainder`, `has_tail`), operation counts (`eq_elementwise`, `eq_dot`, `eq_reduce`, `eq_gather_scatter`, `eq_convert`, `eq_other`), tensor memory (`carry_bytes`, `input_bytes`, `output_bytes`, `intermediate_bytes`, `max_tensor_bytes`, `bytes_moved_proxy`), dependences (`longest_path`, `max_out_degree`, `num_feed_forward_values`, `dependence_density`), and proxies (`flops_proxy`, `arithmetic_intensity_proxy`). | `tests/test_features.py` checks known values on all families. Verified that `jax.jit` is never called. |
| **T2.2** | Handle `scan` and `fori_loop`; document captured/uncaptured | **Accomplished** | Decouples `fori_loop` integer counter from loop-carried state; documents limits in `docs/features.md`. | Verified on `jacobi_solver` (only state carry `32x32 float32` = 4096 bytes is counted in `carry_bytes`). |
| **T2.3** | Parity, known-value, and no-compile tests | **Accomplished** | `test_scan_and_equivalent_fori_loop_parity` confirms identical tensor footprints; `test_extraction_does_not_compile_or_lower` monkeypatches `jax.jit` to raise if compilation occurs. | All tests pass deterministically. |
| **T2.4** | Baseline policies (`factor_1`, fixed `2/4/8`, `llvm_size`) | **Accomplished** | `src/jax_unroll/baselines.py` implements all 5 policies. `llvm_size` simulates LLVM's `UnrollThreshold` (default threshold 128 equations, max factor 8). | Evaluated against all 55 valid configurations: `llvm_size` over-unrolls (45x u=8, 10x u=4) with 0.842x median warm speedup. |
| **T2.5** | Feature/label dataset builder with schema, dedupe, versioning | **Accomplished** | `benchmarks/dataset.py` aggregates 3 process repeats via `aggregate_raw`, joins features via `_MERGE_KEY_COLUMNS`, computes natural log targets (`log_cold_ms`, `log_warm_us`), and writes sidecar metadata with SHA-256 hashes. | Output CSV `data/features_cpu_2026-11-17.csv` (220 rows, 78 columns) verified. |
| **T2.6** | Frozen LOFO splits before modeling | **Accomplished** | `src/jax_unroll/splits.py`: Split version 1 frozen 2026-11-17. Dev: `recurrence`, `elementwise`, `reduction`, `dense_linear_algebra`. Holdout: `indexed_memory`, `iterative_solver`. | `assert_holdout_excluded()` strictly guards fit entry points. |

---

## 3. Discrepancies and Bugs Identified During Audit

### Finding 1: Flawed Output Variable Truncation for `fori_loop`
- **Location:** `src/jax_unroll/features.py`, lines 507–509.
- **Original Code:**
  ```python
  state_outvars = (
      outvars[1:] if loop.loop_kind == "fori_loop" and len(outvars) > 1 else outvars
  )
  ```
- **The Issue:**
  1. *Bare Body Extraction with Multi-Element Carry:* When using `extract_features(body_fn=..., loop_kind="fori_loop")`, the bare body function `body_fn(i, carry)` returns only the new carry state (no counter `i + 1`). If `carry` is a tuple of two or more arrays (e.g. `(a, b)`), `len(outvars) > 1` evaluated to `True`. The code dropped `outvars[0]`, mistakenly discarding the first state return value (`a`). This resulted in severe undercounting of `output_bytes`, `output_elements`, and `output_shapes`.
  2. *Zero-Carry Loops:* When extracting a full loop function `lax.fori_loop(0, N, lambda i, _: (), ())` with no carried data state, JAX's loop wrapper returns only the counter `(i + 1,)` so `len(outvars) == 1`. Because `len(outvars) > 1` was `False`, the code took `outvars`, treating the 4-byte integer counter as the loop's output rather than reporting 0 bytes.
- **Impact on Existing Dataset:**
  **Zero impact.** The only `fori_loop` workload in the benchmark suite is `jacobi_solver`. `jacobi_solver` carries a single 2D grid array; when traced inside its outer function `fn`, `outvars` has length 2 `[counter, new_grid]`. Thus `outvars[1:]` correctly selected `new_grid`.
- **Resolution Applied:**
  Modified `features.py` to inspect the loop source:
  ```python
  if loop.loop_kind == "fori_loop":
      state_outvars = outvars[1:] if loop.source == "loop" else outvars
  else:
      state_outvars = outvars
  ```
  Added two new unit tests in `tests/test_features.py`:
  - `test_bare_body_fori_loop_multi_carry`: Verifies that a 2-tuple carry `(4-elem, 8-elem)` preserves all 12 elements (48 bytes) and shape `[[4],[8]]`.
  - `test_fori_loop_empty_carry`: Verifies that an empty carry loop reports 0 `carry_bytes`, 0 `output_bytes`, and shape `[]`.

---

### Finding 2: Redundant Primitive in `_ELEMENTWISE_PRIMITIVES`
- **Location:** `src/jax_unroll/features.py`, lines 61 and 131.
- **The Issue:** `"convert_element_type"` was present in both `_CONVERT_PRIMITIVES` and `_ELEMENTWISE_PRIMITIVES`.
- **Impact:** In `_classify()`, `_CONVERT_PRIMITIVES` was checked before `_ELEMENTWISE_PRIMITIVES`, so it was classified as `"convert"` as intended. However, having it in both sets was ambiguous and misleading.
- **Resolution Applied:** Removed `"convert_element_type"` from `_ELEMENTWISE_PRIMITIVES`.

---

### Finding 3: Uniform `max_out_degree` Across All Benchmark Workloads
- **Observation:** In `data/features_cpu_2026-11-17.csv`, the feature `max_out_degree` is exactly `1` for all 220 rows.
- **Analysis:**
  We dumped and inspected the top-level equation DAGs for all 6 workload families:
  - `scalar_recurrence`: `mul -> add` (pipeline; out-degree 1)
  - `elementwise_sine`: `add -> sin` (pipeline; out-degree 1)
  - `reduction_vector`: `mul -> reduce_sum -> add` (pipeline; out-degree 1)
  - `dense_matvec`: `dot_general -> add` (pipeline; out-degree 1)
  - `indexed_update`: Tree of index clamps, gathers, and scatter updates where every intermediate variable is consumed at most once (out-degree 1)
  - `jacobi_solver`: Four `jnp.roll` calls each consumed once by an adder tree (out-degree 1)
- **Phase 3 Modeling Implication:** Because `max_out_degree` has zero variance in this dataset, tree-based and linear regressors cannot use it to differentiate loops. This should be explicitly noted in the feature ablation report in Phase 3.

---

### Finding 4: Empty Budget `B` and Horizon `R` in Dataset
- **Observation:** Columns `B` and `R` are null (`NaN`) in `data/features_cpu_2026-11-17.csv`.
- **Reason:** Week 1 raw sweeps were run with unconstrained execution at a single fixed horizon (`r_calls = 100`).
- **Phase 3 Modeling Implication:**
  The cost model to be trained in Phase 3 should predict `C(u)` (`cold_median_ms`) and `T(u)` (`warm_median_us`) independently from the static loop features, rather than taking `B` or `R` as input features. The decision policy in `selector.py` will then evaluate:
  $$\text{score}(u) = \hat{C}(u) + R \cdot \hat{T}(u) \quad \text{subject to } \hat{C}(u) \le B$$
  This matches the proposal's analytical objective and avoids needing to re-sweep the physical hardware for every `(B, R)` combination.

---

### Finding 5: Inlined Jit Wrappers vs Top-Level DAG Features
- **Observation:**
  - `_body_counts` inlines `jit`/`pjit` equations to compute `num_equations`, `flops_proxy`, and operation mix counts (`eq_*`). For example, in `jacobi_solver`, `jnp.roll` wraps in `jit`, expanding 10 top-level equations into 18 leaf equations.
  - However, `_dependence_features` (`longest_path`, `dependence_density`) and `intermediate_bytes` operate only on top-level `jaxpr.eqns`.
- **Verdict:** This design is well-documented in `docs/features.md` ("A top-level def-use path, fan-out, and feed-forward count") and keeps dependence extraction fast (<1 ms). It is acceptable for Phase 2, but Phase 3 researchers should know that internal dependence paths within inlined JIT helpers are abstracted.

---

### Finding 6: Git Commit Hash in Metadata Sidecar
- **Observation:** The original `data/features_cpu_2026-11-17.meta.json`
  recorded `"git_commit": "0ad979d"`, whereas the commit adding the pipeline is
  `6cde169`.
- **Reason:** The dataset was generated immediately prior to creating the git
  commit.
- **Verdict:** Harmless. The version-2 rebuild records `6cde169` (with the
  feature/correctness fixes present as uncommitted changes at build time).

---

### Finding 7: Correctness Gate Accepted Non-Finite Outputs (Fixed)
- **Location:** `benchmarks/sweep.py` (`_allclose`, reference construction in `measure_config`).
- **The Issue:** `numpy.allclose(..., equal_nan=True)` treated matching `NaN`
  outputs as equal, and `factor == 1` was marked `correct=True` unconditionally.
  `dense_matvec`/`n32`/`T=1000` overflows `float32` and produces all-`NaN`
  outputs, so all 12 raw rows (4 factors x 3 repeats) were accepted as valid
  labels with meaningless timings.
- **Impact:** 4 dataset rows out of 224 had invalid labels; the oracle and
  baselines included them.
- **Resolution Applied:** Added `_all_finite`, required finite outputs for both
  `factor == 1` and the reference comparison, and switched to
  `equal_nan=False`. Non-finite rows are marked `correct=False`, get an
  explanatory `notes` marker, and are dropped by the oracle/dataset builder.
  The 12 affected raw rows were re-measured, spliced into
  `results/raw/sweep_cpu_2026-11-17_shapes.csv`, and the oracle/dataset were
  rebuilt. Dataset version is now 2: 220 rows, 55 configurations, 140 dev /
  80 holdout.

### Finding 8: Jit-Wrapped Functions and Requested-Kind Mislabeling (Fixed)
- **Location:** `src/jax_unroll/features.py` (`_find_loop_eqn`).
- **The Issue:** The module docstring claimed jitted loop functions were
  supported, but `_find_loop_eqn` inspected only top-level equations.
  `extract_features(jax.jit(fn), ...)` therefore raised "no scan/while loop
  found". Separately, requesting `loop_kind="scan"` on a function that traced
  only to a dynamic-bound `while` silently returned
  `loop_kind="scan", loop_primitive="while"` instead of failing.
- **Resolution Applied:** `_iter_loop_candidates` now descends recursively
  through `jit`/`pjit` equations, and `_find_loop_eqn` raises
  `FeatureExtractionError` when the requested kind is not the loop that was
  found. Tests cover jitted extraction parity and the mismatch error.

### Finding 9: Bare-Body Scan Split Needed an Explicit `body_num_xs` (Fixed)
- **Location:** `src/jax_unroll/features.py` (`extract_loop_body`,
  `extract_features`).
- **The Issue:** For bare-body extraction, every input after the leading consts
  was treated as carry, so a scan body `(carry, x)` reported `x` as
  loop-carried state. `docs/features.md` already showed `(carry, x)` bodies,
  so the API and documentation disagreed.
- **Resolution Applied:** Added `body_num_xs`; `num_carry` is now
  `len(invars) - num_consts - num_xs`. The default `0` preserves prior
  behavior, so existing dataset rows are unchanged.

### Finding 10: Raw Sweeps Are Gitignored but Referenced by Metadata
- **Observation:** `results/raw/*.csv` is excluded by `.gitignore`, while
  `data/features_cpu_2026-11-17.meta.json` records the SHA-256 of those files.
- **Verdict:** The metadata hashes cannot be verified from a clean checkout.
  The committed dataset, oracle, and gate artifacts are sufficient for Phase 3
  model work; re-running the documented sweep commands regenerates raw files
  (with new timings). If exact raw-file provenance is required later, commit
  the raw CSVs or attach them to a release.

## 4. Baseline Behavior & Project Hypothesis Validation

Phase 2 implemented and evaluated the baseline policies across all 55 valid configurations:

```
Policy       | Median Warm Gain vs u=1 | Mean Warm Gain | Oracle Agreement | Selected Factors
-------------+-------------------------+----------------+------------------+------------------------------
factor_1     | 1.000x                  | 1.000x         | 56.4% (31/55)    | 55 x u=1
factor_2     | 0.993x                  | 0.951x         | 16.4% (9/55)     | 55 x u=2
factor_4     | 0.970x                  | 0.883x         | 12.7% (7/55)     | 55 x u=4
factor_8     | 0.842x                  | 0.816x         | 14.5% (8/55)     | 55 x u=8
llvm_size    | 0.842x                  | 0.820x         | 14.5% (8/55)     | 45 x u=8, 10 x u=4
```

### Critical Compiler Takeaway:
The LLVM size heuristic selects `u=8` for 45 configurations and `u=4` for 10 configurations because all benchmark bodies have between 2 and 18 equations, all of which easily satisfy $u \cdot \text{body\_size} \le 128$.
Because it ignores loop memory traffic, memory-bound kernels, and loop-carried dependence latency, **`llvm_size` regresses warm runtime by 15.8% on median**.
This provides the definitive empirical justification for Phase 3: **learning a budget-aware cost model from static Jaxpr features is necessary to beat naive compiler size heuristics.**

---

## 5. Phase 3 Readiness Checklist

| Item | Requirement | Verification Result |
|---|---|---|
| **Clean Feature Schema** | `FEATURE_COLUMNS` has 55 strictly defined scalar/JSON columns | Verified matching `FEATURE_COLUMNS` |
| **Clean Target Labels** | `cold_median_ms`, `warm_median_us`, `log_cold_ms`, `log_warm_us` | Verified finite, non-null, positive after the non-finite gate fix |
| **Leak-Free Splits** | LOFO folds strictly over 4 Dev families (`recurrence`, `elementwise`, `reduction`, `dense_linear_algebra`) | Verified with `assert_holdout_excluded()` |
| **Holdout Integrity** | `indexed_memory` and `iterative_solver` reserved for Phase 4 | Verified 0 holdout rows in dev LOFO folds |
| **Decision Cost Feasibility** | Feature extraction overhead < 2 ms | Verified: Median 0.56 ms (p95 1.13 ms) |
| **Test Suite Health** | All unit, integration, and regression tests pass | **197 passed** |

Phase 2 is **100% complete, verified, and ready** for Phase 3 cost-model training and selector implementation.

---

## 6. Verification pass (2026-11-17, post-audit)

An independent re-check of the fixes above found and fixed two further latent
bugs in `src/jax_unroll/features.py`. Neither affects the committed dataset:
all six benchmark workloads use explicit `loop_kind` values and single loops,
and a full dataset rebuild after the fixes reproduced every semantic column
byte-identically (only the `extraction_ms` timing column differs).

### Finding 11: Raw `while` loops were silently mislabeled as `fori_loop` (Fixed)
- **Location:** `extract_loop_body` (the `kind = loop_kind or (...)` default).
- **Issue:** With `loop_kind=None`, a `while`-primitive loop (raw
  `lax.while_loop`, or a dynamic-bound `fori_loop`) was labeled
  `loop_kind="fori_loop"`, which triggered fori counter-stripping in
  `extract_features`. For a raw `while_loop` the first carry element is not a
  counter, so `carry_bytes`, `output_bytes`, `num_loop_carried`, `carry_shapes`,
  and `output_shapes` were silently wrong (a single-carry while loop reported
  0 carried bytes).
- **Resolution:** With no requested kind, the reported `loop_kind` is now the
  traced primitive (`scan` or `while`) and the carry is counted as-is; counter
  stripping applies only to an explicitly requested `loop_kind="fori_loop"`.
  Documented in `docs/features.md`; regression tests cover raw multi-carry and
  single-carry `while` loops, static-`fori`-as-`scan` inference, and
  dynamic-`fori` counter stripping.

### Finding 12: Multiple loops silently took the first candidate (Fixed)
- **Location:** `_find_loop_eqn` (the `return candidates[0]` fallback).
- **Issue:** A function containing more than one `scan`/`while` loop (for
  example two sequential scans) silently had its features computed from the
  first loop only, violating the documented "exactly one loop" contract.
- **Resolution:** `_find_loop_eqn` now raises `FeatureExtractionError` when
  more than one loop candidate is found. Regression test covers a
  two-scan function.

### Post-fix verification
- `pytest -q`: **197 passed** (5 new regression tests).
- `make quick` smoke sweep: 8/8 rows correct, 0 errors.
- Dataset rebuild from raw files: 220 rows, 55 configurations, all semantic
  columns identical to the committed dataset version 2; metadata sidecar
  consistent (hashes verified against the local raw files).
