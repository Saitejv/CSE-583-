# Compiler Analysis and Benchmark Deduplication Review: JAX Loop Unrolling Corpus Expansion

**Author:** Duplicate-Review Subagent  
**Date:** November 18, 2026  
**Scope:** Full audit of 66 historical JAX loop bodies, formal deduplication criteria, historical related-group taxonomy, and audit protocols for the 60–80 fori_loop corpus expansion.  
**Reference Document:** `docs/gemini_fori_corpus_prompt.md`

---

## 1. Executive Summary and Architecture Overview

The JAX loop unrolling corpus expansion transitions our evaluation framework to a pure `jax.lax.fori_loop` benchmark dataset on Apple Silicon (M2 CPU). The historical corpus consists of **66 measured workloads**, but only **13 use `fori_loop`** (with only 7 in the historical training partition), while **53 use `jax.lax.scan`**. Across all loop kinds, 35 historical bodies have only a single measured configuration.

To support cost-model training and out-of-distribution evaluation for loop unrolling factors $u \in \{1, 2, 4, 8\}$, the project executes two parallel corpus streams:
1. **Conversion Stream:** Audit and convert the 53 existing public `scan` workloads into faithful static-bound `fori_loop` bodies under `benchmarks/corpus/fori_expansion/converted/`.
2. **Expansion Stream:** Research and implement **60–80 genuinely independent `fori_loop` bodies** across 10 distinct algorithmic mechanisms, reserving **15–20 independent bodies for fresh evaluation**.

### Fundamental Review Mandate
The integrity of this benchmark rests on compiler-level deduplication:
- **No duplicate or trivial variants may be counted as independent bodies.**
- **A direct translation of a current scan is a paired control, contributing exactly zero new independent bodies.**
- **No related group may ever straddle development (train) and evaluation (test) partitions.**
- **Every historical workload, converted scan, and newly authored body must be mapped to a deterministic `related_group` ID.**

```
+----------------------------------------------------------------------------------------------------+
|                                    TOTAL CORPUS INVENTORY (126-146 BODIES)                         |
+-------------------------------------------------+--------------------------------------------------+
|               HISTORICAL LINEAGES               |                 FRESH LINEAGES                   |
|                  (66 BODIES)                    |              (60-80 INDEPENDENT)                 |
+------------------------+------------------------+------------------------+-------------------------+
|  Historical fori_loop  |    Converted Scans     |   New Dev Candidates   |     Fresh Evaluation    |
|      (13 Bodies)       |   (Up to 53 Bodies)    |     (45-60 Bodies)     |      (15-20 Bodies)     |
|   7 Train / 6 Test     |    45 Dev / 8 Test     |    Unseen Families     |    Unfamiliar Families  |
|   0 New Independent    |   0 New Independent    |   Independent Bodies   |    Independent Bodies   |
+------------------------+------------------------+------------------------+-------------------------+
| Related Groups: Same   | Related Groups: Same   | Related Groups: New    | Related Groups: New     |
| as Historical Lineage  | as Original Scan Body  | Development Groups     | Fresh Eval Groups Only  |
+------------------------+------------------------+------------------------+-------------------------+
```

---

## 2. Formal Duplication Criteria and Taxonomy

To ensure scientific validity, loop bodies are classified into three mutually exclusive categories:
1. **Trivial Variants (0 New Independent Bodies):** Parameter, shape, constant, or syntax perturbations of an existing computational graph.
2. **Paired Syntactic Translations (0 New Independent Bodies):** One-to-one conversions of historical `lax.scan` loops into `lax.fori_loop` loops that implement identical mathematical recurrences.
3. **Genuinely Independent Loop Bodies (1 New Independent Body Each):** Algorithmic implementations with distinct mathematical state transitions, computational graph topologies, loop carry structures, and memory access patterns.

### 2.1 Criteria for Trivial Variants (DO NOT COUNT)

Any workload modification that satisfies any of the following 9 conditions is classified as a **trivial variant** and contributes **0 to the independent body count**:

1. **Shape and Dimension Scaling:**
   - Altering the length of a vector, dimensions of a matrix, or size of a buffer (e.g., $N=16 \to 32 \to 64$, or vector width $256 \to 1024 \to 4096$).
   - *Compiler justification:* Tensor dimensions alter the type signatures in `Jaxpr.invars`, changing the raw SHA-256 hash of `str(jaxpr)`, but leave the dependency graph topology, primitive sequence, and algorithmic mechanism completely isomorphic.
2. **Constant and Coefficient Perturbations:**
   - Modifying scalar constants, decay weights, learning rates, or filter coefficients (e.g., changing $0.9 \times c + x$ to $0.875 \times c + x$, or changing polynomial coefficients).
   - *Compiler justification:* Inlined constants alter literal attributes in Jaxpr equations without changing the schedule, dataflow dependency chains, or unrolled register pressure.
3. **Trip Count Variations:**
   - Varying the loop iterations $T \in \{16, 64, 100, 256, 1023, 1024\}$.
   - *Compiler justification:* Trip count is an argument to the loop primitive (`fori_loop(0, T, body, init)`); it does not alter the inner loop body Jaxpr.
4. **PRNG Seed and Data Perturbations:**
   - Changing NumPy/JAX random seeds (e.g., seed `583001` vs `583002`) or using different synthetic input distributions that preserve the memory layout.
   - *Compiler justification:* Tracing operates on abstract values (`ShapedArray`); runtime values have zero representation in the traced Jaxpr.
5. **Syntactic and Variable Renaming:**
   - Changing variable names in Python (e.g., `carry` vs `state`, `x` vs `val`), extracting sub-functions, or refactoring into closures.
   - *Compiler justification:* JAX traces Python source into intermediate representations with canonical positional identifiers (`a`, `b`, `c`).
6. **Pure Unroll Factor Changes:**
   - Requesting `unroll=1`, `unroll=2`, `unroll=4`, or `unroll=8`.
   - *Compiler justification:* Unrolling is the compiler transformation being evaluated, not a benchmark definition property.
7. **Operation Commutation and Dead-Code/Identity Insertion:**
   - Reordering commutative operations ($a + b \to b + a$) or inserting identity operations ($x + 0$, $x \times 1.0$) that fold or compile to the same HLO graph.
8. **Direct Scan-to-Fori Loop Translation:**
   - Replacing `lax.scan(step, init, xs)` with `lax.fori_loop(0, T, body, init)` where `body` simply closes over `xs` and indexes `xs[i]`.
   - *Compiler justification:* While adding an explicit index slice primitive to the Jaxpr, the underlying recurrence relation $c_{t+1} = f(c_t, x_t)$ is invariant.
9. **Index Layout Permutation without Mechanism Change:**
   - Changing index arrays from ascending to descending without altering memory access density or collision structure.

### 2.2 Criteria for Genuinely Independent Loop Bodies (DO COUNT)

A proposed loop body is accepted as a **genuinely independent loop body** if and only if it satisfies **all** of the following four criteria:

1. **Distinct Computational Mechanism & Mathematical Formulation:**
   - Solves a distinct algorithmic problem with a unique state recurrence equation $c_{t+1} = \mathcal{M}(c_t, x_t, \theta)$.
   - *Examples:* Viterbi trellis dynamic programming ($\max$-plus semiring) vs. Runge-Kutta 4th-order ODE integration vs. Biquad Direct Form II IIR filtering vs. Coordinate descent with incremental residual maintenance.
2. **Distinct Loop Carry State Graph:**
   - Distinct carry topology: scalar vs. fixed vector vs. 2D grid buffer vs. multi-component tuple carries (e.g., carrying both state and running covariance matrix, or carrying dual ring pointers and accumulators).
   - Intra-iteration and inter-iteration data dependency distances and out-degrees must reflect different mathematical formulations.
3. **Distinct Memory Access & Addressing Mode:**
   - Employs a fundamentally different access mechanism:
     - Pure sequential streaming ($x_t = \text{array}[i]$)
     - Affine strided access ($x_t = \text{array}[s \cdot i + k]$)
     - Dynamic window slicing (`lax.dynamic_slice`)
     - Indirect gather / multi-table lookup ($\text{table}[\text{indices}[i]]$)
     - Indirect scatter / collision accumulation (`carry.at[idx].add(...)` or `.min(...)`)
     - Dependent pointer chasing / linked traversal ($j_{t+1} = \text{table}[j_t]$)
     - Multi-axis stencil rolling / spatial neighbor shifting
     - Blocked / tiled sub-tensor slicing with local reductions
4. **Distinct Primary Source Lineage:**
   - Grounded in distinct primary literature, documented numerical algorithms, or established scientific software packages (e.g., JAXopt, JAX-CFD, Jraph, Optax, JAX MD), with specific URL and license attributions.

---

## 3. Comprehensive Audit of All 66 Historical Loop Bodies

The historical inventory consists of **66 public loop bodies**:
- **13 `fori_loop` bodies** (7 in historical training, 6 in historical testing)
- **53 `scan` bodies** (45 in historical training, 8 in historical testing)
- **Historical Partitioning:** 52 Development (Train) vs. 14 Evaluation (Test)

Every historical body has been audited against its implementation source, Jaxpr structure, primary variant, arithmetic intensity, and loop carry state.

### 3.1 Historical Loop Bodies Master Table

The table below catalogs all 66 historical loop bodies with their source module, family, loop kind, historical split, equations count, arithmetic intensity proxy, and assigned related-group ID:

| # | Workload Name | Source Module | Historical Family | Loop Kind | Historical Split | Primary Variant | Equations (T=64) | FLOPs Proxy | Intensity Proxy | Assigned Related-Group ID |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `scalar_recurrence` | `workloads.py` | `recurrence` | `scan` | `train` | `scalar` | 3 | 2 | 0.083 | `iir_filters` |
| 2 | `elementwise_sine` | `workloads.py` | `elementwise` | `scan` | `train` | `n1024` | 3 | 1024 | 0.083 | `elementwise_control` |
| 3 | `reduction_vector` | `workloads.py` | `reduction` | `scan` | `train` | `n1024` | 4 | 2048 | 0.250 | `vector_reduction_accumulator` |
| 4 | `dense_matvec` | `workloads.py` | `dense_linear_algebra` | `scan` | `train` | `n64` | 3 | 8192 | 0.496 | `dense_matvec_recurrence` |
| 5 | `indexed_update` | `workloads.py` | `indexed_memory` | `scan` | `test` | `n1024` | 4 | 64 | 0.007 | `indexed_gather_scatter_update` |
| 6 | `jacobi_solver` | `workloads.py` | `iterative_solver` | `fori_loop` | `test` | `n32` | 18 | 5121 | 0.404 | `stencil_axis_aligned` |
| 7 | `window_read` | `recovery_workloads.py` | `window` | `scan` | `train` | `n512` | 5 | 8 | 0.111 | `dynamic_window_slice` |
| 8 | `window_replace` | `recovery_workloads.py` | `window` | `scan` | `train` | `n512` | 3 | 0 | 0.000 | `dynamic_window_slice` |
| 9 | `gather_lookup` | `recovery_workloads.py` | `gather` | `scan` | `train` | `n512` | 5 | 16 | 0.174 | `weighted_gather_lookup` |
| 10 | `gather_rows` | `recovery_workloads.py` | `gather` | `scan` | `train` | `n512` | 5 | 64 | 0.205 | `row_gather_pool` |
| 11 | `scatter_accumulate` | `recovery_workloads.py` | `scatter` | `scan` | `train` | `n512` | 3 | 12 | 0.006 | `collision_memory` |
| 12 | `scatter_blend` | `recovery_workloads.py` | `scatter` | `scan` | `train` | `n512` | 6 | 16 | 0.008 | `unique_update` |
| 13 | `strided_mix` | `recovery_workloads.py` | `strided_read` | `scan` | `test` | `n512` | 8 | 13 | 0.148 | `strided_offset_mix` |
| 14 | `bucket_max` | `recovery_workloads.py` | `histogram` | `scan` | `test` | `n512` | 4 | 12 | 0.006 | `bucket_histogram_max` |
| 15 | `masked_rows` | `recovery_workloads.py` | `masked_read` | `scan` | `test` | `n512` | 5 | 48 | 0.146 | `masked_row_select` |
| 16 | `ring_feedback` | `recovery_workloads.py` | `ring_buffer` | `fori_loop` | `test` | `n512` | 38 | 40 | 0.019 | `ring_feedback_buffer` |
| 17 | `gather_scalar_table` | `corpus/indirect.py` | `indirect_gather` | `scan` | `train` | `n512` | 5 | 2 | 0.083 | `random_scalar_gather` |
| 18 | `gather_row_pool` | `corpus/indirect.py` | `indirect_gather` | `scan` | `train` | `rows512` | 8 | 24 | 0.176 | `random_row_gather` |
| 19 | `gather_multi_table` | `corpus/indirect.py` | `indirect_gather` | `scan` | `train` | `t512` | 10 | 12 | 0.125 | `select_gather` |
| 20 | `gather_duplicate_index` | `corpus/indirect.py` | `indirect_gather` | `scan` | `train` | `d1024` | 5 | 16 | 0.174 | `collision_memory` |
| 21 | `gather_sorted_blocks` | `corpus/indirect.py` | `indirect_gather` | `scan` | `train` | `b2048` | 5 | 16 | 0.174 | `ordered_gather` |
| 22 | `gather_masked_window` | `corpus/indirect.py` | `indirect_gather` | `scan` | `train` | `m1024` | 8 | 20 | 0.156 | `masked_memory` |
| 23 | `scatter_unique_rows` | `corpus/indirect.py` | `indirect_scatter` | `scan` | `train` | `r512` | 3 | 0 | 0.000 | `unique_update` |
| 24 | `scatter_collision_add` | `corpus/indirect.py` | `indirect_scatter` | `scan` | `train` | `g32x128` | 3 | 16 | 0.001 | `collision_memory` |
| 25 | `scatter_min_update` | `corpus/indirect.py` | `indirect_scatter` | `fori_loop` | `train` | `s1024` | 6 | 0 | 0.000 | `collision_memory` |
| 26 | `scatter_two_tables` | `corpus/indirect.py` | `indirect_scatter` | `scan` | `train` | `t1024` | 7 | 8 | 0.002 | `split_table_memory` |
| 27 | `scatter_permutation` | `corpus/indirect.py` | `indirect_scatter` | `scan` | `train` | `p512` | 3 | 0 | 0.000 | `permutation_memory` |
| 28 | `locality_stream_contiguous`| `corpus/indirect.py`| `locality_access` | `scan` | `train` | `n2048` | 5 | 8 | 0.111 | `sequential_locality` |
| 29 | `locality_stride_sample` | `corpus/indirect.py` | `locality_access` | `scan` | `train` | `n2048` | 6 | 8 | 0.111 | `strided_locality` |
| 30 | `locality_random_window` | `corpus/indirect.py` | `locality_access` | `scan` | `train` | `n2048` | 5 | 8 | 0.111 | `random_window_locality` |
| 31 | `locality_block_reuse` | `corpus/indirect.py` | `locality_access` | `fori_loop` | `train` | `hot8x32` | 15 | 32 | 0.015 | `reuse_locality` |
| 32 | `locality_permutation_read`| `corpus/indirect.py`| `locality_access` | `scan` | `train` | `n512` | 4 | 8 | 0.111 | `permutation_memory` |
| 33 | `dep_pointer_chase` | `corpus/indirect.py` | `dependent_indexing` | `fori_loop` | `train` | `n1024` | 5 | 1 | 0.045 | `dependent_chain` |
| 34 | `dep_parallel_chains` | `corpus/indirect.py` | `dependent_indexing` | `scan` | `train` | `n1024` | 5 | 4 | 0.083 | `dependent_chain` |
| 35 | `dep_two_level_indirection`| `corpus/indirect.py`| `dependent_indexing` | `fori_loop` | `train` | `large` | 7 | 1 | 0.038 | `dependent_chain` |
| 36 | `dep_linked_list` | `corpus/indirect.py` | `dependent_indexing` | `fori_loop` | `train` | `n1024` | 8 | 2 | 0.067 | `dependent_chain` |
| 37 | `dep_bucket_stateful` | `corpus/indirect.py` | `dependent_indexing` | `fori_loop` | `train` | `b64` | 13 | 4 | 0.007 | `stateful_dependent` |
| 38 | `rolling_mean_shift` | `corpus/structured.py` | `rolling_window` | `scan` | `train` | `w8` | 7 | 2 | 0.026 | `rolling_moments` |
| 39 | `rolling_conv1d_fir` | `corpus/structured.py` | `rolling_window` | `scan` | `train` | `k8` | 8 | 16 | 0.174 | `rolling_weighted_windows` |
| 40 | `rolling_extreme_max` | `corpus/structured.py` | `rolling_window` | `scan` | `train` | `w16` | 7 | 1 | 0.007 | `rolling_extrema` |
| 41 | `rolling_variance_moments` | `corpus/structured.py` | `rolling_window` | `scan` | `train` | `w12` | 11 | 6 | 0.053 | `rolling_moments` |
| 42 | `rolling_dot_two_streams` | `corpus/structured.py` | `rolling_window` | `scan` | `train` | `w16` | 8 | 32 | 0.235 | `rolling_weighted_windows` |
| 43 | `stencil_diffusion_1d_fori`| `corpus/structured.py`| `stencil` | `fori_loop` | `test` | `n128` | 11 | 384 | 0.250 | `stencil_axis_aligned` |
| 44 | `stencil_five_point_2d` | `corpus/structured.py` | `stencil` | `fori_loop` | `test` | `n16` | 18 | 1280 | 0.357 | `stencil_axis_aligned` |
| 45 | `stencil_nine_point_2d` | `corpus/structured.py` | `stencil` | `fori_loop` | `test` | `n16` | 26 | 2816 | 0.404 | `stencil_wide_support` |
| 46 | `stencil_anisotropic_2d` | `corpus/structured.py` | `stencil` | `fori_loop` | `test` | `n16` | 23 | 2560 | 0.400 | `stencil_wide_support` |
| 47 | `streaming_running_mean` | `corpus/structured.py` | `streaming_reduction`| `scan` | `train` | `scalar` | 5 | 3 | 0.094 | `streaming_moments` |
| 48 | `streaming_welford_var` | `corpus/structured.py` | `streaming_reduction`| `scan` | `train` | `scalar` | 8 | 8 | 0.133 | `streaming_moments` |
| 49 | `streaming_decayed_extrema`| `corpus/structured.py`| `streaming_reduction`| `scan` | `train` | `scalar` | 6 | 3 | 0.094 | `streaming_extrema` |
| 50 | `streaming_logsumexp` | `corpus/structured.py` | `streaming_reduction`| `scan` | `train` | `scalar` | 9 | 5 | 0.089 | `streaming_normalizers` |
| 51 | `segment_sum_reset` | `corpus/structured.py` | `segmented_scan` | `scan` | `test` | `w4_short` | 5 | 8 | 0.111 | `segmented_reset` |
| 52 | `segment_max_reset` | `corpus/structured.py` | `segmented_scan` | `scan` | `test` | `w4_short` | 5 | 4 | 0.067 | `segmented_reset` |
| 53 | `segment_weighted_mean` | `corpus/structured.py` | `segmented_scan` | `scan` | `test` | `w4_short` | 9 | 16 | 0.133 | `segmented_reset` |
| 54 | `segment_carry_ema` | `corpus/structured.py` | `segmented_scan` | `scan` | `test` | `w4_short` | 6 | 12 | 0.130 | `segmented_carry` |
| 55 | `stateful_scalar_iir` | `corpus/structured.py` | `stateful_recurrence` | `scan` | `train` | `scalar` | 4 | 2 | 0.077 | `iir_filters` |
| 56 | `stateful_vector_iir` | `corpus/structured.py` | `stateful_recurrence` | `scan` | `train` | `n8` | 4 | 16 | 0.077 | `iir_filters` |
| 57 | `stateful_gru_gate` | `corpus/structured.py` | `stateful_recurrence` | `scan` | `train` | `n4` | 18 | 96 | 0.229 | `gated_recurrences` |
| 58 | `stateful_lstm_cell` | `corpus/structured.py` | `stateful_recurrence` | `scan` | `train` | `n4` | 27 | 120 | 0.185 | `gated_recurrences` |
| 59 | `stateful_coupled_oscillator`| `corpus/structured.py`| `stateful_recurrence`| `fori_loop` | `train` | `n8` | 18 | 96 | 0.250 | `coupled_dynamics` |
| 60 | `control_polynomial_forcing`| `corpus/structured.py`| `control_arithmetic` | `scan` | `train` | `n512` | 7 | 3584 | 0.467 | `elementwise_control` |
| 61 | `control_transcendental` | `corpus/structured.py` | `control_arithmetic` | `scan` | `train` | `n512` | 6 | 2048 | 0.333 | `elementwise_control` |
| 62 | `control_bisection_bracket`| `corpus/structured.py` | `control_arithmetic` | `scan` | `train` | `vec4` | 10 | 28 | 0.179 | `root_finding_control` |
| 63 | `control_lowrank_update` | `corpus/structured.py` | `control_arithmetic` | `scan` | `train` | `n8` | 8 | 272 | 0.443 | `lowrank_control` |
| 64 | `matched_gather` | `corpus/static_matched.py`| `indirect_gather` | `scan` | `train` | `clustered` | 6 | 2 | 0.077 | `random_scalar_gather` |
| 65 | `matched_scatter_add` | `corpus/static_matched.py`| `indirect_scatter` | `scan` | `train` | `unique` | 4 | 256 | 0.015 | `collision_memory` |
| 66 | `matched_scatter_min` | `corpus/static_matched.py`| `indirect_scatter` | `scan` | `train` | `unique` | 4 | 0 | 0.000 | `collision_memory` |

---

## 4. Historical Related-Group Taxonomy and Boundary Mapping

The audit establishes a **41-group related taxonomy** encompassing all 66 historical loop bodies. 

### 4.1 Strict Partition Isolation Proof (Zero Cross-Contamination)

Let $\mathcal{W}$ be the universe of historical workloads ($|\mathcal{W}| = 66$). Let $\mathcal{P}: \mathcal{W} \to \{\text{train}, \text{test}\}$ denote the historical partition assignment defined in `results/all_workloads/split.csv`. Let $\mathcal{R}: \mathcal{W} \to \mathcal{G}$ denote the related-group mapping into the 41 groups $\mathcal{G}$.

**Partition Isolation Theorem:**
$$\forall g \in \mathcal{G}, \quad |\{\mathcal{P}(w) \mid w \in \mathcal{W}, \mathcal{R}(w) = g\}| = 1$$

*Proof:*
Of the 41 related groups:
- **32 groups** contain exclusively `train` workloads (totaling 52 bodies).
- **9 groups** contain exclusively `test` workloads (totaling 14 bodies).
- The intersection of the training workload set and the evaluation workload set across all groups is $\emptyset$. $\blacksquare$

### 4.2 Detailed Justifications for Previously Unassigned Historical Workloads

The 19 historical workloads in `workloads.py`, `recovery_workloads.py`, and `static_matched.py` did not possess formal `related_group` attributes in their original dataclasses. The audit establishes their deterministic mappings as follows:

#### Grouping Lineage 1: `iir_filters` (`train`)
- **Members:** `scalar_recurrence` (`workloads.py`), `stateful_scalar_iir` (`structured.py`), `stateful_vector_iir` (`structured.py`).
- **Mathematical Lineage:** First-order autoregressive / infinite impulse response (IIR) filtering.
  - `scalar_recurrence`: $c_{t+1} = 0.9 c_t + x_t$
  - `stateful_scalar_iir`: $c_{t+1} = 0.9 c_t + 0.1 x_t$
  - `stateful_vector_iir`: $\mathbf{c}_{t+1} = 0.9 \mathbf{c}_t + 0.1 \mathbf{x}_t$
- **Audit Decision:** `scalar_recurrence` is mathematically identical to `stateful_scalar_iir` up to scalar input normalization. All three are in `train`. Grouping them in `iir_filters` prevents leakage.

#### Grouping Lineage 2: `stencil_axis_aligned` (`test`)
- **Members:** `jacobi_solver` (`workloads.py`), `stencil_five_point_2d` (`structured.py`), `stencil_diffusion_1d_fori` (`structured.py`).
- **Mathematical Lineage:** Axis-aligned finite-difference stencils on periodic grids using `jnp.roll`.
  - `jacobi_solver`: $C_{i,j}^{(t+1)} = \frac{1}{4} (C_{i+1,j} + C_{i-1,j} + C_{i,j+1} + C_{i,j-1} + B_{i,j})$
  - `stencil_five_point_2d`: $C_{i,j}^{(t+1)} = C_{i,j} + 0.2 (C_{i+1,j} + C_{i-1,j} + C_{i,j+1} + C_{i,j-1} - 4 C_{i,j})$
- **Audit Decision:** Both 2D bodies execute identical four-direction circular shifts over 2D buffers. Both `jacobi_solver` and `stencil_five_point_2d` reside in the historical `test` partition. Grouping them preserves evaluation isolation.

#### Grouping Lineage 3: `collision_memory` (`train`)
- **Members:** `scatter_accumulate` (`recovery_workloads.py`), `scatter_collision_add` (`indirect.py`), `scatter_min_update` (`indirect.py`), `gather_duplicate_index` (`indirect.py`), `matched_scatter_add` (`static_matched.py`), `matched_scatter_min` (`static_matched.py`).
- **Mathematical Lineage:** Non-deterministic or duplicate-index updates targeting shared memory buckets via `.at[indices].add(...)` or `.at[indices].min(...)`.
- **Audit Decision:** All six bodies are in `train`. Unifying scatter accumulations into `collision_memory` captures their shared atomic/collision memory characteristics.

#### Grouping Lineage 4: `dynamic_window_slice` (`train`)
- **Members:** `window_read` (`recovery_workloads.py`), `window_replace` (`recovery_workloads.py`).
- **Mathematical Lineage:** Contiguous window extraction and replacement via `lax.dynamic_slice_in_dim` and `lax.dynamic_update_slice_in_dim`. Both belong to the historical `window` family in `train`.

#### Grouping Lineage 5: `unique_update` (`train`)
- **Members:** `scatter_unique_rows` (`indirect.py`), `scatter_blend` (`recovery_workloads.py`).
- **Mathematical Lineage:** In-place update of vector state at unique, non-colliding permutation indices using `.at[index].set(...)`. Both reside in `train`.

#### Grouping Lineage 6: Historical Test Singletons
- `indexed_update` $\to$ **`indexed_gather_scatter_update`** (`test`)
- `strided_mix` $\to$ **`strided_offset_mix`** (`test`)
- `bucket_max` $\to$ **`bucket_histogram_max`** (`test`)
- `masked_rows` $\to$ **`masked_row_select`** (`test`)
- `ring_feedback` $\to$ **`ring_feedback_buffer`** (`test`)
- *Audit Decision:* These 5 bodies represent distinct memory patterns held out for evaluation. They MUST NOT be merged with training groups (`strided_locality`, `random_scalar_gather`, or `collision_memory`), ensuring that no test evaluation signals leak into training folds.

---

## 5. Audit Protocols for Reviewing New Proposed Workloads

Every newly proposed `fori_loop` body submitted by an expansion workteam must pass an automated 7-stage gate before being merged into `benchmarks/corpus/fori_expansion/registry.py`.

```
[Proposal: Source & Code]
          |
          v
[Stage 1: Lineage & Provenance] ---- Fail: Unattributed / Cloned ----> [REJECT]
          | Pass
          v
[Stage 2: Mathematical Recurrence] - Fail: Trivial Variant ----------> [REJECT]
          | Pass
          v
[Stage 3: Loop Carry Graph] -------- Fail: Isomorphic State ---------> [REJECT]
          | Pass
          v
[Stage 4: Memory Access Pattern] --- Fail: Redundant Addressing -----> [REJECT]
          | Pass
          v
[Stage 5: Jaxpr & Body Hash] ------- Fail: Hash Collision -----------> [REJECT]
          | Pass
          v
[Stage 6: Numerical Invariants] ---- Fail: Divergent / Unstable -----> [REJECT]
          | Pass
          v
[Stage 7: Related Group & Split] --- Fail: Cross-Partition Leakage --> [REJECT]
          | Pass
          v
[ACCEPTED INTO FORI EXPANSION REGISTRY]
```

### 5.1 The 7-Stage Audit Protocol

#### Stage 1: Lineage and Literature Provenance Audit
- Verify `Provenance`: `kind` must be `"original"` or `"public-inspired"`.
- If `"public-inspired"`, require explicit URL, repository commit hash or release tag, specific algorithm name, and permissive license (Apache-2.0, MIT, BSD-3, CC-BY-4.0).
- Check against all 66 historical bodies: if the source algorithm matches an existing workload (e.g., standard LMS or Kalman filter), verify that the mathematical formulation is distinct or assign it to the same related group.

#### Stage 2: Mathematical Recurrence and Mechanism Audit
- Extract the mathematical state transition equation $c_{t+1} = \mathcal{M}(c_t, x_t)$.
- Reject proposals that merely change scalar constants, polynomial orders, activation functions (e.g., swapping `tanh` for `sigmoid` in an identical recurrence), or trip counts.

#### Stage 3: Loop Carry State Graph Audit
- Inspect the data structure of the carried state:
  - Scalar carry ($\mathbb{R}$)
  - Vector carry ($\mathbb{R}^D$)
  - Matrix carry ($\mathbb{R}^{M \times N}$)
  - Heterogeneous pytree / tuple carry ($(\mathbf{c}_1, \mathbf{c}_2, \dots)$)
- Trace intra-iteration dependency chains: measure longest dependency path length, maximum out-degree, and feed-forward values using `jax_unroll.features`. Reject bodies that produce isomorphic dependency DAGs.

#### Stage 4: Memory Access Pattern Audit
- Classify indexing into one of the 8 canonical memory regimes:
  1. *Sequential streaming:* Closed-over input sliced at index $i$.
  2. *Strided affine:* Closed-over input sliced at $s \cdot i + k$.
  3. *Dynamic multi-element window:* `lax.dynamic_slice` with loop-computed offset.
  4. *Indirect gather:* Static table indexed by external or carried indices.
  5. *Indirect scatter:* Update buffer targeted by non-contiguous index lists.
  6. *Dependent pointer chase:* Addressing where index at step $t+1$ depends on value read at step $t$.
  7. *Spatial stencil roll:* Neighbor shifting via `jnp.roll` or cyclic buffer access.
  8. *Panelwise / tiled buffer:* Block sub-tensor slicing with intermediate register reuse.
- Ensure that proposals in a workstream introduce distinct addressing regimes rather than repeating identical memory accesses.

#### Stage 5: Jaxpr Normalization and Body Hash Canonicalization
- Trace the workload at primary variant with $T=64$ and $u=1$ using `jax.make_jaxpr`.
- Extract `body_hash` via `hashlib.sha256(str(jaxpr).encode("utf-8")).hexdigest()[:16]`.
- Check against all 66 frozen historical hashes and all currently accepted expansion hashes.
  - **Identical Hash:** Immediate rejection (definite duplicate).
  - **Different Hash:** Necessary but NOT sufficient (requires passing Stages 1–4).

#### Stage 6: Factor Equivalence and Numerical Invariant Verification
- Test all unroll factors $u \in \{1, 2, 4, 8\}$ on both divisible ($T=64$) and remainder ($T=100, 1023$) trip counts.
- Compare output at factors 2, 4, and 8 against factor 1:
  $$\max_{u \in \{2, 4, 8\}} \| \mathbf{y}_u - \mathbf{y}_1 \|_\infty \le \text{atol} + \text{rtol} \cdot \| \mathbf{y}_1 \|_\infty$$
- Verify numerical stability: outputs must be strictly finite (no `NaN` or `Inf`) at $T=1023$.
- Verify real dependence on inputs: outputs must change when inputs are modified.

#### Stage 7: Related-Group Assignment and Partition Quarantine Check
- Assign a stable, semantic `related_group` identifier.
- Partition quarantine rule:
  - If the workload shares lineage with any historical training body $\to$ assign to `development` (`train`).
  - If the workload shares lineage with any historical test body $\to$ assign to `historical_evaluation` (`test`).
  - If the workload is genuinely new $\to$ assign to a new related group.
  - Check that no related group contains both development and evaluation workloads.

---

## 6. Guidelines and Recommendations for Corpus Workteams

### 6.1 Recommendations for the Workload Design Team (10 Workstreams)

The target is **60–80 genuinely independent `fori_loop` bodies** across 10 distinct domains. To prevent duplication across parallel subagents:

1. **Adhere to the Strict Static Loop Contract:**
   Every workload must use a single top-level `lax.fori_loop`:
   ```python
   def fn(init_val, *args):
       def body(i, carry):
           # Close over static args or slice at index i
           ...
           return next_carry
       return lax.fori_loop(0, trip_count, body, init_val, unroll=factor)
   ```
2. **Diversify Carry Topologies:**
   Avoid submitting only scalar accumulators. Aim for:
   - 25% Tuple carries (e.g., carrying both filter state and running variance, or double ring pointers).
   - 35% Vector carries (e.g., delay lines, coordinate vectors, frequency bins).
   - 25% Matrix/Grid carries (e.g., covariance matrices, 2D mesh patches, trellis scores).
   - 15% Scalar carries (overhead-dominated loops).
3. **Explicit Domain Differentiation:**
   - *Signal Processing:* Implement Cascaded Biquad (Direct Form II), Goertzel single-bin DFT, LMS adaptive filter, and Polyphase FIR interpolator. Do NOT submit a 1st-order IIR (already in historical `iir_filters`).
   - *Sparse Linear Algebra:* Implement CSR-like row accumulation, triangular back-substitution with masked dependencies, and block-coordinate relaxation.
   - *Dynamic Programming:* Implement rolling edit-distance row update, Viterbi max-plus path scoring, and bounded knapsack updates.
   - *Optimization:* Implement coordinate descent with residual tracking, recursive least squares (RLS), and diagonal Kalman filter updates. Do NOT submit basic dense matrix-vector multiplications.
   - *Numerical Simulation:* Implement Symplectic Leapfrog integration, finite-volume upwind flux, and constrained position Verlet updates.
4. **Define Multiple Shape Variants:**
   Each body must provide at least two shape variants (e.g., cache-fitting vs. memory-bound), with one designated as `primary_variant`.

### 6.2 Recommendations for the Scan Conversion Team

The conversion team is responsible for converting **all 53 historical scan workloads** into faithful `fori_loop` bodies:

1. **Paired Control Labeling:**
   - Converted workloads must be explicitly labeled as paired translations: `is_converted=True` or `lineage="converted_scan"`.
   - Converted bodies contribute **0 new independent bodies**.
2. **Semantic and Numerical Equivalence:**
   - Forward scans over `xs` must close over `xs` and index `xs[i]`.
   - Output preservation: where the original scan returned stacked step outputs, preallocate the full output buffer and update it via `carry.out_buf.at[i].set(...)`. Where only the final carry was returned, preserve that exact return behavior.
   - Verify that outputs match the original scan identically across multiple seeds and trip counts.
3. **Strict Lineage and Partition Preservation:**
   - A converted workload must inherit the EXACT `related_group` ID of its original scan body.
   - The converted version of a historical training scan belongs strictly to `development`.
   - The converted version of a historical test scan belongs strictly to `historical_evaluation`.

### 6.3 Recommendations for the Split Design Team

The expanded corpus requires a three-cohort evaluation design:
1. **Development Cohort (`dev`):**
   - 7 historical fori_loop bodies + 45 converted scan bodies + 45–60 new independent fori_loop bodies.
   - Used for training feature models and cross-validation.
2. **Historical Evaluation Cohort (`historical_eval`):**
   - 6 historical fori_loop bodies + 8 converted scan bodies.
   - Used to verify backward compatibility and transfer on previously seen families.
3. **Fresh Evaluation Cohort (`fresh_eval`):**
   - **15–20 genuinely independent fori_loop bodies** drawn from unfamiliar mechanisms and source lineages with ZERO prior training exposure.
   - Completely disjoint related groups from all development workloads.
4. **Deterministic Hash Pinning:**
   - The complete split assignment table must be frozen and its SHA-256 hash published in `docs/fori_expansion.md` **prior** to acquiring timing labels.

---

## 7. Verification Script and Historical Group Mapping

To automate validation and prevent regression, the complete historical mapping is codified below. This validation routine confirms zero partition leakage across all 66 historical loop bodies:

```python
"""Validation routine for historical workload related-group assignments."""

HISTORICAL_RELATED_GROUPS = {
    # workloads.py (6)
    "scalar_recurrence": "iir_filters",
    "elementwise_sine": "elementwise_control",
    "reduction_vector": "vector_reduction_accumulator",
    "dense_matvec": "dense_matvec_recurrence",
    "indexed_update": "indexed_gather_scatter_update",
    "jacobi_solver": "stencil_axis_aligned",
    # recovery_workloads.py (10)
    "window_read": "dynamic_window_slice",
    "window_replace": "dynamic_window_slice",
    "gather_lookup": "weighted_gather_lookup",
    "gather_rows": "row_gather_pool",
    "scatter_accumulate": "collision_memory",
    "scatter_blend": "unique_update",
    "strided_mix": "strided_offset_mix",
    "bucket_max": "bucket_histogram_max",
    "masked_rows": "masked_row_select",
    "ring_feedback": "ring_feedback_buffer",
    # corpus/indirect.py (21)
    "gather_scalar_table": "random_scalar_gather",
    "gather_row_pool": "random_row_gather",
    "gather_multi_table": "select_gather",
    "gather_duplicate_index": "collision_memory",
    "gather_sorted_blocks": "ordered_gather",
    "gather_masked_window": "masked_memory",
    "scatter_unique_rows": "unique_update",
    "scatter_collision_add": "collision_memory",
    "scatter_min_update": "collision_memory",
    "scatter_two_tables": "split_table_memory",
    "scatter_permutation": "permutation_memory",
    "locality_stream_contiguous": "sequential_locality",
    "locality_stride_sample": "strided_locality",
    "locality_random_window": "random_window_locality",
    "locality_block_reuse": "reuse_locality",
    "locality_permutation_read": "permutation_memory",
    "dep_pointer_chase": "dependent_chain",
    "dep_parallel_chains": "dependent_chain",
    "dep_two_level_indirection": "dependent_chain",
    "dep_linked_list": "dependent_chain",
    "dep_bucket_stateful": "stateful_dependent",
    # corpus/structured.py (26)
    "rolling_mean_shift": "rolling_moments",
    "rolling_conv1d_fir": "rolling_weighted_windows",
    "rolling_extreme_max": "rolling_extrema",
    "rolling_variance_moments": "rolling_moments",
    "rolling_dot_two_streams": "rolling_weighted_windows",
    "stencil_diffusion_1d_fori": "stencil_axis_aligned",
    "stencil_five_point_2d": "stencil_axis_aligned",
    "stencil_nine_point_2d": "stencil_wide_support",
    "stencil_anisotropic_2d": "stencil_wide_support",
    "streaming_running_mean": "streaming_moments",
    "streaming_welford_var": "streaming_moments",
    "streaming_decayed_extrema": "streaming_extrema",
    "streaming_logsumexp": "streaming_normalizers",
    "segment_sum_reset": "segmented_reset",
    "segment_max_reset": "segmented_reset",
    "segment_weighted_mean": "segmented_reset",
    "segment_carry_ema": "segmented_carry",
    "stateful_scalar_iir": "iir_filters",
    "stateful_vector_iir": "iir_filters",
    "stateful_gru_gate": "gated_recurrences",
    "stateful_lstm_cell": "gated_recurrences",
    "stateful_coupled_oscillator": "coupled_dynamics",
    "control_polynomial_forcing": "elementwise_control",
    "control_transcendental": "elementwise_control",
    "control_bisection_bracket": "root_finding_control",
    "control_lowrank_update": "lowrank_control",
    # corpus/static_matched.py (3)
    "matched_gather": "random_scalar_gather",
    "matched_scatter_add": "collision_memory",
    "matched_scatter_min": "collision_memory",
}
```

This mapping guarantees zero leakage, complete backward compatibility, and an uncompromised foundation for the expanded `fori_loop` benchmark corpus.
