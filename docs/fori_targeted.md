# JAX fori_loop Targeted Benchmark Collection: Apple M2 CPU Report

This document reports the end-to-end implementation, verification, frozen workload split, and serial timed collection for the targeted `jax.lax.fori_loop` unrolling study on the Apple M2 CPU, pursuant to [docs/gemini_fori_targeted_collection.md](file:///Users/saitejvelaga/Documents/umich/CSE583/project/docs/gemini_fori_targeted_collection.md).

---

## 1. Executive Summary & Protocol Overview

The goal of this round is to determine whether choosing an unroll factor $u \in \{1, 2, 4, 8\}$ makes static-bound `jax.lax.fori_loop` bodies execute faster on Apple M2 CPU after compilation, isolating trip-count effects from unroll-tail remainder effects.

### Fixed Measurement Matrix
- **Cohorts**:
  - **12 Existing Development Bodies**: Reused from genuine distinct training lineages in `benchmarks/corpus/fori_expansion`.
  - **12 New Development Bodies**: Broadening the development set across 6 core runtime variation mechanisms.
  - **24 New Sealed Evaluation Bodies**: Entirely new algorithm lineages with zero prior overlap, including ordinary workloads where factor 1 can win. At least 12 (all 24 here) are traceable to published, peer-reviewed public application algorithms.
- **Total Loop Bodies**: **48 distinct bodies** across 6 domain modules under `benchmarks/corpus/fori_targeted/`.
- **Working-Set Sizing**: Exactly **2 meaningfully different sizes** (`small` and `large`) for each of the 48 bodies, with verified and documented memory byte differences.
- **Trip-Count Pairs**: Four static trips — **255, 256, 1023, 1024** — pairing divisible trips (256, 1024) against prime/remainder trips (255, 1023) at nearly identical work.
- **Deterministic Prefix Inputs**: Base input buffers generated at maximum trip ($T=1024$) and sliced (`[:trip]`), ensuring iterations $0 \le i < 255$ encounter bit-for-bit identical inputs whether running at $T=255$ or $T=256$.
- **Process Measurements**: 48 bodies $\times$ 2 sizes $\times$ 4 trip counts $\times$ 4 factors $\times$ 3 independent process repeats = **4,608 planned measurements**.
- **Model-Ready Configurations**:
  - Development Table: 24 bodies $\times$ 2 sizes $\times$ 4 trips = **192 configurations** (768 rows across 4 factors).
  - Sealed Evaluation Table: 24 bodies $\times$ 2 sizes $\times$ 4 trips = **192 configurations** (768 rows across 4 factors).

---

## 2. Six Domain Workstreams & 48-Body Inventory

All 48 loop bodies are self-contained JAX functions containing exactly **one** top-level `jax.lax.fori_loop` with static bounds and `unroll=factor`.

### Domain 1: Dependency Depth & Instruction Parallelism (`dependency_parallelism.py`)
| Workload | Cohort | Description | Small Size (Bytes) | Large Size (Bytes) |
|---|---|---|---|---|
| `dep_cascaded_biquad` | Existing Train | Biquad filter cascade | stages=4 (state 32 B, sos 80 B) | stages=16 (state 128 B, sos 320 B) |
| `dep_coupled_oscillator` | Existing Train | Harmonic oscillator chain | N=16 (state 128 B) | N=64 (state 512 B) |
| `dep_dual_accumulator_iir` | New Train | Interleaved dual recursive accumulator | channels=4 (state 64 B) | channels=16 (state 256 B) |
| `dep_coupled_ekf_covariance` | New Train | EKF state and covariance time propagation | dim=4 (x 16 B, P 64 B) | dim=8 (x 32 B, P 256 B) |
| `eval_dep_levinson_durbin` | Sealed Eval | Levinson-Durbin lattice LPC recursion | order=8 (coeffs 36 B) | order=32 (coeffs 132 B) |
| `eval_dep_savitzky_golay_stream` | Sealed Eval | Real-time Savitzky-Golay polynomial filter | window=7 (buffer 28 B) | window=31 (buffer 124 B) |
| `eval_dep_carrier_pll_costas` | Sealed Eval | Costas loop carrier phase/freq recovery | channels=1 (state 16 B) | channels=4 (state 64 B) |
| `eval_dep_unscented_sigma_prop` | Sealed Eval | UKF sigma-point non-linear propagation | dim=3 (mean 12 B, cov 36 B) | dim=6 (mean 24 B, cov 144 B) |

### Domain 2: Locality and Indexing (`locality_indexing.py`)
| Workload | Cohort | Description | Small Size (Bytes) | Large Size (Bytes) |
|---|---|---|---|---|
| `loc_strided_transpose` | Existing Train | Strided 2D matrix transpose copy | tile 32x32 (4 KB) | tile 64x64 (16 KB) |
| `loc_double_indirect` | Existing Train | Two-level pointer dereferencing table lookup | table_size=256 (1 KB) | table_size=1024 (4 KB) |
| `loc_morton_z_order_lookup` | New Train | Morton Z-order curve coordinate mapping | grid 32x32 (4 KB) | grid 128x128 (64 KB) |
| `loc_runlength_compressed_scan` | New Train | RLE compressed column segment scan | segments=64 (512 B) | segments=256 (2 KB) |
| `eval_loc_btree_search_step` | Sealed Eval | Cache-oblivious Eytzinger B-tree search | depth=6 (63 nodes, 252 B) | depth=10 (1023 nodes, 4 KB) |
| `eval_loc_bresenham_voxel_walk` | Sealed Eval | 3D Amanatides-Woo fast voxel ray traversal | grid 32x32x32 (128 KB) | grid 64x64x64 (1024 KB) |
| `eval_loc_quadtree_point_query` | Sealed Eval | Quadtree hierarchical quadrant point search | depth=5 (341 nodes, 4 KB) | depth=7 (5461 nodes, 65 KB) |
| `eval_loc_hilbert_curve_gather` | Sealed Eval | 2D Hilbert space-filling curve gather | order=4 (grid 16x16 = 1 KB) | order=6 (grid 64x64 = 16 KB) |

### Domain 3: Writes and State Size (`writes_state.py`)
| Workload | Cohort | Description | Small Size (Bytes) | Large Size (Bytes) |
|---|---|---|---|---|
| `write_patch_blend` | Existing Train | Overlapping 2D patch update blending | canvas 32x32, patch 8x8 (4 KB) | canvas 64x64, patch 16x16 (16 KB) |
| `write_scatter_accumulate` | Existing Train | Dynamic update scatter addition | dim=64 (256 B) | dim=256 (1 KB) |
| `write_multitap_delay_ring` | New Train | Circular delay line ring buffer with 4 taps | ring 256 floats (1 KB) | ring 2048 floats (8 KB) |
| `write_tiled_stencil_accum` | New Train | 2D grid accumulator with stencil writes | grid 16x16 (1 KB) | grid 48x48 (9.2 KB) |
| `eval_write_waveguide_mesh` | Sealed Eval | Digital waveguide mesh acoustic scattering | mesh 16x16 (1 KB) | mesh 48x48 (9.2 KB) |
| `eval_write_pic_charge_deposit` | Sealed Eval | PIC bilinear charge scatter deposition | grid 16x16 (1 KB) | grid 64x64 (16 KB) |
| `eval_write_pingpong_frame_swap` | Sealed Eval | Dual-buffer ping-pong state swap | 2 buffers x 64 floats (512 B) | 2 buffers x 256 floats (2 KB) |
| `eval_write_circular_sliding_hist` | Sealed Eval | Circular buffer rolling histogram update | window=32, buckets=16 (192 B) | window=128, buckets=64 (768 B) |

### Domain 4: Dense Arithmetic and Reductions (`dense_reductions.py`)
| Workload | Cohort | Description | Small Size (Bytes) | Large Size (Bytes) |
|---|---|---|---|---|
| `dense_matvec_step` | Existing Train | Matrix-vector row dot-product reduction | dim=64 (matrix 64x64 = 16 KB) | dim=128 (matrix 128x128 = 64 KB) |
| `dense_gram_accum` | Existing Train | Outer-product Gram matrix accumulation | dim=16 (matrix 16x16 = 1 KB) | dim=32 (matrix 32x32 = 4 KB) |
| `dense_outer_product_cov` | New Train | Streaming sample covariance update | dim=12 (cov 12x12 = 576 B) | dim=28 (cov 28x28 = 3136 B) |
| `dense_givens_qr_step` | New Train | Givens orthogonal plane rotation | cols=16 (1 KB) | cols=48 (3 KB) |
| `eval_dense_lanczos_tridiag` | Sealed Eval | Lanczos three-term vector orthogonalization | dim=32 (matrix 32x32 = 4 KB) | dim=64 (matrix 64x64 = 16 KB) |
| `eval_dense_horner_matrix_poly` | Sealed Eval | Horner nested matrix polynomial evaluation | matrix 16x16 (1 KB) | matrix 32x32 (4 KB) |
| `eval_dense_attention_block_step` | Sealed Eval | Scaled dot-product attention score reduction | head_dim=16, seq=16 (1 KB) | head_dim=32, seq=32 (4 KB) |
| `eval_dense_fft_radix2_butterfly` | Sealed Eval | Cooley-Tukey radix-2 FFT complex butterfly | pairs=16 complex (128 B) | pairs=64 complex (512 B) |

### Domain 5: Branches, Masks and Active State (`branches_masks.py`)
| Workload | Cohort | Description | Small Size (Bytes) | Large Size (Bytes) |
|---|---|---|---|---|
| `branch_leaky_bucket` | Existing Train | Token bucket packet policing state machine | classes=4 (64 B) | classes=16 (256 B) |
| `branch_masked_active_set` | Existing Train | Active set boolean indicator update | dim=32 (128 B) | dim=128 (512 B) |
| `branch_hysteresis_schmitt` | New Train | Dual-threshold Schmitt trigger FSM | channels=4 (64 B) | channels=16 (256 B) |
| `branch_sparse_grad_clip` | New Train | Deadband gradient thresholding & norm clip | dim=64 (256 B) | dim=256 (1 KB) |
| `eval_branch_huffman_zero_rle` | Sealed Eval | FSM decoding zero runs into carried buffer | buffer=32 ints (128 B) | buffer=128 ints (512 B) |
| `eval_branch_tcp_congestion_fsm` | Sealed Eval | TCP AIMD congestion control state machine | sessions=4 (64 B) | sessions=16 (256 B) |
| `eval_branch_particle_boundary_bounce` | Sealed Eval | 2D particle specular boundary bounce | particles=8 (128 B) | particles=32 (512 B) |
| `eval_branch_ray_slab_intersect` | Sealed Eval | Kay-Kajiya slab method ray-AABB test | rays=8 (256 B) | rays=32 (1 KB) |

### Domain 6: Application Kernels (`application_kernels.py`)
| Workload | Cohort | Description | Small Size (Bytes) | Large Size (Bytes) |
|---|---|---|---|---|
| `app_rigid_quaternion` | Existing Train | Rigid body quaternion rotation mechanics | bodies=1 (state 28 B) | bodies=4 (state 112 B) |
| `app_coordinate_descent` | Existing Train | Cyclic coordinate descent with soft-threshold | features=16 (64 B) | features=64 (256 B) |
| `app_molecular_verlet` | New Train | Velocity-Verlet for Lennard-Jones chain | particles=8 (pos 32 B + vel 32 B = 64 B) | particles=32 (pos 128 B + vel 128 B = 256 B) |
| `app_robot_kinematic_chain` | New Train | Forward kinematics SO(3) rotation chain | joints=6 (48 B) | joints=18 (48 B, link_len 72 B) |
| `eval_app_rk4_lorenz_step` | Sealed Eval | 4th-order Runge-Kutta Lorenz system step | systems=2 (state 24 B) | systems=8 (state 96 B) |
| `eval_app_trajectory_shooting` | Sealed Eval | Spacecraft trajectory shooting under gravity | vehicles=1 (state 24 B) | vehicles=4 (state 96 B) |
| `eval_app_ct_backprojection_step` | Sealed Eval | CT filtered backprojection ray accumulation | image 16x16 (1 KB) | image 32x32 (4 KB) |
| `eval_app_epidemic_seir_rk4` | Sealed Eval | SEIR epidemic compartmental ODE step | regions=2 (state 32 B) | regions=8 (state 128 B) |

---

## 3. Split Integrity & Lineage Isolation

The partition is frozen prior to execution at:
- **Split CSV Path**: [benchmarks/corpus/fori_targeted/split.csv](file:///Users/saitejvelaga/Documents/umich/CSE583/project/benchmarks/corpus/fori_targeted/split.csv)
- **Planned Matrix**: [results/fori_targeted/planned_matrix.json](file:///Users/saitejvelaga/Documents/umich/CSE583/project/results/fori_targeted/planned_matrix.json)
- **Deterministic Split SHA-256 Hash**: `ef84fb5935d16804cc1264128d093598f862767500e1e77fe141992ca8a6ed27`

### Lineage Isolation Assertions
1. **Disjoint Lineages**: All 24 sealed evaluation bodies belong to completely separate `related_group` identifiers with 0 overlap with the 24 development bodies.
2. **Body Hash Uniqueness**: All 48 factor-1 body Jaxpr hashes are globally distinct with 0 collisions among themselves and 0 collisions against the historical 136 expansion loop bodies.
3. **No Leakage**: No evaluation body is a renamed, parameter-tweaked, or converted derivative of any historical or new training body.

---

## 4. Hardware and Environment Specification

Recorded dynamically into [results/fori_targeted/hardware_env.json](file:///Users/saitejvelaga/Documents/umich/CSE583/project/results/fori_targeted/hardware_env.json):
- **CPU**: Apple M2 (arm64, Darwin 24.3.0)
- **Python**: 3.13.2
- **JAX / jaxlib**: 0.12.0.1 (CPU backend, device `cpu:0`)
- **Execution Mode**: Serial fresh-process execution via `benchmarks.sweep` child processes.

---

## 5. Artifact Deliverables & Exact Paths

| Deliverable | Path | Description |
|---|---|---|
| **Split Specification** | [benchmarks/corpus/fori_targeted/split.csv](file:///Users/saitejvelaga/Documents/umich/CSE583/project/benchmarks/corpus/fori_targeted/split.csv) | Machine-readable partition of all 48 bodies |
| **Planned Matrix** | [results/fori_targeted/planned_matrix.json](file:///Users/saitejvelaga/Documents/umich/CSE583/project/results/fori_targeted/planned_matrix.json) | Planned acquisition keys, seed, and split hash |
| **Hardware Environment** | [results/fori_targeted/hardware_env.json](file:///Users/saitejvelaga/Documents/umich/CSE583/project/results/fori_targeted/hardware_env.json) | Host hardware and software configuration |
| **Raw Timings CSV** | [results/fori_targeted/raw_sweep.csv](file:///Users/saitejvelaga/Documents/umich/CSE583/project/results/fori_targeted/raw_sweep.csv) | All process measurements (cold/warm/e2e/HLO) |
| **Acquisition Log** | [results/fori_targeted/acquisition_log.csv](file:///Users/saitejvelaga/Documents/umich/CSE583/project/results/fori_targeted/acquisition_log.csv) | Order, pass, seed, and per-measurement elapsed |
| **Development Table** | [results/fori_targeted/train_features.csv](file:///Users/saitejvelaga/Documents/umich/CSE583/project/results/fori_targeted/train_features.csv) | 192 configs x 4 factors = 768 rows with static features |
| **Sealed Evaluation Table** | [results/fori_targeted/sealed_eval_features.csv](file:///Users/saitejvelaga/Documents/umich/CSE583/project/results/fori_targeted/sealed_eval_features.csv) | 192 configs x 4 factors = 768 rows with static features |
| **Software Tests** | [tests/test_fori_targeted_registry.py](file:///Users/saitejvelaga/Documents/umich/CSE583/project/tests/test_fori_targeted_registry.py)<br>[tests/test_fori_targeted_split.py](file:///Users/saitejvelaga/Documents/umich/CSE583/project/tests/test_fori_targeted_split.py)<br>[tests/test_fori_targeted_workloads.py](file:///Users/saitejvelaga/Documents/umich/CSE583/project/tests/test_fori_targeted_workloads.py)<br>[tests/test_fori_targeted_dataset.py](file:///Users/saitejvelaga/Documents/umich/CSE583/project/tests/test_fori_targeted_dataset.py) | Verification test suites |

---

## 6. Reproduction Commands

```bash
# 1. Run unit tests
.venv/bin/python -m pytest tests/test_fori_targeted_*.py -v

# 2. Re-verify split freeze and hash
.venv/bin/python -m benchmarks.corpus.fori_targeted.acquire --freeze-only

# 3. Run timed serial collection on Apple M2 CPU
.venv/bin/python -m benchmarks.corpus.fori_targeted.acquire

# 4. Re-export feature tables from existing raw data
.venv/bin/python -m benchmarks.corpus.fori_targeted.acquire --export-only
```
