# JAX fori_loop Corpus Expansion & Conversion Architecture Design Document

**Document Version:** 1.0.0  
**Target Architecture:** Apple Silicon CPU (M2/M3/M4)  
**Target Compiler:** JAX 0.12+ (Lowered to XLA/LLVM loop primitives with static trip bounds)  
**Location:** `benchmarks/corpus/fori_expansion/workload_design.md`  

---

## 1. Executive Summary & Strategic Objectives

The objective of this expansion is to construct an extensive, rigorous, and completely `lax.fori_loop`-based benchmark corpus for loop unrolling policy learning on Apple Silicon CPUs. Currently, our combined benchmark dataset consists of 66 measured workloads, of which only 13 use `fori_loop` (and only 7 of those were in the training partition), with 53 using `lax.scan`. Furthermore, 35 workloads have only a single measured configuration.

To support unroll factor optimization ($u \in \{1, 2, 4, 8\}$) for steady-state execution time on static-bound loops, this project delivers two major contributions:
1. **Conversion Architecture for 53 Historical Scan Workloads:** A deterministic, bit-preserving transformation translating all 53 public `lax.scan` workloads into native `lax.fori_loop` implementations under `benchmarks/corpus/fori_expansion/converted/`, grouped cleanly by historical workload family.
2. **70 New Independent fori_loop Workloads Across 10 Workstreams:** Genuinely distinct algorithms spanning Signal Processing, Optimization/Estimation, Sparse Linear Algebra, Graph/Mesh, Dynamic Programming, Numerical Simulation, Tiled Tensor Workloads, Index/Memory Access, Streaming/Compression, and Scheduling/Control Flow. 

The resulting corpus scales the repository to **136 total fori_loop workloads** (13 historical `fori_loop` + 53 converted `fori_loop` + 70 new independent `fori_loop`), with a clear partitioning scheme that keeps historical lineages intact while reserving 18 fresh independent bodies for held-out evaluation.

---

## 2. Scan-to-fori_loop Conversion Architecture

### 2.1 Theoretical & Operational Equivalence

In JAX, `lax.scan` operates over input sequences by slicing along the leading dimension of input arrays at each timestep:
$$\text{scan}: (c_0, [x_0, x_1, \dots, x_{T-1}]) \mapsto (c_T, [y_0, y_1, \dots, y_{T-1}])$$
where $(c_{t+1}, y_t) = \text{step}(c_t, x_t)$.

In contrast, `lax.fori_loop` operates over an integer index interval $[0, T)$:
$$\text{fori\_loop}: (0, T, \text{body}, c_0) \mapsto c_T$$
where $c_{t+1} = \text{body}(t, c_t)$.

To convert any `lax.scan` to a `lax.fori_loop`:
1. **Forward Indexing (Slicing):** The fori_loop body closes over the input arrays passed into the outer function. Inside the body, for iteration index $i \in [0, T)$, each input array (or leaf of a PyTree) is indexed at position $i$:
   ```python
   # Scalar/array slice
   x_t = xs[i]
   # PyTree of input arrays
   xs_t = jax.tree.map(lambda leaf: leaf[i], xs)
   ```
2. **Carry Preservation:**
   - **Carry-Only Workloads:** When the original scan consumes only the final carry (52 of 53 historical scan workloads), the fori_loop carry is identical to the scan carry:
     ```python
     def body(i, carry):
         x_t = xs[i]
         next_carry, _ = step(carry, x_t)
         return next_carry
     final_carry = lax.fori_loop(0, trip, body, init, unroll=factor)
     ```
   - **Stacked-Output Workloads:** When the caller consumes stacked outputs $y_t \in \mathbb{R}^{S}$, an output buffer of shape $(T, S)$ is preallocated:
     ```python
     out_buf = jnp.zeros((trip, *y_shape), dtype=y_dtype)
     fori_init = (init, out_buf)
     def body(i, carry_and_out):
         carry, out = carry_and_out
         x_t = xs[i]
         next_carry, y_t = step(carry, x_t)
         next_out = out.at[i].set(y_t)
         return (next_carry, next_out)
     final_carry, final_ys = lax.fori_loop(0, trip, body, fori_init, unroll=factor)
     ```
3. **Tuple Carries:** Where the carry is a tuple (e.g. `(left, right)` in `scatter_two_tables` or `(history, running)` in `rolling_mean_shift`), the tuple structure is preserved directly in the fori_loop carry.
4. **Preserved Signatures & Numerics:** The converted workload function accepts the exact same arguments as the original scan workload. Bit-level numerical equivalence is enforced against the original scan output at all unroll factors ($u \in \{1, 2, 4, 8\}$) on both divisible ($T=64, 1024$) and remainder ($T=100, 1023$) trip counts.

### 2.2 Converted Module Layout

The 53 scan workloads are partitioned into 11 dedicated modules under `benchmarks/corpus/fori_expansion/converted/`:
- `converted_workloads.py`: 5 Phase 1 core workloads (`dense_matvec`, `elementwise_sine`, `indexed_update`, `reduction_vector`, `scalar_recurrence`)
- `converted_recovery.py`: 9 Phase 2 recovery workloads (`bucket_max`, `gather_lookup`, `gather_rows`, `masked_rows`, `scatter_accumulate`, `scatter_blend`, `strided_mix`, `window_read`, `window_replace`)
- `converted_indirect_gather.py`: 7 workloads (`gather_scalar_table`, `gather_row_pool`, `gather_multi_table`, `gather_duplicate_index`, `gather_sorted_blocks`, `gather_masked_window`, `matched_gather`)
- `converted_indirect_scatter.py`: 6 workloads (`scatter_unique_rows`, `scatter_collision_add`, `scatter_two_tables`, `scatter_permutation`, `matched_scatter_add`, `matched_scatter_min`)
- `converted_locality_access.py`: 4 workloads (`locality_stream_contiguous`, `locality_stride_sample`, `locality_random_window`, `locality_permutation_read`)
- `converted_dependent_indexing.py`: 1 workload (`dep_parallel_chains`)
- `converted_rolling_window.py`: 5 workloads (`rolling_mean_shift`, `rolling_conv1d_fir`, `rolling_extreme_max`, `rolling_variance_moments`, `rolling_dot_two_streams`)
- `converted_streaming_reduction.py`: 4 workloads (`streaming_running_mean`, `streaming_welford_var`, `streaming_decayed_extrema`, `streaming_logsumexp`)
- `converted_segmented_scan.py`: 4 workloads (`segment_sum_reset`, `segment_max_reset`, `segment_weighted_mean`, `segment_carry_ema`)
- `converted_stateful_recurrence.py`: 4 workloads (`stateful_scalar_iir`, `stateful_vector_iir`, `stateful_gru_gate`, `stateful_lstm_cell`)
- `converted_control_arithmetic.py`: 4 workloads (`control_polynomial_forcing`, `control_transcendental`, `control_bisection_bracket`, `control_lowrank_update`)

---

### 2.3 Comprehensive Conversion Audit Table (All 53 Scan Workloads)

| Converted ID | Original Workload ID | Historical Source | Historical Family | Historical Split | Related Group ID | Forward Indexing & Carry Transformation | Return Preservation |
|---|---|---|---|---|---|---|---|
| `conv_dense_matvec` | `dense_matvec` | `workloads.py` | `dense_linear_algebra` | Train | `dense_matvec` | `x[i]`; carry: `(width,)`; body: `matrix @ carry + x[i]` | Final carry array `(width,)` |
| `conv_elementwise_sine` | `elementwise_sine` | `workloads.py` | `elementwise` | Train | `elementwise_sine` | `x[i]`; carry: `(width,)`; body: `sin(carry + x[i])` | Final carry array `(width,)` |
| `conv_indexed_update` | `indexed_update` | `workloads.py` | `indexed_memory` | Test | `indexed_update` | `x[i]`; carry: `(width,)`; body: `carry.at[idx].set(...)` | Final carry array `(width,)` |
| `conv_reduction_vector` | `reduction_vector` | `workloads.py` | `reduction` | Train | `reduction_vector` | `x[i]`; carry: `()`; body: `carry + sum(x[i]*weights)` | Final scalar carry `()` |
| `conv_scalar_recurrence` | `scalar_recurrence` | `workloads.py` | `recurrence` | Train | `scalar_recurrence` | `x[i]`; carry: `()` or `(8,)`; body: `carry*0.9 + x[i]` | Final carry `()` or `(8,)` |
| `conv_bucket_max` | `bucket_max` | `recovery_workloads.py` | `histogram` | Test | `bucket_max` | `(keys[i], vals[i])`; carry: `(width,)`; body: `carry.at[k].max(v)` | Final carry array `(width,)` |
| `conv_gather_lookup` | `gather_lookup` | `recovery_workloads.py` | `gather` | Train | `gather_lookup` | `(idx[i], w[i])`; carry: `()`; body: `carry + sum(table[idx]*w)` | Final scalar carry `()` |
| `conv_gather_rows` | `gather_rows` | `recovery_workloads.py` | `gather` | Train | `gather_rows` | `idx[i]`; carry: `(16,)`; body: `max(carry*0.875, sum(table[idx]))` | Final carry array `(16,)` |
| `conv_masked_rows` | `masked_rows` | `recovery_workloads.py` | `masked_read` | Test | `masked_rows` | `(l[i], r[i], m[i])`; carry: `(8,)`; body: masked row blend + add | Final carry array `(8,)` |
| `conv_scatter_accumulate` | `scatter_accumulate` | `recovery_workloads.py` | `scatter` | Train | `scatter_accumulate` | `(idx[i], u[i])`; carry: `(width,)`; body: `carry.at[idx].add(u)` | Final carry array `(width,)` |
| `conv_scatter_blend` | `scatter_blend` | `recovery_workloads.py` | `scatter` | Train | `scatter_blend` | `(idx[i], s[i])`; carry: `(width,)`; body: `carry.at[idx].set(tanh)` | Final carry array `(width,)` |
| `conv_strided_mix` | `strided_mix` | `recovery_workloads.py` | `strided_read` | Test | `strided_mix` | `starts[i]`; carry: `()`; body: strided table dot product | Final scalar carry `()` |
| `conv_window_read` | `window_read` | `recovery_workloads.py` | `window` | Train | `window_read` | `starts[i]`; carry: `()`; body: `dynamic_slice_in_dim` sum | Final scalar carry `()` |
| `conv_window_replace` | `window_replace` | `recovery_workloads.py` | `window` | Train | `window_replace` | `(starts[i], patch[i])`; carry: `(width,)`; dynamic slice update | Final carry array `(width,)` |
| `conv_gather_scalar_table` | `gather_scalar_table` | `indirect.py` | `indirect_gather` | Train | `random_scalar_gather` | `(idx[i], gain[i])`; carry: `()`; body: `carry*0.9 + table[idx]*gain` | Final scalar carry `()` |
| `conv_gather_row_pool` | `gather_row_pool` | `indirect.py` | `indirect_gather` | Train | `random_row_gather` | `(idx[i], gain[i])`; carry: `(8,)`; pooled row addition | Final carry array `(8,)` |
| `conv_gather_multi_table` | `gather_multi_table` | `indirect.py` | `indirect_gather` | Train | `select_gather` | `(idx[i], sel[i], gain[i])`; carry: `(4,)`; table selection | Final carry array `(4,)` |
| `conv_gather_duplicate_index` | `gather_duplicate_index` | `indirect.py` | `indirect_gather` | Train | `collision_memory` | `(dup_idx[i], gain[i])`; carry: `()`; paired gather add | Final scalar carry `()` |
| `conv_gather_sorted_blocks` | `gather_sorted_blocks` | `indirect.py` | `indirect_gather` | Train | `ordered_gather` | `(starts[i], gain[i])`; carry: `(4,)`; block gather + tanh | Final carry array `(4,)` |
| `conv_gather_masked_window` | `gather_masked_window` | `indirect.py` | `indirect_gather` | Train | `masked_memory` | `(idx[i], mask[i], gain[i])`; carry: `()`; masked take fill | Final scalar carry `()` |
| `conv_matched_gather` | `matched_gather` | `static_matched.py` | `indirect_gather` | Train | `indirect_gather+...` | `idx[i]`; carry: `()`; body: `carry*0.99 + mean(table[idx])` | Final scalar carry `()` |
| `conv_scatter_unique_rows` | `scatter_unique_rows` | `indirect.py` | `indirect_scatter` | Train | `unique_update` | `(idx[i], up[i])`; carry: `(rows, 8)`; unique index set | Final carry array `(rows, 8)` |
| `conv_scatter_collision_add` | `scatter_collision_add` | `indirect.py` | `indirect_scatter` | Train | `collision_memory` | `(r[i], c[i], up[i])`; carry: `(rows, cols)`; 2D collision add | Final carry 2D array |
| `conv_scatter_two_tables` | `scatter_two_tables` | `indirect.py` | `indirect_scatter` | Train | `split_table_memory` | `(idx[i], up[i], mask[i])`; carry: `(left, right)`; masked set | Final 2-tuple `(left, right)` |
| `conv_scatter_permutation` | `scatter_permutation` | `indirect.py` | `indirect_scatter` | Train | `permutation_memory` | `drive[i]`; carry: `(size,)`; permuted roll + tanh update | Final carry array `(size,)` |
| `conv_matched_scatter_add` | `matched_scatter_add` | `static_matched.py` | `indirect_scatter` | Train | `indirect_gather+...` | `(idx[i], up[i])`; carry: `(4096,)`; `carry.at[idx].add(up)` | Final carry array `(4096,)` |
| `conv_matched_scatter_min` | `matched_scatter_min` | `static_matched.py` | `indirect_scatter` | Train | `indirect_gather+...` | `(idx[i], up[i])`; carry: `(4096,)`; `carry.at[idx].min(up)` | Final carry array `(4096,)` |
| `conv_locality_stream_contiguous` | `locality_stream_contiguous` | `indirect.py` | `locality_access` | Train | `sequential_locality` | `(starts[i], gain[i])`; carry: `()`; contiguous slice sum | Final scalar carry `()` |
| `conv_locality_stride_sample` | `locality_stride_sample` | `indirect.py` | `locality_access` | Train | `strided_locality` | `(starts[i], stride[i])`; carry: `(8,)`; strided gather | Final carry array `(8,)` |
| `conv_locality_random_window` | `locality_random_window` | `indirect.py` | `locality_access` | Train | `random_window_locality` | `(starts[i], gain[i])`; carry: `(8,)`; dynamic slice sum | Final carry array `(8,)` |
| `conv_locality_permutation_read` | `locality_permutation_read` | `indirect.py` | `locality_access` | Train | `permutation_memory` | `offsets[i]`; carry: `()`; permutation gather dot product | Final scalar carry `()` |
| `conv_dep_parallel_chains` | `dep_parallel_chains` | `indirect.py` | `dependent_indexing` | Train | `dependent_chain` | `(lane_idx[i], hop[i])`; carry: `(8,)`; parallel pointer chase | Final carry array `(8,)` |
| `conv_rolling_mean_shift` | `rolling_mean_shift` | `structured.py` | `rolling_window` | Train | `rolling_moments` | `x[i]`; carry: `(window, running)`; rolling sum shift | Final scalar `running / width` |
| `conv_rolling_conv1d_fir` | `rolling_conv1d_fir` | `structured.py` | `rolling_window` | Train | `rolling_weighted_windows` | `x[i]`; carry: `(history, last)`; FIR dot product | Final scalar `last` |
| `conv_rolling_extreme_max` | `rolling_extreme_max` | `structured.py` | `rolling_window` | Train | `rolling_extrema` | `x[i]`; carry: `(window, max_val)`; rolling max | Final scalar `max_val` |
| `conv_rolling_variance_moments` | `rolling_variance_moments` | `structured.py` | `rolling_window` | Train | `rolling_moments` | `x[i]`; carry: `(win, s1, s2)`; rolling 1st/2nd moments | Final scalar sample variance |
| `conv_rolling_dot_two_streams` | `rolling_dot_two_streams` | `structured.py` | `rolling_window` | Train | `rolling_weighted_windows` | `(a[i], b[i])`; carry: `(win_a, win_b)`; rolling dot | Final scalar dot product |
| `conv_streaming_running_mean` | `streaming_running_mean` | `structured.py` | `streaming_reduction` | Train | `streaming_moments` | `x[i]`; carry: `(count, mean)`; online mean recurrence | Final scalar mean |
| `conv_streaming_welford_var` | `streaming_welford_var` | `structured.py` | `streaming_reduction` | Train | `streaming_moments` | `x[i]`; carry: `(count, M, S)`; Welford variance update | Final scalar variance `S / count` |
| `conv_streaming_decayed_extrema` | `streaming_decayed_extrema` | `structured.py` | `streaming_reduction` | Train | `streaming_extrema` | `x[i]`; carry: `(lo, hi)`; exponential decay envelope | Final scalar `hi - lo` |
| `conv_streaming_logsumexp` | `streaming_logsumexp` | `structured.py` | `streaming_reduction` | Train | `streaming_normalizers` | `x[i]`; carry: `(m, s)`; online log-sum-exp normalization | Final scalar `m + log(s)` |
| `conv_segment_sum_reset` | `segment_sum_reset` | `structured.py` | `segmented_scan` | Test | `segmented_reset` | `(seg_id[i], x[i])`; carry: `(last_seg, accum)`; reset sum | Final carry array `(4,)` |
| `conv_segment_max_reset` | `segment_max_reset` | `structured.py` | `segmented_scan` | Test | `segmented_reset` | `(seg_id[i], x[i])`; carry: `(last_seg, accum)`; reset max | Final carry array `(4,)` |
| `conv_segment_weighted_mean` | `segment_weighted_mean` | `structured.py` | `segmented_scan` | Test | `segmented_reset` | `(seg_id[i], x[i], w[i])`; carry: `(last_seg, sw, sx)` | Final carry array `(4,)` |
| `conv_segment_carry_ema` | `segment_carry_ema` | `structured.py` | `segmented_scan` | Test | `segmented_carry` | `(seg_id[i], x[i])`; carry: `(last_seg, ema)`; carry EMA | Final carry array `(4,)` |
| `conv_stateful_scalar_iir` | `stateful_scalar_iir` | `structured.py` | `stateful_recurrence` | Train | `iir_filters` | `x[i]`; carry: `(y1, y2)`; 2nd order autoregression | Final scalar `y1` |
| `conv_stateful_vector_iir` | `stateful_vector_iir` | `structured.py` | `stateful_recurrence` | Train | `iir_filters` | `x[i]`; carry: `(y1, y2)` of shape `(8,)`; vector AR(2) | Final carry array `(8,)` |
| `conv_stateful_gru_gate` | `stateful_gru_gate` | `structured.py` | `stateful_recurrence` | Train | `gated_recurrences` | `x[i]`; carry: `h (4,)`; reset and update gates | Final hidden state `h (4,)` |
| `conv_stateful_lstm_cell` | `stateful_lstm_cell` | `structured.py` | `stateful_recurrence` | Train | `gated_recurrences` | `x[i]`; carry: `(c, h)` of shape `(4,)`; input/forget/output gates | Final hidden state `h (4,)` |
| `conv_control_polynomial_forcing` | `control_polynomial_forcing` | `structured.py` | `control_arithmetic` | Train | `elementwise_control` | `(x[i], t[i])`; carry: `(512,)`; polynomial recurrence | Final state array `(512,)` |
| `conv_control_transcendental` | `control_transcendental` | `structured.py` | `control_arithmetic` | Train | `elementwise_control` | `(x[i], t[i])`; carry: `(512,)`; sin/exp coupling | Final state array `(512,)` |
| `conv_control_bisection_bracket` | `control_bisection_bracket` | `structured.py` | `control_arithmetic` | Train | `root_finding_control` | `target[i]`; carry: `(lo, hi)`; branchless bracket update | Final bracket midpoint `(4,)` |
| `conv_control_lowrank_update` | `control_lowrank_update` | `structured.py` | `control_arithmetic` | Train | `lowrank_control` | `(u[i], v[i])`; carry: `(8, 8)`; rank-1 update $M + u v^T$ | Final matrix `(8, 8)` |

---

## 3. New Independent fori_loop Bodies Design (70 Algorithms)

We propose 70 genuinely distinct, mathematically rigorous, independent algorithms across the 10 required workstreams (7 algorithms per workstream). 

Each algorithm is specified below with complete architectural and operational fidelity.

---

### Workstream 1: Signal Processing

#### 1. `sig_cascaded_biquad`
- **Mechanism & Family:** Cascaded Direct-Form II Transposed Biquad Filter Bank (`signal_biquad_cascade`)
- **Loop Carry & State:** `states`: float32 array `(num_stages, 2)`, holding the internal delay states $w_{1, s}, w_{2, s}$ for each biquad section $s \in [0, S)$.
- **Input Arrays & Generation:** Input audio stream `x`: `(trip,)` float32 uniformly distributed in $[-0.5, 0.5]$ (seed 583_801); filter coefficients `sos`: `(num_stages, 5)` containing $[b_{0}, b_{1}, b_{2}, a_{1}, a_{2}]$ normalized such that poles lie strictly inside the unit circle ($|p| < 0.95$) for numerical stability.
- **Primary Source / Attribution:** Oppenheim & Schafer, *Discrete-Time Signal Processing*; SciPy `scipy.signal.sosfilt` (BSD-3-Clause). Independent implementation in pure JAX without external imports.
- **Closest Existing Body & Key Structural Difference:** Differs from `stateful_scalar_iir` (which is a single scalar 2nd-order IIR). Here, $S$ stages are connected in series within each iteration: stage $s$ consumes output $y_{s-1}$, updating its internal delay registers and producing $y_s$.
- **Reference Check / Invariant:** Stable bounded output ($||y||_\infty < M$); exact match to SciPy `scipy.signal.sosfilt(sos, x)`.

#### 2. `sig_goertzel`
- **Mechanism & Family:** Goertzel Tone Detection Second-Order Resonator (`signal_spectral_detection`)
- **Loop Carry & State:** `(s_prev, s_prev2)`: two float32 scalar delay states.
- **Input Arrays & Generation:** Input signal `x`: `(trip,)` float32 synthesized tone with additive Gaussian noise (seed 583_802); target frequency bin coefficient `coeff = 2.0 * jnp.cos(2.0 * jnp.pi * k / N)`.
- **Primary Source / Attribution:** Goertzel, G. (1958), "An Algorithm for the Evaluation of Finite Trigonometric Series", *The American Mathematical Monthly*.
- **Closest Existing Body & Key Structural Difference:** Differs from `scalar_recurrence`. Goertzel executes a real second-order recurrence $s = x_t + \text{coeff} \cdot s_{\text{prev}} - s_{\text{prev2}}$, followed by an end-of-loop power extraction $P = s_1^2 + s_2^2 - \text{coeff} \cdot s_1 s_2$.
- **Reference Check / Invariant:** Evaluated power $P$ matches the discrete Fourier transform magnitude squared $|X[k]|^2 = |\sum_{n=0}^{N-1} x[n] e^{-j 2\pi k n / N}|^2$ to within floating-point tolerance.

#### 3. `sig_lms_adaptive`
- **Mechanism & Family:** Normalized Least Mean Squares (NLMS) Adaptive Filter (`signal_adaptive_filter`)
- **Loop Carry & State:** `(weights, history)`: `weights` float32 array `(taps,)`, `history` ring buffer `(taps,)`.
- **Input Arrays & Generation:** Reference signal `ref`: `(trip,)` AR(1) process; desired signal `desired`: `(trip,)` filtered reference plus independent noise (seed 583_803); step size $\mu = 0.05$.
- **Primary Source / Attribution:** Widrow & Hoff (1960), "Adaptive switching circuits"; Haykin, *Adaptive Filter Theory*.
- **Closest Existing Body & Key Structural Difference:** Differs from `rolling_conv1d_fir`. In FIR convolution, weights are static constants. Here, weights mutate at every single timestep based on instantaneous prediction error $e_t = d_t - w_t^T x_t$: $w_{t+1} = w_t + \frac{\mu}{\epsilon + ||x_t||^2} e_t x_t$.
- **Reference Check / Invariant:** Mean squared error $e_t^2$ is non-increasing over stationary intervals; weights converge toward optimal Wiener filter solution.

#### 4. `sig_polyphase_resample`
- **Mechanism & Family:** Polyphase FIR Fractional Rate Interpolator (`signal_multirate`)
- **Loop Carry & State:** `(history, phase_idx)`: `history` buffer `(taps_per_phase,)` float32, `phase_idx` scalar int32 tracking commutator rotation $0 \le \phi < M$.
- **Input Arrays & Generation:** Input stream `x`: `(trip,)` float32; polyphase subfilter bank `filters`: `(num_phases, taps_per_phase)` float32 sinc-windowed filter (seed 583_804).
- **Primary Source / Attribution:** Crochiere & Rabiner (1983), *Multirate Digital Signal Processing*; SciPy `scipy.signal.resample_poly` (BSD-3-Clause).
- **Closest Existing Body & Key Structural Difference:** Differs from `rolling_conv1d_fir`. The active filter kernel index cycles dynamically ($\phi_{t+1} = (\phi_t + L) \pmod M$), resulting in periodic memory gather patterns across the filter bank.
- **Reference Check / Invariant:** Output preserves low-frequency passband energy without aliasing distortion; linear phase delay verified.

#### 5. `sig_hilbert_analytic`
- **Mechanism & Family:** Dual Allpass Half-Band Analytic Signal Transformer (`signal_quadrature`)
- **Loop Carry & State:** `(state_path0, state_path1)`: two float32 arrays `(order,)` representing states of two parallel allpass filter branches.
- **Input Arrays & Generation:** Input stream `x`: `(trip,)` float32 wideband signal (seed 583_805); allpass pole coefficients `a0, a1`: `(order,)` designed to maintain a 90-degree relative phase difference.
- **Primary Source / Attribution:** Vaidyanathan (1993), *Multirate Systems and Filter Banks*, Prentice Hall.
- **Closest Existing Body & Key Structural Difference:** Differs from `stateful_vector_iir`. Dual parallel branch structure where the real part is delayed and the imaginary part is phase-shifted by $\pi/2$, reconstructing an analytic signal $s(t) = x(t) + j \hat{x}(t)$.
- **Reference Check / Invariant:** Quadrature orthogonality: cross-correlation between in-phase and quadrature branches at lag zero vanishes; energy conservation $||y_0||^2 + ||y_1||^2 \approx ||x||^2$.

#### 6. `sig_pll_carrier`
- **Mechanism & Family:** Costas Loop Phase-Locked Carrier Tracker (`signal_tracking_loop`)
- **Loop Carry & State:** `(phase_est, freq_est, integrator)`: three float32 scalars tracking instantaneous carrier phase, frequency error, and loop filter integral state.
- **Input Arrays & Generation:** Baseband input `iq`: `(trip, 2)` representing in-phase and quadrature samples of a BPSK/QPSK carrier with frequency offset $\Delta \omega$ (seed 583_806); loop filter gains $\alpha=0.02, \beta=0.001$.
- **Primary Source / Attribution:** Costas, J. P. (1956), "Synchronous Communications", *Proc. IRE*; Gardner, *Phaselock Techniques*.
- **Closest Existing Body & Key Structural Difference:** Differs from `control_transcendental`. Contains coupled non-linear phase error detector $e = I_t \cdot Q_t = \text{rot}(iq_t, -\theta_t)$, feeding a PI loop filter that adjusts carrier phase.
- **Reference Check / Invariant:** Phase error variance $\text{Var}(\hat{\theta} - \theta_{\text{true}})$ strictly contracts into locked state.

#### 7. `sig_overlap_add_step`
- **Mechanism & Family:** Overlap-Add Spectral Framing Buffer (`signal_synthesis_buffer`)
- **Loop Carry & State:** `overlap_buffer`: float32 array `(hop_size,)` storing unconsumed overlapping samples from prior frames.
- **Input Arrays & Generation:** Frame spectrum `frames`: `(trip, frame_size)` float32 representing time-domain windowed IFFT blocks (seed 583_807); synthesis window `window`: `(frame_size,)` satisfying COLA (constant overlap-add).
- **Primary Source / Attribution:** Allen, J. B. (1977), "Short Time Spectral Analysis, Synthesis, and Modification by Discrete Fourier Transform", *IEEE Trans. ASSP*.
- **Closest Existing Body & Key Structural Difference:** Differs from `window_replace`. In `window_replace`, memory is overwritten. In overlap-add, overlapping boundary segments are added arithmetically: $y[0..H] = \text{prev}[0..H] + \text{frame}[0..H]$, and trailing samples roll forward.
- **Reference Check / Invariant:** Constant Overlap-Add (COLA) property: flat unity reconstruction for constant input spectra.

---

### Workstream 2: Optimization and Estimation

#### 1. `opt_coordinate_descent`
- **Mechanism & Family:** Cyclic Coordinate Descent with Incremental Residual Maintenance (`opt_coordinate_descent`)
- **Loop Carry & State:** `(coef, residual)`: `coef` float32 array `(P,)`, `residual` float32 array `(N,)`.
- **Input Arrays & Generation:** Design matrix $X$: `(N, P)` normalized columns ($||X_j||_2 = 1$), target vector $y$: `(N,)` (seed 583_808); penalty parameter $\lambda = 0.1$.
- **Primary Source / Attribution:** Friedman, Hastie, Tibshirani (2010), "Regularization Paths for Generalized Linear Models via Coordinate Descent", *JSS*; scikit-learn `ElasticNet` (BSD-3-Clause).
- **Closest Existing Body & Key Structural Difference:** Differs from `dense_matvec`. Does not perform full matrix multiplications; updates only single coordinate $j = i \pmod P$ via soft-thresholding $\mathcal{S}(X_j^T r + \beta_j, \lambda)$, and updates the residual via rank-1 increment $r \leftarrow r - (\beta_j^{\text{new}} - \beta_j^{\text{old}}) X_j$.
- **Reference Check / Invariant:** Lasso objective $\frac{1}{2N} ||r||_2^2 + \lambda ||\beta||_1$ is strictly monotonically non-increasing at every coordinate step.

#### 2. `opt_primal_dual_chambolle`
- **Mechanism & Family:** First-Order Primal-Dual Chambolle-Pock TV Minimizer (`opt_primal_dual`)
- **Loop Carry & State:** `(primal, dual, primal_bar)`: three float32 arrays `(N,)` tracking primal signal, dual gradient vector, and over-relaxed primal variable.
- **Input Arrays & Generation:** Noisy input signal $g$: `(N,)` float32 step-function corrupted by Gaussian noise (seed 583_809); step sizes $\tau = 0.25, \sigma = 0.25, \theta = 1.0$.
- **Primary Source / Attribution:** Chambolle & Pock (2011), "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging", *JMIV*.
- **Closest Existing Body & Key Structural Difference:** Differs from `control_lowrank_update`. Coupled saddle-point updates: dual update with projection onto $L_\infty$ unit ball $\Pi_{[-1, 1]}$, primal proximal update with quadratic data fidelity, and extrapolation $\bar{x} = x_{k+1} + \theta (x_{k+1} - x_k)$.
- **Reference Check / Invariant:** Dual feasibility $||y||_\infty \le 1.0$ holds unconditionally; total variation objective converges.

#### 3. `opt_recursive_least_squares`
- **Mechanism & Family:** Recursive Least Squares (RLS) Sherman-Morrison Covariance Tracking (`opt_recursive_estimation`)
- **Loop Carry & State:** `(w, P_mat)`: parameter vector `w` of shape `(P,)`, inverse correlation matrix `P_mat` of shape `(P, P)`.
- **Input Arrays & Generation:** Regressor stream $x$: `(trip, P)` float32; measurement stream $d$: `(trip,)` float32 (seed 583_810); forgetting factor $\lambda = 0.99$.
- **Primary Source / Attribution:** Ljung (1999), *System Identification: Theory for the User*; Haykin, *Adaptive Filter Theory*.
- **Closest Existing Body & Key Structural Difference:** Differs from `control_lowrank_update`. Implements the full matrix Riccati update: computes gain vector $k_t = \frac{P x_t}{\lambda + x_t^T P x_t}$, updates weights $w \leftarrow w + k_t (d_t - x_t^T w)$, and downdates $P \leftarrow \frac{1}{\lambda} (P - k_t x_t^T P)$.
- **Reference Check / Invariant:** Matrix $P$ remains symmetric positive definite ($x^T P x > 0$); estimates converge to batch OLS $(X^T X)^{-1} X^T y$.

#### 4. `opt_kalman_diagonal`
- **Mechanism & Family:** Diagonalized State-Space Kalman Estimator (`opt_state_estimation`)
- **Loop Carry & State:** `(mean, cov_diag)`: two float32 arrays `(D,)` tracking state estimate and diagonal covariance.
- **Input Arrays & Generation:** Noisy observations $z$: `(trip, D)` (seed 583_811); process noise variance $Q$: `(D,)`, measurement noise variance $R$: `(D,)`, transition factors $F$: `(D,)`.
- **Primary Source / Attribution:** Kalman, R. E. (1960), "A New Approach to Linear Filtering and Prediction Problems", *J. Basic Eng.*
- **Closest Existing Body & Key Structural Difference:** Differs from `stateful_vector_iir`. Incorporates explicit probabilistic two-phase estimation: predict step ($P^- = F P F^T + Q$) and measurement update step ($K = P^- / (P^- + R)$, $x = x^- + K(z - x^-)$, $P = (1 - K)P^-$).
- **Reference Check / Invariant:** Covariance elements contract toward the steady-state discrete algebraic Riccati limit $P_\infty$.

#### 5. `opt_mirror_descent`
- **Mechanism & Family:** Entropic Mirror Descent on Probability Simplex (`opt_mirror_descent`)
- **Loop Carry & State:** `weights`: float32 array `(D,)` constrained to the probability simplex $\Delta^{D-1}$.
- **Input Arrays & Generation:** Streaming loss gradient vectors $g$: `(trip, D)` bounded in $[-1, 1]$ (seed 583_812); learning rate $\eta = 0.05$.
- **Primary Source / Attribution:** Beck & Teboulle (2003), "Mirror descent and nonlinear projected subgradient methods for convex optimization", *Operations Research Letters*.
- **Closest Existing Body & Key Structural Difference:** Differs from `streaming_running_mean`. Multiplicative exponential weight update $w_{t+1} \propto w_t \exp(-\eta g_t)$ followed by exact log-sum-exp normalization to project back onto the probability simplex.
- **Reference Check / Invariant:** Simplex invariant: $\sum_{i=1}^D w_i = 1.0$ and $w_i > 0$ for all iterations.

#### 6. `opt_anderson_acceleration`
- **Mechanism & Family:** Anderson Acceleration Fixed-Point Depth-2 Extrapolator (`opt_acceleration`)
- **Loop Carry & State:** `(x_curr, x_prev, res_curr, res_prev)`: four float32 arrays `(D,)` maintaining current/previous iterates and residual vectors $g(x) = T(x) - x$.
- **Input Arrays & Generation:** Fixed-point operator parameters: symmetric matrix $A$: `(D, D)` with spectral radius $\rho(A) < 1.0$, bias vector $b$: `(D,)` (seed 583_813).
- **Primary Source / Attribution:** Walker & Ni (2011), "Anderson Acceleration for Fixed-Point Iterations", *SIAM J. Numer. Anal.*
- **Closest Existing Body & Key Structural Difference:** Differs from `jacobi_solver`. In standard Picard iteration, $x_{k+1} = T(x_k)$. Anderson acceleration computes optimal mixing parameter $\alpha = \frac{\langle \Delta r, r_k \rangle}{||\Delta r||^2}$ and extrapolates $x_{k+1} = (1 - \alpha) T(x_k) + \alpha T(x_{k-1})$.
- **Reference Check / Invariant:** Residual norm $||T(x) - x||_2$ converges faster than unaccelerated linear iteration.

#### 7. `opt_projected_subgradient_box`
- **Mechanism & Family:** Polyak Projected Subgradient Descent with Box Constraints (`opt_constrained_descent`)
- **Loop Carry & State:** `(x, best_x, best_val)`: current iterate `x` `(D,)`, best-iterate `best_x` `(D,)`, minimum function value `best_val` scalar float32.
- **Input Arrays & Generation:** Subgradient stream $g$: `(trip, D)` (seed 583_814); box constraint limits `lower = -1.0`, `upper = 1.0`.
- **Primary Source / Attribution:** Shor, N. Z. (1985), *Minimization Methods for Non-Differentiable Functions*, Springer.
- **Closest Existing Body & Key Structural Difference:** Differs from `control_bisection_bracket`. Multi-dimensional projection operator $\text{clip}(x - \gamma g, l, u)$ coupled with running optimal value tracking.
- **Reference Check / Invariant:** Box feasibility: $l_j \le x_j \le u_j$ strictly satisfied for all $j \in [0, D)$ across all timesteps.

---

### Workstream 3: Sparse Linear Algebra

#### 1. `sp_csr_matvec_row`
- **Mechanism & Family:** Segmented CSR Sparse Matrix-Vector Product (`sparse_linear_algebra`)
- **Loop Carry & State:** `y`: float32 array `(num_rows,)` accumulating output row dot products.
- **Input Arrays & Generation:** Sparse matrix in CSR format: `row_ptr`: `(num_rows + 1,)` int32, `col_indices`: `(nnz,)` int32, `values`: `(nnz,)` float32; dense input vector $x$: `(num_cols,)` float32 (seed 583_815).
- **Primary Source / Attribution:** Saad, Y. (2003), *Iterative Methods for Sparse Linear Systems*, SIAM; SciPy `scipy.sparse.csr_matrix` (BSD-3-Clause).
- **Closest Existing Body & Key Structural Difference:** Differs from `gather_lookup`. Iteration index $i$ corresponds to matrix row $i$. Slices variable-length column/value segments $[row\_ptr[i], row\_ptr[i+1])$ and accumulates dynamic inner products.
- **Reference Check / Invariant:** Output $y$ exactly matches dense matrix multiplication $A_{\text{dense}} x$.

#### 2. `sp_triangular_solve`
- **Mechanism & Family:** Forward Substitution Sparse Unit Lower Triangular Solver (`sparse_triangular_solve`)
- **Loop Carry & State:** `x`: float32 array `(N,)` storing solved variables.
- **Input Arrays & Generation:** Lower triangular CSR matrix `row_ptr`: `(N+1,)`, `col_indices`: `(nnz,)`, `values`: `(nnz,)` with strictly subdiagonal entries ($col < row$); right-hand side $b$: `(N,)` float32 (seed 583_816).
- **Primary Source / Attribution:** Golub & Van Loan (2013), *Matrix Computations*, Johns Hopkins.
- **Closest Existing Body & Key Structural Difference:** Differs from `sp_csr_matvec_row`. Causal recurrent data dependency: solving for $x_i = b_i - \sum_{j < i} L_{ij} x_j$ requires previously computed elements $x_0, \dots, x_{i-1}$ from earlier loop iterations.
- **Reference Check / Invariant:** Residual check: $||L x - b||_\infty < 10^{-4}$.

#### 3. `sp_bcr_relaxation`
- **Mechanism & Family:** Block-Coordinate Relaxation (Block Gauss-Seidel) (`sparse_block_solver`)
- **Loop Carry & State:** `x_blocks`: float32 array `(num_blocks, block_size)` storing partitioned block unknowns.
- **Input Arrays & Generation:** Tridiagonal block matrix: inverse diagonal blocks `D_inv`: `(num_blocks, block_size, block_size)`, off-diagonal coupling blocks `A_off`: `(num_blocks - 1, block_size, block_size)`, rhs blocks `b`: `(num_blocks, block_size)` (seed 583_817).
- **Primary Source / Attribution:** Hackbusch, W. (1994), *Iterative Solution of Large Sparse Systems of Equations*, Springer.
- **Closest Existing Body & Key Structural Difference:** Differs from `dense_matvec`. Sweeps across block coordinates, solving $x_k \leftarrow D_k^{-1} (b_k - A_{k, k-1} x_{k-1} - A_{k, k+1} x_{k+1})$ in an alternating forward-backward relaxation pass.
- **Reference Check / Invariant:** Monotonic decay of block residual norm $||A x - b||_2$.

#### 4. `sp_residual_propagation`
- **Mechanism & Family:** Sparse Graph Residual Mass Diffusion (`sparse_diffusion`)
- **Loop Carry & State:** `(residual, accumulated)`: two float32 arrays `(num_nodes,)` holding active residual mass and finalized page scores.
- **Input Arrays & Generation:** Directed edge list `(src, dst)`: `(trip, 2)` int32; edge transition probabilities `weights`: `(trip,)` float32; push fraction $\alpha = 0.85$ (seed 583_818).
- **Primary Source / Attribution:** Page et al. (1999), "The PageRank Citation Ranking: Bringing Order to the Web", Stanford InfoLab.
- **Closest Existing Body & Key Structural Difference:** Differs from `scatter_collision_add`. As residual mass is pushed along edge $u \to v$, source node $u$ decreases its residue while destination node $v$ increases its residue, conserving total mass.
- **Reference Check / Invariant:** Mass conservation: $\sum_{u} (\text{residual}[u] + \text{accumulated}[u]) = \text{total\_initial\_mass}$ up to numerical precision.

#### 5. `sp_ellpack_gemv`
- **Mechanism & Family:** ELLPACK Formatted Fixed-Width Sparse Matvec (`sparse_ellpack_format`)
- **Loop Carry & State:** `y`: float32 array `(nrows,)` holding output vector.
- **Input Arrays & Generation:** Column index table `indices`: `(nrows, K)` int32; coefficient table `data`: `(nrows, K)` float32 (where $K$ is the maximum nonzeros per row); dense vector $x$: `(ncols,)` float32 (seed 583_819).
- **Primary Source / Attribution:** Grimes, Kincaid, Young (1979), "ITPACK 2.0 User's Guide", Univ. Texas at Austin; cuSPARSE ELLPACK documentation.
- **Closest Existing Body & Key Structural Difference:** Differs from `gather_row_pool`. Regular 2D memory access pattern across padded sparse columns with sentinel zero-padding, avoiding pointer indirections of CSR.
- **Reference Check / Invariant:** Identical output to full dense dot product $A_{\text{dense}} x$.

#### 6. `sp_ilu0_precond_apply`
- **Mechanism & Family:** Incomplete LU Zero-Fill Preconditioner Sweep (`sparse_preconditioning`)
- **Loop Carry & State:** `y`: float32 array `(N,)` storing intermediate solution $U^{-1} L^{-1} r$.
- **Input Arrays & Generation:** Factored ILU(0) matrix representation `(L_csr, U_csr)` with matching sparsity pattern of $A$; residual vector $r$: `(N,)` float32 (seed 583_820).
- **Primary Source / Attribution:** Saad, Y. (2003), *Iterative Methods for Sparse Linear Systems*, SIAM.
- **Closest Existing Body & Key Structural Difference:** Differs from `sp_triangular_solve`. Executes a compound forward solve ($L y = r$) followed by a backward solve ($U z = y$) across static CSR sparsity patterns.
- **Reference Check / Invariant:** Exact identity when applied to $L U$ product vectors ($M^{-1} (L U v) = v$).

#### 7. `sp_jacobi_sparse_row`
- **Mechanism & Family:** Sparse Jacobi Iterative Relaxation Step (`sparse_jacobi_solver`)
- **Loop Carry & State:** `x`: float32 array `(N,)` representing iterative field estimate.
- **Input Arrays & Generation:** Diagonal reciprocal vector `diag_inv`: `(N,)` float32; strictly off-diagonal CSR sparsity structure `(row_ptr, col_indices, values)`; rhs vector $b$: `(N,)` float32 (seed 583_821).
- **Primary Source / Attribution:** Demmel, J. W. (1997), *Applied Numerical Linear Algebra*, SIAM.
- **Closest Existing Body & Key Structural Difference:** Differs from `jacobi_solver`. `jacobi_solver` assumes a structured 2D Cartesian 5-point grid. `sp_jacobi_sparse_row` executes Jacobi relaxation on arbitrary irregular sparse graphs.
- **Reference Check / Invariant:** Contraction mapping: error $||x^{(k)} - x^*||$ strictly decreases for strictly diagonally dominant matrices.

---

### Workstream 4: Graph and Mesh

#### 1. `graph_edge_message_passing`
- **Mechanism & Family:** Edge-to-Node Message Passing Neural Aggregation (`graph_neural_network`)
- **Loop Carry & State:** `node_features`: float32 array `(num_nodes, D)` representing latent vertex features.
- **Input Arrays & Generation:** Edge index lists `senders`: `(num_edges,)` int32, `receivers`: `(num_edges,)` int32; edge attribute vectors `edge_attr`: `(num_edges, E)` float32; linear projection matrix $W$: `(2D + E, D)` (seed 583_822).
- **Primary Source / Attribution:** Battaglia et al. (2018), "Relational inductive biases, deep learning, and graph networks", arXiv:1806.01261; Jraph (Apache-2.0).
- **Closest Existing Body & Key Structural Difference:** Differs from `scatter_collision_add`. In `scatter_collision_add`, raw values are summed. Here, edge messages are formed by gathering sender and receiver states, concatenating with edge attributes, evaluating a linear transformation, and scatter-reducing to target nodes.
- **Reference Check / Invariant:** Permutation equivariance: relabeling vertex indices produces identically permuted feature outputs.

#### 2. `graph_frontier_relaxation`
- **Mechanism & Family:** Fixed-Capacity Bellman-Ford Shortest-Path Relaxation (`graph_shortest_path`)
- **Loop Carry & State:** `(distances, updated_mask)`: `distances` float32 array `(num_nodes,)` initialized to $\infty$ (except source node at 0), `updated_mask` bool array `(num_nodes,)`.
- **Input Arrays & Generation:** Directed edge endpoints `(u, v)`: `(num_edges, 2)` int32; edge weights `w`: `(num_edges,)` positive float32 (seed 583_823).
- **Primary Source / Attribution:** Bellman, R. (1958), "On a routing problem", *Quarterly of Applied Mathematics*; NetworkX (BSD-3-Clause).
- **Closest Existing Body & Key Structural Difference:** Differs from `dep_pointer_chase`. Vectorized min-plus tropical semiring relaxation: $d[v] = \min(d[v], d[u] + w_{uv})$ across edge batches.
- **Reference Check / Invariant:** Triangle inequality: upon convergence, $d[v] \le d[u] + w_{uv}$ for all edges $(u, v) \in E$.

#### 3. `graph_greedy_coloring`
- **Mechanism & Family:** Parallel Greedy Graph Vertex Coloring Conflict Resolution (`graph_coloring`)
- **Loop Carry & State:** `(colors, conflicts)`: `colors` int32 array `(num_nodes,)` (initialized to -1), `conflicts` scalar int32 tracking unresolved conflicting edges.
- **Input Arrays & Generation:** Symmetric edge list `edges`: `(num_edges, 2)` int32; random node priorities `priority`: `(num_nodes,)` float32 (seed 583_824).
- **Primary Source / Attribution:** Jones & Plassmann (1993), "A parallel graph coloring heuristic", *SIAM J. Sci. Comput.*
- **Closest Existing Body & Key Structural Difference:** Differs from `scatter_min_update`. Uses random priority numbers to select independent sets; nodes with higher priority than all neighbors choose minimum available color and freeze.
- **Reference Check / Invariant:** Validity invariant: for all edges $(u, v)$, if both endpoints are colored, $color[u] \neq color[v]$.

#### 4. `graph_mesh_flux_assembly`
- **Mechanism & Family:** Unstructured Finite-Volume Mesh Edge Flux Assembly (`mesh_finite_volume`)
- **Loop Carry & State:** `cell_conserved`: float32 array `(num_cells, 3)` storing conserved mass, x-momentum, and y-momentum.
- **Input Arrays & Generation:** Mesh topology `face_cells`: `(num_faces, 2)` int32 mapping face to left and right cell neighbors; geometric face normal vectors `face_normals`: `(num_faces, 2)` float32 and lengths `face_lengths`: `(num_faces,)` (seed 583_825).
- **Primary Source / Attribution:** Toro, E. F. (2009), *Riemann Solvers and Numerical Methods for Fluid Dynamics*, Springer; PyFR (BSD-3-Clause).
- **Closest Existing Body & Key Structural Difference:** Differs from `stencil_five_point_2d`. Operates on unstructured polygonal mesh topologies: computes numerical flux across face $e$ and simultaneously adds $+F_e$ to left cell and $-F_e$ to right cell.
- **Reference Check / Invariant:** Global conservation: sum of conserved variables across all cells is strictly preserved due to telescoping face flux cancellation.

#### 5. `graph_pagerank_iter`
- **Mechanism & Family:** PageRank Power Iteration with Dangling Node Redistribution (`graph_pagerank`)
- **Loop Carry & State:** `ranks`: float32 array `(num_nodes,)` representing vertex probability distribution.
- **Input Arrays & Generation:** Column-stochastic graph matrix in CSR format `(row_ptr, col_indices, weights)`; dangling node mask `is_dangling`: `(num_nodes,)` bool; damping factor $\alpha = 0.85$ (seed 583_826).
- **Primary Source / Attribution:** Page et al. (1999), "The PageRank Citation Ranking: Bringing Order to the Web", Stanford InfoLab; SciPy `scipy.sparse.csgraph`.
- **Closest Existing Body & Key Structural Difference:** Differs from `dense_matvec`. Combines sparse link propagation with uniform distribution of mass leaked by dangling nodes: $p_{k+1} = \alpha (A p_k + \frac{m_{\text{dangling}}}{N}) + \frac{1-\alpha}{N} \mathbf{1}$.
- **Reference Check / Invariant:** Stochasticity invariant: $\sum_{i=1}^N \text{ranks}[i] = 1.0 \pm 10^{-5}$ at every iteration.

#### 6. `graph_label_propagation`
- **Mechanism & Family:** Semi-Supervised Edge Affinity Label Propagation (`graph_label_propagation`)
- **Loop Carry & State:** `soft_labels`: float32 array `(num_nodes, num_classes)` holding class probability distributions.
- **Input Arrays & Generation:** Edge list `edges`: `(num_edges, 2)` int32; edge affinities `affinities`: `(num_edges,)` float32; seed label mask `is_seed`: `(num_nodes,)` bool (seed 583_827).
- **Primary Source / Attribution:** Zhu & Ghahramani (2002), "Learning from Labeled and Unlabeled Data with Label Propagation", *CMU Tech Report*.
- **Closest Existing Body & Key Structural Difference:** Differs from `graph_edge_message_passing`. Incorporates boundary clamping: after diffusing continuous probabilities along edges, seed vertices are clamped back to ground-truth one-hot vectors.
- **Reference Check / Invariant:** Clamped seeds maintain ground truth distributions ($soft\_labels[seed] == ground\_truth$) exactly.

#### 7. `graph_degree_normalized_conv`
- **Mechanism & Family:** Kipf-Welling GCN Symmetric Normalized Graph Convolution (`graph_convolution`)
- **Loop Carry & State:** `H_out`: float32 array `(num_nodes, out_dim)` holding filtered feature map.
- **Input Arrays & Generation:** Adjacency edges `(src, dst)`: `(num_edges, 2)` int32 with self-loops; precomputed symmetric normalization coefficients $c_{ij} = (d_i d_j)^{-1/2}$: `(num_edges,)` float32; input node features $X$: `(num_nodes, in_dim)` float32; weight matrix $W$: `(in_dim, out_dim)` (seed 583_828).
- **Primary Source / Attribution:** Kipf & Welling (2017), "Semi-Supervised Classification with Graph Convolutional Networks", *ICLR*; PyTorch Geometric (MIT).
- **Closest Existing Body & Key Structural Difference:** Differs from `graph_edge_message_passing`. Implements the standard spectral GCN formulation: feature transform $X W$, symmetric edge scaling, scatter-add accumulation, and bias addition with ReLU activation.
- **Reference Check / Invariant:** Matches reference matrix calculation $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X W$.

---

### Workstream 5: Dynamic Programming

#### 1. `dp_banded_smith_waterman`
- **Mechanism & Family:** Banded Smith-Waterman Sequence Alignment Wavefront (`dp_sequence_alignment`)
- **Loop Carry & State:** `(diag_prev, diag_curr)`: two float32 arrays `(band_width,)` storing diagonal wavefront score profiles.
- **Input Arrays & Generation:** Query/target similarity scores along diagonal strip `scores`: `(trip, band_width)` float32 (seed 583_829); gap open penalty $g_o = -2.0$, gap extend penalty $g_e = -0.5$.
- **Primary Source / Attribution:** Smith & Waterman (1981), "Identification of Common Molecular Subsequences", *J. Mol. Biol.*; Durbin et al., *Biological Sequence Analysis*.
- **Closest Existing Body & Key Structural Difference:** Differs from `rolling_extreme_max`. 2D wavefront antidiagonal recurrence where cell $(i, j)$ depends on $(i-1, j-1)$, $(i-1, j)$, and $(i, j-1)$, bounded below by zero: $H_{i, j} = \max(0, H_{i-1, j-1} + S_{i, j}, H_{i-1, j} + g, H_{i, j-1} + g)$.
- **Reference Check / Invariant:** Non-negativity invariant ($H_{i, j} \ge 0$); score matches unbanded alignment when band covers the optimal alignment path.

#### 2. `dp_viterbi_max_plus`
- **Mechanism & Family:** Viterbi Trellis Decoding in Max-Plus Semiring (`dp_viterbi_trellis`)
- **Loop Carry & State:** `log_probs`: float32 array `(num_states,)` holding maximum log-likelihood of paths reaching each hidden state.
- **Input Arrays & Generation:** State transition log-probabilities `log_trans`: `(num_states, num_states)` float32; emission log-probability stream `log_emissions`: `(trip, num_states)` float32 (seed 583_830).
- **Primary Source / Attribution:** Viterbi, A. J. (1967), "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm", *IEEE Trans. Inf. Theory*; Rabiner, L. R. (1989), "A tutorial on hidden Markov models".
- **Closest Existing Body & Key Structural Difference:** Differs from `streaming_logsumexp`. `streaming_logsumexp` executes sum-product algebra ($\log \sum \exp$). Viterbi executes max-plus semiring algebra: $V_{t+1}(j) = \max_k (V_t(k) + \log A_{k, j}) + \log B_j(x_t)$.
- **Reference Check / Invariant:** Monotonic log-likelihood bounds: $V_{t+1}(j) \le \max_k V_t(k) + \max \log B$.

#### 3. `dp_edit_distance_row`
- **Mechanism & Family:** Rolling Wagner-Fischer Levenshtein Edit Distance (`dp_edit_distance`)
- **Loop Carry & State:** `dp_row`: int32 array `(target_len + 1,)` representing the active row of the edit distance dynamic programming matrix.
- **Input Arrays & Generation:** Query token stream `query`: `(trip,)` int32; target token sequence `target`: `(target_len,)` int32 (seed 583_831).
- **Primary Source / Attribution:** Wagner & Fischer (1974), "The String-to-String Correction Problem", *J. ACM*.
- **Closest Existing Body & Key Structural Difference:** Differs from `rolling_mean_shift`. Maintains a 1D DP wavefront: in each step, updates row entries sequentially using diagonal register `prev_diag`, evaluating $\min(dp[j] + 1, dp[j-1] + 1, prev\_diag + \text{cost})$.
- **Reference Check / Invariant:** Metric properties: $D(s, t) = 0 \iff s = t$, and triangle inequality $D(a, c) \le D(a, b) + D(b, c)$.

#### 4. `dp_knapsack_bounded`
- **Mechanism & Family:** Bounded 0/1 Knapsack Dynamic Programming Row Transition (`dp_knapsack`)
- **Loop Carry & State:** `dp_table`: float32 array `(capacity + 1,)` tracking maximum value achievable for each weight capacity.
- **Input Arrays & Generation:** Item values `values`: `(trip,)` float32; item weights `weights`: `(trip,)` int32 in $[1, 20]$ (seed 583_832); maximum bag capacity $W = 128$.
- **Primary Source / Attribution:** Bellman, R. (1957), *Dynamic Programming*, Princeton University Press; CLRS Chapter 16.
- **Closest Existing Body & Key Structural Difference:** Differs from `window_replace`. Executes a reverse-capacity loop sweep over $w \in [W, \text{weight}_i]$: $dp[w] = \max(dp[w], dp[w - \text{weight}_i] + \text{value}_i)$ to enforce that each item is consumed at most once.
- **Reference Check / Invariant:** Monotonicity: $dp[w] \le dp[w+1]$ for all capacities $w \in [0, W)$; optimal value matches brute-force search on small configurations.

#### 5. `dp_longest_common_subsequence`
- **Mechanism & Family:** Rolling Longest Common Subsequence Length Vector (`dp_lcs`)
- **Loop Carry & State:** `curr_row`: int32 array `(seq2_len + 1,)` holding LCS values.
- **Input Arrays & Generation:** Character stream 1 `seq1`: `(trip,)` int32; reference sequence `seq2`: `(seq2_len,)` int32 (seed 583_833).
- **Primary Source / Attribution:** Hirschberg, D. S. (1975), "A linear space algorithm for computing maximal common subsequences", *Comm. ACM*.
- **Closest Existing Body & Key Structural Difference:** Differs from `dp_edit_distance_row`. Pure matching logic: increments diagonal value when characters match, else takes maximum of horizontal and vertical neighbors without substitution cost.
- **Reference Check / Invariant:** Bounded length invariant: $0 \le LCS(s_1, s_2) \le \min(|s_1|, |s_2|)$.

#### 6. `dp_matrix_chain_tile`
- **Mechanism & Family:** Matrix Chain Multiplication Optimal Substructure Tile Sweep (`dp_matrix_chain`)
- **Loop Carry & State:** `cost_matrix`: float32 array `(N, N)` holding minimum multiplication cost for subchains $A_i \dots A_j$.
- **Input Arrays & Generation:** Matrix dimension sequence `dims`: `(trip + 1,)` int32 (seed 583_834).
- **Primary Source / Attribution:** Cormen, Leiserson, Rivest, Stein (2009), *Introduction to Algorithms (CLRS)*, MIT Press.
- **Closest Existing Body & Key Structural Difference:** Differs from `dp_banded_smith_waterman`. Subproblem length sweep where cell $(i, j)$ minimizes over all intermediate split points $k$: $\min_{i \le k < j} (m[i, k] + m[k+1, j] + p_{i-1} p_k p_j)$.
- **Reference Check / Invariant:** Zero cost on diagonal ($m[i, i] = 0$); matches CLRS reference tables.

#### 7. `dp_dtw_banded_step`
- **Mechanism & Family:** Banded Dynamic Time Warping (DTW) with Sakoe-Chiba Constraint (`dp_dynamic_time_warping`)
- **Loop Carry & State:** `band_costs`: float32 array `(band_width,)` storing accumulated warping distances across the active band.
- **Input Arrays & Generation:** Time series feature streams $X$: `(trip, D)` float32, $Y$: `(trip, D)` float32 (seed 583_835); Sakoe-Chiba band width $W = 16$.
- **Primary Source / Attribution:** Sakoe & Chiba (1978), "Dynamic programming algorithm optimization for spoken word recognition", *IEEE Trans. ASSP*; `fastdtw`.
- **Closest Existing Body & Key Structural Difference:** Differs from `dp_banded_smith_waterman`. Computes local Euclidean feature distances and evaluates 3-way predecessor minimum: $D(i, j) = ||x_i - y_j||_2 + \min(D(i-1, j), D(i, j-1), D(i-1, j-1))$.
- **Reference Check / Invariant:** Symmetry: $DTW(X, Y) = DTW(Y, X)$; identity: $DTW(X, X) = 0$.

---

### Workstream 6: Numerical Simulation

#### 1. `sim_verlet_symplectic`
- **Mechanism & Family:** Velocity Verlet / Leapfrog Symplectic N-Body Integrator (`sim_molecular_dynamics`)
- **Loop Carry & State:** `(pos, vel, force)`: `pos` float32 array `(N, 3)`, `vel` float32 array `(N, 3)`, `force` float32 array `(N, 3)`.
- **Input Arrays & Generation:** Particle masses `mass`: `(N,)` float32; Lennard-Jones potential parameters $\epsilon, \sigma$; integration time step $dt = 0.001$ (seed 583_836).
- **Primary Source / Attribution:** Verlet, L. (1967), "Computer 'Experiments' on Classical Fluids", *Phys. Rev.*; JAX MD (Apache-2.0).
- **Closest Existing Body & Key Structural Difference:** Differs from `stateful_coupled_oscillator`. `stateful_coupled_oscillator` implements simple linear Hookean coupling. Velocity Verlet implements a symplectic two-stage integration of non-linear $1/r^6 - 1/r^{12}$ Lennard-Jones inter-particle forces.
- **Reference Check / Invariant:** Symplectic energy conservation: total Hamiltonian $H = T(\text{vel}) + V(\text{pos})$ fluctuates within bounded $O(dt^2)$ energy envelope without secular drift.

#### 2. `sim_reaction_diffusion_1d`
- **Mechanism & Family:** Gray-Scott Coupled Reaction-Diffusion PDE Ring (`sim_reaction_diffusion`)
- **Loop Carry & State:** `(U, V)`: two float32 arrays `(grid_size,)` tracking chemical species concentrations on a 1D spatial periodic domain.
- **Input Arrays & Generation:** Diffusion coefficients $D_u = 0.16, D_v = 0.08$; feed rate $F = 0.035$; kill rate $k = 0.065$; spatial step $dx = 1.0$; time step $dt = 1.0$ (seed 583_837).
- **Primary Source / Attribution:** Pearson, J. E. (1993), "Complex Patterns in a Simple System", *Science*; JAX-CFD.
- **Closest Existing Body & Key Structural Difference:** Differs from `stencil_diffusion_1d_fori`. In `stencil_diffusion_1d_fori`, a single scalar field diffuses linearly. Gray-Scott integrates two coupled non-linear PDE fields: $\partial_t U = D_u \nabla^2 U - U V^2 + F(1 - U)$ and $\partial_t V = D_v \nabla^2 V + U V^2 - (F + k)V$.
- **Reference Check / Invariant:** Positivity: $U(x), V(x) \ge 0$ for all $x$; total integrated mass matches expected source/sink equilibrium.

#### 3. `sim_finite_volume_burgers`
- **Mechanism & Family:** Godunov Upwind Non-Linear Burgers' Shock Capturing Solver (`sim_hyperbolic_conservation`)
- **Loop Carry & State:** `u`: float32 array `(grid_size,)` storing cell-averaged conserved velocity values.
- **Input Arrays & Generation:** Initial wave profile: `(grid_size,)` float32 sine wave; spatial resolution $dx = 0.01$; time step $dt$ chosen to satisfy CFL condition ($CFL = 0.4$) (seed 583_838).
- **Primary Source / Attribution:** Godunov, S. K. (1959), "A Difference Scheme for Numerical Computation of Discontinuous Solutions", *Math. Sbornik*; LeVeque, *Finite Volume Methods for Hyperbolic Problems*.
- **Closest Existing Body & Key Structural Difference:** Differs from `stencil_five_point_2d`. Solves non-linear hyperbolic conservation law $\partial_t u + \partial_x (\frac{1}{2} u^2) = 0$ using an exact Godunov Riemann solver at cell interfaces, capturing shock waves and rarefaction fans.
- **Reference Check / Invariant:** Total momentum conservation: $\int u(x) dx = \sum_i u_i dx$ is strictly constant under periodic boundary conditions.

#### 4. `sim_pbd_cloth_spring`
- **Mechanism & Family:** Position-Based Dynamics (PBD) Distance Constraint Projection (`sim_position_based_dynamics`)
- **Loop Carry & State:** `(positions, prev_positions)`: two float32 arrays `(num_particles, 3)` tracking current and previous particle coordinates.
- **Input Arrays & Generation:** Constraint spring topology `edges`: `(num_edges, 2)` int32; rest lengths `rest_lengths`: `(num_edges,)` float32; external gravity vector $g = [0, -9.81, 0]$; compliance stiffness parameter $k = 0.8$ (seed 583_839).
- **Primary Source / Attribution:** Müller et al. (2007), "Position Based Dynamics", *J. Vis. Commun. Image R.*
- **Closest Existing Body & Key Structural Difference:** Differs from `graph_frontier_relaxation`. Unconditionally stable geometric constraint projection: computes distance error $C(p_1, p_2) = ||p_1 - p_2|| - d_0$ and applies non-linear positional corrections $\Delta p = \pm \frac{k}{2} C \frac{p_1 - p_2}{||p_1 - p_2||}$.
- **Reference Check / Invariant:** Constraint satisfaction: mean constraint violation $\frac{1}{|E|} \sum_e | ||p_u - p_v|| - d_{0, e} |$ strictly decreases after projection iterations.

#### 5. `sim_shallow_water_1d`
- **Mechanism & Family:** 1D Saint-Venant Shallow Water Equations with Lax-Friedrichs Flux (`sim_fluid_dynamics`)
- **Loop Carry & State:** `(height, momentum)`: two float32 arrays `(grid_size,)` tracking water column height $h$ and discharge $h u$.
- **Input Arrays & Generation:** Bottom bathymetry `bed`: `(grid_size,)` float32; gravity $g = 9.81$; spatial step $dx$; time step $dt$ (seed 583_840).
- **Primary Source / Attribution:** Toro, E. F. (2001), *Shock-Capturing Methods for Free-Surface Shallow Flows*, Wiley.
- **Closest Existing Body & Key Structural Difference:** Differs from `sim_finite_volume_burgers`. Coupled 2x2 system of hyperbolic conservation laws with characteristic wave speeds $u \pm \sqrt{g h}$, handling non-conservative bed topography source terms.
- **Reference Check / Invariant:** Well-balanced property: "lake at rest" equilibrium ($h(x) + b(x) = \text{const}, u = 0$) is preserved exactly to machine precision.

#### 6. `sim_heat_crank_nicolson_step`
- **Mechanism & Family:** Crank-Nicolson Implicit-Explicit Thermal Diffusion Stepping (`sim_implicit_pde`)
- **Loop Carry & State:** `u`: float32 array `(N,)` storing temperature distribution across spatial domain.
- **Input Arrays & Generation:** Thermal diffusivity constant $\alpha$; grid step $dx$; tridiagonal Thomas solver pre-factored coefficients `(c_prime, d_prime)` (seed 583_841).
- **Primary Source / Attribution:** Crank & Nicolson (1947), "A Practical Method for Numerical Evaluation of Solutions of P.D.E.", *Proc. Camb. Phil. Soc.*
- **Closest Existing Body & Key Structural Difference:** Differs from `stencil_diffusion_1d_fori`. `stencil_diffusion_1d_fori` uses explicit Euler stepping (which requires $dt \le \frac{dx^2}{2\alpha}$ for stability). Crank-Nicolson is unconditionally stable, solving a linear tridiagonal system on every time step.
- **Reference Check / Invariant:** Energy dissipation: $L_2$ norm $||u||_2$ is strictly monotonically decreasing for zero-flux boundary conditions.

#### 7. `sim_rigid_body_quaternion`
- **Mechanism & Family:** Rigid Body Rotational Kinematics via Unit Quaternion Integration (`sim_rotational_dynamics`)
- **Loop Carry & State:** `(quat, omega)`: `quat` float32 array `(4,)` (unit quaternion), `omega` float32 array `(3,)` (body-frame angular velocity).
- **Input Arrays & Generation:** External torque profile `torques`: `(trip, 3)` float32; principal moments of inertia diagonal `I`: `(3,)` float32; time step $dt = 0.01$ (seed 583_842).
- **Primary Source / Attribution:** Baraff, D. (1997), "An Introduction to Physically Based Modeling: Rigid Body Simulation", *SIGGRAPH Course Notes*.
- **Closest Existing Body & Key Structural Difference:** Differs from `control_transcendental`. Couples Euler's non-linear equations for angular acceleration $I \dot{\omega} + \omega \times (I \omega) = \tau$ with quaternion kinematic derivative $\dot{q} = \frac{1}{2} q \otimes [0, \omega]$ and explicit quaternion normalization.
- **Reference Check / Invariant:** Unit norm invariant: $||q||_2 = 1.0 \pm 10^{-6}$ maintained at all steps.

---

### Workstream 7: Tiled Tensor Workloads

#### 1. `tiled_online_softmax`
- **Mechanism & Family:** Block-Tiled Online Softmax with Rescaling Recurrence (`tensor_tiled_attention`)
- **Loop Carry & State:** `(out_tile, running_max, running_sum)`: `out_tile` float32 array `(tile_dim, d)`, `running_max` float32 array `(tile_dim,)`, `running_sum` float32 array `(tile_dim,)`.
- **Input Arrays & Generation:** Query block $Q$: `(tile_dim, d)` float32; key/value panel stream $K, V$: `(trip, tile_dim, d)` float32 (seed 583_843).
- **Primary Source / Attribution:** Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient Exact Attention", *NeurIPS*; Milakov & Gimelshein (2018), "Online normalizer calculation for softmax".
- **Closest Existing Body & Key Structural Difference:** Differs from `streaming_logsumexp`. `streaming_logsumexp` computes scalar log-sum-exp. `tiled_online_softmax` performs matrix tile operations ($Q K_t^T$), rescales existing accumulated attention outputs when a new block maximum is encountered ($O \leftarrow O \cdot \exp(m_{\text{old}} - m_{\text{new}})$), and accumulates value projections.
- **Reference Check / Invariant:** Exact mathematical equivalence to standard full-context softmax attention: $\text{softmax}(Q K^T) V$.

#### 2. `tiled_gram_accumulate`
- **Mechanism & Family:** Panelwise Batched Gram Matrix Outer-Product Accumulation (`tensor_gram_matrix`)
- **Loop Carry & State:** `Gram`: float32 array `(K, K)` accumulating symmetric covariance matrix.
- **Input Arrays & Generation:** Feature panel stream $X$: `(trip, B, K)` float32 (seed 583_844).
- **Primary Source / Attribution:** Golub & Van Loan (2013), *Matrix Computations*; BLAS `syrk` routine.
- **Closest Existing Body & Key Structural Difference:** Differs from `control_lowrank_update`. `control_lowrank_update` performs rank-1 vector outer products $u v^T$. `tiled_gram_accumulate` performs block matrix multiplications $G \leftarrow G + X_t^T X_t$ across batches of vectors.
- **Reference Check / Invariant:** Symmetry and positive semi-definiteness: $G = G^T$ and all eigenvalues $\lambda_i(G) \ge 0$.

#### 3. `tiled_cholesky_update`
- **Mechanism & Family:** Tiled Rank-k Cholesky Factor Modification (`tensor_cholesky_update`)
- **Loop Carry & State:** `L`: float32 array `(N, N)` lower triangular Cholesky matrix factor.
- **Input Arrays & Generation:** Rank-$k$ update panels $V$: `(trip, N, k)` float32 (seed 583_845).
- **Primary Source / Attribution:** Seeger, M. (2004), "Low Rank Updates to Cholesky Factorizations", *Tech Report*; LINPACK `dchud`.
- **Closest Existing Body & Key Structural Difference:** Differs from `control_lowrank_update`. Directly updates the triangular factor $L$ such that $L_{\text{new}} L_{\text{new}}^T = L L^T + v v^T$ using a sequence of Givens rotations without re-factorizing from scratch.
- **Reference Check / Invariant:** Factorization consistency: $||L_{\text{new}} L_{\text{new}}^T - (A + \sum v_t v_t^T)||_\infty < 10^{-4}$.

#### 4. `tiled_blocked_pairwise_dist`
- **Mechanism & Family:** Blocked Euclidean Pairwise Distance with Running Top-K Nearest Neighbors (`tensor_similarity_search`)
- **Loop Carry & State:** `(topk_dists, topk_indices)`: two float32/int32 arrays `(num_queries, K_nn)` tracking closest database neighbors.
- **Input Arrays & Generation:** Query block $Q$: `(num_queries, D)` float32; database panel stream $X$: `(trip, block_size, D)` float32 (seed 583_846).
- **Primary Source / Attribution:** Johnson et al. (2019), "Billion-scale similarity search with GPUs" (Faiss), *IEEE Trans. Big Data*.
- **Closest Existing Body & Key Structural Difference:** Differs from `scatter_min_update`. Computes dense block distance matrices $||q_i - x_j||^2 = ||q_i||^2 + ||x_j||^2 - 2 q_i^T x_j$ and merges the top-$K$ smallest distances into running sorted candidate buffers.
- **Reference Check / Invariant:** Distances match brute-force full pairwise distance calculations.

#### 5. `tiled_conv2d_im2col_tile`
- **Mechanism & Family:** Block-Tiled 2D Convolution via im2col GEMM Panel Accumulation (`tensor_tiled_convolution`)
- **Loop Carry & State:** `out_tile`: float32 array `(tile_h, tile_w, out_channels)` accumulating spatial convolution output.
- **Input Arrays & Generation:** Input image feature window: `(tile_h + K - 1, tile_w + K - 1, in_channels)` float32; filter kernels `weights`: `(K, K, in_channels, out_channels)` float32 (seed 583_847).
- **Primary Source / Attribution:** Chetlur et al. (2014), "cuDNN: Efficient Primitives for Deep Learning", arXiv:1410.0759.
- **Closest Existing Body & Key Structural Difference:** Differs from `rolling_conv1d_fir`. Operates in 2D multi-channel tensor space: unrolls spatial patches into GEMM panels and accumulates products along channel dimensions.
- **Reference Check / Invariant:** Output matches standard `jax.lax.conv_general_dilated` on identical image windows.

#### 6. `tiled_tensor_contract_chain`
- **Mechanism & Family:** Tiled Tensor Core Contraction Chain (`tensor_train_contraction`)
- **Loop Carry & State:** `core_accum`: float32 array `(R_left, R_right, D)` representing contracted tensor core.
- **Input Arrays & Generation:** Tensor slice stream: `(trip, R_left, R_mid, D)` float32; factor matrix $B$: `(R_mid, R_right)` float32 (seed 583_848).
- **Primary Source / Attribution:** Oseledets, I. V. (2011), "Tensor-Train Decomposition", *SIAM J. Sci. Comput.*; TensorNetwork library (Apache-2.0).
- **Closest Existing Body & Key Structural Difference:** Differs from `dense_matvec`. Evaluates 3-way multi-index tensor contraction $\sum_k A[i, k, d] \cdot B[k, j]$ across intermediate tensor dimensions.
- **Reference Check / Invariant:** Exactly matches NumPy `np.einsum('ikd,kj->ijd', A, B)`.

#### 7. `tiled_matrix_power_trace`
- **Mechanism & Family:** Blocked Krylov Matrix Power Iteration with Rayleigh Trace (`tensor_eigen_subspace`)
- **Loop Carry & State:** `(V_block, rayleigh_trace)`: `V_block` float32 array `(N, K)` (orthonormal subspace basis), `rayleigh_trace` scalar float32.
- **Input Arrays & Generation:** Symmetric matrix $A$: `(N, N)` float32; initial random orthogonal basis $V_0$: `(N, K)` (seed 583_849).
- **Primary Source / Attribution:** Saad, Y. (2011), *Numerical Methods for Large Eigenvalue Problems*, SIAM.
- **Closest Existing Body & Key Structural Difference:** Differs from `opt_anderson_acceleration`. Operates on a $K$-dimensional subspace: evaluates $W = A V$, computes Rayleigh quotient matrix $H = V^T W$, and performs in-loop Gram-Schmidt re-orthonormalization of $V$.
- **Reference Check / Invariant:** Trace of Rayleigh quotient monotonically converges toward the sum of the $K$ largest eigenvalues of $A$.

---

### Workstream 8: Index and Memory Access

#### 1. `idx_permutation_cycle_walk`
- **Mechanism & Family:** Permutation Orbit Cycle Walk with Visited Mask (`index_permutation_orbit`)
- **Loop Carry & State:** `(curr_node, orbit_length, visited_mask)`: `curr_node` scalar int32, `orbit_length` scalar int32, `visited_mask` bool array `(N,)`.
- **Input Arrays & Generation:** Permutation table `perm`: `(N,)` int32 generated via random permutation (seed 583_850); seed query stream: `(trip,)` int32.
- **Primary Source / Attribution:** Knuth, D. E. (1997), *The Art of Computer Programming, Vol 1: Fundamental Algorithms*; NetworkX.
- **Closest Existing Body & Key Structural Difference:** Differs from `dep_pointer_chase`. `dep_pointer_chase` traverses an unconstrained pointer chain. `idx_permutation_cycle_walk` tracks cycle decomposition: detects when the walk re-enters a previously visited vertex and resets cycle metrics.
- **Reference Check / Invariant:** Sum of disjoint cycle lengths over all orbits equals total table size $N$.

#### 2. `idx_double_indirect_gather`
- **Mechanism & Family:** Two-Level B-Tree Columnar Leaf Retrieval (`index_multi_level_gather`)
- **Loop Carry & State:** `accum`: float32 array `(D,)` accumulating retrieved leaf vectors.
- **Input Arrays & Generation:** Data leaf table `leaves`: `(M, D)` float32; pointer dictionary `pointers`: `(P,)` int32; query stream `queries`: `(trip,)` int32 (seed 583_851).
- **Primary Source / Attribution:** Original; models nested pointer indexing in columnar database storage (e.g. Apache Arrow, DuckDB).
- **Closest Existing Body & Key Structural Difference:** Differs from `dep_two_level_indirection`. `dep_two_level_indirection` performs scalar chained indexing. Here, multi-lane vector records are gathered ($leaves[pointers[queries[i]]]$) with out-of-bounds bounds-check masking.
- **Reference Check / Invariant:** Matches reference array indexing $leaves[pointers[queries]]$.

#### 3. `idx_gather_transform_scatter`
- **Mechanism & Family:** Particle-in-Cell (PIC) Gather-Weight-Scatter Pipeline (`index_gather_scatter_pipeline`)
- **Loop Carry & State:** `grid_charge`: float32 array `(grid_size, grid_size)` accumulating field charges.
- **Input Arrays & Generation:** Source particle coordinates `particles`: `(trip, 2)` float32; particle charges `q`: `(trip,)` float32; grid dimensions (seed 583_852).
- **Primary Source / Attribution:** Birdsall & Langdon (1985), *Plasma Physics via Computer Simulation*, McGraw-Hill.
- **Closest Existing Body & Key Structural Difference:** Differs from `scatter_accumulate`. Computes 4-point bilinear interpolation weights based on particle sub-grid coordinates and scatters fractions into adjacent grid cells: $\Delta \rho_{i, j} = q \cdot (1 - \Delta x)(1 - \Delta y)$.
- **Reference Check / Invariant:** Charge conservation: total grid charge $\sum_{i, j} \text{grid\_charge}[i, j]$ matches sum of particle charges $\sum_p q_p$ exactly.

#### 4. `idx_overlapping_block_update`
- **Mechanism & Family:** 2D Overlapping Patch Stitching with Partition of Unity Blending (`index_overlapping_patches`)
- **Loop Carry & State:** `canvas`: float32 array `(H, W)` storing blended composite image.
- **Input Arrays & Generation:** Patch stream `patches`: `(trip, patch_h, patch_w)` float32; patch placement coordinates `coords`: `(trip, 2)` int32; blend weight mask `weight`: `(patch_h, patch_w)` float32 (seed 583_853).
- **Primary Source / Attribution:** Original; represents overlap-save synthesis and patch-based texture synthesis (Efros & Freeman).
- **Closest Existing Body & Key Structural Difference:** Differs from `window_replace`. `window_replace` performs non-overlapping overwrites in 1D. `idx_overlapping_block_update` performs 2D weighted blend accumulation $canvas[x, y] \leftarrow canvas[x, y] + w \cdot \text{patch}$.
- **Reference Check / Invariant:** Partition of unity: uniform patches with valid blend weights produce a flat, uniform composite field.

#### 5. `idx_compressed_sparse_lookup`
- **Mechanism & Family:** In-Loop Binary Search over Sorted Index Arrays (`index_binary_search`)
- **Loop Carry & State:** `(found_sum, total_probes)`: two int32 scalars accumulating match values and binary search probe counts.
- **Input Arrays & Generation:** Sorted key array `keys`: `(N,)` int32; value payload array `values`: `(N,)` float32; search probe stream `queries`: `(trip,)` int32 (seed 583_854).
- **Primary Source / Attribution:** Original; foundational binary search probe routine in columnar storage engines.
- **Closest Existing Body & Key Structural Difference:** Differs from `control_bisection_bracket`. `control_bisection_bracket` performs bisection over continuous mathematical intervals without memory access. This algorithm performs discrete binary search over memory-backed arrays.
- **Reference Check / Invariant:** Returned index matches Python `np.searchsorted(keys, queries)`.

#### 6. `idx_strided_transpose_copy`
- **Mechanism & Family:** Multi-Dimensional Strided Coordinate Remapping (`index_strided_remap`)
- **Loop Carry & State:** `out_tile`: float32 array `(tile_size, tile_size)` holding transposed memory panel.
- **Input Arrays & Generation:** Source matrix $A$: `(H, W)` float32; origin tile coordinates `origins`: `(trip, 2)` int32 (seed 583_855).
- **Primary Source / Attribution:** Original; represents tensor transpose and memory striding kernels.
- **Closest Existing Body & Key Structural Difference:** Differs from `strided_mix`. Involves coordinate transformation: reads matrix panel with striding along axis 0 and writes into memory with transposed layout along axis 1 ($out[j, i] = in[i, j]$).
- **Reference Check / Invariant:** Transposed equality: $out\_tile == in\_tile^T$.

#### 7. `idx_circular_fifo_buffer`
- **Mechanism & Family:** Decoupled Circular FIFO Queue Buffer (`index_circular_queue`)
- **Loop Carry & State:** `(buffer, head_ptr, tail_ptr, count)`: `buffer` float32 array `(capacity, D)`, `head_ptr` scalar int32, `tail_ptr` scalar int32, `count` scalar int32.
- **Input Arrays & Generation:** Enqueue data records `items`: `(trip, D)` float32; operation command stream `cmd`: `(trip,)` int32 (0 for push, 1 for pop) (seed 583_856).
- **Primary Source / Attribution:** Original; foundational data structure in stream processing and operating system ring buffers.
- **Closest Existing Body & Key Structural Difference:** Differs from `ring_feedback`. Decouples read and write pointers with wrap-around modulo indexing and dynamic occupancy checking.
- **Reference Check / Invariant:** FIFO invariant: popped sequence matches pushed sequence in first-in-first-out order; count never exceeds capacity.

---

### Workstream 9: Streaming and Compression

#### 1. `stream_count_min_sketch`
- **Mechanism & Family:** Count-Min Sketch Frequency Estimator (`stream_count_min_sketch`)
- **Loop Carry & State:** `table`: int32 array `(depth, width)` storing frequency counter matrix.
- **Input Arrays & Generation:** Streaming token keys `keys`: `(trip,)` int32; pairwise independent hash seeds `(a, b)`: `(depth, 2)` int32 (seed 583_857).
- **Primary Source / Attribution:** Cormode & Muthukrishnan (2005), "An improved data stream summary: the count-min sketch and its applications", *J. Algorithms*.
- **Closest Existing Body & Key Structural Difference:** Differs from `bucket_max`. Evaluates $D$ independent universal hash functions $h_d(k) = (a_d k + b_d) \pmod W$ per token and increments counters at $(d, h_d(k))$.
- **Reference Check / Invariant:** One-sided error bound: estimated frequency $\hat{f}(k) = \min_d table[d, h_d(k)] \ge f_{\text{true}}(k)$.

#### 2. `stream_run_length_encode`
- **Mechanism & Family:** Streaming Run-Length Encoder with Preallocated Buffer (`stream_rle_compression`)
- **Loop Carry & State:** `(symbols, lengths, out_idx, current_sym, current_len)`: `symbols` int32 array `(capacity,)`, `lengths` int32 array `(capacity,)`, `out_idx` scalar int32, `current_sym` scalar int32, `current_len` scalar int32.
- **Input Arrays & Generation:** Token stream with repeated run distributions `tokens`: `(trip,)` int32 (seed 583_858).
- **Primary Source / Attribution:** Original; standard lossless compression primitive (ITU-T T.4, TIFF).
- **Closest Existing Body & Key Structural Difference:** Differs from `stateful_scalar_iir`. Dynamic run tracking: when token $x_t == current\_sym$, increments count; on symbol change, commits $(symbol, length)$ tuple to output buffer and resets counter.
- **Reference Check / Invariant:** Lossless reconstruction: expanding $(symbol_i, length_i)$ pairs exactly recovers the original input token stream.

#### 3. `stream_bit_pack_u8_to_u32`
- **Mechanism & Family:** Bit-Packing Sub-Byte Quantized Stream Compressor (`stream_bit_packing`)
- **Loop Carry & State:** `(words, accum_bits, bit_count, word_idx)`: `words` uint32 array `(capacity,)`, `accum_bits` scalar uint32, `bit_count` scalar int32, `word_idx` scalar int32.
- **Input Arrays & Generation:** Stream of 4-bit integers `symbols`: `(trip,)` uint32 with values in $[0, 15]$ (seed 583_859).
- **Primary Source / Attribution:** Original; ubiquitous primitive in quantized deep learning inference (e.g. `bitsandbytes`, GPTQ).
- **Closest Existing Body & Key Structural Difference:** Differs from `control_transcendental`. Executes bitwise logic: shifts incoming bits ($\text{accum} \mid (s \ll \text{count})$), flushing to the 32-bit word buffer whenever 32 bits are accumulated.
- **Reference Check / Invariant:** Invertibility: bit-unpacking the 32-bit word stream reproduces the exact input 4-bit symbol stream.

#### 4. `stream_hyperloglog_step`
- **Mechanism & Family:** HyperLogLog Distinct Cardinality Sketch (`stream_cardinality_sketch`)
- **Loop Carry & State:** `registers`: uint8 array `(num_buckets,)` tracking maximum leading zero counts.
- **Input Arrays & Generation:** High-entropy 64-bit integer hash stream `hashes`: `(trip,)` uint32 (seed 583_860); number of register buckets $m = 64$.
- **Primary Source / Attribution:** Flajolet et al. (2007), "HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm", *AOFA*.
- **Closest Existing Body & Key Structural Difference:** Differs from `bucket_max`. Evaluates leading zeros of hash suffix ($\text{clz}(x)$) and updates bucket register $M[b] = \max(M[b], \rho(\text{suffix}))$.
- **Reference Check / Invariant:** Harmonic mean estimate $E = \alpha_m m^2 (\sum_{j=1}^m 2^{-M[j]})^{-1}$ approximates true distinct count within standard error $\frac{1.04}{\sqrt{m}}$.

#### 5. `stream_lzw_bounded_dict`
- **Mechanism & Family:** Bounded LZW Sliding Dictionary Token Matcher (`stream_lzw_dictionary`)
- **Loop Carry & State:** `(dict_keys, dict_prefixes, dict_size, match_prefix)`: dictionary prefix table `(dict_cap,)` int32, character table `(dict_cap,)` int32, `dict_size` scalar int32, `match_prefix` scalar int32.
- **Input Arrays & Generation:** Character byte stream `input_bytes`: `(trip,)` int32 in $[0, 255]$ (seed 583_861).
- **Primary Source / Attribution:** Welch, T. A. (1984), "A Technique for High-Performance Data Compression", *IEEE Computer*; Zlib.
- **Closest Existing Body & Key Structural Difference:** Differs from `dep_linked_list`. Hash-probes prefix-extension table: on dictionary hit, extends prefix; on miss, emits dictionary code, adds new prefix string to dictionary, and resets.
- **Reference Check / Invariant:** Deterministic dictionary growth and prefix chain consistency.

#### 6. `stream_exponential_decay_histogram`
- **Mechanism & Family:** Time-Decayed Continuous Streaming Histogram (`stream_decayed_histogram`)
- **Loop Carry & State:** `(buckets, last_timestamp)`: `buckets` float32 array `(num_bins,)`, `last_timestamp` scalar float32.
- **Input Arrays & Generation:** Arrival timestamp stream `timestamps`: `(trip,)` float32 monotonically increasing; observed value stream `values`: `(trip,)` float32 (seed 583_862); decay half-life $\tau = 10.0$.
- **Primary Source / Attribution:** Original; standard pattern in real-time telemetry systems and streaming quantile monitors.
- **Closest Existing Body & Key Structural Difference:** Differs from `streaming_decayed_extrema`. `streaming_decayed_extrema` decays a scalar min/max. This algorithm decays an entire vector of histogram bins exponentially by $\exp(-\Delta t / \tau)$ before incrementing the bucket corresponding to the incoming value.
- **Reference Check / Invariant:** Integrated weight matches theoretical continuous decay sum $\sum_i \exp(-(T_{\text{final}} - t_i)/\tau)$.

#### 7. `stream_reservoir_sampling`
- **Mechanism & Family:** Streaming Reservoir Sampling with Linear Congruential State (`stream_reservoir_sampling`)
- **Loop Carry & State:** `(reservoir, total_seen, rng_state)`: `reservoir` float32 array `(K, D)`, `total_seen` scalar int32, `rng_state` scalar uint32.
- **Input Arrays & Generation:** Input record stream `items`: `(trip, D)` float32 (seed 583_863); reservoir size $K = 16$.
- **Primary Source / Attribution:** Vitter, J. S. (1985), "Random Sampling with a Reservoir", *ACM Trans. Math. Softw.*
- **Closest Existing Body & Key Structural Difference:** Differs from `window_replace`. In `window_replace`, writes occur at deterministic indices. Reservoir sampling evaluates dynamic replacement probability: item $t$ replaces a random reservoir slot with probability $K / t$.
- **Reference Check / Invariant:** Uniformity invariant: each seen stream item has an identical probability $K / N$ of residing in the reservoir after $N$ timesteps.

---

### Workstream 10: Scheduling and Control Flow

#### 1. `sched_priority_queue_step`
- **Mechanism & Family:** Fixed-Capacity Array Binary Min-Heap Priority Queue (`control_priority_heap`)
- **Loop Carry & State:** `(heap_keys, heap_vals, heap_size)`: `heap_keys` float32 array `(capacity,)`, `heap_vals` int32 array `(capacity,)`, `heap_size` scalar int32.
- **Input Arrays & Generation:** Command stream `op`: `(trip,)` int32 (0 for insert, 1 for extract-min); key stream `keys`: `(trip,)` float32; payload stream `vals`: `(trip,)` int32 (seed 583_864).
- **Primary Source / Attribution:** Williams, J. W. J. (1964), "Algorithm 232: Heapsort", *Comm. ACM*; Python `heapq`.
- **Closest Existing Body & Key Structural Difference:** Differs from `dep_linked_list`. Tree-structured index stepping ($2k+1, 2k+2, \lfloor (k-1)/2 \rfloor$) with conditional element swapping maintaining the binary min-heap property.
- **Reference Check / Invariant:** Min-heap property invariant: $heap\_keys[k] \le \min(heap\_keys[2k+1], heap\_keys[2k+2])$ holds for all valid nodes $k < heap\_size$.

#### 2. `sched_fsm_traffic_controller`
- **Mechanism & Family:** Hierarchical Finite State Machine Traffic Controller (`control_fsm_controller`)
- **Loop Carry & State:** `(state, timer, lights)`: `state` scalar int32, `timer` scalar int32, `lights` int32 array `(num_directions,)`.
- **Input Arrays & Generation:** Vehicle presence detector stream `sensors`: `(trip, num_directions)` bool; pedestrian call button stream `ped_calls`: `(trip,)` bool (seed 583_865).
- **Primary Source / Attribution:** Original; classic embedded digital controller design (Lee & Seshia, *Introduction to Embedded Systems*).
- **Closest Existing Body & Key Structural Difference:** Differs from `stateful_scalar_iir`. Implements multi-way discrete state transitions with timer countdown guards, minimum dwell times, and sensor override branches.
- **Reference Check / Invariant:** Safety invariant: opposing traffic directions never exhibit green lights simultaneously.

#### 3. `sched_masked_active_set`
- **Mechanism & Family:** Active-Set Quadratic Programming Constraint Selection (`control_active_set`)
- **Loop Carry & State:** `(x, active_mask, lagrange_mults)`: primal vector `x` `(D,)`, `active_mask` bool array `(num_constraints,)`, `lagrange_mults` float32 array `(num_constraints,)`.
- **Input Arrays & Generation:** Constraint matrix $A$: `(num_constraints, D)` float32; rhs $b$: `(num_constraints,)` float32; objective Hessian diagonal $H$: `(D,)` float32 (seed 583_866).
- **Primary Source / Attribution:** Nocedal & Wright (2006), *Numerical Optimization*, Springer.
- **Closest Existing Body & Key Structural Difference:** Differs from `gather_masked_window`. Dynamically toggles active constraints: adds the most violated constraint to `active_mask` or drops a constraint with negative Lagrange multiplier $\lambda_i < 0$.
- **Reference Check / Invariant:** Karush-Kuhn-Tucker (KKT) complementary slackness: $\lambda_i (A_i x - b_i) = 0$ for all $i$.

#### 4. `sched_leaky_bucket_policer`
- **Mechanism & Family:** Dual-Rate Token Bucket Traffic Policer (RFC 2698) (`control_traffic_policer`)
- **Loop Carry & State:** `(committed_tokens, peak_tokens, last_time)`: three float32 scalars.
- **Input Arrays & Generation:** Packet arrival times `times`: `(trip,)` float32; packet byte sizes `sizes`: `(trip,)` int32 (seed 583_867); committed/peak information rates (CIR, PIR) and burst sizes (CBS, PBS).
- **Primary Source / Attribution:** Heinanen & Guerin (1999), "A Two Rate Three Color Marker", *IETF RFC 2698*.
- **Closest Existing Body & Key Structural Difference:** Differs from `segment_carry_ema`. Continuous token replenishment bounded by bucket capacities, performing multi-tier packet classification (Green, Yellow, Red) and conditional token subtraction.
- **Reference Check / Invariant:** Conformance bound: conforming green byte throughput strictly respects committed information rate envelope.

#### 5. `sched_round_robin_arbiter`
- **Mechanism & Family:** Deficit Weighted Round-Robin Task Arbiter (`control_fair_queueing`)
- **Loop Carry & State:** `(deficit_counters, active_queue_ptr, served_counts)`: `deficit_counters` int32 array `(num_queues,)`, `active_queue_ptr` scalar int32, `served_counts` int32 array `(num_queues,)`.
- **Input Arrays & Generation:** Queue demand lengths `demands`: `(trip, num_queues)` int32; queue quantum weights `quanta`: `(num_queues,)` int32 (seed 583_868).
- **Primary Source / Attribution:** Katevenis et al. (1991), "Weighted round-robin cell multiplexing in a general-purpose ATM switch", *IEEE JSAC*.
- **Closest Existing Body & Key Structural Difference:** Differs from `idx_circular_fifo_buffer`. Implements multi-queue credit scheduling: cycles through queues, increments deficit counters by quantum, serves packets while deficit is positive, and advances pointer.
- **Reference Check / Invariant:** Long-term fairness: ratio of served packets $\frac{\text{served}[i]}{\text{served}[j]}$ converges to the weight ratio $\frac{quantum[i]}{quantum[j]}$ under continuous saturation.

#### 6. `sched_pid_anti_windup`
- **Mechanism & Family:** Discrete PID Controller with Conditional Anti-Windup Clamping (`control_pid_regulator`)
- **Loop Carry & State:** `(integral_state, prev_error, filtered_derivative)`: three float32 scalars.
- **Input Arrays & Generation:** Setpoint stream `sp`: `(trip,)` float32; plant measurement stream `pv`: `(trip,)` float32 (seed 583_869); PID gains $K_p = 1.2, K_i = 0.5, K_d = 0.1$; actuator saturation limits $[-1.0, 1.0]$.
- **Primary Source / Attribution:** Åström & Hägglund (2006), *Advanced PID Control*, ISA.
- **Closest Existing Body & Key Structural Difference:** Differs from `stateful_scalar_iir`. Integrates non-linear conditional clamping: halts integration whenever the control actuator saturates and error sign drives it further into saturation.
- **Reference Check / Invariant:** Anti-windup boundedness: $|integral\_state| \le I_{\max}$ under persistent step disturbances.

#### 7. `sched_branching_tree_walker`
- **Mechanism & Family:** Static-Bound Decision Tree Feature Traversal (`control_decision_tree`)
- **Loop Carry & State:** `(curr_node, prediction)`: `curr_node` scalar int32, `prediction` scalar float32.
- **Input Arrays & Generation:** Sample feature vector `features`: `(num_features,)` float32; tree split feature indices `split_feature`: `(num_nodes,)` int32; split thresholds `threshold`: `(num_nodes,)` float32; left/right child pointers `left_child, right_child`: `(num_nodes,)` int32 (seed 583_870).
- **Primary Source / Attribution:** Breiman et al. (1984), *Classification and Regression Trees*; scikit-learn `DecisionTreeClassifier` (BSD-3-Clause).
- **Closest Existing Body & Key Structural Difference:** Differs from `dep_pointer_chase`. Branching pointer traversal guided by runtime feature evaluations: $next\_node = \text{where}(features[split\_feature[node]] \le threshold[node], left[node], right[node])$.
- **Reference Check / Invariant:** Traversal terminates at a leaf node; prediction matches scikit-learn tree prediction on identical feature vectors.

---

## 4. Verification, Validation & Invariant Testing

Every converted workload and newly authored algorithm must satisfy rigorous correctness and stability gates prior to benchmark timing:

1. **Four-Factor Numerical Equivalence:**
   Outputs generated with unroll factors $u \in \{1, 2, 4, 8\}$ must match factor 1:
   $$\max |y^{(u)} - y^{(1)}| \le \text{atol} + \text{rtol} \cdot |y^{(1)}|$$
   with standard tolerances $\text{atol} = 10^{-4}, \text{rtol} = 10^{-5}$.
2. **Divisible & Remainder Trip Counts:**
   Every body is validated at both divisible trips ($T \in \{64, 256, 1024\}$) and remainder trips ($T \in \{100, 255, 1023\}$).
3. **Exact Scan-to-fori_loop Match:**
   For all 53 converted workloads, the output of the converted `fori_loop` function must match the original `lax.scan` implementation on identical arguments to machine precision.
4. **Independent Reference & Invariant Checks:**
   Each new algorithm is tested against an independent reference computation (e.g. NumPy, SciPy, or an explicit Python loop) and verifies a fundamental physical/mathematical invariant (such as energy conservation, mass balance, simplex constraint, or triangle inequality).
5. **Factor-1 Tracing & Lowering:**
   Verified that JAX traces and lowers the workload cleanly to XLA HLO without shape errors or unsupported dynamic primitives.

---

## 5. Evaluation Partitioning & Lineage Strategy

To prevent data leakage and evaluate generalization on unseen static-bound loops:

1. **Lineage Preservation:**
   - Workloads belonging to historical training lineages (including all 45 converted training scans and the 7 historical training fori_loops) remain strictly in the **Development / Training Partition**.
   - Workloads belonging to historical test lineages (including the 8 converted test scans and the 6 historical test fori_loops) remain strictly in the **Historical Evaluation Partition**.
2. **Fresh Evaluation Cohort (18 New Independent Bodies):**
   18 of the 70 newly proposed algorithms are held out in a dedicated **Fresh Evaluation Cohort** from unfamiliar mechanism families:
   - Signal Processing: `sig_pll_carrier`, `sig_hilbert_analytic`
   - Optimization: `opt_mirror_descent`, `opt_anderson_acceleration`
   - Sparse Linear Algebra: `sp_ellpack_gemv`, `sp_ilu0_precond_apply`
   - Graph & Mesh: `graph_mesh_flux_assembly`, `graph_greedy_coloring`
   - Dynamic Programming: `dp_banded_smith_waterman`, `dp_knapsack_bounded`
   - Numerical Simulation: `sim_verlet_symplectic`, `sim_shallow_water_1d`
   - Tiled Tensor: `tiled_online_softmax`, `tiled_cholesky_update`
   - Index & Memory: `idx_permutation_cycle_walk`, `idx_circular_fifo_buffer`
   - Streaming: `stream_hyperloglog_step`
   - Scheduling: `sched_fsm_traffic_controller`
3. **Deterministic Partition Hash:**
   The train/test split is deterministically assigned and frozen before any performance data is collected, ensuring an uncontaminated test of loop unroll factor prediction.

---

## 6. Deliverables & File Structure

```
benchmarks/corpus/fori_expansion/
├── workload_design.md                   # This comprehensive specification
├── converted/                           # 53 converted historical scan workloads
│   ├── __init__.py
│   ├── converted_workloads.py           # 5 Phase 1 workloads
│   ├── converted_recovery.py            # 9 Phase 2 recovery workloads
│   ├── converted_indirect_gather.py     # 7 indirect gather workloads
│   ├── converted_indirect_scatter.py    # 6 indirect scatter workloads
│   ├── converted_locality_access.py     # 4 locality access workloads
│   ├── converted_dependent_indexing.py  # 1 dependent indexing workload
│   ├── converted_rolling_window.py      # 5 rolling window workloads
│   ├── converted_streaming_reduction.py # 4 streaming reduction workloads
│   ├── converted_segmented_scan.py      # 4 segmented scan workloads
│   ├── converted_stateful_recurrence.py # 4 stateful recurrence workloads
│   └── converted_control_arithmetic.py  # 4 control arithmetic workloads
├── signal_processing.py                 # Workstream 1 (7 bodies)
├── optimization_estimation.py           # Workstream 2 (7 bodies)
├── sparse_linear_algebra.py             # Workstream 3 (7 bodies)
├── graph_mesh.py                        # Workstream 4 (7 bodies)
├── dynamic_programming.py               # Workstream 5 (7 bodies)
├── numerical_simulation.py              # Workstream 6 (7 bodies)
├── tiled_tensor.py                      # Workstream 7 (7 bodies)
├── index_memory.py                      # Workstream 8 (7 bodies)
├── streaming_compression.py             # Workstream 9 (7 bodies)
├── scheduling_control.py                # Workstream 10 (7 bodies)
├── registry.py                          # Master expansion registry
├── manifest.py                          # Machine-readable manifest generator
├── validate.py                          # Equivalence & invariant validation suite
└── split.py                             # Deterministic development vs eval split
```
