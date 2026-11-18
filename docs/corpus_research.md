# Corpus research: public loop-pattern sources

Scope: research only. No public code was cloned or executed and no benchmark
was run. Sources were chosen for credible, permissively licensed descriptions
of loop or access patterns that can motivate original corpus bodies. All URLs
were accessed 2026-11-17.

## Sources

| # | Source and exact URL | License and verification | Loop / access pattern demonstrated | Corpus family | Concrete adaptation idea (own words) | Safe to adapt? |
|---|---|---|---|---|---|---|
| 1 | **JAX docs: `jax.lax.scan`** — https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html | Apache-2.0. Verified on the `jax-ml/jax` GitHub repo sidebar and its `LICENSE`; docs are built from that repo. | Fixed-length scan with explicit carry and stacked outputs; lowers to one WhileOp; `unroll` controls how many iterations are fused. | `streaming_reduction`, `stateful_recurrence`, `segmented_scan` | Use as the baseline contract for all scan bodies: a tuple carry `(state, accumulator)` that never changes shape, with per-step output. A prefix-mean body carrying `(sum, count)` and a lagged recurrence carrying `(y_prev, y_prev2)` are both direct applications. | Direct; Apache-2.0. Keep attribution/notice if code is copied; an independent reimplementation is preferable. |
| 2 | **JAX docs: `jax.lax.fori_loop`** — https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html | Apache-2.0. Same JAX repo verification as above. | Static `for i in range(lower, upper)` loop; static bounds lower to `scan`, dynamic bounds to `while_loop`; carry is fixed-shape. | `stencil`, `locality_access`, `control_arithmetic` | Time-step a 1D advection or 2D Jacobi stencil inside `fori_loop`; the body reads `jnp.roll`-shifted neighbors and writes the next grid. A second variant can make the update coefficient data-dependent inside the body. | Direct; Apache-2.0. Keep attribution if copied. |
| 3 | **JAX docs: `jax.lax.while_loop`** — https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html | Apache-2.0. Same JAX repo verification as above. | Data-dependent condition with a fixed-shape carry; one WhileOp; not reverse-mode differentiable. | `control_arithmetic`, `stateful_recurrence` | Iterative refinement with a residual condition: carry `(x, r, it)` and stop when `r < tol` or `it == maxiter`. A second body can use the condition to switch between two arithmetic updates without a Python branch. | Direct; Apache-2.0. Keep attribution if copied. |
| 4 | **JAX docs: `jax.lax.associative_scan`** — https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html | Apache-2.0. Same JAX repo verification as above. | Parallel prefix scan with an associative binary operator; examples include cumulative sums and matrix-product prefixes; cites Blelloch 1990. | `segmented_scan`, `streaming_reduction` | Build a segmented inclusive scan by carrying `(value, segment_id)` and combining only when the segment ids agree, otherwise taking the right operand. Alternatively use the documented Blelloch prefix-sum structure as inspiration for a scan that is mathematically a prefix operation but implemented sequentially in the corpus. | Direct; Apache-2.0. Keep attribution if copied. |
| 5 | **JAX docs: `jax.ops.segment_sum`** — https://docs.jax.dev/en/latest/_autosummary/jax.ops.segment_sum.html | Apache-2.0. Same JAX repo verification as above. | Sums values sharing a segment id; ids may repeat and need not be sorted; `num_segments` must be static under JIT. | `segmented_scan`, `indirect_scatter` | Histogram-by-index or segment-normalization body: per step, scatter-add a vector of values into fixed bins, then divide by per-bin counts. A segmented variant can use the bin id as the segment id and emit per-segment prefix sums. | Direct; Apache-2.0. Keep attribution if copied. |
| 6 | **JAX docs: `jax.lax.scatter`** — https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html | Apache-2.0. Same JAX repo verification as above. | Indexed update with `scatter_indices` and `updates`; duplicate-index order is unspecified; modes `clip`, `fill`, `drop`, `promise_in_bounds`. | `indirect_scatter` | Per-step sparse deposit into a fixed-size state array using `state.at[idx].add(values)` or `lax.scatter` directly. Use non-unique indices in one variant to force a deterministic reduction order (e.g. sort or bucket) rather than relying on scatter order. | Direct; Apache-2.0. Keep attribution if copied. |
| 7 | **JAX docs: `jax.lax.gather`** — https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html | Apache-2.0. Same JAX repo verification as above. | Gathers slices at index arrays; `slice_sizes`, `unique_indices`, `indices_are_sorted`, and out-of-bounds modes. | `indirect_gather`, `dependent_indexing` | Table-lookup body: carry an index vector, gather rows from a fixed table each step, compute a new index from the gathered value, and repeat. Boundary variants use `mode="clip"` versus `mode="fill"`. | Direct; Apache-2.0. Keep attribution if copied. |
| 8 | **JAX docs: `jax.lax.reduce_window`** — https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_window.html | Apache-2.0. Same JAX repo verification as above. | Windowed reduction with window dimensions, strides, padding, and dilation; example computes a windowed product over adjacent pairs. | `rolling_window`, `stencil` | Use the documented window semantics as the oracle for a rolling-window scan: carry a ring buffer of width `W`, overwrite one slot per step, and emit sum/max/min over the buffer. The `reduce_window` result checks the scan implementation. | Direct; Apache-2.0. Keep attribution if copied. |
| 9 | **Flax seq2seq LSTM example** — https://github.com/google/flax/blob/main/examples/seq2seq/models.py | Apache-2.0. Verified on the `google/flax` repo sidebar and the Apache-2.0 header in the example source. | LSTM recurrence over time with carry `(c, h)`; encoder/decoder `nn.RNN`; a decoder variant feeds the previous prediction back as the next input. | `stateful_recurrence` | Reimplement a small LSTM cell: four gates from `x_t` and `h_{t-1}`, update `c_t` and `h_t`, and emit `h_t`. Keep the carry a nested tuple and use fixed trip counts; a second body can feed the previous output back as input. | Direct; Apache-2.0. Prefer a simplified, dependency-free LSTM over copying Flax modules. |
| 10 | **NumPyro AR2 example** — https://num.pyro.ai/en/stable/examples/ar2.html | Apache-2.0. Verified on the `pyro-ppl/numpyro` repo sidebar and README. | Second-order autoregressive recurrence: carry `(y_{t-1}, y_{t-2})`, compute `c + a1*y_{t-1} + a2*y_{t-2}`, emit the mean and roll the carry. | `stateful_recurrence` | An order-2 linear recurrence with two lagged state slots. Variants can use time-varying coefficients, a nonlinear term, or a three-slot carry to change dependence depth without changing the loop kind. | Direct; Apache-2.0. Reimplement the small transition function; no need for NumPyro. |
| 11 | **FlashAttention paper and official repo** — https://arxiv.org/abs/2205.14135 and https://github.com/Dao-AILab/flash-attention | Paper: algorithm description, no code license needed for reimplementation. Repo: BSD-3-Clause, verified on the repo `LICENSE` page. | Tiled online softmax: keep running max `m`, normalizer `l`, and output accumulator `O`; update and rescale as each K/V block arrives. | `streaming_reduction` | Remove the attention matmuls and keep the online-softmax core: per chunk, compute a local max and local sum, update `m_new = max(m, m_chunk)`, rescale `l` and `acc` by `exp(m - m_new)`, and add the chunk contribution. This gives a stable streaming reduction with three coupled carries. | Reimplement from the paper/algorithm. The repo is permissive but CUDA-specific, so copying is not useful; if any code is copied, retain the BSD-3 notice. |
| 12 | **JAX-CFD** — https://github.com/google/jax-cfd | Apache-2.0. Verified on the repo sidebar and `LICENSE`; PyPI metadata also says Apache-2.0. | Finite-volume/finite-difference time stepping on staggered grids; neighbor stencils via shifted array slices; explicit advection and diffusion loops. | `stencil`, `locality_access` | A 1D periodic advection body: `u_new[i] = u[i] - c*dt/dx*(u[i] - u[i-1])` built from `jnp.roll`; a 2D variant computes a five-point Laplacian with four rolls. Both are local stencils with a fixed neighborhood and a time-step scan. | Direct; Apache-2.0. Reimplement a small stencil rather than importing the CFD stack. |
| 13 | **Jraph graph neural network library** — https://github.com/google-deepmind/jraph/blob/master/jraph/_src/models.py | Apache-2.0. Verified on the repo sidebar and `LICENSE`. | GraphNetwork message passing: gather sender/receiver node features by edge index, update edges, aggregate with `segment_sum`, update nodes; optional attention uses `segment_softmax`. | `indirect_gather`, `indirect_scatter`, `dependent_indexing`, `segmented_scan` | A per-step message-passing body: `sent = node_feat[sender]`, `recv = node_feat[receiver]`, `msg = f(sent, recv, edge_feat)`, then `segment_sum(msg, receiver)` into new node features. Indices are data-dependent arrays, so this is a clean dependent-indexing pattern. | Direct; Apache-2.0. Reimplement a two-layer message-passing body without the library. |
| 14 | **JAX MD partition / neighbor list** — https://github.com/jax-md/jax-md/blob/main/jax_md/partition.py | Apache-2.0. Verified on the repo `LICENSE`; file header carries the Apache-2.0 notice. | Cell-list spatial partition: compute cell ids from positions, scatter particle ids into a dense fixed-capacity cell buffer, then gather neighboring cells for a neighbor list. | `dependent_indexing`, `indirect_gather`, `indirect_scatter`, `locality_access` | Fixed-capacity cell-list body: `cell = floor(pos / cell_size)`, scatter particle ids with `.at[cell].add` or a masked slot update, then for each particle gather the 3x3 neighboring cells and mask by distance. This combines data-dependent indices with local access. | Direct; Apache-2.0. Reimplement a 1D or 2D fixed-capacity version; do not copy the full MD stack. |
| 15 | **JAXopt Bisection** — https://jaxopt.github.io/stable/_autosummary/jaxopt.Bisection.html | Apache-2.0. Verified on the `google/jaxopt` repo sidebar and `LICENSE`. | Root-finding loop: evaluate at the midpoint, update `low`/`high` with `jnp.where` based on the sign, cap iterations, and optionally run under `while_loop`. | `control_arithmetic`, `dependent_indexing` | A branchless bracketing body: `mid = 0.5*(lo+hi)`, `v = f(mid)`, `lo = where(sign*v > 0, mid, lo)`, `hi = where(sign*v > 0, hi, mid)`. Variants can use a fixed `fori_loop` budget or a residual-driven `while_loop`. | Direct; Apache-2.0. Reimplement the small update rule; no JAXopt dependency needed. |
| 16 | **jvf/welford: online variance** — https://github.com/jvf/welford | BSD-2-Clause. Verified on the repo sidebar and `LICENSE`; README cites Welford's 1962 method and Knuth. | One-pass online mean/variance: carry `(count, mean, M2)`; per value compute `delta = x - mean`, update `mean`, then `M2 += delta*(x-mean)`. | `streaming_reduction` | A Welford scan body with a three-field carry and per-step variance output. A second variant can process chunks and merge `(n, mean, M2)` states, giving a two-level streaming reduction. | Direct/reimplement; BSD-2 permits reuse with notice. A from-scratch implementation is trivial and preferable. |
| 17 | **CUB `DeviceSegmentedScan` documentation** — https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceSegmentedScan.html | CUB repo: BSD-3-Clause, verified on the archived `NVIDIA/cub` sidebar and README. Current `NVIDIA/cccl` source files use Apache-2.0 WITH LLVM-exception. | Segmented inclusive/exclusive prefix scan with per-segment begin/end offsets; associative operator; separate segment boundaries. | `segmented_scan` | A sequential segmented scan body: carry `(acc, previous_segment)`; if the new segment id equals the previous one, add to `acc`, otherwise reset `acc` to the current value. Variants use explicit begin/end offsets or a per-element segment-id array. | Reimplement the documented semantics. Do not copy the C++ implementation; it is permissive but not directly usable from JAX. |

## Coverage check

| Family | Primary sources |
|---|---|
| `indirect_gather` | 7, 13, 14 |
| `indirect_scatter` | 5, 6, 13, 14 |
| `locality_access` | 2, 12, 14 |
| `dependent_indexing` | 7, 13, 14, 15 |
| `rolling_window` | 8 |
| `stencil` | 2, 8, 12 |
| `streaming_reduction` | 1, 4, 11, 16 |
| `segmented_scan` | 4, 5, 17 |
| `stateful_recurrence` | 1, 3, 9, 10 |
| `control_arithmetic` | 2, 3, 15 |

## Top adaptation ideas

1. **Online-softmax streaming reduction (source 11).** Carry `(m, l, acc)` and
   process chunks; rescale the accumulator only when the running max changes.
   This creates a genuinely coupled reduction, unlike a simple sum, and the
   `m`/`l` rescaling makes partial unrolling nontrivial.
2. **Fixed-capacity cell list / neighbor gather (source 14).** Compute cell ids
   from data, scatter particle ids into a dense buffer, then gather neighboring
   cells with a distance mask. This exercises dependent indexing, scatter, and
   local access in one body.
3. **Graph message passing with `segment_sum` (source 13).** Gather sender and
   receiver features by edge arrays, form messages, and scatter-add by receiver.
   A two-round variant adds a second dependent gather.
4. **1D/2D finite-volume stencil (source 12).** Use shifted slices or `jnp.roll`
   for nearest-neighbor fluxes and time-step with `fori_loop`; a 2D five-point
   Laplacian is a compact locality-heavy variant.
5. **LSTM recurrence (source 9).** Four gates, carry `(c, h)`, and a fixed trip
   count. A feedback variant consumes the previous output, which changes the
   dependence structure without changing the loop kind.
6. **Branchless bisection (source 15).** A data-dependent bracket update with
   `jnp.where`, run under a fixed iteration budget or a residual condition.
   Variants can alternate bracket updates with a local polynomial evaluation.
7. **Segmented prefix scan (sources 4, 5, 17).** Carry `(acc, segment_id)` and
   reset on a boundary; or use explicit begin/end offsets. This is a clean
   alternative to plain prefix sum and tests whether unrolling respects
   segment resets.
8. **Rolling-window ring buffer (source 8).** Carry a width-`W` buffer, overwrite
   one slot per step, and reduce the buffer. Compare against `reduce_window`
   for correctness; vary `W` and boundary handling.

## License cautions

- **No GPL/AGPL or non-commercial source is in the table.** The recommended
  sources are Apache-2.0, BSD-2-Clause, or BSD-3-Clause.
- **CUB:** the archived `NVIDIA/cub` repo is BSD-3-Clause; the active
  `NVIDIA/cccl` tree is Apache-2.0 WITH LLVM-exception. Either is permissive,
  but do not copy C++ code into a JAX corpus; reimplement the documented
  segmented-scan semantics.
- **FlashAttention:** the paper algorithm can be reimplemented freely; the
  repository is BSD-3-Clause, so retain its copyright/notice if any code is
  copied. The CUDA code is not directly useful for a JAX body.
- **Apache-2.0 sources** (JAX, Flax, NumPyro, JAX-CFD, Jraph, JAX MD, JAXopt):
  direct adaptation is allowed with attribution/notice. For corpus provenance,
  prefer `kind="public-inspired"` and an independent implementation, recording
  the source and license rather than copying files.
- **Avoid code from pages with no explicit license.** Tutorial/Q&A/blog pages
  that describe a pattern (for example scan-based running-mean exercises) may
  be useful for intuition, but their code should not be copied without a clear
  license. Wikipedia is CC BY-SA; use the equations and reimplement.
- **No source in the final table is marked "unverified."** If a source is added
  later and its license cannot be confirmed, mark it unverified and exclude its
  code from the corpus.
