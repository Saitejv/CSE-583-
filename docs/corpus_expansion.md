# Phase 3 corpus expansion: 47 distinct loop bodies

Status: **complete for corpus construction, validation, and a CPU smoke sweep.**
The full 4,560-row timing tier is defined but was not run.

This corpus is separate from the frozen Phase 1/Phase 2 workloads and the
recovery split. It adds development candidates only: no model was trained, no
gate was run, and no historical holdout or scored confirmation labels were
inspected. Factor 1 remains the default policy.

## Target and final counts

| Requirement | Final result |
| --- | --- |
| 40-60 genuinely distinct loop bodies | **47 bodies** |
| 8-12 families | **10 families** |
| Machine-readable manifest | `benchmarks/corpus/manifest.json` |
| Every body supports factors `{1,2,4,8}` | 47/47 validated |
| Divisible and remainder trips | `T=64` and `T=100`/`T=1023` checks |
| Smoke tier | **12 bodies, 96 measurements** |
| Full tier | **47 bodies, 95 variants, 4,560 measurements** |
| Provenance | 32 original, 15 public-inspired |
| Related groups | 31 groups |
| Rejected out-of-scope cases | 7, kept outside the performance registry |

The row counts are shape/trip/factor/process measurements, not unique loop
bodies. The 47-body count is the distinct-body count.

## Family coverage

| Family | Bodies | Smoke | Main mechanisms |
| --- | ---: | ---: | --- |
| `indirect_gather` | 6 | 1 | scalar/row gathers, duplicate rows, sorted blocks, masked out-of-bounds fill, multi-table select |
| `indirect_scatter` | 5 | 2 | unique rows, 2D collision add, min reduction, permutation, predicate-directed two-table writes |
| `locality_access` | 5 | 1 | adjacent-window streaming, strided samples, random windows, block reuse, permutation reads |
| `dependent_indexing` | 5 | 1 | pointer chase, multi-level indirection, linked list, parallel chains, stateful buckets |
| `rolling_window` | 5 | 2 | running mean, FIR convolution, sliding max, variance moments, two-stream dot product |
| `stencil` | 4 | 1 | 1D diffusion, five-point, nine-point, anisotropic 2D |
| `streaming_reduction` | 4 | 1 | running mean, Welford variance, decayed extrema, online log-sum-exp |
| `segmented_scan` | 4 | 1 | reset sum/max, weighted means, segment-carry EMA |
| `stateful_recurrence` | 5 | 1 | scalar/vector IIR, GRU, LSTM, coupled oscillator |
| `control_arithmetic` | 4 | 1 | polynomial forcing, transcendental update, low-rank matrix, branchless bisection |

## Body index

Every entry below is a fixed-trip, statically bounded loop. `scan` entries use
`lax.scan`; `fori_loop` entries use a static-bound `lax.fori_loop`, which traces
to `scan` in JAX 0.12.

<details>
<summary>All 47 bodies</summary>

| Body | Family | Loop | Primary variant | Provenance | Related group |
| --- | --- | --- | --- | --- | --- |
| `control_bisection_bracket` | `control_arithmetic` | scan | vec4 | public-inspired | root_finding_control |
| `control_lowrank_update` | `control_arithmetic` | scan | n8 | original | lowrank_control |
| `control_polynomial_forcing` | `control_arithmetic` | scan | n512 | original | elementwise_control |
| `control_transcendental` | `control_arithmetic` | scan | n512 | original | elementwise_control |
| `dep_bucket_stateful` | `dependent_indexing` | fori_loop | b64 | original | stateful_dependent |
| `dep_linked_list` | `dependent_indexing` | fori_loop | n1024 | original | dependent_chain |
| `dep_parallel_chains` | `dependent_indexing` | scan | n1024 | original | dependent_chain |
| `dep_pointer_chase` | `dependent_indexing` | fori_loop | n1024 | public-inspired | dependent_chain |
| `dep_two_level_indirection` | `dependent_indexing` | fori_loop | large | original | dependent_chain |
| `gather_duplicate_index` | `indirect_gather` | scan | d1024 | original | collision_memory |
| `gather_masked_window` | `indirect_gather` | scan | m1024 | original | masked_memory |
| `gather_multi_table` | `indirect_gather` | scan | t512 | original | select_gather |
| `gather_row_pool` | `indirect_gather` | scan | rows512 | public-inspired | random_row_gather |
| `gather_scalar_table` | `indirect_gather` | scan | n512 | public-inspired | random_scalar_gather |
| `gather_sorted_blocks` | `indirect_gather` | scan | b2048 | original | ordered_gather |
| `scatter_collision_add` | `indirect_scatter` | scan | g32x128 | public-inspired | collision_memory |
| `scatter_min_update` | `indirect_scatter` | fori_loop | s1024 | original | collision_memory |
| `scatter_permutation` | `indirect_scatter` | scan | p512 | original | permutation_memory |
| `scatter_two_tables` | `indirect_scatter` | scan | t1024 | original | split_table_memory |
| `scatter_unique_rows` | `indirect_scatter` | scan | r512 | public-inspired | unique_update |
| `locality_block_reuse` | `locality_access` | fori_loop | hot8x32 | original | reuse_locality |
| `locality_permutation_read` | `locality_access` | scan | n512 | original | permutation_memory |
| `locality_random_window` | `locality_access` | scan | n2048 | original | random_window_locality |
| `locality_stream_contiguous` | `locality_access` | scan | n2048 | public-inspired | sequential_locality |
| `locality_stride_sample` | `locality_access` | scan | n2048 | original | strided_locality |
| `rolling_conv1d_fir` | `rolling_window` | scan | k8 | public-inspired | rolling_weighted_windows |
| `rolling_dot_two_streams` | `rolling_window` | scan | w16 | original | rolling_weighted_windows |
| `rolling_extreme_max` | `rolling_window` | scan | w16 | original | rolling_extrema |
| `rolling_mean_shift` | `rolling_window` | scan | w8 | original | rolling_moments |
| `rolling_variance_moments` | `rolling_window` | scan | w12 | original | rolling_moments |
| `segment_carry_ema` | `segmented_scan` | scan | w4_short | original | segmented_carry |
| `segment_max_reset` | `segmented_scan` | scan | w4_short | original | segmented_reset |
| `segment_sum_reset` | `segmented_scan` | scan | w4_short | public-inspired | segmented_reset |
| `segment_weighted_mean` | `segmented_scan` | scan | w4_short | original | segmented_reset |
| `stateful_coupled_oscillator` | `stateful_recurrence` | fori_loop | n8 | original | coupled_dynamics |
| `stateful_gru_gate` | `stateful_recurrence` | scan | n4 | public-inspired | gated_recurrences |
| `stateful_lstm_cell` | `stateful_recurrence` | scan | n4 | public-inspired | gated_recurrences |
| `stateful_scalar_iir` | `stateful_recurrence` | scan | scalar | original | iir_filters |
| `stateful_vector_iir` | `stateful_recurrence` | scan | n8 | original | iir_filters |
| `stencil_anisotropic_2d` | `stencil` | fori_loop | n16 | original | stencil_wide_support |
| `stencil_diffusion_1d_fori` | `stencil` | fori_loop | n128 | public-inspired | stencil_axis_aligned |
| `stencil_five_point_2d` | `stencil` | fori_loop | n16 | public-inspired | stencil_axis_aligned |
| `stencil_nine_point_2d` | `stencil` | fori_loop | n16 | original | stencil_wide_support |
| `streaming_decayed_extrema` | `streaming_reduction` | scan | scalar | original | streaming_extrema |
| `streaming_logsumexp` | `streaming_reduction` | scan | scalar | public-inspired | streaming_normalizers |
| `streaming_running_mean` | `streaming_reduction` | scan | scalar | original | streaming_moments |
| `streaming_welford_var` | `streaming_reduction` | scan | scalar | public-inspired | streaming_moments |

</details>

## Tiers, manifest, and commands

`benchmarks/corpus/manifest.json` is generated from the registry and records
families, variants, loop kinds, provenance, related groups, body hashes, tier
arithmetic, commands, unsupported cases, and known feature caveats.

| Tier | Selection | Measurements |
| --- | --- | --- |
| `smoke` | 12 bodies covering all 10 families, primary variants, trips 64/100, all factors, one repeat | 96 |
| `full` | all 47 bodies, 95 variants, trips 64/100/256/1023, all factors, three repeats | 4,560 |

Reproduce the metadata and run the tiers:

```bash
source .venv/bin/activate
make corpus-validate
make corpus-manifest
make corpus-smoke
make corpus-full
```

Equivalent direct commands:

```bash
PYTHONPATH=src:. .venv/bin/python -m benchmarks.corpus.validate
PYTHONPATH=src:. .venv/bin/python -m benchmarks.corpus.manifest
PYTHONPATH=src:. .venv/bin/python -m benchmarks.corpus.sweep \
  --tier smoke --out results/raw/corpus_smoke_cpu_$(date +%F).csv
PYTHONPATH=src:. .venv/bin/python -m benchmarks.corpus.sweep \
  --tier full --out results/raw/corpus_full_cpu_$(date +%F).csv
```

The corpus sweep reuses the existing isolated process-per-configuration harness
in `benchmarks/sweep.py`. A small `--workload-module` bridge lets the child
process resolve `benchmarks.corpus.registry.build_workload` instead of the
frozen registry. Raw CSVs stay under `results/raw/` and are gitignored.

## Validation evidence

Run on 2026-11-17 with JAX/jaxlib `0.12.0.1`, Python 3.13, CPU:

- `make corpus-validate`: 10 families, 47 bodies, 12 smoke bodies, 47
  static-scan bodies, hashes checked, smoke rows 96, full rows 4,560.
- Every body is a single static `scan` after tracing; `--workload-module` child
  resolution is covered by an end-to-end one-row test.
- Factor equivalence: every body at `T=64` for factors 2/4/8 against factor 1;
  every body at `T=100/u=8`; all smoke bodies at `T=1023` for 2/4/8; all
  non-smoke bodies at `T=1023/u=8`; every secondary variant at `T=64/u=8`.
- Independent oracle: the rolling-mean body is checked against a NumPy mean of
  the final window.
- Loop metadata: the traced scan `unroll` parameter equals the requested factor
  for all bodies and factors.
- Manifest: the committed JSON equals a fresh manifest, including body hashes.
- Cross-registry check: corpus body hashes must not equal any frozen or
  recovery body hash; the check has a regression test.
- Unsupported cases: each case is rejected with its expected reason category.
- `pytest -q tests/test_corpus.py`: **656 passed**.
- `pytest -q`: **890 passed**.
- `make quick`: legacy smoke sweep, 8 measurements, 0 errors.
- CPU smoke sweep: **96/96 rows correct, 0 errors, 0.7-1.2 min** on the target
  Apple M2 CPU process-per-configuration harness.
- `make quick` and the legacy frozen sweeps still pass; frozen workload,
  recovery, split, dataset, model, and gate files are untouched.

The initial smoke run exposed a real integration bug: child processes imported
the frozen workload registry and could not resolve corpus names. The generic
`--workload-module` bridge plus the integration test fix that path. The full
4,560-row tier is intentionally not run here.

## Independent review

A separate reviewer inspected the merged result, reproduced the child-process
bug, and found one exact frozen-body duplicate and one false mechanism claim.
The fixes were:

- Add `--workload-module` to `benchmarks/sweep.py`; corpus sweeps pass
  `benchmarks.corpus.registry`. Added a one-row child integration test.
- Replace `control_dense_matvec` with `control_bisection_bracket`; the removed
  body shared an exact Jaxpr hash with frozen `dense_matvec`. Cross-registry
  body-hash rejection is now part of `body_hashes()`.
- Move `rolling_extreme_max`'s max reduction inside the loop body and update its
  metadata.
- Differentiate the former near-duplicate masked scatter/gather and 1D collision
  bodies: row-granularity unique scatter, predicate-directed two-table scatter,
  2D collision add, out-of-bounds fill gather, adjacent-window streaming, and
  duplicate-row gather.
- Add remainder, secondary-variant, loop-unroll, full-manifest, child-process,
  unsupported-reason, and independent-oracle tests.
- Flag the pre-existing feature-schema limitation for `scatter-add` and
  `scatter-min` in the manifest rather than changing frozen feature semantics.

Re-review verdict: **no blocking issue remains**. Remaining limitations are
listed below.

## Provenance and licensing

`docs/corpus_research.md` records 17 sources. Public-inspired bodies are
Apache-2.0, BSD-2-Clause, or BSD-3-Clause; no GPL/AGPL, non-commercial, or
unverified source contributed code. Implementations were written from
descriptions or algorithm papers; no external code was copied or executed.
The repository has no root `LICENSE` file, so original bodies are marked
accordingly rather than assigned a license.

Strongest sources: JAX `scan`/`fori_loop`/`gather`/`scatter`/`associative_scan`/
`reduce_window`/`segment_sum` documentation, FlashAttention, JAX-CFD, JAX MD,
Flax LSTM, Welford, Jraph, and JAXopt Bisection.

## Unsupported cases

These are permanently rejected and are not part of the performance corpus:

| Case | Reason |
| --- | --- |
| `dynamic_trip_fori` | `fori_loop` bounds must be static integers |
| `while_pointer_chase` | `lax.while_loop` has a data-dependent trip count |
| `nested_scan` | scan inside scan is nested control flow |
| `nested_cond` | `lax.cond` inside scan is nested control flow |
| `dynamic_scan_length` | scan length must be a static Python integer |
| `dynamic_nonzero_shape` | dynamic output shapes are not jittable in this corpus |
| `debug_print_effect` | effectful debug printing is outside the timing domain |

## Coverage gaps and known limitations

- The feature schema v1 classifies `lax.scatter` but not scatter-add or
  scatter-min. `scatter_collision_add` and `scatter_min_update` are therefore
  flagged in the manifest's `feature_caveats`; do not use their
  `eq_gather_scatter` count for model training without a feature-version bump.
- There is no 2D row+column gather, graph/segment-scatter message passing, AR(2)
  recurrence, or nonlinear-solver family in this expansion. Adjacent-window
  streaming and branchless bisection cover two of the research ideas.
- Some bodies are purposefully close to recovery development workloads as
  data-pattern contrasts: duplicate-row gather vs `gather_lookup`,
  adjacent-window streaming vs `window_read`, and `stateful_scalar_iir` vs
  frozen `scalar_recurrence`. They were differentiated structurally; they are
  not confirmation bodies and carry no labels.
- The full timing tier is defined but not executed. `R`/`B` grids and policy
  evaluation remain future work.
- Raw smoke results are local only (`results/raw/` is ignored); the manifest and
  this report are the committed evidence.

## Subagent contributions

| Contributor | Contribution |
| --- | --- |
| Research subagent | `docs/corpus_research.md`: 17 sources, permissive-license audit, concrete body ideas, prohibited-source list. No code copied. |
| Indirect-memory subagent | `benchmarks/corpus/indirect.py`: 21 gather/scatter/locality/dependent bodies, validation table, 6 `fori_loop` bodies, smoke selection. |
| Structured/numerical subagent | `benchmarks/corpus/structured.py`: 26 rolling/stencil/streaming/segmented/stateful/control bodies, validation table. |
| Independent reviewer | Reviewed the merged corpus, reproduced the frozen-registry child bug, found the exact `dense_matvec` duplicate and the out-of-loop max, checked similarity/leakage/metadata/tests, and re-reviewed the fixes. |

Shared scaffolding (`spec.py`, `common.py`, `registry.py`, `manifest.py`,
`validate.py`, `sweep.py`, `unsupported.py`), tests, and integration were
written by the main agent; subagent files were merged without overlapping
edits.

## Recommendation

Run the **smoke tier first**. It costs 96 fresh-process measurements, covers all
10 families and 12 representative bodies, and completed in 0.7-1.2 minutes with
96/96 correctness passes on the target CPU. Use it to verify the harness and
compare factor shapes before spending on the full tier. If the smoke results
are stable, run the full tier in family batches rather than as one 4,560-row
job, and keep every raw CSV outside the frozen experiment paths.
