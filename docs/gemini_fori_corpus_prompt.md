# Gemini CLI assignment — expand the JAX loop-unrolling benchmark

Work in /Users/saitejvelaga/Documents/umich/CSE583/project. Act as the lead implementer. Spawn parallel subagents to research and implement workloads, then integrate, validate, and collect real timings. Deliver runnable code and measured datasets for subsequent training and evaluation. Model fitting belongs to the next experiment.

## Objective

Our Apple M2 CPU model chooses a JAX loop unroll factor from 1, 2, 4 and 8. The immediate question is whether it can reduce steady-state execution time for previously unseen static-bound jax.lax.fori_loop bodies. The combined dataset has 66 measured workloads, but only 13 use fori_loop, and only seven of those were in the last training partition. Across all loop kinds, 35 bodies have just one measured configuration.

Aim for 60–80 additional independent fori_loop bodies across at least ten mechanisms. Aim to reserve 15–20 independent bodies for fresh evaluation. These are planning targets. Report the actual distinct, valid, measured counts. A change to a size, constant, seed, trip count, spelling, or unroll factor is a variant of an existing body. A direct translation of a current scan is a useful paired control if labeled as such, but contributes zero new independent bodies.

## Convert the existing scan inventory to fori_loop

The deliverable is a fori_loop-only dataset. In addition to the new independent bodies, audit **all 53 existing public scan workloads** for conversion. Assign conversion agents by historical workload family and give each agent a separate module under benchmarks/corpus/fori_expansion/converted/. Produce a conversion manifest with one entry per original workload, including source path, original workload and related-group IDs, converted ID, partition history, preserved variants, reference checks, and either measured coverage or a concrete reason conversion was unsuitable. Include the **13 existing public fori_loop workloads** in the final inventory and reuse their compatible measured rows. Report historical fori_loop bodies, converted algorithms, and genuinely new algorithms separately.

For a forward scan, close over its input arrays, read the timestep at index i (including each leaf of a pytree), and implement the same state transition inside lax.fori_loop(0, T, body, init, unroll=factor). Preserve the original return value: where the caller consumes stacked outputs, carry preallocated fixed-shape output buffers and write each timestep to the correct slot. Where the caller consumes only the final carry, preserve that behavior. Handle reverse traversal, scans without input arrays, tuple carries, and index-dependent behavior according to the original semantics. Compare the converted public function against the original scan on identical inputs, including every observable output, on multiple seeds, sizes, and divisible/remainder trips. Add a small independent reference or invariant for new algorithms. Document any new buffer or indexing costs. If a body cannot be converted faithfully within the supported loop contract, record why and move on.

Collect **fresh timings for each accepted conversion** at all four factors using the same acquisition rules as new bodies. A scan timing is not a fori_loop label. Keep original scan implementations and historical results available as correctness references. Exclude public scan rows from **both** model-ready tables. Check loop kind from the builder and the public API used in source; a static-bound fori_loop can legitimately appear as a scan primitive in its lowered representation.

Keep each conversion in its original algorithm lineage. A historical training relative belongs on the development side. A historical test relative stays evaluation-only and is labeled as historical evaluation. Reserve the fresh evaluation cohort from genuinely new lineages with no prior training or benchmark-driven design. Publish separate historical/fresh evaluation tags. Do not count a scan and its converted version twice toward independent coverage. Recalculate the acquisition plan after the conversion audit: at two trips, four factors and three repeats, 53 accepted conversions would add 1,272 raw rows on top of the new-body matrix.

## Start here and coordinate parallel work

Read benchmarks/workloads.py, benchmarks/recovery_workloads.py, benchmarks/corpus/{spec,registry,structured,indirect,static_matched,sweep,all_workloads}.py, benchmarks/sweep.py, docs/{corpus_expansion,corpus_research,fori_dev_review,all_workload_prediction}.md, and the current split in results/all_workloads/split.csv. Inspect git status and preserve existing work. Inventory all 66 historical bodies, including the matched-index cases.

Dispatch one worker for each of the ten workstreams below, plus independent workers for source/duplicate review, split design, and correctness/measurement review. Use waves if concurrency is limited. Each implementation worker owns a separate module and reference checks. Have each propose mechanisms, closest existing bodies, structural differences, and source leads before coding. The lead reconciles overlaps and owns the registry, validator, manifest, sweep, and final merge. Allow parallel implementation and correctness checks. Give a single process exclusive use of the benchmark CPU during timed runs.

## Workstreams

Each worker should implement several genuinely different algorithms. These are starting points; replace overlaps discovered in the inventory.

| Workstream | Candidate mechanisms |
|---|---|
| Signal processing | Cascaded biquad states, Goertzel frequency detector, adaptive LMS coefficient updates, polyphase resampling into a fixed output buffer |
| Optimization and estimation | Coordinate descent with incremental residuals, projected primal–dual updates, recursive least squares, diagonal Kalman updates |
| Sparse linear algebra | Fixed-width sparse row accumulation, triangular substitution with masked dependencies, block-coordinate relaxation, sparse residual propagation |
| Graph and mesh | Edge-to-node messages, fixed-capacity frontier relaxation, graph coloring updates, mesh flux assembly with shared destinations |
| Dynamic programming | Banded alignment, Viterbi max-plus updates with backpointers, rolling edit-distance rows, fixed-capacity knapsack updates |
| Numerical simulation | Symplectic integration, coupled reaction–diffusion, finite-volume upwind flux, constrained position updates |
| Tiled tensor workloads | Block online softmax with rescaling, panelwise Gram accumulation, low-rank updates, blocked pairwise distances with running nearest neighbors |
| Index and memory access | Permutation cycles, two-level indirection, gather–transform–scatter, overlapping block updates; vary locality and collision distributions as variants |
| Streaming and compression | Bit packing into word buffers, run-length encoding with fixed capacity, streaming sketch updates, bounded dictionary lookup with stateful writes |
| Scheduling and control flow | Fixed-capacity queue simulation, state-machine transitions, masked active-set updates, piecewise vector carry with sparse state changes |

Cover small bodies dominated by loop overhead and heavier bodies dominated by computation or memory. Include scalar, vector and tuple carries, dependent and independent iterations, cheap and expensive branches, predictable and data-dependent addresses. Include ordinary workloads where unrolling may be neutral or slower. Preserve these outcomes in the dataset. For distribution-sensitive bodies, record the input generation rule and group the variants together.

## Sources and implementation

Use either original algorithms or independently implemented adaptations from primary sources. Search official JAX examples, scientific library documentation, papers, and permissively licensed repositories such as JAXopt, Optax, Jraph, JAX MD and JAX-CFD. Find the specific algorithm or function. Record its exact URL, revision when available, what was adapted, and the license of any reused implementation. Implement compact dependency-free JAX bodies where practical. Label invented workloads as original and explain the application they represent. Follow applicable attribution and license terms for any reused code.

Place the expansion under benchmarks/corpus/fori_expansion/ with domain modules, registry.py, validate.py, sweep.py, and a machine-readable manifest and split. Use tests/test_fori_expansion_*.py for correctness, registry, split, and harness tests. Place generated measurements under results/fori_expansion/ and the acquisition report in docs/fori_expansion.md. Keep the historical registries, datasets, trained model and split intact.

Reuse CorpusSpec, WorkloadSpec and the existing fresh-process sweep. Implement build_workload(name, trip_count, factor, variant=None) in the expansion registry. Ensure the sweep passes the expansion specs to the parent and the expansion module through --workload-module to every child. Smoke-test that a workload unique to the new registry runs from the command line in a fresh child.

Use a real lax.fori_loop(0, static_trip_count, body, init, unroll=factor) with a fixed-shape carry. Generate deterministic inputs before timing and pass input arrays as arguments. Make outputs depend on the loop work. Use stable numerics at the longest trip count and keep shapes compatible with the current CPU feature extractor and selector. Prefer one eligible target loop per workload. Document unsupported patterns separately. Check that factor-1 tracing succeeds, including for public fori_loop functions represented internally by a scan primitive.

## Freeze a useful evaluation set before timing

Audit proposed bodies against all historical workloads and each other. Compare source lineage, algorithm, dependence structure, access pattern, normalized body features and Jaxpr hashes. Different hashes alone do not prove independence.

Assign a stable related-group ID. Keep all members of a related group on one side, including translated scans and closely related historical bodies. If a new proposed test body is closely related to historical training data, move it to development or replace it before timing. Preserve historical test relatives for evaluation. Reserve several whole mechanisms or source lineages for an explicitly labeled unfamiliar-family test. Record counts by mechanism and partition, the deterministic assignment rule, and a hash of the split before inspecting performance. Keep eval labels separate from training exports. Select shapes and trips using workload semantics and resource costs. Never select or discard a body because of which factor wins.

## Collect labels and verify them

1. Check every body against an independent small reference calculation or a meaningful numerical invariant. Compare outputs at all four factors with factor 1. Test divisible and remainder trips. Check finite values, reproducible input generation, real dependence on inputs, and usable output. Record failures with reasons.
2. Run a serial, timed primary acquisition on **every accepted body**, including evaluation bodies. Use one primary shape, two predeclared trip counts (default 64 and 255), four factors, and three independent process repeats. For 60 bodies this is 1,440 raw timing rows. Use a justified alternate matrix for mathematically constrained algorithms and record it before timing. Use the pinned CPU environment, the existing synchronized warm timing routine, and retain cold timings for later analysis.
3. Acquire depth as well as breadth. Preselect representative bodies in each partition and mechanism for a second input size and longer divisible/remainder trips such as 512 and 1023. Include cache-fitting and larger working sets where meaningful. Add short-trip coverage for overhead-dominated loops. Record the planned matrix and estimated runtime after smoke checks. Run in resumable batches with unique keys so a restart cannot silently duplicate timings. If hardware differs from the historical Apple M2 environment, label the hardware and keep that dataset separate.
4. Retain raw per-process rows, repeat IDs, seed, versions, device, machine, shape, trip, factor, correctness, and failures. Use the existing aggregation/feature pipeline to output separate development and evaluation feature/label tables. Keep timing values, source identifiers, partition tags, and workload names out of predictor features. Verify a four-factor prediction group can be reconstructed for every accepted configuration and report repeat spread and unstable factor rankings.
5. Run the new tests and the existing test suite. Independently verify the raw-key counts, train/eval disjointness, duplicate report, and a sample of aggregates recomputed from raw rows. Deliver exact reproduction commands, manifest and split hashes, expected versus actual coverage, and all unfinished or rejected cases. Distinguish software tests, distinct bodies, variants, configurations, raw timings, and aggregated label rows in the final report.

Assert that every row in both model-ready tables has public loop_kind equal to fori_loop. Reconcile the 53 conversion dispositions, the 13 historical fori_loop bodies, and the newly authored bodies against the manifest. Finish with paths for the training table, evaluation table, raw data, and split manifest. The next agent should be able to train on the development set and compare warm-only choices with factor 1 on the held-out set without reconstructing the experiment.
