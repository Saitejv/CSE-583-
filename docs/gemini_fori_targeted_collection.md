# Gemini CLI assignment — one targeted JAX fori_loop timing round

Work in /Users/saitejvelaga/Documents/umich/CSE583/project. Implement and measure this round, then hand the data back for modeling. Our question is whether choosing an unroll factor from {1, 2, 4, 8} makes static-bound jax.lax.fori_loop execute faster on the Apple M2 CPU, after compilation. Keep the current model, prior raw files, and existing partitions intact.

## Why this round and how much to collect

The previous fresh evaluation has 18 bodies and 40 configurations. Of those configurations, 38 are at 64 or 255 iterations, and only five have a measured best factor yielding 1.05× speedup or more over factor 1. The current predictor achieved 1.005× geometric-mean speedup, and perfect factor selection would have achieved 1.021× there. We want to distinguish lack of opportunity from missed opportunity, and trip-count effects from unroll-tail effects. Set a fixed budget before inspecting timings:

| Cohort | Bodies | Purpose |
|---|---:|---|
| Existing development bodies | 12 | Reuse genuinely distinct training bodies and measure new sizes and trips. Preserve their train assignment. |
| New development bodies | 12 | Add realistic, distinct loop mechanisms that broaden the training set. |
| New sealed evaluation bodies | 24 | Measure entirely new algorithm lineages. Include ordinary workloads where factor 1 can win. |

This is 36 newly authored bodies and 12 existing bodies. For each of the 48, use two meaningfully different working-set sizes and four trip counts: **255, 256, 1023, 1024**. The neighboring pairs isolate divisible vs. remainder behavior at nearly equal work. Use four factors and three independent process repeats: **384 configurations, 1,536 factor-level median labels, 4,608 planned process measurements**. The development and evaluation tables should each contain 192 configurations. Counts describe coverage; count a compatible, reused historical row only once. Record reuse explicitly and use the new balanced, randomized acquisition for the primary paired comparison. Do not expand the matrix after seeing favorable or unfavorable results.

## Organize the work

Read docs/fori_predictor.md, docs/fori_expansion.md, benchmarks/corpus/fori_expansion/{registry,acquire,split}.py, its split.csv, benchmarks/sweep.py, benchmarks/corpus/train_fori.py and the original corpus specs. Verify the actual code and CSV schemas. Preserve uncommitted work. Spawn six domain agents in parallel, plus separate duplicate/split and measurement/correctness reviewers. Use waves if necessary. Each domain agent owns one module and proposes two existing training bodies, two new training bodies, and four new evaluation bodies. The lead reconciles duplicates and owns integration. Run timed jobs serially on one quiet machine after parallel coding and correctness work has stopped.

Use these six workstreams to cover different causes of runtime variation:

1. **Dependency depth and instruction parallelism.** Compare long carried dependencies, multiple independent accumulators, and coupled states in practical DSP or estimation algorithms. Coefficients alone do not make a distinct body.
2. **Locality and indexing.** Include contiguous/block access, strided access, and data-dependent indirection from real sparse, image, or table algorithms. Pass indices as inputs; record their distribution.
3. **Writes and state size.** Exercise overlapping updates, conflict-free writes, ring state, and carried buffers. Include an application where changing the buffer size changes the live footprint substantially.
4. **Dense arithmetic and reductions.** Include block updates, reductions and mixed scalar/vector work. Cover both modest loop bodies and bodies where the inner operation dominates.
5. **Branches, masks and active state.** Exercise state machines, masked updates and input-dependent paths with deterministic, documented branch distributions. Keep the loop bounds static.
6. **Application kernels.** Source recognizable use cases from actual JAX applications, such as fixed-step solvers, signal reconstruction or trajectory processing. At least 12 of the 24 sealed bodies across the six workstreams should be traceable to independent public application algorithms. Document why repeated execution at the chosen sizes is plausible.

Use primary sources such as official projects, application repositories and papers. Record URL, version or revision, method or function, license for reused code, and adaptation. Original kernels are welcome when tied to a concrete use case. Implement compact self-contained JAX kernels. Avoid counting parameter tweaks, translations of existing scans, and renamed existing expansion kernels as new bodies.

## Freeze the split and matrix first

Audit the full prior inventory, including all 136 fori_loop bodies, converted scan lineages and prior holdouts. Check algorithm/source lineage, dependencies, access pattern and factor-1 body hashes. Keep related bodies and all variants, trips, seeds and repeats on one side. Reserve entirely new evaluation lineages; replace any proposed evaluation relative of an existing workload before timing. Broad domains may overlap. Select workloads on semantic coverage, independently of the previous oracle gains or the predictor's errors. Publish a machine-readable split, input-generation specs, shape dimensions and planned acquisition keys plus hashes before timing.

Choose the two sizes using the application and realistic CPU memory costs. Verify that a size change materially changes array dimensions or live state and document actual bytes. Keep input-distribution assumptions explicit and consistent within each trip-count pair. Use a deterministic prefix or equivalent matched input generation so changing 255 to 256 does not independently randomize the whole problem. If a kernel cannot meaningfully support the matrix, replace it before the freeze. If a frozen case fails, retain the failure and disposition.

## Implementation and acquisition

- Put new domain modules, registry, split and acquisition driver under benchmarks/corpus/fori_targeted/. Put registry/split/correctness tests under tests/test_fori_targeted_*.py, outputs under results/fori_targeted/, and the collection report in docs/fori_targeted.md.
- Reuse CorpusSpec, the dataset aggregator, factor-1 static feature extraction and fresh-process sweep. Verify the subprocess actually builds from the new registry. Use a single eligible public lax.fori_loop, static bounds and unroll=factor. Check factor-1 tracing through the same path the predictor uses. Save inferred hardware, backend, JAX/jaxlib versions and compiler settings. Keep any other machine's data separate.
- For each new body check a small independent reference or mathematical invariant, finite values, factor invariance, deterministic inputs, and dependence of the result on the loop work. Keep preprocessing outside timing. Use synchronized warm samples from the existing harness; retain cold timing for compatibility.
- Randomize execution order deterministically. Use three passes so repeats of a body/size/trip are separated in time. Randomize the four factors' order within each matched block. Record pass, timestamp, order, seed and raw measurement keys. Keep paired cases on the same hardware and record interruptions. Support restart without duplicate rows. Label smoke/debug runs separately.
- Validate raw correctness flags, finite positive timings, repeat completeness, unique keys and medians independently. Check agreement of the fastest factor across repeats. A possible speedup smaller than observed repeat variation merits an uncertainty flag. Keep easy, neutral, noisy and unfavorable valid rows.
- Generate train_features.csv and sealed_eval_features.csv with the same predictor-compatible columns used in the last expansion. Each configuration needs all four factors and static inputs available before compilation. Keep timings, source/lineage IDs, filenames and partition labels outside predictor features. Include a compatibility check showing that the existing training code can reconstruct groups and the saved predictor can consume representative feature rows. Keep sealed evaluation labels out of model selection.

## Finish and hand off

Deliver the raw timing CSV, development table, sealed evaluation table, manifest, split and hashes, coverage/failures, correctness results, reproduction commands, and a numerical reconciliation of planned versus completed measurements. Give the exact file paths. Report development-only opportunity by size and trip pair, along with repeat variability. Leave model fitting and sealed performance scoring to the next agent.

This round ends at the declared matrix. At handoff, the next agent will choose the model with grouped development validation, then score the current model, the new one, factor 1 and the measured oracle on the same sealed bodies. The scorecard will include workload-weighted speedup, total warm time at equal invocation counts, downside, and a workload-bootstrap interval. If even perfect choices offer little benefit on the application mix, report that result and stop broad synthetic collection.
