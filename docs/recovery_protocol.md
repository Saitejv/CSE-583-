# Phase 3 recovery protocol (recovery split v1)

Freeze this protocol before collecting any recovery timing labels. The Week 3
records at `c3a4ce1` are historical, not a source of new validation labels.
Both freezes selected factor 1 throughout their holdout; at R=100,000 and
unlimited budget the oracle chose above 1 on 8/10 indexed-memory configurations
and the revised model's median regret there was 6.39%. Its 87.4% grid agreement
includes budgets where neither policy had a feasible candidate. The first
development question is whether **within-configuration warm ratios** predict
unroll effects on unseen bodies, not whether a model predicts absolute warm
latency or agrees in forced fallback cases.

## Data and grouping

Retain split v1, its CSV, the two frozen JSON models, and gate records verbatim.
Use only its 35 dev configurations as legacy recovery data; neither
`indexed_update` nor `jacobi_solver` enters recovery fitting, selection or
confirmation. The recovery manifest is separate from the historical split.

| Recovery partition | Family | Workload IDs and bodies |
|---|---|---|
| Dev (legacy) | recurrence, elementwise, reduction, dense_linear_algebra | Original dev bodies only |
| Dev (new) | window | `window_read` (dynamic contiguous load + reduction), `window_replace` (dynamic contiguous replacement) |
| Dev (new) | gather | `gather_lookup` (1D indirect read + weighted reduction), `gather_rows` (row reads with vector recurrence) |
| Dev (new) | scatter | `scatter_accumulate` (duplicated indexed add), `scatter_blend` (unique indexed read + nonlinear indexed set) |
| Confirmation | strided_read | `strided_mix` (two computed-offset reads + scalar recurrence) |
| Confirmation | histogram | `bucket_max` (indirect max update with repeated indices) |
| Confirmation | masked_read | `masked_rows` (two row gathers + mask-select + reduction) |
| Confirmation | ring_buffer | `ring_feedback` (loop counter, dynamic slice and feedback update via `fori_loop`) |

New workloads each have variants n64/n512 and trip counts 64, 256 and 1023;
all four factors, 3 fresh processes per setting, default sweep warmup and
sampling, CPU, JAX/jaxlib 0.12.0.1. Use the existing trace/lower/compile/run
measurement and finite numerical correctness check against factor 1. If a
factor lacks three valid repeats, drop its whole configuration and record it;
confirmation requires all 24 configurations or is inconclusive. No shape,
trip, or factor from a workload may cross a split or fold. Group dev validation
by family (seven LOFO folds); hold out both new bodies together in each new
family. Use configuration-level summaries, not replicate-level sample sizes.

## Development experiment and selection

- Compare factor-1-relative **log warm** prediction errors first: bias, MAE,
  tolerance, and ability to predict the direction of the effect for factors
  2/4/8. Compare the frozen Week 3 policy (legacy dev is in-sample for that
  artifact), a Week 3 design refit within each fold, and zero warm effect.
- Freeze the small candidate menu: refit `memory_work` at its existing alpha
  0.025; two warm-only extensions of that design at alpha 0.025 and 0.1,
  adding either indexed/addressing density or indexed/addressing density plus
  carry-traffic share and tail fraction interactions. Keep the same cold and
  factor-1 regressions and fit relative effects against the measured factor 1.
  No identifiers, split, B or R enter the models.
- On the new-dev out-of-fold rows, shortlist candidates with relative warm log
  MAE no worse than the refit. Select the smallest worst-new-family mean
  positive total-cost regret at feasible long horizons where the measured
  oracle improves on factor 1. Ignore families with no such settings. Break
  ties by relative warm log MAE, then smaller warm design, then stronger
  regularization. Refit once on the full recovery dev set. If no settings
  qualify, select by relative warm log MAE with the same remaining tie-breaks.
- Replay the original grid R = 1, 10, 100, 1,000, 10,000, 100,000 and B = 1,
  10, 100 ms or unlimited. Report absolute and relative calibration and paired
  measured agreement, violations, signed/positive regret and speedups vs factor
  1 and `llvm_size`; break out all settings, long settings (R >= 10,000) with
  at least two measured feasible factors, and that latter subset where the
  oracle improves on measured factor 1 (factor 1 must be feasible). Dev
  selection excludes decision overhead; confirm end-to-end with overhead.

## One-shot confirmation

Commit the selected versioned artifact, hash, dev inputs/results and the
confirmation runner before starting the confirmation sweep. Do not use any
confirmation costs to revise models, features, thresholds or workloads. Score
the committed model once and retain an append-only start/outcome record. Permit
read-only verification of the recorded result, never a second selection gate.

Replay the same R/B grid. A win is a strict observed net total-cost reduction
over **both** factor 1 and `llvm_size` on the **same** configuration and
setting, with all three measured cold costs within budget. Count wins on the
predeclared feasible long-horizon subset; report wins anywhere else separately,
the number of distinct winning configurations, their paired percent savings
against both baselines (including wins of at least 1% against both), and
regressions. The observed gate passes with at least one such long-horizon win;
this is not a statistical claim and does not by itself justify a new default.
Charge the learned selector a measured per-configuration uncached decision
median once per R-call session. Charge zero to both baselines (conservative to
the learned policy); include overhead in regret, not in the cold budget. Report
cache-hit timing separately. Use 2,000 percentile bootstrap resamples with
seed 583, sampling configurations, not settings or process repeats; combine
repeated settings within configuration before pooled resampling. Inspect
optimized HLO of winning configurations offline before interpreting wins.
If the gate fails, record it and stop tuning on this set. Factor 1 remains
the Phase 4 default pending a broader evidence base; learned policy is opt-in.
