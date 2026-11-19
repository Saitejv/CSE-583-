# Expanded corpus pilot: more bodies, same features

**Do not launch the full sweep for model training yet.** The new corpus is
useful: every body ran correctly, and larger unroll factors offer meaningful
gains on some of them. But refitting the existing warm-effect terms on the
expanded corpus made predictions worse across held-out family groups.
Keep factor 1 as the default and recovery v1 unchanged as opt-in.

## What ran

OpenCode delivered 47 bodies in 10 families and a reproducible manifest.
`make corpus-validate` independently verified the registry, unique body hashes,
and static loop support. The [predeclared pilot](corpus_pilot_protocol.md)
ran all 47 bodies at their primary shapes and T=1023, four factors and three
fresh-process repeats: **564/564 correct**, no exclusions, 8.2 minutes on
Apple M2 CPU with JAX/jaxlib 0.12.0.1. Related-group links merged gather,
scatter and locality into one test component; the other seven families
remain separate. Neither recovery confirmation nor historical holdout labels
enter this pilot. The committed recovery artifact was used read-only.

## Warm-ratio prediction

| Comparison | Mean absolute log-ratio error ↓ | Difference vs zero [95% body-bootstrap CI] |
|---|---:|---:|
| Frozen recovery | 0.337 | −0.022 [−0.043, +0.001] |
| Zero effect | 0.359 | — |
| Per-factor median (grouped CV) | 0.385 | +0.026 [+0.015, +0.036] |
| Warm-effect ridge refit (grouped CV) | 0.460 | +0.101 [+0.019, +0.198] |

The refit lost to both simple baselines and beat zero in **3 of 8** held-out
components. The frozen recovery model's small advantage over zero is not
clear at this sample size. The timing labels themselves showed signal:
**133/141** factor effects kept the same sign in all three repeat blocks.
Median within-body log-ratio range across repeats was 0.018, compared
with 0.155 for the median absolute measured effect. This suggests
repeat-to-repeat noise is not the main limit in this matrix; it does not
establish transfer to other trips, shapes, or machines.

## Does it select useful factors?

Of 180 feasible long-horizon settings, the oracle improved on factor 1 in
89 settings spanning 27 bodies. On that useful subset, factor 1 had 16.24%
median oracle regret, frozen recovery 9.12%, and the warm-only refit 2.94%.
But the refit lowered agreement over **all feasible long settings** from
48.9% to 42.8%, raised mean positive regret from 10.91% to 13.12%, and still
had 1.000× median speedup vs factor 1. It picked the measured oracle in
only 20.2% of the useful settings, up from 7.9% frozen. Across the full grid
564 settings had no feasible measured choice. The frozen policy exceeded
budget in those 564 **plus five settings with a feasible measured choice**;
the refit exceeded budget only in the 564. This replay charges no decision
overhead and is not a confirmation gate.

## Next experiment

Target the **static representation** before adding thousands of labels.
As a post-hoc sanity check, factor 8 made `stencil_anisotropic_2d` 29.7×
slower warm (frozen forecast: 1.36× slower) but made
`rolling_dot_two_streams` 2.46× faster (forecast: 1.07× slower). Inspecting
optimized HLO offline, each still had an outer `while`; the stencil's
optimized IR grew from about 11 to 638 kB, with fusion ops rising from
3 to 82, while the rolling workload grew from about 10 to 25 kB.
This contrast is descriptive, not proof of which optimization caused the
timing difference.

First test a small, versioned feature experiment around stencil slice/
concatenate structure, carry traffic and dependencies, and gather/scatter
kind and index provenance. Feature schema v1 misses `scatter-add` and
`scatter-min` in its gather/scatter count; fix that only in a new schema,
not in frozen features. Use matched dev workloads that vary index locality
without changing the Jaxpr to find where static-only inference cannot
distinguish cases. Include one additional divisible trip point in a small
diagnostic subset before considering the full grid. Keep group separation
and a zero-effect baseline, and keep lowering/compilation offline.

## Reproduce

- `make corpus-pilot-report` rebuilds the report from the retained CSV, without new timings.
- [Report JSON](../results/corpus_pilot/report.json) includes folds, per-fold calibration, budgets and paired baseline summaries.
- [Warm ratios](../results/corpus_pilot/warm_ratios.csv) and [raw timings](../results/corpus_pilot/primary_1023_raw.csv) retain per-body results and sweep evidence.
- `make recovery-verify` and `make verify-week3` verify prior evidence separately, without a new gate.
