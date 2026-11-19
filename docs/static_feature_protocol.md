# Static feature experiment v2: premeasurement protocol

Declared 2026-11-17 before new timings. Development only; stop before any
confirmation gate. Existing 47-body pilot is development data. Do not read
historical holdout or scored confirmation labels for tuning. Frozen recovery
artifact and feature v1 remain unchanged; factor 1 stays default.

## Representation and fixed statistical design

A separate static extension (experimental version 2) traces factor 1 only.
Structure block: log1p of carry_slice_count, carry_concatenate_count,
carry_slice_bytes (per read proxy, not distinct memory traffic),
carry_updated_bytes, carry_dependency_pairs, carry_to_carry_max_depth.
Index block: log1p of index_gather_count, index_scatter_count,
index_scatter_add_count, index_scatter_min_count, index_scatter_other_count,
index_from_constant, index_from_counter, index_from_scanned,
index_from_carry, index_from_unknown, index_mixed_sources_count.
Each log feature is divided by 8 and interacted with (1-1/u) and log(u).
Append to the existing eight memory_work warm-effect terms. Count auxiliary
opaque-call fields for audit; do not use them in regressions. Propagate SSA
roots through supported jit calls; distinguish loop-invariant inputs from
compile-time literals conservatively. Carry provenance means dependence,
not proof of changing values. No runtime locality, collision, names, hashes,
B, or R as model inputs. No candidate lowering, compiling or executing at
decision time.

Primary candidate: all structure+index terms, ridge alpha .025, no intercept.
Predeclared ablations: structure only; index only. All use the same .025 and
fixed transforms. No hyperparameter or feature selection from these outputs;
therefore no nested model selection is required. Report every candidate even
if the primary fails; no switching the headline winner or tuning to pass.
Compare zero effect, training-fold per-factor median, frozen recovery, and
same-design warm-only refit first. Keep cold and factor-1 warm forecasts frozen.

## New matched measurements (120 rows, not full sweep)

Three bodies with two runtime-only variants each:

* matched_gather: float32 table[65536], scanned int32[T,256], scalar carry;
  clustered indices in 64 adjacent entries vs dispersed stride 251 mod65536.
* matched_scatter_add: float32 carry[4096], scanned indices and float32 updates
  [T,256]; unique permutation slices vs 256 updates to eight buckets.
* matched_scatter_min: same abstract arguments, unique vs colliding indices.

T=1023 for all three; T=1024 additionally for gather and scatter-add only.
Factors 1,2,4,8, three fresh-process repeats; seed583 shuffled order, default
sweep warmups/adaptive samples. Pin CPU JAX/jaxlib0.12.0.1; disable persistent
cache and x64. Same finite-output and factor-1 allclose protocol as pilot.
Verify exact body Jaxpr and every deterministic decision feature is identical
within each runtime pair at every trip/factor before measuring. Preserve raw
rows. Missing/invalid matrix drops the whole matched body, never a performance
filter. Analyze repeatwise differences of factor-relative warm log ratios;
report their signs and ranges, without treating sign consistency as a
significance test or dropping unstable pairs. Locality and collision patterns
are controlled inputs, never inferred static features.

## Folds, weighting and replay

Use exactly the pilot's eight transitive family components. All related
bodies/shapes/trips/factors stay together. All three matched bodies and both
variants join the indirect_gather+indirect_scatter+locality_access component.
Headline evaluation uses the original 47-body matrix; separately report the
augmented matrix as a sensitivity diagnostic. Both exclude their entire test
component during fitting. Repeated shapes/trips/variants share their body
bootstrap identity; reduce paired errors within body first. Report row-pooled
MAE and per-fold MAE, and paired mean absolute-error differences with 2,000
body bootstrap draws seed583 (vs zero, median, and same-design refit).
Repeat stability is separate from aggregate median labels.

Replay the existing B/R grid on held-out forecasts. Report all settings,
feasible long horizons, and oracle-beats-factor1 long horizons separately:
regret, agreement, speedup, measured violations including those where a
measured feasible factor exists. Save paired rows. No decision overhead is
charged: label all replays diagnostic, not end-to-end.

Promising iff fixed primary beats BOTH zero and per-factor median in pooled
held-out MAE, beats zero in >4 of 8 folds, and the 95% paired body-bootstrap
interval vs zero has upper bound <0. Failure ends this experiment; explain
representation limits without further tuning. Passing does not replace the
frozen policy/default and does not authorize a new confirmation gate.
