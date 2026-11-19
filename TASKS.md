# Next tasks for `auto_fori_loop`

Improve end-to-end runtime on unchanged `jax.lax.fori_loop` bodies. Prioritize training on costly prediction mistakes, then use the improved forecasts to decide when unrolling pays off.

## 1. Train on costly prediction mistakes

- [x] Rank 517 historical development configurations by warm regret, including harmful unrolling and missed speedups at factor 1. [Ranked cases](results/fori_regret/development_regret.csv)
- [x] Review new drafts from OpenCode and Anti-Gravity, freeze the family split, and sweep factors 1, 2, 4, and 8 on 56 configurations in three CPU processes. Verified identical inputs, equivalent outputs, and Jaxprs differing only in unroll. Retained all cold and warm samples. [Frozen split](results/fori_regret/split.json)
- [x] Train on 549 development configurations and score against the incumbent on 24 configurations from three fresh families. Effect log MAE fell from 0.414 to 0.409. Pooled warm speedup versus factor 1 improved from 0.869× to 0.981×, and slowdowns above 3% fell from six to four. Native factor 1 was fastest on every held-out configuration. [Holdout metrics](results/fori_regret/holdout.json)

## 2. Unroll only when the expected savings justify setup

- [x] Add an opt-in payoff rule using the expected call count, incremental first-call cost, trace-time decision cost, and development residual margins. Use factor 1 when the conservative payoff is nonpositive or the candidate exceeds the cold budget. [Calibration](results/fori_regret/payoff.json)
- [x] Measure the actual wrapper at 1, 300, and 5,000 synchronized calls on all 24 held-out configurations, including model load, tracing, decision, compilation, and execution. At 5,000 calls the refit saved 6.7% versus the incumbent, and the payoff rule saved about 9%. Native factor 1 had the lowest total runtime at every horizon and the payoff rule had no pooled break-even. Keep the incumbent as the general static choice. [Results and commands](docs/fori_regret.md)

Verified the full test suite with `PYTHONPATH=src:. JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q` (**1,200 passed**).

## 3. Broaden evaluation around the proposal

- [x] Freeze eight fresh bodies covering loop-control-heavy recurrences and lane-wise arithmetic, dense updates, indexing, and simulation. Preserve the proposal-style, historical, and prior regression workloads as separate controls. [Frozen matrix](results/fori_proposal/matrix.json)
- [x] Sweep native factors 1, 2, 4, and 8. Verified repeatable warm wins on 16 of 32 fresh configurations, reaching 2.44× on lane-wise moments. Verified equivalent outputs and Jaxprs differing only by unroll. [Warm results](results/fori_proposal/native_summary.json)
- [x] Export the incumbent's numerical forecast data for low-overhead inference. Verified exact forecast parity across all 41 swept configurations. [Compact model calibration](results/fori_proposal/compact_payoff.json)
- [x] Directly time the wrapper at 1, 300, and 5,000 calls on 24 fresh and nine control configurations. Confirm 20,000 calls separately on all 12 live fresh cases with a measured warm opportunity. The compact model was 2.7% slower pooled on all fresh cases at 5,000 calls and 7.7% faster pooled on the opportunity cohort at 20,000 calls. [Evaluation and commands](docs/fori_proposal_evaluation.md)

Replayed both runtime summaries from 1,890 raw live rows and verified the full CPU suite (**1,204 passed**).

## 4. Run five measured optimization cycles

- [x] Compare model storage, trace reuse, proposal-augmented refitting, per-call savings calibration, and deferred loading. Keep the uncompressed export and deferred loading. Keep trace reuse in the benchmark and retain the warm incumbent after the accuracy/runtime comparison. [Five-cycle report](docs/fori_iterations.md)
- [x] Validate on six frozen families with native factors 1, 2, 4, and 8 and direct totals through 20,000 calls. The incumbent saved 8.8% pooled at 20,000 calls versus native factor 1. The savings rule chose two larger factors at that horizon and was effectively tied with native. [Confirmation results](results/fori_iterations/confirmation_summary.json)

Replayed all 864 live measurements and verified the full CPU suite (**1,209 passed**).

[Current predictor results](docs/fori_prediction_update.md) · [Live wrapper benchmark](docs/fori_predictor.md)

## 5. Optimize the horizon-specific decision

- [x] Ranked the compact incumbent's factor-choice regret on 626 historical configurations at 300, 5,000, and 20,000 calls. The prior mixed confirmation left only 87 ms at the long horizon. Two older dense-update bodies accounted for 15.685 s of the targeted suite's 24.349 s gap. [Replay rankings](results/fori_horizon/development_regret.csv)
- [x] Calibrated an opt-in matrix rule using family-grouped development forecasts and compared it to a simple crossover. The warm incumbent remains the general long-horizon selector. The short bypass saved 0.101 s at one call and 0.092 s at 300 on the 14 originally frozen fresh configurations.
- [x] Verified native factors and measured 1,035 end-to-end wrapper/native/oracle totals in 105 fresh CPU processes, with every frozen case retained. On three separately frozen matrix-update families, the targeted decision took 2.949 s versus 5.713 s for the incumbent and 5.700 s for native at 20,000 calls. It recovered 80.7% of direct oracle headroom. [Report, limitations, and reproduction](docs/fori_horizon.md)

Verified the full CPU test suite (**1,225 passed**) and reproduced the summaries from the saved direct measurements.

## 6. Expand the family-grouped training and evaluation corpus

- [x] Used three OpenCode free-model sessions and two agy CLI sessions to draft native loop bodies. Retained 13 bodies, froze 28 training and 24 held-out configurations, and collected 624 per-factor process measurements on CPU. On training data, 20 of 28 configurations had a repeat-supported warm gain of at least 1.05×. Verified outputs, graph equivalence, label units and split isolation. All four focused tests passed. [Corpus, data loader and commands](docs/fori_agent_cases.md)
