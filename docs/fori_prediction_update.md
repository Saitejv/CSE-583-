# Unroll prediction on unchanged JAX fori_loop bodies

A shorter-trip profile improved both prediction accuracy and average warm execution on the tested recurring loops. At 767 and 768 iterations, correcting the static forecast with timings from 511 and 512 iterations cut factor-effect error by **46.1%** and improved workload-weighted warm speedup from **1.054× to 1.080×** versus native factor 1. The static refit improved some metrics and left the incumbent as the general static selector.

## Data and evaluation

OpenCode with DeepSeek V4.1 Flash and Anti-Gravity CLI with Gemini 3.8 Flash each drafted 12 workloads. Review removed five close analogues of existing benchmarks: coupled oscillator, stencil diffusion, thermostat hysteresis, EMA, and histogram. The 19 retained bodies cover eight groups. Nine bodies supplied 72 new development configurations. Ten bodies from four whole groups supplied the fresh test. The static fit used 517 configurations from 137 bodies after adding the two historical development datasets.

The new acquisition recorded **2,304 factor timings** across 192 distinct configurations and three fresh CPU processes per configuration. It compiled native factors 1, 2, 4 and 8 and interleaved synchronized warm calls. Every factor passed the corpus tolerance and had the same Jaxpr after substituting its unroll factor. The hardware was Apple M2 with JAX and jaxlib 0.12.0.1. [OpenCode bodies](../benchmarks/corpus/fori_prediction/opencode_cases.py) · [Anti-Gravity bodies](../benchmarks/corpus/fori_prediction/agy_cases.py)

Each speedup uses native factor 1 as the baseline. Workload-weighted speedup is the geometric mean with equal weight per body. Pooled speedup compares the sum of warm times with one call per configuration. Factor-effect log MAE measures how accurately the model predicts the change from factor 1 for factors 2, 4 and 8. It compares predicted and measured `log(T_u / T_1)`.

## Static refit

Five-fold validation kept related families together. The selected predictor combines 75% of the incumbent's log forecast with 25% of a new forecast. The new model fits baseline and cold costs with extra trees, and uses a shared Huber regressor for warm factor effects. It was selected ahead of holdout measurement. Its grouped development effect log MAE fell from 0.279 to 0.271 with equal weight per collection and workload. Its development workload-weighted warm speedup rose from 1.027× to 1.036×.

| Evaluation | Effect log MAE, incumbent / refit | Workload-weighted warm speedup | Pooled warm speedup | Worst warm slowdown |
|---|---:|---:|---:|---:|
| Fresh, 10 bodies and 80 configurations | 0.286 / 0.289 | 1.032× / 1.029× | 1.007× / 1.024× | 1.048× / 1.048× |
| Archived targeted, 24 bodies and 192 configurations | 0.230 / 0.227 | 1.043× / 1.041× | 1.130× / 1.112× | 1.372× / 1.096× |

The same refit trained on historical development data alone reached 1.038× workload-weighted speedup on the fresh set and 1.031× on the archived set. Adding the generated development data had different effects across these cohorts. Both paired intervals on the fresh set include their neutral value. The speedup over the incumbent was 0.997× [0.985, 1.009] and the log MAE reduction was −0.003 [−0.014, 0.005]. The incumbent remains the static choice. [Validation grid](../results/fori_prediction/validation.csv) · [Static evaluation and data ablation](../results/fori_prediction/evaluation.json)

The expansion archive and current traces use different Jaxpr fingerprints on 246 configurations. All recorded base features used in the fit matched the current traces on those rows. The overlap audit checks both fingerprints. [Training keys and fingerprints](../results/fori_prediction/training_keys.csv)

## Follow-up with a shorter-trip profile

For a recurring workload, the model can use four warm timings from the same loop at a shorter trip count. It adds half of the per-factor log-time residual from the profile to the static prediction for the longer trip and applies a 0.03 log margin before selecting a larger factor. The profile and target share the same loop signature and trip-count remainder modulo 8. This keeps the remainder the same for all four factors. The profile design and its parameters were chosen on the generated development bodies. The 767/768 confirmation used new target timings collected after the profile correction was fixed.

| Target and profile trip counts | Factor-effect log MAE, static / profiled | Workload-weighted warm speedup | Pooled warm speedup | Within 5% of fastest factor |
|---|---:|---:|---:|---:|
| 511/512 from 127/256 | 0.304 / 0.184 | 1.045× / 1.058× | 1.009× / 1.011× | 65.0% / 77.5% |
| 767/768 from 511/512 | 0.309 / 0.167 | 1.054× / 1.080× | 1.009× / 1.018× | 50.0% / 80.0% |

Each row covers 40 target configurations from the same 10 held-out bodies. The 767/768 experiment measures transfer to new trip counts on those bodies. At 767/768, workload-bootstrap intervals for the paired improvements over the static model were **1.010–1.043×** for speedup and **0.059–0.286** for factor-effect log MAE reduction. The pooled improvement over the static model was **0.9%**. Zero profiled configurations were slower than factor 1 in the median timings at either target pair. [Profile evaluation and development choices](../results/fori_prediction/profile_evaluation.json)

The benchmark replays selections against measured compiled kernels and independently checks that the wrapper chooses the same factor at a live trace. Warm comparisons exclude model inference and profile setup. On the fresh test, the median sum of the four first-call times ranged from 127 to 172 ms across the four profile counts, with additional time for warm sampling. A profile already available on the target machine avoids that acquisition cost. Keep each record scoped to the compiler version and input distribution used when collecting it.

## Reproduction and integration

```sh
export PYTHONPATH=src:.
export JAX_PLATFORMS=cpu
.venv/bin/python -m benchmarks.corpus.fori_prediction.acquire train
.venv/bin/python -m benchmarks.corpus.fori_prediction.train fit
.venv/bin/python -m benchmarks.corpus.fori_prediction.train ablate
.venv/bin/python -m benchmarks.corpus.fori_prediction.acquire holdout
.venv/bin/python -m benchmarks.corpus.fori_prediction.train score
.venv/bin/python -m benchmarks.corpus.fori_prediction.acquire confirm
.venv/bin/python -m benchmarks.corpus.fori_prediction.profile
```

Give `WarmResidualProfile` the existing `ForiCostPredictor`, the factor-1 feature row at the profile trip count, and four measured warm times in factor order. Wrap it in `DecisionEngine`, then pass that engine to `auto_fori_loop` with `throughput_only=True` as in the [wrapper guide](fori_predictor.md). When the profile signature or remainder does not match, the engine uses the reference forecasts. [Profile model implementation](../src/jax_unroll/profile_model.py)
