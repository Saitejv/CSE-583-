# CPU unroll selector evaluation

Frozen model `cpu-ridge-v1-7864d652dd6de258`. CPU sweep labels use JAX 0.12.0.1 and jaxlib 0.12.0.1. Selector total includes a 0.4984 ms decision charge per configuration.

Totals use measured cold ms + R × measured warm µs / 1,000. The oracle chooses the fastest measured feasible factor and uses factor 1 when every measured factor is infeasible. Violation uses measured cold time. A budget-violating choice can have negative signed regret. The CSV includes paired speedups and 95% percentile intervals from 2,000 bootstrap draws of entire configurations within each group (seed 583).

## Dev leave-one-family-out

Factor agreement, budget violations, and mean positive regret pool the R/B grid before overhead. Relative MAE uses measured cost as denominator.

| Fold | Configs | C log MAE | T log MAE | C rel. MAE | T rel. MAE | Agreement | Violations | Mean positive regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recurrence | 8 | 0.471 | 0.782 | 61.7% | 57.7% | 82.3% | 50.0% | 2.45% |
| elementwise | 10 | 0.204 | 1.700 | 22.4% | 76.9% | 95.0% | 50.0% | 1.26% |
| reduction | 8 | 0.414 | 1.025 | 29.8% | 184.3% | 99.0% | 50.0% | 0.17% |
| dense_linear_algebra | 9 | 0.316 | 0.618 | 25.4% | 88.6% | 78.7% | 50.0% | 0.43% |
| pooled | 35 | 0.342 | 1.058 | 33.8% | 100.1% | 88.8% | 50.0% | 1.07% |

## Held-out policy comparison

Unlimited budget rows show all seven policies. The CSV includes every budget and family.

| R | B (ms) | Policy | Median total (ms) | Median regret [95% CI] | Violations | Agreement | Factor × count |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | unlimited | factor_1 | 34.133 | 0.00% [0.00%, 0.00%] | 0.0% | 100.0% | 1×20 |
| 1 | unlimited | factor_2 | 44.019 | 40.40% [14.48%, 100.85%] | 0.0% | 0.0% | 2×20 |
| 1 | unlimited | factor_4 | 187.542 | 168.80% [41.00%, 304.05%] | 0.0% | 0.0% | 4×20 |
| 1 | unlimited | factor_8 | 365.465 | 409.10% [83.17%, 701.76%] | 0.0% | 0.0% | 8×20 |
| 1 | unlimited | llvm_size | 206.877 | 228.35% [83.17%, 304.05%] | 0.0% | 0.0% | 4×10, 8×10 |
| 1 | unlimited | selector | 34.631 | 1.46% [0.36%, 1.53%] | 0.0% | 100.0% | 1×20 |
| 1 | unlimited | measured_oracle | 34.133 | 0.00% [0.00%, 0.00%] | 0.0% | 100.0% | 1×20 |
| 10 | unlimited | factor_1 | 35.065 | 0.00% [0.00%, 0.00%] | 0.0% | 100.0% | 1×20 |
| 10 | unlimited | factor_2 | 45.352 | 41.86% [13.95%, 97.12%] | 0.0% | 0.0% | 2×20 |
| 10 | unlimited | factor_4 | 194.618 | 141.59% [41.11%, 299.93%] | 0.0% | 0.0% | 4×20 |
| 10 | unlimited | factor_8 | 372.559 | 335.00% [81.63%, 691.47%] | 0.0% | 0.0% | 8×20 |
| 10 | unlimited | llvm_size | 213.801 | 201.81% [81.63%, 299.93%] | 0.0% | 0.0% | 4×10, 8×10 |
| 10 | unlimited | selector | 35.563 | 1.42% [0.28%, 1.50%] | 0.0% | 100.0% | 1×20 |
| 10 | unlimited | measured_oracle | 35.065 | 0.00% [0.00%, 0.00%] | 0.0% | 100.0% | 1×20 |
| 100 | unlimited | factor_1 | 49.902 | 0.00% [0.00%, 0.00%] | 0.0% | 100.0% | 1×20 |
| 100 | unlimited | factor_2 | 69.849 | 30.80% [12.28%, 91.80%] | 0.0% | 0.0% | 2×20 |
| 100 | unlimited | factor_4 | 261.419 | 80.42% [39.89%, 266.65%] | 0.0% | 0.0% | 4×20 |
| 100 | unlimited | factor_8 | 444.379 | 183.07% [73.70%, 607.82%] | 0.0% | 0.0% | 8×20 |
| 100 | unlimited | llvm_size | 267.423 | 139.54% [73.70%, 266.65%] | 0.0% | 0.0% | 4×10, 8×10 |
| 100 | unlimited | selector | 50.401 | 1.00% [0.15%, 1.27%] | 0.0% | 100.0% | 1×20 |
| 100 | unlimited | measured_oracle | 49.902 | 0.00% [0.00%, 0.00%] | 0.0% | 100.0% | 1×20 |
| 1,000 | unlimited | factor_1 | 213.533 | 0.00% [0.00%, 0.00%] | 0.0% | 85.0% | 1×20 |
| 1,000 | unlimited | factor_2 | 318.370 | 22.95% [3.87%, 68.92%] | 0.0% | 15.0% | 2×20 |
| 1,000 | unlimited | factor_4 | 563.257 | 63.23% [26.21%, 154.89%] | 0.0% | 0.0% | 4×20 |
| 1,000 | unlimited | factor_8 | 873.482 | 96.89% [53.91%, 322.03%] | 0.0% | 0.0% | 8×20 |
| 1,000 | unlimited | llvm_size | 559.838 | 73.26% [53.91%, 166.00%] | 0.0% | 0.0% | 4×10, 8×10 |
| 1,000 | unlimited | selector | 214.032 | 0.28% [0.12%, 0.85%] | 0.0% | 85.0% | 1×20 |
| 1,000 | unlimited | measured_oracle | 208.532 | 0.00% [0.00%, 0.00%] | 0.0% | 100.0% | 1×17, 2×3 |
| 10,000 | unlimited | factor_1 | 1240.308 | 0.00% [0.00%, 2.21%] | 0.0% | 60.0% | 1×20 |
| 10,000 | unlimited | factor_2 | 1496.860 | 20.93% [0.07%, 47.02%] | 0.0% | 30.0% | 2×20 |
| 10,000 | unlimited | factor_4 | 2297.717 | 60.55% [15.04%, 72.99%] | 0.0% | 5.0% | 4×20 |
| 10,000 | unlimited | factor_8 | 3133.586 | 68.91% [38.62%, 104.20%] | 0.0% | 5.0% | 8×20 |
| 10,000 | unlimited | llvm_size | 2200.972 | 60.55% [38.62%, 77.38%] | 0.0% | 5.0% | 4×10, 8×10 |
| 10,000 | unlimited | selector | 1240.807 | 0.06% [0.02%, 2.44%] | 0.0% | 60.0% | 1×20 |
| 10,000 | unlimited | measured_oracle | 1183.991 | 0.00% [0.00%, 0.00%] | 0.0% | 100.0% | 1×12, 2×6, 4×1, 8×1 |
| 100,000 | unlimited | factor_1 | 10864.011 | 0.00% [0.00%, 4.75%] | 0.0% | 60.0% | 1×20 |
| 100,000 | unlimited | factor_2 | 12651.018 | 18.96% [0.05%, 37.11%] | 0.0% | 30.0% | 2×20 |
| 100,000 | unlimited | factor_4 | 18593.763 | 52.34% [12.66%, 56.88%] | 0.0% | 5.0% | 4×20 |
| 100,000 | unlimited | factor_8 | 19199.917 | 60.25% [36.89%, 66.83%] | 0.0% | 5.0% | 8×20 |
| 100,000 | unlimited | llvm_size | 18099.420 | 52.34% [36.36%, 56.80%] | 0.0% | 5.0% | 4×10, 8×10 |
| 100,000 | unlimited | selector | 10864.510 | 0.01% [0.00%, 4.78%] | 0.0% | 60.0% | 1×20 |
| 100,000 | unlimited | measured_oracle | 10864.011 | 0.00% [0.00%, 0.00%] | 0.0% | 100.0% | 1×12, 2×6, 4×1, 8×1 |

## Frozen Week 3 gate

**FAIL** on evaluation #1. 20 configurations scored once. A win requires strictly lower measured total including selector overhead against both factor 1 and LLVM size on the same configuration, with all three choices within any finite budget.

| R | B (ms) | Indexed memory wins | Iterative solver wins |
|---:|---:|---:|---:|
| 1 | unlimited | 0 | 0 |
| 1 | 1 | 0 | 0 |
| 1 | 10 | 0 | 0 |
| 1 | 100 | 0 | 0 |
| 10 | unlimited | 0 | 0 |
| 10 | 1 | 0 | 0 |
| 10 | 10 | 0 | 0 |
| 10 | 100 | 0 | 0 |
| 100 | unlimited | 0 | 0 |
| 100 | 1 | 0 | 0 |
| 100 | 10 | 0 | 0 |
| 100 | 100 | 0 | 0 |
| 1,000 | unlimited | 0 | 0 |
| 1,000 | 1 | 0 | 0 |
| 1,000 | 10 | 0 | 0 |
| 1,000 | 100 | 0 | 0 |
| 10,000 | unlimited | 0 | 0 |
| 10,000 | 1 | 0 | 0 |
| 10,000 | 10 | 0 | 0 |
| 10,000 | 100 | 0 | 0 |
| 100,000 | unlimited | 0 | 0 |
| 100,000 | 1 | 0 | 0 |
| 100,000 | 10 | 0 | 0 |
| 100,000 | 100 | 0 | 0 |

## Held-out prediction diagnostics

| Family | C log MAE | T log MAE | C median prediction / measurement | T median prediction / measurement |
|---|---:|---:|---:|---:|
| indexed_memory | 0.139 | 3.061 | 0.926 | 23.021 |
| iterative_solver | 2.262 | 1.130 | 0.084 | 3.108 |
