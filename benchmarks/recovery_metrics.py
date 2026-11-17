"""Matched-setting recovery diagnostics, selection strata and configuration CIs."""

from collections import Counter, defaultdict
from math import log

import numpy as np

from benchmarks.recovery_data import config_key
from jax_unroll.schema import FACTORS
from jax_unroll.selector import total_ms
from jax_unroll.validation import B_GRID, LONG_R, R_GRID, choices, prediction_errors

SEED = 583
BOOTSTRAPS = 2_000
POLICIES = ("selector", "factor_1", "llvm_size", "measured_oracle")
STRATA = ("all", "feasible_long", "oracle_improves_long")


def calibration(cases):
    actual = np.array([[log(case.measured[factor][1] / case.measured[1][1]) for factor in FACTORS[1:]] for case in cases])
    predicted = np.array([[
        case.predictions[factor].log_warm_us - case.predictions[1].log_warm_us for factor in FACTORS[1:]
    ] for case in cases])

    def relative(errors, estimates, targets):
        return {
            "bias_log": float(errors.mean()), "mae_log": float(abs(errors).mean()),
            "within_10_percent": float((abs(np.expm1(errors)) <= 0.10).mean()),
            "direction_agreement": float((np.sign(estimates) == np.sign(targets)).mean()),
        }

    return {
        "configurations": len(cases), "absolute": prediction_errors(cases),
        "warm_relative": relative(predicted - actual, predicted, actual),
        "warm_relative_zero": relative(-actual, np.zeros_like(actual), actual),
        "measured_warm_faster_fraction": float((actual < 0).mean()),
        "warm_relative_by_factor": {
            factor: relative(predicted[:, index] - actual[:, index], predicted[:, index], actual[:, index])
            for index, factor in enumerate(FACTORS[1:])
        },
    }


def paired_rows(cases, overhead=None):
    rows = []
    for case in cases:
        name, variant, trip = config_key(case.features)
        miss_ms = 0.0 if overhead is None else overhead[(name, variant, trip)]
        for budget in B_GRID:
            measured_feasible = {factor for factor in FACTORS if budget is None or case.measured[factor][0] <= budget}
            for horizon in R_GRID:
                factors, _ = choices(case, horizon, budget)
                oracle = factors["measured_oracle"]
                totals = {policy: total_ms(*case.measured[factors[policy]], horizon) for policy in POLICIES}
                totals["selector"] += miss_ms
                long_feasible = horizon >= LONG_R and len(measured_feasible) >= 2
                useful = long_feasible and 1 in measured_feasible and totals["measured_oracle"] < totals["factor_1"]
                eligible = bool(measured_feasible) and all(factors[policy] in measured_feasible for policy in POLICIES)
                row = {
                    "workload": name, "family": case.features["family"], "variant": variant, "trip_count": trip,
                    "R": horizon, "B_ms": "unlimited" if budget is None else int(budget), "decision_ms": miss_ms,
                    "measured_feasible_factors": len(measured_feasible), "feasible_long": long_feasible,
                    "oracle_improves_long": useful, "both_baselines_and_selector_feasible": eligible,
                    "paired_win": eligible and totals["selector"] < min(totals["factor_1"], totals["llvm_size"]),
                }
                for policy in POLICIES:
                    factor = factors[policy]
                    row.update({
                        f"{policy}_factor": factor, f"{policy}_total_ms": totals[policy],
                        f"{policy}_cold_ms": case.measured[factor][0],
                        f"{policy}_violates_budget": factor not in measured_feasible,
                        f"{policy}_agreement": factor == oracle,
                    })
                rows.append(row)
    return rows


def select_stratum(rows, name):
    return rows if name == "all" else [row for row in rows if row[name]]


def _matrix(rows, policy):
    return np.array([[
        row[f"{policy}_total_ms"] / row["measured_oracle_total_ms"] - 1,
        row["factor_1_total_ms"] / row[f"{policy}_total_ms"],
        row["llvm_size_total_ms"] / row[f"{policy}_total_ms"],
    ] for row in rows])


def metrics(rows):
    if not rows:
        return {"settings": 0, "configurations": 0}
    report = {
        "settings": len(rows), "configurations": len({config_key(row) for row in rows}),
        "no_measured_feasible_settings": sum(not row["measured_feasible_factors"] for row in rows),
        "paired_feasible_settings": sum(row["both_baselines_and_selector_feasible"] for row in rows),
        "paired_wins": sum(row["paired_win"] for row in rows),
        "policies": {},
    }
    for policy in POLICIES:
        regret, speedup_one, speedup_llvm = _matrix(rows, policy).T
        report["policies"][policy] = {
            "factor_counts": dict(Counter(str(row[f"{policy}_factor"]) for row in rows)),
            "agreement": sum(row[f"{policy}_agreement"] for row in rows) / len(rows),
            "budget_violations": sum(row[f"{policy}_violates_budget"] for row in rows),
            "budget_violations_when_feasible": sum(
                row[f"{policy}_violates_budget"] and row["measured_feasible_factors"] > 0 for row in rows
            ),
            "budget_violation_rate": sum(row[f"{policy}_violates_budget"] for row in rows) / len(rows),
            "median_regret": float(np.median(regret)),
            "mean_positive_regret": float(np.maximum(regret, 0).mean()),
            "median_speedup_vs_factor_1": float(np.median(speedup_one)),
            "median_speedup_vs_llvm_size": float(np.median(speedup_llvm)),
        }
    return report


def describe(cases, rows):
    return {
        "calibration": calibration(cases),
        "strata": {name: metrics(select_stratum(rows, name)) for name in STRATA},
    }


def bootstrap(rows):
    if not rows:
        return None
    by_config = defaultdict(list)
    for row in rows:
        by_config[config_key(row)].append(row)
    values = np.array([np.median(_matrix(by_config[key], "selector"), axis=0) for key in sorted(by_config)])
    draws = np.random.default_rng(SEED).integers(len(values), size=(BOOTSTRAPS, len(values)))
    lower, upper = np.quantile(np.median(values[draws], axis=1), (0.025, 0.975), axis=0)
    return {
        field: {"median": float(median), "ci95": [float(low), float(high)]}
        for field, median, low, high in zip(
            ("regret", "speedup_vs_factor_1", "speedup_vs_llvm_size"), np.median(values, axis=0), lower, upper, strict=True,
        )
    }


def _paired_summary(rows):
    paired = [row for row in rows if row["both_baselines_and_selector_feasible"]]
    wins = [row for row in paired if row["paired_win"]]
    savings = [
        [1 - row["selector_total_ms"] / row[f"{policy}_total_ms"] for policy in ("factor_1", "llvm_size")]
        for row in wins
    ]
    regressions = {}
    for baseline in ("factor_1", "llvm_size"):
        losses = [(row, row["selector_total_ms"] / row[f"{baseline}_total_ms"] - 1) for row in paired]
        losses = [(row, loss) for row, loss in losses if loss > 0]
        regressions[baseline] = {
            "settings": len(losses), "configurations": len({config_key(row) for row, _ in losses}),
            "at_least_1_percent": sum(loss >= 0.01 for _, loss in losses),
            "worst_fraction": max((loss for _, loss in losses), default=0.0),
            "median_fraction": float(np.median([loss for _, loss in losses])) if losses else None,
        }
    return {
        "settings": len(paired), "configurations": len({config_key(row) for row in paired}),
        "win_settings": len(wins), "win_configurations": len({config_key(row) for row in wins}),
        "win_settings_at_least_1_percent_vs_both": sum(min(pair) >= 0.01 for pair in savings),
        "win_savings_fraction": {
            baseline: {"min": float(min(values)), "median": float(np.median(values)), "max": float(max(values))}
            for baseline, values in zip(("factor_1", "llvm_size"), zip(*savings))
        },
        "regressions": regressions, "bootstrap": bootstrap(paired),
    }


def confirmation_summary(rows):
    strata = {name: select_stratum(rows, name) for name in STRATA}
    strata["outside_feasible_long"] = [row for row in rows if not row["feasible_long"]]
    groups = {name: [row for row in rows if row["family"] == name] for name in sorted({row["family"] for row in rows})}
    return {
        "gate_passed": any(row["paired_win"] for row in strata["feasible_long"]),
        "bootstrap_config_draws": BOOTSTRAPS, "bootstrap_seed": SEED,
        "strata": {name: {"metrics": metrics(samples), "paired": _paired_summary(samples)} for name, samples in strata.items()},
        "families_feasible_long": {name: _paired_summary(select_stratum(samples, "feasible_long")) for name, samples in groups.items()},
        "settings": [
            {"R": horizon, "B_ms": budget, "metrics": metrics(samples), "paired": _paired_summary(samples)}
            for budget in ("unlimited", 1, 10, 100) for horizon in R_GRID
            for samples in [[row for row in rows if row["R"] == horizon and row["B_ms"] == budget]]
        ],
    }
