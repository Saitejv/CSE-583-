"""Offline prediction diagnostics and measured policy comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jax_unroll.baselines import BASELINES
from jax_unroll.model import LABELS, CostModel, LogCosts, Row, group_configs, lofo_predictions
from jax_unroll.schema import FACTORS
from jax_unroll.selector import select_factor, total_ms

R_GRID = (1, 10, 100, 1_000, 10_000, 100_000)
B_GRID = (None, 1.0, 10.0, 100.0)
LONG_R = 10_000
POLICIES = (*BASELINES, "selector", "measured_oracle")


@dataclass(frozen=True)
class Forecast:
    features: Row
    measured: dict[int, tuple[float, float]]
    predictions: dict[int, LogCosts]
    uncertain: bool


def forecast(model: CostModel, rows: list[Row]) -> list[Forecast]:
    cases = []
    for group in group_configs(rows):
        features = group[1]
        measured = {
            u: (float(row["cold_median_ms"]), float(row["warm_median_us"]))
            for u, row in group.items()
        }
        if any(not np.isfinite(cost) or cost <= 0 for pair in measured.values() for cost in pair):
            raise ValueError("measured costs must be finite and positive")
        cases.append(Forecast(features, measured, model.predict_candidates(features), not model.supports(features)))
    return cases


def lofo_forecasts(rows: list[Row], *, warm_terms: str, warm_alpha: float) -> dict[str, list[Forecast]]:
    folds = lofo_predictions(rows, warm_terms=warm_terms, warm_alpha=warm_alpha)
    return {fold: forecast(model, test) for fold, (model, test) in folds.items()}


def prediction_errors(cases: list[Forecast]) -> dict:
    actual = np.array([[case.measured[u] for u in FACTORS] for case in cases])
    predicted = np.array([[case.predictions[u].linear() for u in FACTORS] for case in cases])
    log_error = np.log(predicted / actual)
    # Subtract the factor-1 residual to measure calibration of counterfactual ratios.
    delta_error = log_error[:, 1:] - log_error[:, :1]
    return {
        label: {
            "log_mae": float(np.mean(abs(log_error[..., i]))),
            "mean_absolute_relative_error": float(np.mean(abs(predicted[..., i] / actual[..., i] - 1))),
            "median_predicted_over_measured": float(np.median(predicted[..., i] / actual[..., i])),
            "p90_absolute_log_error": float(np.quantile(abs(log_error[..., i]), 0.9)),
            "within_25_percent": float(np.mean(abs(predicted[..., i] / actual[..., i] - 1) <= 0.25)),
            "factor_ratio_log_mae": float(np.mean(abs(delta_error[..., i]))),
        }
        for i, label in enumerate(LABELS)
    }


def choices(case: Forecast, R: int, B: float | None) -> tuple[dict[str, int], bool]:
    feasible = [u for u in FACTORS if B is None or case.measured[u][0] <= B]
    oracle = min(
        feasible, key=lambda u: (total_ms(*case.measured[u], R), u),
    ) if feasible else 1
    return {
        **{policy: choose(case.features) for policy, choose in BASELINES.items()},
        "selector": select_factor(case.predictions, B=B, R=R, uncertain=case.uncertain),
        "measured_oracle": oracle,
    }, bool(feasible)


def cv_metrics(folds: dict[str, list[Forecast]]) -> dict:
    all_cases = [case for cases in folds.values() for case in cases]
    reports = {}
    for fold, cases in {**folds, "pooled": all_cases}.items():
        observations = []
        for case in cases:
            for B in B_GRID:
                for R in R_GRID:
                    factors, _ = choices(case, R, B)
                    u, oracle = factors["selector"], factors["measured_oracle"]
                    regret = total_ms(*case.measured[u], R) / total_ms(*case.measured[oracle], R) - 1
                    observations.append((regret, u == oracle, B is not None and case.measured[u][0] > B, R >= LONG_R))
        regret, agreement, violation, long_horizon = np.asarray(observations).T
        reports[fold] = {
            "configurations": len(cases), "prediction_errors": prediction_errors(cases),
            "median_regret": float(np.median(regret)),
            "mean_positive_regret": float(np.mean(np.maximum(regret, 0))),
            "mean_positive_regret_long": float(np.mean(np.maximum(regret[long_horizon.astype(bool)], 0))),
            "factor_agreement": float(np.mean(agreement)),
            "budget_violation_rate": float(np.mean(violation)),
        }
    return reports
