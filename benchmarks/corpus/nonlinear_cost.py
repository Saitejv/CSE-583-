"""Evaluate static multi-output CPU cost models on family-disjoint pilot folds.

Run from the project root with python -m benchmarks.corpus.nonlinear_cost.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.corpus.static_experiment import components
from benchmarks.recovery_data import MODEL
from benchmarks.recovery_metrics import paired_rows
from jax_unroll.model import CostModel, LogCosts
from jax_unroll.schema import FACTORS
from jax_unroll.validation import forecast, prediction_errors

FEATURES = (
    "num_equations", "eq_elementwise", "eq_dot", "eq_reduce", "eq_gather_scatter",
    "carry_bytes", "const_bytes", "xs_bytes", "intermediate_bytes", "max_tensor_bytes",
    "longest_path", "max_out_degree", "carry_updated_bytes", "carry_dependency_bytes",
    "index_from_carry", "index_from_counter", "index_scatter_add_count", "index_gather_count",
)
METHODS = ("extra_trees", "forest", "rbf")


def features(row):
    return np.log1p([float(row[k]) for k in FEATURES] + [float(row["loop_kind"] == "scan")])


class PilotCostModel:
    """Four factor forecasts for static CPU loops at the pilot trip count."""
    def __init__(self, method):
        if method == "rbf":
            self.regressor = make_pipeline(StandardScaler(), KernelRidge(alpha=1, kernel="rbf"))
        elif method in ("forest", "extra_trees"):
            tree = RandomForestRegressor if method == "forest" else ExtraTreesRegressor
            self.regressor = tree(n_estimators=96, min_samples_leaf=2, random_state=583)
        else:
            raise ValueError(method)

    def fit(self, cases, *, residual=False):
        x = [features(c.features) for c in cases]
        y = np.log([[c.measured[u] for u in FACTORS] for c in cases])
        if residual:
            y -= np.array([[[c.predictions[u].log_cold_ms, c.predictions[u].log_warm_us]
                            for u in FACTORS] for c in cases])
        # Model the baseline separately from the factor effects.
        y[:, 1:] -= y[:, :1]
        targets = y.reshape(len(cases), -1)
        self.center = np.median(targets, axis=0)
        self.regressor.fit(x, targets - self.center)
        return self

    def predict_candidates(self, row):
        if row["backend"] != "cpu" or int(row["trip_count"]) != 1023:
            raise ValueError("pilot model requires CPU and 1023 iterations")
        pred = (self.center + self.regressor.predict([features(row)])[0]).reshape(-1, 2)
        pred[1:] += pred[:1]
        return {u: LogCosts(*pair) for u, pair in zip(FACTORS, pred, strict=True)}


def score(cases):
    errors = prediction_errors(cases)
    grid = pd.DataFrame(paired_rows(cases))
    long = grid[grid.feasible_long]
    return {
        "cold_mae": errors["log_cold_ms"]["log_mae"],
        "warm_mae": errors["log_warm_us"]["log_mae"],
        "effect_mae": errors["log_warm_us"]["factor_ratio_log_mae"],
        "long_regret": (long.selector_total_ms / long.measured_oracle_total_ms - 1).clip(lower=0).mean(),
        "long_agreement": long.selector_agreement.mean(),
        "avoidable_violations": int((grid.selector_violates_budget & (grid.measured_feasible_factors > 0)).sum()),
    }


def main():
    rows = pd.read_csv("results/static_features/pilot_features.csv").to_dict("records")
    cases = forecast(CostModel.load(MODEL), rows)
    evaluated = {"frozen": cases, **{method: [] for method in METHODS}}
    evaluated["rbf_cold_frozen_warm"] = []
    evaluated["rbf_residual"] = []
    for members in components().values():
        train = [c for c in cases if c.features["workload"] not in members]
        test = [c for c in cases if c.features["workload"] in members]
        for method in METHODS:
            model = PilotCostModel(method).fit(train)
            for case in test:
                pred = model.predict_candidates(case.features)
                evaluated[method].append(replace(case, predictions=pred))
                if method == "rbf":
                    hybrid = {u: LogCosts(pred[u].log_cold_ms, case.predictions[u].log_warm_us)
                              for u in FACTORS}
                    evaluated["rbf_cold_frozen_warm"].append(replace(case, predictions=hybrid))
        correction = PilotCostModel("rbf").fit(train, residual=True)
        for case in test:
            delta = correction.predict_candidates(case.features)
            corrected = {u: LogCosts(case.predictions[u].log_cold_ms + delta[u].log_cold_ms,
                                     case.predictions[u].log_warm_us + delta[u].log_warm_us) for u in FACTORS}
            evaluated["rbf_residual"].append(replace(case, predictions=corrected))
    table = pd.DataFrame({name: score(batch) for name, batch in evaluated.items()}).T
    print(table.to_string())
    table.to_csv(Path("results") / "nonlinear_pilot.csv", index_label="model")


if __name__ == "__main__":
    main()
