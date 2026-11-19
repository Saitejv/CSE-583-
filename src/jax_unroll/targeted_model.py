"""Warm-throughput candidates for static CPU fori_loop selection."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jax_unroll.corpus_model import design
from jax_unroll.fori_model import ForiCostPredictor
from jax_unroll.schema import FACTORS


class TargetedCostPredictor(ForiCostPredictor):
    """Estimate time at each factor; the selector may require a minimum warm gain."""

    def __init__(self, method, margin=0):
        super().__init__("trees", margin)
        self.method = method
        if method == "joint_forest":
            self.fit_model = RandomForestRegressor(n_estimators=128, min_samples_leaf=2, random_state=583)
        elif method in ("cost_ratio", "blend"):
            self.effect_model = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=3, random_state=583)
        elif method == "kernel":
            self.effect_model = make_pipeline(StandardScaler(), KernelRidge(alpha=2, kernel="rbf"))
        elif method == "shared_effect":
            self.effect_model = GradientBoostingRegressor(loss="huber", n_estimators=100, max_depth=2, random_state=583)
        elif method == "boosted":
            regressor = GradientBoostingRegressor(loss="huber", n_estimators=60, max_depth=2, random_state=583)
            self.effect_model = MultiOutputRegressor(regressor)
        elif method not in ("trees", "upper_quartile"):
            raise ValueError(f"unknown targeted model {method}")

    def _effect_design(self, frame):
        static = self.effect_features(frame)
        if self.method != "shared_effect":
            return static
        # Share one regression across choices, exposing work left in the tail
        # and the approximate size of each candidate unrolled body.
        count = len(FACTORS) - 1
        factors = np.tile(FACTORS[1:], len(frame))
        trips = np.repeat(frame.trip_count.to_numpy(), count)
        eqns = np.repeat(frame.num_equations.to_numpy(), count)
        traffic = np.repeat(frame.bytes_moved_proxy.to_numpy(), count)
        return np.column_stack((np.repeat(static, count, axis=0), np.log2(factors),
                                trips % factors, np.log1p(trips // factors),
                                np.log1p(eqns * factors), np.log1p(traffic * factors)))

    def fit(self, frame, measured):
        super().fit(frame, measured)
        if hasattr(self, "effect_model"):
            ratios = measured[:, 1:, 1] / measured[:, :1, 1]
            # Arithmetic ratios give large slowdowns proportionally more weight.
            target = ratios if self.method in ("cost_ratio", "blend") else np.log(ratios)
            if self.method == "shared_effect":
                target = target.ravel()
            self.effect_model.fit(self._effect_design(frame), target)
        return self

    def predict(self, frame):
        costs = super().predict(frame)
        if self.method == "upper_quartile":
            # Each tree estimates three log warm ratios relative to factor 1.
            # Prefer larger factors only when most trees predict a gain.
            x = self.scale.transform(design(frame))
            ratios = np.stack([tree.predict(x)[:, 3::2] for tree in self.fit_model.estimators_])
            upper = self.center[3::2] + np.quantile(ratios, .75, axis=0)
            costs[:, 1:, 1] = costs[:, :1, 1] * np.exp(upper)
        if hasattr(self, "effect_model"):
            effect = self.effect_model.predict(self._effect_design(frame)).reshape(len(frame), -1)
            if self.method not in ("cost_ratio", "blend"):
                effect = np.exp(effect)
            warm = costs[:, :1, 1] * effect
            if self.method == "blend":
                warm = np.sqrt(warm * costs[:, 1:, 1])
            costs[:, 1:, 1] = warm
        return costs

    def predict_candidates(self, row):
        if self.method in ("trees", "joint_forest"):
            return super().predict_candidates(row)
        import pandas as pd
        from jax_unroll.model import LogCosts

        pred = self.predict(pd.DataFrame([row]))[0]
        pred[1:, 1] *= np.exp(self.margin)
        return {u: LogCosts(*np.log(pair)) for u, pair in zip(FACTORS, pred, strict=True)}
