"""Static-bound CPU fori_loop cost model with a dedicated warm-effect regressor."""
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jax_unroll.corpus_model import CorpusPredictor, FIELDS
from jax_unroll.model import LogCosts
from jax_unroll.schema import FACTORS


class ForiCostPredictor(CorpusPredictor):
    def __init__(self, effect="trees", margin=0):
        super().__init__("trees", margin)
        self.effect = effect
        if effect in ("effect_trees", "effect_forest"):
            estimator = ExtraTreesRegressor if effect == "effect_trees" else RandomForestRegressor
            self.ratios = estimator(n_estimators=128, min_samples_leaf=3, random_state=583)
        elif effect == "effect_linear":
            self.ratios = make_pipeline(StandardScaler(), Ridge(alpha=25))
        elif effect != "trees":
            raise ValueError(effect)

    def supports(self, row):
        return row["loop_kind"] == "fori_loop" and super().supports(row)

    @staticmethod
    def effect_features(frame):
        fields = [*FIELDS, "eq_elementwise", "eq_convert", "eq_other", "carry_elements",
                  "dependence_density", "num_outvars", "num_invars"]
        return np.column_stack((np.log1p(frame[fields].to_numpy(dtype=float)), frame.trip_count % 8))

    def fit(self, frame, measured):
        super().fit(frame, measured)
        if self.effect != "trees":
            warm = measured[:, :, 1]
            self.ratios.fit(self.effect_features(frame), np.log(warm[:, 1:] / warm[:, :1]))
        return self

    def predict(self, frame):
        costs = super().predict(frame)
        if self.effect != "trees":
            ratios = self.ratios.predict(self.effect_features(frame))
            costs[:, 1:, 1] = costs[:, :1, 1] * np.exp(ratios)
        return costs

    def predict_candidates(self, row):
        if self.effect == "trees":
            return super().predict_candidates(row)

        import pandas as pd

        costs = self.predict(pd.DataFrame([row]))[0]
        costs[1:, 1] *= np.exp(self.margin)
        return {u: LogCosts(*np.log(pair)) for u, pair in zip(FACTORS, costs, strict=True)}

    @classmethod
    def load(cls, path):
        path = Path(path)
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError("expected a fitted fori_loop predictor")
        if model.effect == "trees":
            model._prepare_trees()
        model.version = sha256(path.read_bytes()).hexdigest()
        return model
