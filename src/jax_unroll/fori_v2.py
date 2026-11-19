"""Static cost models for selecting unroll factors on CPU fori_loop bodies."""

from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

from jax_unroll.corpus_model import FIELDS
from jax_unroll.fori_model import ForiCostPredictor
from jax_unroll.model import LogCosts
from jax_unroll.schema import FACTORS
from jax_unroll.static_features import INDEX_TERMS, STRUCTURE_TERMS
from jax_unroll.static_indices import index_features
from jax_unroll.static_structure import structure_features


ADDITIONAL_FIELDS = (
    "eq_elementwise", "eq_convert", "eq_other", "output_bytes",
    "dependence_density", "max_out_degree", "num_loop_carried",
)
EXTENDED_FIELDS = (*STRUCTURE_TERMS, *INDEX_TERMS)


def loop_features(loop):
    """Extract carry feedback and index provenance from the already traced body."""
    return {**structure_features(loop), **index_features(loop)}


def design(rows, extended):
    fields = FIELDS + ADDITIONAL_FIELDS
    if extended:
        fields += EXTENDED_FIELDS
    values = np.asarray([[row[name] for name in fields] for row in rows], dtype=float)
    trips = np.asarray([row["trip_count"] for row in rows], dtype=int)
    return np.column_stack((
        np.log1p(values),
        trips[:, None] % np.asarray(FACTORS[1:]),
        np.log1p([row["bytes_moved_proxy"] / max(1, row["num_equations"]) for row in rows]),
        np.log1p([row["flops_proxy"] / max(1, row["carry_bytes"]) for row in rows]),
    )), trips


def effect_design(x, trips):
    factors = np.asarray(FACTORS[1:])
    lengths = trips[:, None]
    return np.column_stack((
        np.repeat(x, len(factors), axis=0),
        np.tile(np.log2(factors), len(trips)),
        (lengths % factors).ravel(),
        np.log1p(lengths // factors).ravel(),
    ))


class ForiPrediction:
    """Warm-effect and absolute latency estimates with a conservative policy margin."""

    backend = ForiCostPredictor.backend
    feature_version = ForiCostPredictor.feature_version
    jax_versions = ForiCostPredictor.jax_versions

    def __init__(self, method, extended, seed=583):
        self.method = method
        self.extended = extended
        self.margin = 0.0
        self.anchor_weight = 0.0
        self.reference = None
        self.loop_features = loop_features if extended else None
        estimator = RandomForestRegressor if method == "forest" else ExtraTreesRegressor
        self.joint = estimator(n_estimators=96, min_samples_leaf=3, random_state=seed)
        if method.startswith("shared"):
            loss = "huber" if method == "shared_huber" else "absolute_error"
            self.effect = GradientBoostingRegressor(
                loss=loss, n_estimators=100, max_depth=2, min_samples_leaf=8, random_state=seed,
            )

    def supports(self, row):
        return (
            row["backend"] == self.backend and row["loop_kind"] == "fori_loop"
            and row["loop_primitive"] == "scan" and 16 <= row["trip_count"] <= 1024
            and row["dtype_bits"] == 32
            and (not self.extended or all(name in row for name in EXTENDED_FIELDS))
        )

    def fit(self, rows, costs, weights=None):
        x, trips = design(rows, self.extended)
        logs = np.log(costs)
        logs[:, 1:] -= logs[:, :1]
        self.joint.fit(x, logs.reshape(len(x), -1), sample_weight=weights)
        if self.method.startswith("shared"):
            sample_weight = None if weights is None else np.repeat(weights, len(FACTORS) - 1)
            self.effect.fit(effect_design(x, trips), logs[:, 1:, 1].ravel(), sample_weight=sample_weight)
        return self

    def predict_new(self, rows):
        x, trips = design(rows, self.extended)
        logs = self.joint.predict(x).reshape(len(x), len(FACTORS), 2)
        if self.method.startswith("shared"):
            logs[:, 1:, 1] = self.effect.predict(effect_design(x, trips)).reshape(len(x), len(FACTORS) - 1)
        logs[:, 1:] += logs[:, :1]
        return logs

    def predict_log(self, rows):
        logs = self.predict_new(rows)
        if self.anchor_weight:
            old = np.asarray([
                [[cost.log_cold_ms, cost.log_warm_us] for cost in self.reference.predict_candidates(row).values()]
                for row in rows
            ])
            logs = (1 - self.anchor_weight) * logs + self.anchor_weight * old
        return logs

    def predict_candidates(self, row):
        logs = self.predict_log([row])[0]
        logs[1:, 1] += self.margin
        return {factor: LogCosts(*pair) for factor, pair in zip(FACTORS, logs, strict=True)}

    @classmethod
    def load(cls, path):
        path = Path(path)
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError("expected a trained fori predictor")
        model.version = sha256(path.read_bytes()).hexdigest()
        return model
