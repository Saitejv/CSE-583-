"""CPU cost predictor trained on grouped loop workloads."""
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from jax_unroll.compact_fori import FIELDS, tree_costs
from jax_unroll.model import LogCosts
from jax_unroll.schema import FACTORS, FEATURE_VERSION

def design(frame):
    values = np.log1p(frame[list(FIELDS)].to_numpy(dtype=float))
    return np.column_stack((values, frame.loop_kind == "scan", frame.trip_count % 8 / 8))


class CorpusPredictor:
    backend = "cpu"
    feature_version = FEATURE_VERSION
    jax_versions = ("0.12.0.1", "0.12.0.1")

    @classmethod
    def load(cls, path):
        """Load a trusted project-trained joblib artifact for DecisionEngine."""
        path = Path(path)
        method, margin, scale, center, estimator = joblib.load(path)
        model = cls(method, margin)
        model.scale, model.center, model.fit_model = scale, center, estimator
        if method in ("trees", "forest"):
            model._prepare_trees()
        model.version = sha256(path.read_bytes()).hexdigest()
        return model

    def supports(self, row):
        return (row["backend"] == self.backend and row["loop_primitive"] == "scan"
                and row["loop_kind"] in ("scan", "fori_loop")
                and 16 <= int(row["trip_count"]) <= 1024 and int(row["dtype_bits"]) == 32)

    def __init__(self, method, margin):
        self.margin = margin
        self.scale = StandardScaler()
        if method == "trees":
            self.fit_model = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=2, random_state=583)
        elif method == "forest":
            self.fit_model = RandomForestRegressor(n_estimators=128, min_samples_leaf=2, random_state=583)
        elif method == "kernel":
            self.fit_model = KernelRidge(alpha=.5, kernel="rbf")
        elif method == "linear":
            self.fit_model = Ridge(alpha=10)
        else:
            raise ValueError(f"unknown model {method}")

    def fit(self, frame, measured, *, sample_weight=None):
        target = np.log(measured)
        target[:, 1:] -= target[:, :1]
        y = target.reshape(len(frame), -1)
        self.center = np.median(y, axis=0)
        self.fit_model.fit(self.scale.fit_transform(design(frame)), y - self.center, sample_weight=sample_weight)
        return self

    def predict(self, frame):
        y = self.fit_model.predict(self.scale.transform(design(frame))) + self.center
        logs = y.reshape(-1, len(FACTORS), 2)
        logs[:, 1:] += logs[:, :1]
        return np.exp(logs)

    def _prepare_trees(self):
        # Concatenate the fitted trees so one array lookup advances the entire forest.
        trees = [tree.tree_ for tree in self.fit_model.estimators_]
        offsets = np.cumsum([0] + [tree.node_count for tree in trees[:-1]])
        self._roots = offsets
        self._feature = np.concatenate([tree.feature for tree in trees])
        self._threshold = np.concatenate([tree.threshold for tree in trees])
        self._values = np.concatenate([tree.value[:, :, 0] for tree in trees])
        self._depth = max(tree.max_depth for tree in trees)
        children = []
        for attr in ("children_left", "children_right"):
            parts = []
            for tree, offset in zip(trees, offsets):
                child = getattr(tree, attr)
                parts.append(offset + np.where(child < 0, np.arange(tree.node_count), child))
            children.append(np.concatenate(parts))
        self._left, self._right = children

    def _tree_arrays(self):
        return {
            "mean": self.scale.mean_, "scale": self.scale.scale_, "center": self.center,
            "roots": self._roots, "feature": self._feature, "threshold": self._threshold,
            "values": self._values, "depth": self._depth, "left": self._left, "right": self._right,
        }

    def _tree_costs(self, row):
        return tree_costs(row, **self._tree_arrays())

    def predict_candidates(self, row):
        pred = self._tree_costs(row) if hasattr(self, "_roots") else self.predict(pd.DataFrame([row]))[0]
        # Require a margin on predicted warm time before picking a larger factor.
        pred[1:, 1] *= np.exp(self.margin)
        return {u: LogCosts(*np.log(pair)) for u, pair in zip(FACTORS, pred, strict=True)}
