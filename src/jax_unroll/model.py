"""Small CPU regressors for natural-log cold milliseconds and warm microseconds."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from jax_unroll.schema import DATASET_KEY_COLUMNS, DATASET_VERSION, FACTORS, FEATURE_COLUMNS, FEATURE_VERSION
from jax_unroll.splits import DEV_FAMILIES, SPLIT_VERSION, assert_holdout_excluded, lofo_folds, split_for_family

MODEL_FORMAT_VERSION = 1
LABELS = ("log_cold_ms", "log_warm_us")
CONFIG_KEY = tuple(column for column in DATASET_KEY_COLUMNS if column != "factor")
FACTOR_FEATURES = (
    "factor", "full_chunks", "remainder", "remainder_ratio", "divides_evenly",
    "has_tail", "factor_over_trip", "unrolled_equations",
)
INDEPENDENT_FEATURES = tuple(column for column in FEATURE_COLUMNS if column not in FACTOR_FEATURES)
MODEL_FEATURES = (
    "trip_count", "factor", "has_tail", "num_equations", "eq_reduce",
    "num_consts", "const_bytes", "carry_bytes", "bytes_moved_proxy", "flops_proxy",
)
WARM_TERMS = ("work_size", "memory_work", "relief", "trip")
BASE_ALPHA = COLD_EFFECT_ALPHA = 1.0
TRIP_SCALE = 128
Row = Mapping[str, object]


@dataclass(frozen=True)
class LogCosts:
    log_cold_ms: float
    log_warm_us: float

    def linear(self) -> tuple[float, float]:
        return math.exp(self.log_cold_ms), math.exp(self.log_warm_us)


def read_rows(path: Path, *, split: str) -> list[Row]:
    """Read one frozen partition and verify the CSV split tags against family."""
    metadata_path = path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for field, expected in (
        ("dataset_version", DATASET_VERSION), ("feature_version", FEATURE_VERSION),
        ("split_version", SPLIT_VERSION), ("backend", "cpu"),
    ):
        if metadata.get(field) != expected:
            raise ValueError(f"expected {field}={expected!r} in {metadata_path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            if split_for_family(row["family"]) != split:
                raise ValueError(f"split tag disagrees with family {row['family']!r}")
            rows.append({**row, "jax_version": metadata["jax_version"], "jaxlib_version": metadata["jaxlib_version"]})
    return rows


def group_configs(rows: Iterable[Row]) -> list[dict[int, Row]]:
    groups: dict[tuple[object, ...], dict[int, Row]] = defaultdict(dict)
    for row in rows:
        group = groups[tuple(row[column] for column in CONFIG_KEY)]
        factor = int(row["factor"])
        if factor in group:
            raise ValueError("duplicate configuration/factor row")
        group[factor] = row
    for group in groups.values():
        if set(group) != set(FACTORS):
            raise ValueError(f"configuration must contain exactly the factors {FACTORS}")
    return list(groups.values())


def candidate_features(features: Row, factor: int) -> dict[str, object]:
    """Recompute all candidate-dependent fields from one static extraction."""
    trip = int(features["trip_count"])
    chunks, remainder = divmod(trip, factor)
    return {
        **features,
        "factor": factor,
        "full_chunks": chunks,
        "remainder": remainder,
        "remainder_ratio": remainder / trip,
        "divides_evenly": remainder == 0,
        "has_tail": remainder != 0,
        "factor_over_trip": factor / trip,
        "unrolled_equations": int(features["num_equations"]) * factor,
    }


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def dev_rows_digest(rows: list[Row]) -> str:
    _check_dev(rows)
    return _digest(sorted(_canonical(dict(row)) for row in rows))


def _check_dev(rows: list[Row]) -> None:
    families = {str(row["family"]) for row in rows}
    assert_holdout_excluded(families)
    if not rows or not families.issubset(DEV_FAMILIES) or any(row["split"] != "dev" for row in rows):
        raise ValueError("training and LOFO require development rows only")
    if {row["backend"] for row in rows} != {"cpu"}:
        raise ValueError("the Phase 3 cost model supports CPU rows only")


def _design(row: Row, warm_terms: str) -> tuple[tuple[float, ...], ...]:
    trip = int(row["trip_count"])
    factor = int(row["factor"])
    if trip <= 0 or factor not in FACTORS:
        raise ValueError("expected a positive trip count and a supported factor")
    log_factor = math.log2(factor)
    work = float(row["flops_proxy"])
    footprint = float(row["bytes_moved_proxy"])
    log_work = math.log1p(work)
    log_size = math.log1p(footprint)
    log_equations = math.log2(1 + float(row["num_equations"]))
    tail = row["has_tail"] in (True, "True", "true", "1")
    cold_base = (
        math.log(trip), log_work, log_size, math.log1p(float(row["carry_bytes"])),
        log_equations, float(row["num_consts"]),
    )
    cold_effect = (
        log_factor, log_factor * log_equations, log_factor * math.log1p(float(row["const_bytes"])) / 8,
        log_factor * float(row["eq_reduce"]), float(tail),
    )
    warm_base = (math.log1p(trip * work / 1e3), math.log1p(trip * footprint / 1e4))
    trip_weight = trip / (trip + TRIP_SCALE)
    shape = (1.0, trip_weight)
    if warm_terms == "work_size":
        shape += (log_work / 8, log_size / 8)
    elif warm_terms == "memory_work":
        shape += (log_work / 8, math.log1p(footprint / (1 + work)) / 8)
    elif warm_terms == "relief":
        shape += (trip_weight / (1 + log_work), log_size / 8)
    warm_effect = tuple(basis * value for basis in (1 - 1 / factor, math.log(factor)) for value in shape)
    return cold_base, cold_effect, warm_base, warm_effect


class CostModel:
    """Fit four tiny ridge equations and run inference directly from JSON weights."""

    def __init__(self, *, warm_terms: str = "memory_work", warm_alpha: float = 0.025) -> None:
        if warm_terms not in WARM_TERMS or not math.isfinite(warm_alpha) or warm_alpha <= 0:
            raise ValueError("invalid warm regression hyperparameters")
        self.warm_terms = warm_terms
        self.warm_alpha = warm_alpha
        self._state: dict | None = None

    def _fitted(self) -> dict:
        if self._state is None:
            raise ValueError("cost model has not been fitted or loaded")
        return self._state

    @property
    def version(self) -> str:
        return self._fitted()["model_version"]

    @property
    def feature_version(self) -> int:
        return self._fitted()["feature_version"]

    @property
    def backend(self) -> str:
        return self._fitted()["backend"]

    @property
    def jax_versions(self) -> tuple[str, str]:
        state = self._fitted()
        return state["jax_version"], state["jaxlib_version"]

    @property
    def hyperparameters(self) -> dict[str, str | float | int]:
        return {
            "base_alpha": BASE_ALPHA, "cold_effect_alpha": COLD_EFFECT_ALPHA,
            "warm_terms": self.warm_terms, "warm_alpha": self.warm_alpha, "trip_scale": TRIP_SCALE,
        }

    def fit(self, rows: Iterable[Row]) -> CostModel:
        rows = list(rows)
        _check_dev(rows)

        import numpy as np
        from sklearn.linear_model import Ridge

        groups = group_configs(rows)
        ordered = [group[factor] for group in groups for factor in FACTORS]
        x = [np.asarray(section) for section in zip(*(_design(row, self.warm_terms) for row in ordered))]
        y = np.asarray([[float(row[label]) for label in LABELS] for row in ordered])
        if not np.isfinite(y).all() or not all(np.isfinite(section).all() for section in x):
            raise ValueError("training requires finite features and log labels")
        base = y[::len(FACTORS)]
        effects = y - np.repeat(base, len(FACTORS), axis=0)
        regressions = (
            Ridge(alpha=BASE_ALPHA).fit(x[0][::len(FACTORS)], base[:, 0]),
            Ridge(alpha=COLD_EFFECT_ALPHA, fit_intercept=False, positive=True).fit(x[1], effects[:, 0]),
            Ridge(alpha=BASE_ALPHA).fit(x[2][::len(FACTORS)], base[:, 1]),
            Ridge(alpha=self.warm_alpha, fit_intercept=False).fit(x[3], effects[:, 1]),
        )
        versions = {(row["jax_version"], row["jaxlib_version"]) for row in rows}
        if len(versions) != 1:
            raise ValueError("training rows must use one JAX/jaxlib version pair")
        jax_version, jaxlib_version = versions.pop()
        trips = [int(row["trip_count"]) for row in rows]
        state = {
            "format_version": MODEL_FORMAT_VERSION,
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
            "split_version": SPLIT_VERSION,
            "backend": "cpu",
            "jax_version": jax_version,
            "jaxlib_version": jaxlib_version,
            "dev_rows_sha256": dev_rows_digest(rows),
            "features": list(MODEL_FEATURES),
            "hyperparameters": self.hyperparameters,
            "support": {"trip_range": [min(trips), max(trips)], "dtype_bits": sorted({int(row["dtype_bits"]) for row in rows})},
            "regressions": [
                {"intercept": float(fit.intercept_), "weights": fit.coef_.tolist()}
                for fit in regressions
            ],
        }
        self._state = {**state, "model_version": self._version(state)}
        return self

    @staticmethod
    def _version(state: dict) -> str:
        return f"cpu-ridge-v{MODEL_FORMAT_VERSION}-{_digest(state)[:16]}"

    def supports(self, features: Row) -> bool:
        """Check the static domain used for runtime uncertainty fallback."""
        support = self._fitted()["support"]
        low, high = support["trip_range"]
        return (
            features["backend"] == self.backend
            and features["loop_primitive"] == "scan"
            and features["loop_kind"] in ("scan", "fori_loop")
            and low <= int(features["trip_count"]) <= high
            and int(features["dtype_bits"]) in support["dtype_bits"]
        )

    def predict(self, features: Row) -> LogCosts:
        """Predict natural logs for one candidate row."""
        state = self._fitted()
        design = _design(features, state["hyperparameters"]["warm_terms"])
        cold_base, cold_effect, warm_base, warm_effect = (
            regression["intercept"] + math.fsum(weight * value for weight, value in zip(regression["weights"], values, strict=True))
            for regression, values in zip(state["regressions"], design, strict=True)
        )
        return LogCosts(cold_base + cold_effect, warm_base + warm_effect)

    def predict_candidates(self, features: Row) -> dict[int, LogCosts]:
        return {factor: self.predict(candidate_features(features, factor)) for factor in FACTORS}

    def to_dict(self) -> dict:
        return json.loads(_canonical(self._fitted()))

    @classmethod
    def from_dict(cls, artifact: dict) -> CostModel:
        state = artifact["model"].copy()
        version = state.pop("model_version")
        if state["format_version"] != MODEL_FORMAT_VERSION or cls._version(state) != version:
            raise ValueError("incompatible or corrupt model artifact")
        params = state["hyperparameters"]
        model = cls(warm_terms=params["warm_terms"], warm_alpha=params["warm_alpha"])
        model._state = {**state, "model_version": version}
        return model

    @classmethod
    def load(cls, path: Path) -> CostModel:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def lofo_predictions(
    rows: Iterable[Row], *, warm_terms: str, warm_alpha: float,
) -> dict[str, tuple[CostModel, list[Row]]]:
    """Fit each dev-only fold and return its model and unseen validation rows."""
    rows = list(rows)
    _check_dev(rows)
    if {row["family"] for row in rows} != set(DEV_FAMILIES):
        raise ValueError("LOFO requires all frozen dev families")
    return {
        name: (
            CostModel(warm_terms=warm_terms, warm_alpha=warm_alpha).fit(
                row for row in rows if row["family"] in fold["train"]
            ),
            [row for row in rows if row["family"] in fold["test"]],
        )
        for name, fold in lofo_folds().items()
    }
