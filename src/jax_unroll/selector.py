"""Static, budget-aware unroll selection and a versioned decision cache."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping, Sequence

import jax
import jaxlib

from jax_unroll.features import extract_features
from jax_unroll.model import CostModel, LogCosts
from jax_unroll.schema import FACTORS, FEATURE_VERSION


def _context(B: float | None, R: int) -> tuple[float | None, int]:
    if B is not None:
        if math.isnan(B) or B < 0:
            raise ValueError("B must be non-negative milliseconds or unlimited")
        if B == math.inf:
            B = None
    if not isinstance(R, int) or isinstance(R, bool) or R < 0:
        raise ValueError("R must be a non-negative invocation count")
    return B, R


def total_ms(cold_ms: float, warm_us: float, R: int) -> float:
    return cold_ms + R * (warm_us / 1_000)


def select_factor(
    predictions: Mapping[int, LogCosts] | None, *, B: float | None = None,
    R: int = 100, throughput_only: bool = False, uncertain: bool = False,
) -> int:
    """Select from four natural-log forecasts, with a conservative factor-1 fallback."""
    B, R = _context(B, R)
    if uncertain or predictions is None or any(factor not in predictions for factor in FACTORS):
        return 1

    log_budget = math.inf if B is None else math.log(B) if B else -math.inf
    best_factor = 1
    best_score = math.inf
    for factor in FACTORS:
        try:
            cold, warm = predictions[factor].linear()
            score = warm if throughput_only else total_ms(cold, warm, R)
        except (AttributeError, OverflowError, TypeError, ValueError):
            return 1
        if not (math.isfinite(cold) and cold > 0 and math.isfinite(warm) and warm > 0 and math.isfinite(score)):
            return 1
        if predictions[factor].log_cold_ms <= log_budget and score < best_score:
            best_factor, best_score = factor, score
    return best_factor


def _usable_costs(predictions: Mapping[int, LogCosts] | None) -> dict[int, tuple[float, float]] | None:
    if predictions is None or any(factor not in predictions for factor in FACTORS):
        return None
    try:
        costs = {factor: predictions[factor].linear() for factor in FACTORS}
    except (AttributeError, OverflowError, TypeError, ValueError):
        return None
    if any(not math.isfinite(v) or v <= 0 for pair in costs.values() for v in pair):
        return None
    return costs


@dataclass(frozen=True)
class PayoffPolicy:
    """Require a positive conservative total saving relative to native factor 1.

    Cold forecasts include the first synchronized call. The remaining R - 1
    invocations use the warm forecast. Margins are one-sided development
    residuals in log time for warm execution and milliseconds for cold costs.
    model_load_ms optionally charges a per-specialization share of loading the
    model in the payoff calculation. The cold budget covers the specialization
    itself, including its decision.
    """

    warm_effect_margin: float
    baseline_warm_margin: float
    extra_cold_ms: float
    candidate_cold_ms: float
    decision_ms: float
    model_load_ms: float = 0.

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if not math.isfinite(value) or value < 0:
                raise ValueError("payoff margins and decision cost must be finite and non-negative")

    def choose(self, predictions: Mapping[int, LogCosts] | None, *,
               B: float | None = None, R: int = 1) -> int:
        B, R = _context(B, R)
        costs = _usable_costs(predictions) if R > 1 else None
        if costs is None:
            return 1

        available_cold_ms = math.inf if B is None else B - self.candidate_cold_ms - self.decision_ms
        if available_cold_ms <= 0:
            return 1
        log_budget = math.log(available_cold_ms)
        cold1, warm1_us = costs[1]
        warm1_ms_lower = warm1_us / 1_000 * math.exp(-self.baseline_warm_margin)
        best_factor, best_score = 1, 0.
        for factor in FACTORS[1:]:
            cold, _ = costs[factor]
            if predictions[factor].log_cold_ms > log_budget:
                continue
            log_ratio_upper = (
                predictions[factor].log_warm_us - predictions[1].log_warm_us + self.warm_effect_margin
            )
            if log_ratio_upper >= 0:
                continue
            savings_ms = -math.expm1(log_ratio_upper) * warm1_ms_lower
            setup_ms = (
                max(0., cold - cold1 + self.extra_cold_ms) + self.decision_ms + self.model_load_ms
            )
            score = savings_ms - setup_ms * (1 / (R - 1))
            if score > best_score:
                best_factor, best_score = factor, score
        return best_factor


@dataclass(frozen=True)
class SavingsBand:
    """Margins for a latency band, with warm error as a fraction of baseline latency."""

    upper_warm_us: float | None
    warm_margin: float
    extra_cold_ms: float
    candidate_cold_ms: float


@dataclass(frozen=True)
class SavingsPolicy:
    """Calibrate error in warm savings directly, including correlated forecast errors."""

    bands: tuple[SavingsBand, ...]
    decision_ms: float
    model_load_ms: float = 0.

    def __post_init__(self):
        if not self.bands or self.bands[-1].upper_warm_us is not None:
            raise ValueError("savings bands must have an open final interval")
        edge = 0.
        values = [self.decision_ms, self.model_load_ms]
        for band in self.bands[:-1]:
            upper = band.upper_warm_us
            if upper is None or not math.isfinite(upper) or upper <= edge:
                raise ValueError("savings band bounds must be strictly increasing")
            edge = upper
        for band in self.bands:
            values.extend((band.warm_margin, band.extra_cold_ms, band.candidate_cold_ms))
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("savings margins and overhead must be finite and non-negative")

    def choose(self, predictions: Mapping[int, LogCosts] | None, *,
               B: float | None = None, R: int = 1) -> int:
        B, R = _context(B, R)
        costs = _usable_costs(predictions) if R > 1 else None
        if costs is None:
            return 1
        cold1, warm1 = costs[1]
        band = next(b for b in self.bands if b.upper_warm_us is None or warm1 < b.upper_warm_us)
        budget = math.inf if B is None else B - band.candidate_cold_ms - self.decision_ms
        if budget <= 0:
            return 1
        log_budget = math.log(budget)
        setup_overhead = self.decision_ms + self.model_load_ms
        best, surplus = 1, 0.
        for factor in FACTORS[1:]:
            cold, warm = costs[factor]
            if predictions[factor].log_cold_ms > log_budget:
                continue
            per_call = (warm1 - warm - band.warm_margin * warm1) / 1_000
            setup = max(0., cold - cold1 + band.extra_cold_ms) + setup_overhead
            score = per_call - setup * (1 / (R - 1))
            if score > surplus:
                best, surplus = factor, score
        return best


@dataclass(frozen=True)
class HorizonPolicy:
    """Defer warm selection below a call-count crossover.

    An optional calibrated factor applies to sufficiently large matrix carries
    with few indexing operations and no dot products or reductions.
    """

    min_calls: int
    decision_ms: float
    matrix_min_calls: int | None = None
    matrix_bytes: int | None = None
    max_index_ops: int = 0
    matrix_factor: int = 2

    def __post_init__(self):
        if type(self.min_calls) is not int or self.min_calls < 2:
            raise ValueError("min_calls must be an integer of at least two")
        if not math.isfinite(self.decision_ms) or self.decision_ms < 0:
            raise ValueError("decision_ms must be finite and non-negative")
        if self.matrix_min_calls is not None and (type(self.matrix_min_calls) is not int
                                                  or not 2 <= self.matrix_min_calls <= self.min_calls):
            raise ValueError("matrix crossover must lie between two and min_calls")
        if self.matrix_bytes is not None and (type(self.matrix_bytes) is not int or self.matrix_bytes < 1):
            raise ValueError("matrix_bytes must be a positive integer")
        if type(self.max_index_ops) is not int or self.max_index_ops < 0 or self.matrix_factor not in FACTORS[1:]:
            raise ValueError("invalid matrix action")

    def needs_forecast(self, R: int, carry: Any) -> bool:
        if R >= self.min_calls:
            return True
        if self.matrix_bytes is None or self.matrix_min_calls is None or R < self.matrix_min_calls:
            return False
        # Direct DecisionEngine callers may supply a whole program instead of a carry.
        if carry is None:
            return True
        avals = [jax.typeof(x) for x in jax.tree.leaves(carry)]
        return (any(len(a.shape) == 2 for a in avals)
                and sum(a.size * a.dtype.itemsize for a in avals) >= self.matrix_bytes)

    def choose(self, predictions: Mapping[int, LogCosts] | None, *, features: Mapping[str, Any],
               B: float | None = None, R: int = 1) -> int:
        B, R = _context(B, R)
        costs = _usable_costs(predictions) if self.needs_forecast(R, None) else None
        if costs is None:
            return 1
        budget = math.inf if B is None else B - self.decision_ms
        if budget <= 0:
            return 1
        log_budget = math.log(budget)
        feasible = [f for f in FACTORS if predictions[f].log_cold_ms <= log_budget]
        if (self.matrix_bytes is not None and self.matrix_factor in feasible
                and features["carry_bytes"] >= self.matrix_bytes
                and features["eq_dot"] == features["eq_reduce"] == 0
                and features["eq_gather_scatter"] <= self.max_index_ops
                and any(len(s) == 2 for s in json.loads(features["carry_shapes"]))):
            return self.matrix_factor
        if R < self.min_calls:
            return 1
        return min(feasible, key=lambda f: costs[f][1], default=1)


@dataclass(frozen=True)
class LoopSignature:
    body_hash: str
    trip_count: int
    loop_kind: str
    loop_primitive: str
    carry_shapes: str
    const_shapes: str
    input_shapes: str
    output_shapes: str
    dtypes: str

    @classmethod
    def from_features(cls, features: Mapping[str, Any]) -> LoopSignature:
        return cls(
            features["body_hash"], int(features["trip_count"]), features["loop_kind"],
            features["loop_primitive"], features["carry_shapes"], features["const_shapes"],
            features["input_shapes"], features["output_shapes"], features["dtypes"],
        )


@dataclass(frozen=True)
class CacheKey:
    signature: LoopSignature
    B: float | None
    R: int
    throughput_only: bool


class DecisionCache:
    """Thread-safe in-process cache with an optional atomically replaced JSON snapshot."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = RLock()
        self._namespace: tuple[str, int, str] | None = None
        self._entries: dict[CacheKey, int] = {}
        if path is not None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload["format_version"] == 1:
                    for entry in payload["entries"]:
                        key = CacheKey(LoopSignature(**entry["signature"]), *_context(entry["B"], entry["R"]), entry["throughput_only"])
                        factor = entry["factor"]
                        if factor not in FACTORS:
                            raise ValueError("invalid cached factor")
                        self._entries[key] = factor
                    self._namespace = tuple(payload["namespace"])
            except (OSError, ValueError, TypeError, KeyError):
                self._entries.clear()

    def _activate(self, namespace: tuple[str, int, str]) -> None:
        if self._namespace != namespace:
            had_entries = bool(self._entries)
            self._namespace = namespace
            self._entries.clear()
            if had_entries and self.path is not None:
                self._persist()

    def get(self, namespace: tuple[str, int, str], key: CacheKey | None) -> int | None:
        with self._lock:
            self._activate(namespace)
            return self._entries.get(key)

    def put(self, namespace: tuple[str, int, str], key: CacheKey, factor: int) -> None:
        with self._lock:
            self._activate(namespace)
            self._entries[key] = factor
            if self.path is not None:
                self._persist()

    def _persist(self) -> None:
        payload = {
            "format_version": 1,
            "namespace": self._namespace,
            "entries": [dict(asdict(key), factor=factor) for key, factor in self._entries.items()],
        }
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, allow_nan=False)
            os.replace(temporary, self.path)
        except OSError:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


@dataclass(frozen=True)
class Decision:
    factor: int
    signature: LoopSignature | None = None
    cache_hit: bool = False
    fallback_reason: str | None = None


class DecisionEngine:
    """Trace one static loop if needed, predict candidates, and apply the policy.

    A model_loader supplies the model once, on the first decision that needs
    predictions. The model attribute is None until a deferred load succeeds.
    """

    def __init__(
        self, model: CostModel | None = None, cache: DecisionCache | None = None, *,
        payoff: PayoffPolicy | SavingsPolicy | HorizonPolicy | None = None,
        model_loader: Callable[[], CostModel] | None = None,
    ) -> None:
        if (model is None) == (model_loader is None):
            raise ValueError("give either a model or a model_loader")
        self.model = model
        self._model_loader = model_loader
        self._model_lock = RLock() if model_loader is not None else None
        self.cache = cache if cache is not None else DecisionCache()
        self.payoff = payoff

    def _require_model(self) -> CostModel:
        if self._model_loader is not None:
            with self._model_lock:
                if self._model_loader is not None:
                    model = self._model_loader()
                    if model is None:
                        raise TypeError("model_loader returned no model")
                    self.model = model
                    self._model_loader = None
        return self.model

    def decide(
        self, fn: Callable[..., Any] | None = None, args: Sequence[Any] = (), *,
        trip_count: int, loop_kind: str | None = None, B: float | None = None,
        R: int = 100, throughput_only: bool = False, backend: str | None = None,
        signature: LoopSignature | None = None, fori_carry: Any = None,
    ) -> Decision:
        """Reuse a trusted signature from the same loop and abstract argument types.

        With no signature, trace the supplied function to derive one. On a
        known-signature cache miss, use the function to recompute the decision.
        The wrapper can supply its initial carry for the horizon policy's cheap
        shape check before deferred loading or feature extraction.
        """
        B, R = _context(B, R)
        payoff = None if throughput_only else self.payoff
        if payoff is not None and (R <= 1 or (isinstance(payoff, HorizonPolicy)
                                              and not payoff.needs_forecast(R, fori_carry))):
            return Decision(1, fallback_reason="insufficient_calls")
        if payoff is not None and B is not None and B <= payoff.decision_ms:
            return Decision(1, fallback_reason="insufficient_budget")
        model = self._require_model()
        backend = backend if backend is not None else jax.default_backend()
        policy_version = model.version
        if self.payoff is not None:
            policy_version = f"{policy_version}|payoff-v1|{self.payoff!r}"
        namespace = (backend, FEATURE_VERSION, policy_version)
        compatible = (
            backend == model.backend and model.feature_version == FEATURE_VERSION
            and model.jax_versions == (jax.__version__, jaxlib.__version__)
        )
        if not compatible:
            self.cache.get(namespace, None)
            return Decision(1, fallback_reason="incompatible_model")
        if signature is not None:
            if signature.trip_count != trip_count or (loop_kind is not None and signature.loop_kind != loop_kind):
                raise ValueError("provided signature disagrees with loop parameters")
            cached = self.cache.get(namespace, CacheKey(signature, B, R, throughput_only))
            if cached is not None:
                return Decision(cached, signature, cache_hit=True)
        try:
            features = extract_features(
                fn, args, trip_count=trip_count, factor=1, loop_kind=loop_kind, backend=backend,
                feature_block=getattr(model, "loop_features", None),
            )
        except Exception:
            return Decision(1, fallback_reason="feature_extraction_failed")

        signature = LoopSignature.from_features(features)
        key = CacheKey(signature, B, R, throughput_only)
        cached = self.cache.get(namespace, key)
        if cached is not None:
            return Decision(cached, signature, cache_hit=True)
        try:
            if not model.supports(features):
                return Decision(1, signature, fallback_reason="unsupported_features")
            predictions = model.predict_candidates(features)
            if payoff is None:
                factor = select_factor(predictions, B=B, R=R, throughput_only=throughput_only)
            elif isinstance(payoff, HorizonPolicy):
                factor = payoff.choose(predictions, features=features, B=B, R=R)
            else:
                factor = payoff.choose(predictions, B=B, R=R)
        except (ArithmeticError, ValueError, TypeError, KeyError):
            return Decision(1, signature, fallback_reason="prediction_failed")
        self.cache.put(namespace, key, factor)
        return Decision(factor, signature)
