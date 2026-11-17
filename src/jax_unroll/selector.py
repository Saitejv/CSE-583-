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
    """Trace one static loop if needed, predict candidates, and apply the policy."""

    def __init__(self, model: CostModel, cache: DecisionCache | None = None) -> None:
        self.model = model
        self.cache = cache if cache is not None else DecisionCache()

    def decide(
        self, fn: Callable[..., Any] | None = None, args: Sequence[Any] = (), *,
        trip_count: int, loop_kind: str | None = None, B: float | None = None,
        R: int = 100, throughput_only: bool = False, backend: str | None = None,
        signature: LoopSignature | None = None,
    ) -> Decision:
        """Reuse a trusted signature from the same loop and abstract argument types.

        With no signature, trace the supplied function to derive one. On a
        known-signature cache miss, use the function to recompute the decision.
        """
        B, R = _context(B, R)
        backend = backend if backend is not None else jax.default_backend()
        namespace = (backend, FEATURE_VERSION, self.model.version)
        compatible = (
            backend == self.model.backend and self.model.feature_version == FEATURE_VERSION
            and self.model.jax_versions == (jax.__version__, jaxlib.__version__)
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
            )
        except Exception:
            return Decision(1, fallback_reason="feature_extraction_failed")

        signature = LoopSignature.from_features(features)
        key = CacheKey(signature, B, R, throughput_only)
        cached = self.cache.get(namespace, key)
        if cached is not None:
            return Decision(cached, signature, cache_hit=True)
        try:
            if not self.model.supports(features):
                return Decision(1, signature, fallback_reason="unsupported_features")
            predictions = self.model.predict_candidates(features)
            factor = select_factor(predictions, B=B, R=R, throughput_only=throughput_only)
        except (ArithmeticError, ValueError, TypeError, KeyError):
            return Decision(1, signature, fallback_reason="prediction_failed")
        self.cache.put(namespace, key, factor)
        return Decision(factor, signature)
