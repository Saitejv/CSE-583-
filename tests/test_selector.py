from __future__ import annotations

import math
from dataclasses import replace

import jax
import pytest

from benchmarks.workloads import build_workload
from jax_unroll import selector
from jax_unroll.model import CostModel, LogCosts
from jax_unroll.schema import PROJECT_ROOT
from jax_unroll.selector import DecisionCache, DecisionEngine, select_factor


def costs(cold, warm):
    return LogCosts(math.log(cold), math.log(warm))


@pytest.fixture
def predictions():
    return {1: costs(1, 1_000), 2: costs(2, 100), 4: costs(4, 20), 8: costs(8, 10)}


@pytest.mark.parametrize(("budget", "R", "expected"), [(None, 1, 1), (None, 2, 2), (2, 100, 2), (1, 100, 1), (0, 100, 1), (None, 0, 1)])
def test_total_units_feasibility_and_budget_fallback(predictions, budget, R, expected):
    assert select_factor(predictions, B=budget, R=R) == expected


def test_throughput_mode_honors_budget(predictions):
    assert select_factor(predictions, B=4, R=1, throughput_only=True) == 4
    assert select_factor(predictions, R=0, throughput_only=True) == 8


def test_exact_budget_in_log_space(predictions):
    predictions[8] = costs(100, 1)
    assert select_factor(predictions, B=100, throughput_only=True) == 8


def test_deterministic_smaller_factor_tie_break(predictions):
    predictions[2] = predictions[4] = predictions[8] = costs(2, 100)
    assert select_factor(dict(reversed(tuple(predictions.items()))), R=100) == 2
    assert select_factor(predictions, R=100, throughput_only=True) == 2


def test_uncertainty_or_missing_candidates_fall_back(predictions):
    assert select_factor(predictions, uncertain=True) == 1
    assert select_factor({2: predictions[2]}) == select_factor(None) == 1


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -1_000.0])
def test_invalid_log_prediction_falls_back(predictions, invalid):
    predictions[1] = LogCosts(invalid, 1)
    assert select_factor(predictions, throughput_only=True) == 1


@pytest.mark.parametrize("options", [{"B": -1}, {"B": math.nan}, {"R": -1}, {"R": 1.5}])
def test_invalid_context_is_rejected(predictions, options):
    with pytest.raises(ValueError):
        select_factor(predictions, **options)


@pytest.fixture
def engine():
    return DecisionEngine(CostModel.load(PROJECT_ROOT / "models" / "cost_cpu.json"))


def test_decision_traces_once_never_lowers_and_replays_without_trace(engine, monkeypatch):
    workload = build_workload("scalar_recurrence", 64, 1)
    extract = selector.extract_features
    calls = []

    def count(*args, **kwargs):
        calls.append(kwargs["factor"])
        return extract(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("lowering during decision")

    monkeypatch.setattr(selector, "extract_features", count)
    monkeypatch.setattr(type(jax.jit(lambda x: x)), "lower", forbidden)
    options = {"trip_count": 64, "R": 10}
    result = engine.decide(workload.fn, workload.args, **options)
    assert calls == [1] and not result.cache_hit and result.fallback_reason is None
    assert engine.decide(signature=result.signature, **options).cache_hit
    assert calls == [1]
    assert engine.decide(workload.fn, workload.args, **options).cache_hit
    assert calls == [1, 1]


def test_extraction_failure_and_unsupported_trip(engine):
    assert engine.decide(lambda: None, trip_count=32).fallback_reason == "feature_extraction_failed"
    workload = build_workload("scalar_recurrence", 8, 1)
    decision = engine.decide(workload.fn, workload.args, trip_count=8)
    assert decision.factor == 1 and decision.fallback_reason == "unsupported_features"


def test_cache_key_distinguishes_signature_budget_horizon_and_throughput(engine):
    workload = build_workload("scalar_recurrence", 64, 1)
    cold = engine.decide(workload.fn, workload.args, trip_count=64)
    for options in ({"B": 100}, {"R": 20}, {"throughput_only": True}):
        warm = engine.decide(workload.fn, workload.args, trip_count=64, signature=cold.signature, **options)
        assert not warm.cache_hit
        assert engine.decide(signature=cold.signature, trip_count=64, **options).cache_hit
    for change in ({"input_shapes": "[]"}, {"dtypes": "[]"}, {"body_hash": "other"}):
        alternate = replace(cold.signature, **change)
        assert not engine.decide(signature=alternate, trip_count=64).cache_hit
    assert engine.decide(signature=cold.signature, trip_count=64, B=math.inf).cache_hit


def test_cache_invalidates_on_backend_feature_and_model_version(engine, monkeypatch):
    workload = build_workload("scalar_recurrence", 64, 1)
    first = engine.decide(workload.fn, workload.args, trip_count=64)
    options = {"trip_count": 64, "signature": first.signature}
    assert engine.decide(**options).cache_hit
    assert engine.decide(**options, backend="gpu").fallback_reason == "incompatible_model"
    assert not engine.decide(workload.fn, workload.args, **options).cache_hit
    monkeypatch.setattr(selector, "FEATURE_VERSION", engine.model.feature_version + 1)
    assert engine.decide(**options).fallback_reason == "incompatible_model"
    monkeypatch.setattr(selector, "FEATURE_VERSION", engine.model.feature_version)
    assert not engine.decide(workload.fn, workload.args, **options).cache_hit
    monkeypatch.setattr(CostModel, "version", property(lambda _: "another-version"))
    assert not engine.decide(workload.fn, workload.args, **options).cache_hit


def test_json_cache_round_trip_and_corrupt_snapshot(engine, tmp_path):
    path = tmp_path / "decisions.json"
    engine.cache = DecisionCache(path)
    workload = build_workload("scalar_recurrence", 64, 1)
    cold = engine.decide(workload.fn, workload.args, trip_count=64)
    warm = DecisionEngine(engine.model, DecisionCache(path))
    assert warm.decide(signature=cold.signature, trip_count=64).cache_hit
    warm.decide(signature=cold.signature, trip_count=64, backend="gpu")
    assert not DecisionEngine(engine.model, DecisionCache(path)).decide(
        signature=cold.signature, trip_count=64,
    ).cache_hit
    path.write_text("{")
    corrupted = DecisionEngine(engine.model, DecisionCache(path))
    assert not corrupted.decide(workload.fn, workload.args, trip_count=64).cache_hit
