from __future__ import annotations

import copy

import jax
import pytest

from benchmarks.workloads import build_workload
from jax_unroll.baselines import (
    BASELINES,
    factor_one,
    fixed_factor,
    llvm_style_heuristic,
)
from jax_unroll.features import extract_features_from_workload
from jax_unroll.schema import FACTORS


def _features(**overrides):
    features = {
        "trip_count": 100,
        "num_equations": 2,
        "factor": 1,
    }
    features.update(overrides)
    return features


def test_factor_one_always_returns_one():
    assert factor_one({}) == 1
    assert factor_one({"trip_count": 1000, "num_equations": 0}) == 1


@pytest.mark.parametrize("factor", FACTORS)
def test_fixed_factor_returns_supported_factor(factor):
    assert fixed_factor({}, factor) == factor
    assert fixed_factor({"trip_count": 1}, factor) == factor


@pytest.mark.parametrize("factor", [0, 3, 5, 16, -2])
def test_fixed_factor_rejects_unsupported_factor(factor):
    with pytest.raises(ValueError, match="factor must be one of"):
        fixed_factor({}, factor)


def test_llvm_style_selects_largest_feasible_factor():
    assert llvm_style_heuristic(_features(num_equations=2)) == 8
    assert llvm_style_heuristic(_features(num_equations=20)) == 4
    assert llvm_style_heuristic(_features(num_equations=65)) == 1


def test_llvm_style_threshold_can_force_factor_one():
    assert llvm_style_heuristic(_features(num_equations=64)) == 2
    assert llvm_style_heuristic(_features(num_equations=100)) == 1
    assert llvm_style_heuristic(_features(num_equations=1), threshold=1) == 1


def test_llvm_style_respects_max_factor():
    features = _features(num_equations=1)
    assert llvm_style_heuristic(features, max_factor=2) == 2
    assert llvm_style_heuristic(features, max_factor=4) == 4
    assert llvm_style_heuristic(features, max_factor=16) == 8


@pytest.mark.parametrize(
    ("trip_count", "expected"),
    [(1, 1), (3, 2), (6, 4), (7, 4), (8, 8)],
)
def test_llvm_style_respects_trip_count(trip_count, expected):
    features = _features(trip_count=trip_count, num_equations=1)
    assert llvm_style_heuristic(features) == expected


def test_llvm_style_factors_sequence_override():
    features = _features(num_equations=1)
    assert llvm_style_heuristic(features, factors=(1, 3)) == 3
    assert llvm_style_heuristic(features, factors=(2, 4), max_factor=2) == 2
    assert llvm_style_heuristic(features, factors=(8, 2, 1)) == 8
    assert llvm_style_heuristic(features, factors=(-2, 0, 1, 3)) == 3
    assert llvm_style_heuristic(features, factors=(1, 2, 16)) == 2
    assert llvm_style_heuristic(features, factors=()) == 1


@pytest.mark.parametrize(
    ("features", "kwargs", "match"),
    [
        (_features(), {"threshold": 0}, "threshold"),
        (_features(), {"threshold": -5}, "threshold"),
        (_features(), {"max_factor": 0}, "max_factor"),
        (_features(trip_count=0), {}, "trip_count"),
        (_features(trip_count=-4), {}, "trip_count"),
        (_features(num_equations=-1), {}, "num_equations"),
    ],
)
def test_llvm_style_rejects_invalid_arguments(features, kwargs, match):
    with pytest.raises(ValueError, match=match):
        llvm_style_heuristic(features, **kwargs)


def test_baselines_registry_keys_and_callables():
    assert set(BASELINES) == {
        "factor_1",
        "factor_2",
        "factor_4",
        "factor_8",
        "llvm_size",
    }
    features = _features(num_equations=2)
    for name, policy in BASELINES.items():
        assert callable(policy), name
        assert isinstance(policy(features), int), name
    assert BASELINES["factor_1"](features) == 1
    assert BASELINES["factor_2"](features) == 2
    assert BASELINES["factor_4"](features) == 4
    assert BASELINES["factor_8"](features) == 8
    assert BASELINES["llvm_size"](features) == 8


def test_llvm_style_on_scalar_recurrence_workload():
    workload = build_workload("scalar_recurrence", 100, 1)
    features = extract_features_from_workload(workload)

    assert features["trip_count"] == 100
    assert features["num_equations"] == 2
    assert BASELINES["llvm_size"](features) == 8
    assert BASELINES["factor_1"](features) == 1


def test_llvm_style_on_jacobi_workload_respects_threshold():
    workload = build_workload("jacobi_solver", 16, 1)
    features = extract_features_from_workload(workload)

    assert features["trip_count"] == 16
    assert features["num_equations"] == 18
    assert BASELINES["llvm_size"](features) == 4


def test_baselines_are_deterministic_and_do_not_mutate_features(monkeypatch):
    features = _features(num_equations=7)
    snapshot = copy.deepcopy(features)

    def _forbidden(*args, **kwargs):
        raise AssertionError("baselines must not trace or compile JAX")

    monkeypatch.setattr(jax, "jit", _forbidden)
    monkeypatch.setattr(jax, "make_jaxpr", _forbidden)

    for name, policy in BASELINES.items():
        first = policy(features)
        second = policy(features)
        assert first == second, name
        assert features == snapshot, name
