from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from benchmarks.workloads import build_workload
from jax_unroll.features import (
    FEATURE_COLUMNS,
    FeatureExtractionError,
    extract_features,
    extract_features_from_workload,
    extract_loop_body,
)


def _features(name: str, trip: int, factor: int):
    workload = build_workload(name, trip, factor)
    return workload, extract_features_from_workload(workload)


def test_feature_columns_match_feature_dict():
    _, features = _features("scalar_recurrence", 16, 1)
    assert tuple(features) == FEATURE_COLUMNS
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
    json.dumps(features, default=str)


def test_scalar_recurrence_known_features():
    _, features = _features("scalar_recurrence", 16, 1)

    assert features["loop_kind"] == "scan"
    assert features["loop_primitive"] == "scan"
    assert features["num_consts"] == 0
    assert features["num_carry"] == 1
    assert features["num_loop_carried"] == 1
    assert features["num_xs"] == 1
    assert features["num_invars"] == 2
    assert features["num_outvars"] == 1
    assert features["num_equations"] == 2
    assert features["eq_elementwise"] == 2
    assert features["eq_dot"] == features["eq_reduce"] == 0
    assert features["eq_gather_scatter"] == features["eq_convert"] == 0
    assert features["eq_other"] == 0
    assert features["carry_bytes"] == 4
    assert features["carry_elements"] == 1
    assert features["input_bytes"] == 8
    assert features["max_tensor_bytes"] == 4
    assert features["carry_shapes"] == "[[]]"
    assert features["remainder"] == 0
    assert features["divides_evenly"] is True
    assert features["has_tail"] is False
    assert features["full_chunks"] == 16
    assert features["unrolled_equations"] == 2
    assert features["equations_per_iteration"] == 2
    assert features["longest_path"] == 2
    assert features["dependence_density"] == 0.5
    assert features["dtype_bits"] == 32
    assert features["dtypes"] == '["float32"]'
    assert len(features["body_hash"]) == 16
    assert features["extraction_ms"] > 0
    assert features["B"] is None
    assert features["R"] is None


def test_remainder_and_unrolled_features():
    _, features = _features("elementwise_sine", 100, 8)

    assert features["trip_count"] == 100
    assert features["factor"] == 8
    assert features["full_chunks"] == 12
    assert features["remainder"] == 4
    assert features["remainder_ratio"] == pytest.approx(0.04)
    assert features["divides_evenly"] is False
    assert features["has_tail"] is True
    assert features["factor_over_trip"] == pytest.approx(0.08)
    assert features["unrolled_equations"] == features["num_equations"] * 8
    assert features["carry_bytes"] == 1024 * 4


def test_operation_mix_for_reduction_dot_and_indexed_workloads():
    _, reduction = _features("reduction_vector", 16, 1)
    assert reduction["eq_reduce"] == 1
    assert reduction["eq_elementwise"] == 2
    assert reduction["eq_dot"] == 0
    assert reduction["num_consts"] == 1
    assert reduction["const_bytes"] == 1024 * 4

    _, matvec = _features("dense_matvec", 16, 1)
    assert matvec["eq_dot"] == 1
    assert matvec["eq_elementwise"] == 1
    assert matvec["flops_proxy"] > 0

    _, indexed = _features("indexed_update", 16, 1)
    assert indexed["eq_gather_scatter"] == 2
    assert indexed["eq_elementwise"] >= 2
    assert indexed["dtype_bits"] == 32
    assert "int32" in indexed["dtypes"]


def test_fori_loop_features_exclude_counter_from_state():
    _, features = _features("jacobi_solver", 16, 1)

    assert features["loop_kind"] == "fori_loop"
    assert features["loop_primitive"] == "scan"
    assert features["num_consts"] == 1
    assert features["num_carry"] == 2
    assert features["num_loop_carried"] == 1
    assert features["num_xs"] == 0
    assert features["carry_bytes"] == 32 * 32 * 4
    assert features["carry_elements"] == 32 * 32
    assert features["divides_evenly"] is True
    assert features["eq_other"] >= 1


def test_scan_and_equivalent_fori_loop_parity():
    width = 8
    xs = jnp.arange(16 * width, dtype=jnp.float32).reshape((16, width))

    def step(carry, x):
        return jnp.sin(carry + x), None

    def scan_fn(init, xs_val):
        carry, _ = lax.scan(step, init, xs_val, unroll=1)
        return carry

    x0 = xs[0]

    def fori_body(_i, carry):
        return jnp.sin(carry + x0)

    def fori_fn(init):
        return lax.fori_loop(0, 16, fori_body, init, unroll=1)

    init = jnp.zeros((width,), dtype=jnp.float32)
    scan_features = extract_features(
        scan_fn, (init, xs), loop_kind="scan", trip_count=16, factor=1
    )
    fori_features = extract_features(
        fori_fn, (init,), loop_kind="fori_loop", trip_count=16, factor=1
    )

    assert scan_features["carry_bytes"] == fori_features["carry_bytes"]
    assert scan_features["carry_elements"] == fori_features["carry_elements"]
    assert scan_features["num_loop_carried"] == fori_features["num_loop_carried"]
    assert scan_features["max_tensor_bytes"] == fori_features["max_tensor_bytes"]
    assert scan_features["eq_dot"] == fori_features["eq_dot"] == 0
    assert scan_features["eq_reduce"] == fori_features["eq_reduce"] == 0
    assert (
        fori_features["eq_elementwise"]
        == scan_features["eq_elementwise"] + 1  # i + 1 counter update
    )
    assert scan_features["loop_primitive"] == "scan"
    assert fori_features["loop_primitive"] == "scan"
    assert scan_features["body_hash"] != fori_features["body_hash"]


def test_extraction_does_not_compile_or_lower(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("feature extraction must not compile candidates")

    monkeypatch.setattr(jax, "jit", _forbidden)

    workload = build_workload("elementwise_sine", 64, 4)
    features = extract_features_from_workload(workload)
    assert features["num_equations"] > 0


def test_extraction_is_deterministic():
    workload = build_workload("indexed_update", 64, 2)
    first = extract_features_from_workload(workload)
    second = extract_features_from_workload(workload)

    assert first["body_hash"] == second["body_hash"]
    for column in FEATURE_COLUMNS:
        if column == "extraction_ms":
            continue
        assert first[column] == second[column], column


def test_extract_loop_body_without_loop_raises():
    with pytest.raises(FeatureExtractionError, match="no scan/while loop"):
        extract_features(
            lambda x: x + 1.0, (jnp.float32(0.0),), trip_count=16, factor=1
        )


def test_bare_body_extraction():
    width = 4

    def fori_body(i, carry):
        return carry + i.astype(jnp.float32)

    features = extract_features(
        body_fn=fori_body,
        body_args=(jnp.int32(0), jnp.zeros((width,), dtype=jnp.float32)),
        loop_kind="fori_loop",
        trip_count=16,
        factor=2,
    )
    assert features["loop_kind"] == "fori_loop"
    assert features["loop_primitive"] == "body"
    assert features["num_loop_carried"] == 1
    assert features["carry_bytes"] == width * 4
    assert features["num_equations"] >= 1

    loop = extract_loop_body(
        body_fn=lambda c, x: (c + x, None),
        body_args=(jnp.zeros((width,)), jnp.zeros((width,))),
        loop_kind="scan",
    )
    assert loop.source == "body"
    assert loop.num_carry == 2
    assert loop.num_xs == 0


def test_context_arguments_are_recorded():
    workload = build_workload("scalar_recurrence", 16, 1)
    features = extract_features_from_workload(workload, B=12.5, R=1000)
    assert features["B"] == 12.5
    assert features["R"] == 1000
    assert features["backend"] == jax.default_backend()
    assert features["device"] == str(jax.devices()[0])


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="trip_count must be positive"):
        extract_features(lambda x: x, (np.zeros(1),), trip_count=0, factor=1)
    with pytest.raises(ValueError, match="factor must be positive"):
        extract_features(lambda x: x, (np.zeros(1),), trip_count=1, factor=0)
    with pytest.raises(ValueError, match="B must be non-negative"):
        extract_features(
            lambda x: x, (np.zeros(1),), trip_count=1, factor=1, B=-1
        )
    with pytest.raises(ValueError, match="loop_kind"):
        extract_features(
            lambda x: x,
            (np.zeros(1),),
            loop_kind="bogus",
            trip_count=1,
            factor=1,
        )
