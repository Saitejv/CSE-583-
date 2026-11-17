from __future__ import annotations

import jax
import numpy as np
import pytest

from benchmarks.workloads import (
    FAMILIES,
    WORKLOADS,
    build_workload,
    get_spec,
    iter_specs,
    registry_table,
)
from jax_unroll.schema import FACTORS, TRIP_COUNTS

EXPECTED_FAMILIES = {
    "recurrence",
    "elementwise",
    "reduction",
    "dense_linear_algebra",
    "indexed_memory",
    "iterative_solver",
}

_COMPILED: dict[tuple[str, int, int], tuple[object, object]] = {}


def _run(name: str, trip: int, factor: int):
    key = (name, trip, factor)
    if key not in _COMPILED:
        workload = build_workload(name, trip, factor)
        compiled = jax.jit(workload.fn).lower(*workload.args).compile()
        output = compiled(*workload.args)
        jax.block_until_ready(output)
        _COMPILED[key] = (workload, output)
    return _COMPILED[key]


def _leaf_shapes(tree) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(np.shape(leaf)) for leaf in jax.tree.leaves(tree))


def test_registry_has_expected_families_and_shape():
    assert FACTORS == (1, 2, 4, 8)
    assert TRIP_COUNTS == (16, 64, 100, 256, 1000, 1024)
    assert len(FAMILIES) == 6
    assert set(FAMILIES) == EXPECTED_FAMILIES
    assert len(WORKLOADS) >= 6

    for spec in WORKLOADS.values():
        assert spec.family in EXPECTED_FAMILIES
        assert spec.trip_counts == TRIP_COUNTS
        assert spec.primary_variant in spec.shape_variants
        assert len(spec.shape_variants) >= 2

    table = {row["workload"]: row for row in registry_table()}
    assert set(table) == set(WORKLOADS)
    for name, row in table.items():
        assert row["factors"] == list(FACTORS)
        assert row["trip_counts"] == list(TRIP_COUNTS)
        assert row["family"] == WORKLOADS[name].family


def test_iter_specs_filters_by_family():
    subset = iter_specs(["iterative_solver"])
    assert [spec.name for spec in subset] == ["jacobi_solver"]
    assert len(iter_specs()) == len(WORKLOADS)
    with pytest.raises(ValueError, match="unknown families"):
        iter_specs(["not_a_family"])


def test_build_workload_rejects_unknown_workload():
    with pytest.raises(KeyError, match="unknown workload"):
        build_workload("does_not_exist", 16, 1)
    with pytest.raises(KeyError, match="available"):
        get_spec("does_not_exist")


def test_build_workload_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unknown variant"):
        build_workload("scalar_recurrence", 16, 1, variant="bogus")


@pytest.mark.parametrize("trip", [0, -1, -16])
def test_build_workload_rejects_non_positive_trip(trip):
    with pytest.raises(ValueError, match="trip_count must be positive"):
        build_workload("scalar_recurrence", trip, 1)


@pytest.mark.parametrize("factor", [0, -2, -8])
def test_build_workload_rejects_non_positive_factor(factor):
    with pytest.raises(ValueError, match="factor must be positive"):
        build_workload("scalar_recurrence", 16, factor)


@pytest.mark.parametrize("name", sorted(WORKLOADS))
@pytest.mark.parametrize("trip", [16, 100])
@pytest.mark.parametrize("factor", FACTORS)
def test_unroll_factor_matches_factor_one(name, trip, factor):
    workload, output = _run(name, trip, factor)
    reference, expected = _run(name, trip, 1)

    assert workload.loop_kind == reference.loop_kind
    assert workload.trip_count == trip
    assert workload.factor == factor
    assert workload.divides_trip is (trip % factor == 0)

    leaves_out = jax.tree.leaves(output)
    leaves_ref = jax.tree.leaves(expected)
    assert len(leaves_out) == len(leaves_ref)
    for got, want in zip(leaves_out, leaves_ref, strict=True):
        got_array = np.asarray(got)
        want_array = np.asarray(want)
        assert np.all(np.isfinite(got_array))
        np.testing.assert_allclose(
            got_array,
            want_array,
            atol=reference.atol,
            rtol=reference.rtol,
            err_msg=f"{name} T={trip} u={factor} differs from u=1",
        )


def test_divides_trip_semantics():
    assert build_workload("elementwise_sine", 100, 8).divides_trip is False
    assert build_workload("elementwise_sine", 64, 8).divides_trip is True
    assert build_workload("elementwise_sine", 100, 4).divides_trip is True
    assert build_workload("elementwise_sine", 100, 1).divides_trip is True
    assert build_workload("scalar_recurrence", 16, 8).divides_trip is True


def test_fori_loop_family_is_present_and_reports_loop_kind():
    fori_specs = [spec for spec in WORKLOADS.values() if spec.loop_kind == "fori_loop"]
    assert fori_specs
    assert "jacobi_solver" in {spec.name for spec in fori_specs}
    assert {spec.family for spec in fori_specs} == {"iterative_solver"}

    workload = build_workload("jacobi_solver", 16, 4)
    assert workload.loop_kind == "fori_loop"
    assert workload.family == "iterative_solver"

    for spec in WORKLOADS.values():
        assert spec.loop_kind in {"scan", "fori_loop"}


@pytest.mark.parametrize("name", sorted(WORKLOADS))
def test_shape_variants_change_argument_shapes(name):
    spec = WORKLOADS[name]
    variant_shapes = {}
    for variant in sorted(spec.shape_variants):
        workload = build_workload(name, 16, 1, variant=variant)
        shapes = _leaf_shapes(workload.args)
        assert shapes
        assert workload.variant == variant
        variant_shapes[variant] = shapes

    assert len(set(variant_shapes.values())) == len(variant_shapes)
    primary_shapes = variant_shapes[spec.primary_variant]
    for variant, shapes in variant_shapes.items():
        if variant != spec.primary_variant:
            assert shapes != primary_shapes
