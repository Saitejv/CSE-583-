import jax
import numpy as np
import pytest

from benchmarks.corpus.static_matched import MATCHED_COMPONENT, SPECS, fold_assignments, matrix_configs
from jax_unroll.features import extract_features_from_workload


@pytest.mark.parametrize("spec,trip", [(spec, trip) for spec in SPECS for trip in spec.trip_counts], ids=lambda value: value.name if hasattr(value, "name") else str(value))
@pytest.mark.parametrize("factor", (1, 2, 4, 8))
def test_runtime_pairs_have_identical_jaxpr_and_features(spec, trip, factor):
    first, second = [spec.build(trip, factor, variant) for variant in spec.shape_variants]
    assert str(jax.make_jaxpr(first.fn)(*first.args)) == str(jax.make_jaxpr(second.fn)(*second.args))
    assert first.shape == second.shape
    assert [arg.shape for arg in first.args] == [arg.shape for arg in second.args]
    assert [arg.dtype for arg in first.args] == [arg.dtype for arg in second.args]
    index_argument = 2 if spec.name == "matched_gather" else 1
    assert all(np.array_equal(left, right) for index, (left, right) in enumerate(zip(first.args, second.args)) if index != index_argument)
    assert any(not np.array_equal(left, right) for left, right in zip(first.args, second.args))
    a, b = [extract_features_from_workload(workload) for workload in (first, second)]
    a.pop("extraction_ms")
    b.pop("extraction_ms")
    assert a == b


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.name)
def test_runtime_pairs_preserve_finite_factor_equivalence(spec):
    for variant in spec.shape_variants:
        expected_workload = spec.build(17, 1, variant)
        expected = jax.jit(expected_workload.fn)(*expected_workload.args)
        workload = spec.build(17, 8, variant)
        actual = np.asarray(jax.jit(workload.fn)(*workload.args))
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=spec.atol, rtol=spec.rtol)


def test_matched_matrix_and_grouping_keep_all_variants_together():
    from benchmarks.corpus.pilot import family_components
    from benchmarks.corpus.registry import SPECS as PILOT_SPECS, related_groups

    primary, divisible = matrix_configs(), matrix_configs("divisible")
    assert len(primary) == 72
    assert len(divisible) == 48
    assert {row["trip_count"] for row in primary} == {1023}
    assert {row["trip_count"] for row in divisible} == {1024}
    assert {row["workload"] for row in divisible} == {spec.name for spec in SPECS[:2]}
    assert set(fold_assignments().values()) == {MATCHED_COMPONENT}
    components = family_components((*PILOT_SPECS, *SPECS), related_groups())
    assert len(components) == 8
    assert {spec.name for spec in SPECS} <= set(components[MATCHED_COMPONENT])
    for rows in (primary, divisible):
        for name in {row["workload"] for row in rows}:
            body = [row for row in rows if row["workload"] == name]
            assert len(body) == 24
            assert {row["factor"] for row in body} == {1, 2, 4, 8}
            assert {row["process_repeat"] for row in body} == {0, 1, 2}
