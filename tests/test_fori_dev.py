"""Public API scope, grouped predictions and the frozen inference contract."""


import numpy as np
import pandas as pd
import pytest

from benchmarks.corpus import fori_dev as experiment
from benchmarks.corpus import static_matched
from benchmarks.corpus.registry import SPECS
from benchmarks.recovery_data import MODEL, config_key
from jax_unroll.auto_loop import auto_fori_loop, auto_scan
from jax_unroll.model import CostModel
from jax_unroll.schema import FACTORS


@pytest.fixture(scope="module")
def development():
    folds = experiment.eligible_folds()
    rows, repeats, excluded = experiment.prepare(folds)
    assert not excluded
    model = CostModel.load(MODEL)
    forecasts, splits = experiment.predict(rows, folds, model)
    return rows, folds, forecasts, splits, repeats, model


def test_public_loop_scope_and_transitive_components(development):
    rows, folds, _, _, repeats, _ = development
    bodies = {spec.name for spec in SPECS if spec.loop_kind == "fori_loop"}
    assert set(repeats) == {r["workload"] for r in rows} == bodies
    assert len(rows) == len(bodies) * 4 == 44
    assert {r["loop_primitive"] for r in rows} == {"scan"}
    assert {r["loop_kind"] for r in rows} == {"fori_loop"}
    assert not bodies.intersection(s.name for s in static_matched.SPECS)
    assert sorted(map(len, folds.values())) == [1, 2, 4, 4]
    assert folds[static_matched.MATCHED_COMPONENT] == ["locality_block_reuse", "scatter_min_update"]


def test_grouped_splits_keep_every_configuration_together():
    groups = [("a", v, t) for v in ("clustered", "dispersed") for t in (1023, 1024)]
    groups += [("relative", "primary", 1023), ("b", "v", 512)]
    splits = dict((fold, (train, test)) for fold, train, test in experiment.grouped_splits(
        groups, {"related": ["a", "relative"], "other": ["b"]}))
    assert splits["related"][1] == sorted(groups[:-1])
    assert splits["other"][0] == sorted(groups[:-1])
    for folds in ({"one": ["a", "relative"], "two": ["a", "b"]}, {"one": ["a"], "two": ["b"]}):
        with pytest.raises(ValueError, match="exactly one"):
            list(experiment.grouped_splits(groups, folds))


def test_test_component_labels_do_not_influence_its_forecasts(development):
    rows, folds, forecasts, _, _, model = development
    for members in folds.values():
        # Alter both stored log labels and linear outcomes, including the factor-1 reference.
        changed = [dict(r, log_warm_us=r["log_warm_us"] + r["factor"], warm_median_us=5. + r["factor"],
                        log_cold_ms=2., cold_median_ms=np.exp(2.)) if r["workload"] in members else r
                   for r in rows]
        updated, _ = experiment.predict(changed, folds, model)
        for method, cases in forecasts.items():
            for case, other in zip(cases, updated[method], strict=True):
                if case.features["workload"] in members:
                    assert config_key(case.features) == config_key(other.features)
                    assert case.predictions == other.predictions


def test_factor_one_trace_and_absolute_cost_contract(development):
    _, _, forecasts, _, _, model = development
    for i, frozen_case in enumerate(forecasts["frozen"]):
        base = frozen_case.features
        original = model.predict_candidates(base)
        assert frozen_case.predictions == original
        for method, cases in forecasts.items():
            costs = cases[i].predictions
            assert costs[1] == original[1]
            for u in FACTORS:
                assert costs[u].log_cold_ms == original[u].log_cold_ms
                if method == experiment.PRIMARY:
                    assert costs[u].log_warm_us == pytest.approx(
                        (original[u].log_warm_us + original[1].log_warm_us) / 2)
        redacted = dict(base, workload="anon", family="anon", body_hash=None, B=1, R=0,
                        warm_median_us=None, log_warm_us=None, hlo_bytes_median=None)
        assert model.predict_candidates(redacted) == original
        assert experiment.effect_design(redacted) == experiment.effect_design(base)
    with pytest.raises(TypeError, match="body_fun argument should be callable"):
        auto_fori_loop(0, 1, None, 0)
    with pytest.raises(NotImplementedError):
        auto_scan(None, 0, None)


def test_body_bootstrap_averages_before_resampling():
    rows = pd.DataFrame([dict(body=b, measured_log_ratio=1., zero=0., estimate=estimate)
                         for b, estimate in [("a", 1.)] * 20 + [("b", 0.)]])
    result = experiment.paired_intervals(rows, ["zero", "estimate"], "zero")
    assert result["estimate"] == {"mean": -.5, "ci95": [-1., 0.]}
