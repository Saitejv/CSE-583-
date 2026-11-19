import jax
import numpy as np
import pytest

from benchmarks.corpus.fori_application.live import Recorder
from benchmarks.corpus.fori_application.scan_ports import settings as source_settings
from benchmarks.corpus.fori_scalar.cases import settings
from benchmarks.corpus.fori_scalar.dataset import load
from benchmarks.corpus.fori_targeted.auto_fori import through_auto_fori
from jax_unroll.compact_fori import application_engine


def test_incremental_training_includes_all_four_sources_and_separate_test_families():
    train, target = load("train")
    test, _ = load("test")
    source = train[train.variant == "source"]
    assert set(source.workload) == {"scalar", "elementwise", "reduction", "matrix"}
    assert target.shape == (10, 4, 2) and np.all(target > 0)
    assert set(test.family).isdisjoint(train.family)
    assert set(test.body_hash).isdisjoint(train.body_hash)
    i = source.index[source.workload == "scalar"][0]
    oracle = (target[i, :, 0] + 19999 * target[i, :, 1] / 1000).argmin()
    assert (1, 2, 4, 8)[oracle] == 8


@pytest.mark.parametrize("calls,budget,expected,loads", [(300, 125, 1, False), (20000, 0, 1, False),
                                                         (20000, 5, 1, True), (20000, 125, 8, True)])
def test_long_scalar_honors_horizon_and_cold_budget(calls, budget, expected, loads):
    spec, size, trip = source_settings()[0]
    native = spec.build(trip, 1, size)
    engine = Recorder(application_engine())
    fn = through_auto_fori(native.fn, engine, R=calls, B=budget, throughput_only=False)
    jax.make_jaxpr(fn)(*native.args)
    assert engine.decision.factor == expected
    assert (engine.engine.model is not None) == loads


def test_out_of_range_and_expensive_steps_keep_the_native_fallback():
    configs = settings("test")
    for spec, size, trip in [(*configs[0][:2], 1025), (*configs[0][:2], 8193), configs[-1]]:
        native = spec.build(trip, 1, size)
        engine = Recorder(application_engine())
        fn = through_auto_fori(native.fn, engine, R=20000, B=125, throughput_only=False)
        jax.make_jaxpr(fn)(*native.args)
        assert engine.decision.factor == 1 and engine.decision.fallback_reason == "unsupported_features"


def test_legacy_forest_predictions_preserved_on_source_controls():
    from benchmarks.corpus.fori_iterations.storage import UNCOMPRESSED
    from benchmarks.corpus.fori_scalar.fit import BUNDLED
    from jax_unroll.compact_fori import CompactForiPredictor

    base = CompactForiPredictor.load(UNCOMPRESSED)
    updated = CompactForiPredictor.load(BUNDLED)
    assert base.version != updated.version
    rows, _ = load("train")
    for row in rows[rows.trip_count <= 1024].to_dict("records"):
        assert base.supports(row) and updated.supports(row)
        assert base.predict_candidates(row) == updated.predict_candidates(row)
