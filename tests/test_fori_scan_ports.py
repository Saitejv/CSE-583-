from functools import partial

import jax
import numpy as np
import pytest

from benchmarks.corpus.fori_application import scan_ports
from benchmarks.corpus.fori_regret.acquire import assert_same_graph
from jax_unroll.schema import FACTORS


@pytest.mark.parametrize("index", range(4))
def test_original_scan_and_native_fori_agree_at_source_dimensions(index):
    scan_ports.check()
    _, scan, inputs, _ = scan_ports.make_workloads()[index]
    spec, variant, trip = scan_ports.settings()[index]
    reference = jax.jit(partial(scan, unroll=1))(*inputs)
    base = spec.build(trip, 1, variant)
    graph = str(jax.make_jaxpr(base.fn)(*inputs))

    for factor in FACTORS:
        port = spec.build(trip, factor, variant)
        assert_same_graph(port.fn, inputs, graph, factor)
        actual = jax.jit(port.fn)(*inputs)
        assert actual.shape == reference.shape
        assert actual.dtype == reference.dtype == np.float32
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("index, expected", [(0, 8), (1, 1), (2, 1), (3, 1)])
def test_source_cases_use_fitted_engine(index, expected):
    from benchmarks.corpus.fori_application.live import Recorder
    from benchmarks.corpus.fori_targeted.auto_fori import through_auto_fori
    from jax_unroll.compact_fori import application_engine

    spec, variant, trip = scan_ports.settings()[index]
    native = spec.build(trip, 1, variant)
    engine = Recorder(application_engine())
    wrapped = through_auto_fori(native.fn, engine, R=5000, B=125, throughput_only=False)
    assert_same_graph(wrapped, native.args, str(jax.make_jaxpr(native.fn)(*native.args)), expected)
    assert engine.decision.fallback_reason is None
