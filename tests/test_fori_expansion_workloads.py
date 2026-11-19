"""Tests for the 70 newly authored independent fori_loop workloads."""

import jax
import numpy as np
import pytest

from benchmarks.corpus.fori_expansion.registry import NEW_INDEPENDENT_SPECS
from benchmarks.sweep import _all_finite, _allclose


def test_new_independent_count():
    assert len(NEW_INDEPENDENT_SPECS) == 70
    names = {s.name for s in NEW_INDEPENDENT_SPECS}
    assert len(names) == 70
    for s in NEW_INDEPENDENT_SPECS:
        assert s.loop_kind == "fori_loop"


@pytest.mark.parametrize("spec", NEW_INDEPENDENT_SPECS, ids=lambda s: s.name)
def test_workload_factor_invariance(spec):
    trip = 64
    var = spec.primary_variant

    wl1 = spec.build(trip, 1, var)
    out1 = wl1.fn(*wl1.args)
    assert _all_finite(out1), f"{spec.name}: non-finite output at factor=1"

    wl4 = spec.build(trip, 4, var)
    out4 = wl4.fn(*wl4.args)
    assert _all_finite(out4), f"{spec.name}: non-finite output at factor=4"

    assert _allclose(out1, out4, atol=spec.atol, rtol=spec.rtol), (
        f"{spec.name}: output at factor=4 differs from factor=1"
    )
