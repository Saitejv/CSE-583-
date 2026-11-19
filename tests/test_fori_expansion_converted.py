"""Tests for the 53 converted scan-to-fori_loop workloads."""

import jax
import numpy as np
import pytest

from benchmarks.corpus.fori_expansion.converted.manifest import CONVERTED_SPECS, CONVERTED_WORKLOADS
import benchmarks.workloads as bw
import benchmarks.recovery_workloads as rw
import benchmarks.corpus.registry as cr
import benchmarks.corpus.static_matched as sm


def _get_orig_spec(orig_name: str):
    if orig_name in bw.WORKLOADS:
        return bw.WORKLOADS[orig_name]
    if orig_name in rw.RECOVERY_WORKLOADS:
        return rw.RECOVERY_WORKLOADS[orig_name]
    if orig_name in cr.CORPUS_WORKLOADS:
        return cr.CORPUS_WORKLOADS[orig_name]
    if orig_name in sm.WORKLOADS:
        return sm.WORKLOADS[orig_name]
    raise KeyError(f"Unknown original workload {orig_name}")


def test_converted_inventory_count():
    assert len(CONVERTED_SPECS) == 53
    assert len(CONVERTED_WORKLOADS) == 53
    for spec in CONVERTED_SPECS:
        assert spec.loop_kind == "fori_loop"
        assert spec.name.startswith("conv_")


@pytest.mark.parametrize("spec", CONVERTED_SPECS, ids=lambda s: s.name)
def test_converted_against_original(spec):
    orig_name = spec.name.replace("conv_", "")
    orig_spec = _get_orig_spec(orig_name)
    var = spec.primary_variant
    trip = 64
    factor = 1

    orig_wl = orig_spec.build(trip, factor, var)
    conv_wl = spec.build(trip, factor, var)

    assert conv_wl.loop_kind == "fori_loop"

    orig_out = orig_wl.fn(*orig_wl.args)
    conv_out = conv_wl.fn(*conv_wl.args)

    leaves_orig = jax.tree.leaves(orig_out)
    leaves_conv = jax.tree.leaves(conv_out)

    assert len(leaves_orig) == len(leaves_conv)
    for o, c in zip(leaves_orig, leaves_conv):
        assert np.allclose(np.asarray(o), np.asarray(c), atol=spec.atol, rtol=spec.rtol)
