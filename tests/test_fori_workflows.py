"""Check the researched recurrences against independent numerical calculations."""

import jax
import numpy as np
import pytest
from scipy.signal import lfilter

from benchmarks.corpus.fori_application.live import Recorder
from benchmarks.corpus.fori_targeted.auto_fori import through_auto_fori
from benchmarks.corpus.fori_workflows.cases import settings
from benchmarks.corpus.fori_workflows.dataset import OUT, load
from jax_unroll.workflow_fori import workflow_engine


def reference(name, args):
    x = np.asarray(args[0], dtype=np.float64)
    if name == "dc_blocker":
        return [x[-1], lfilter([1, -1], [1, -args[1]], x)[-1]]
    if name == "heikin_ashi":
        close = np.mean(np.stack(args[:4]).astype(float), axis=0)
        opening = float(args[-1][0])
        for prev in close[:-1]:
            opening = (opening + prev) / 2
        return [opening, close[-1]]
    if name == "battery_ecm":
        decay = np.exp(-1 / (0.012 * 8000))
        return [args[1] - x.sum() / 10800, lfilter([0.012 * (1 - decay)], [1, -decay], x)[-1]]
    if name == "holt_trend":
        level, trend = map(float, args[1:])
        for y in x:
            residual = y - level - trend
            level, trend = level + trend + 0.3 * residual, trend + 0.015 * residual
        return [level, trend]
    if name == "alpha_beta_tracker":
        dt, a, b = map(float, args[1:4])
        pos, vel = map(float, args[-1])
        history = []
        for y in x:
            innovation = y - pos - dt * vel
            pos, vel = y - (1 - a) * innovation, vel + (b / dt) * innovation
            history.append([pos, vel])
        return np.asarray(history), history[-1]
    if name == "acoustic_resonator_bank":
        a1, a2, gain = args[1:4]
        hist = np.stack([lfilter([g], [1, -a, -b], x) for a, b, g in zip(a1, a2, gain, strict=True)], axis=1)
        return hist, hist[-1], hist[-2]
    if name == "cumulative_trapezoid_integrator":
        return [np.trapezoid(x, dx=float(args[1])), x[-1]]
    if name == "muskingum":
        init = float(args[1])
        # Constant-input equilibrium initializes the transfer-function filter.
        out = init + lfilter([1 / 21, 9 / 21], [1, -11 / 21], x - init)
        return [x[-1], out[-1]]
    if name == "welford_online_moments":
        return [x.mean(), np.sum((x - x.mean()) ** 2)]
    if name == "rainflow_turning_points":
        unique = x[np.r_[True, np.diff(x) != 0]]
        peaks = unique[1:-1][np.diff(np.sign(np.diff(unique))) != 0]
        return [len(peaks), np.abs(np.diff(np.r_[unique[0], peaks])).sum()]
    if name == "hysteretic_event_detector":
        hi, lo, persist = args[1:4]
        active = count = total = 0
        for value in x:
            if active:
                if value <= lo:
                    active = count = 0
            else:
                count = count + 1 if value >= hi else 0
                if count >= persist:
                    active, count = 1, 0
            total += active
        return [active, count, total]
    raise ValueError(name)


@pytest.mark.parametrize(
    "spec,size,trip", settings("train")[::2] + settings("test")[::2], ids=lambda x: getattr(x, "name", str(x))
)
def test_workflow_outputs(spec, size, trip):
    fn, args = spec.build(trip, 4, size)
    actual = jax.device_get(jax.jit(fn)(*jax.device_put(args)))
    expected = reference(spec.name, args)
    if spec.name in ("alpha_beta_tracker", "acoustic_resonator_bank"):
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_allclose(got, want, rtol=2e-4, atol=5e-5)
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=5e-5)


def test_workflow_labeled_splits():
    train, costs = load("train")
    test, held_costs = load("test")
    assert costs.shape == (12, 4, 2) and held_costs.shape == (10, 4, 2)
    assert train.family.nunique() == 5 and test.family.nunique() == 5
    assert set(train.body_hash).isdisjoint(test.body_hash)
    assert set(train.family).isdisjoint(test.family)
    assert np.all(costs > 0) and np.all(held_costs > 0)


@pytest.mark.parametrize("calls,budget,enabled", [(300, 125, False), (20000, 0, False), (20000, 125, True)])
def test_workflow_routing_gates(calls, budget, enabled):
    spec, size, trip = settings("test")[0]
    fn, args = spec.build(trip, 1, size)
    engine = Recorder(workflow_engine(profile_path=OUT / "profile.json"))
    wrapped = through_auto_fori(fn, engine, R=calls, B=budget, throughput_only=False)
    jax.make_jaxpr(wrapped)(*args)
    assert (engine.engine.model is not None) == enabled
    assert (engine.decision.factor > 1) == enabled
