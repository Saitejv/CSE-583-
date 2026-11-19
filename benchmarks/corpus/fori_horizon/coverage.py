"""Separate action-coverage follow-up, frozen after the original native sweep."""

from hashlib import sha256
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.fori_horizon.cohort import OUT
from benchmarks.corpus.fori_regret.workloads import Case

DIRECT = (1000, 5000, 20000)
DIRECTORY = OUT / "coverage"


def lagged_empirical_cdf(size, trip, factor):
    samples = np.random.default_rng(941).normal(size=trip).astype("float32")
    def run(values, thresholds, initial):
        def step(i, carry):
            counts, previous = carry
            current = values[i]
            event = (previous <= thresholds)[:, None] & (current <= thresholds)[None, :]
            return counts + event, current
        return lax.fori_loop(0, trip, step, (initial, values[0]), unroll=factor)
    return run, (samples, np.linspace(-2, 2, 32, dtype=np.float32), np.zeros((32, 32), np.int32))


def saturating_photodetector(size, trip, factor):
    rng = np.random.default_rng(942)
    power = rng.uniform(0, .02, trip).astype("float32")
    response = rng.uniform(.1, 1., (24, 32)).astype("float32")
    def run(initial, gain, beam):
        def step(i, charge):
            return .992 * charge + (1. - charge) * gain * beam[i]
        return lax.fori_loop(0, trip, step, initial, unroll=factor)
    return run, (np.zeros_like(response), response, power)


def panel_drawdown(size, trip, factor):
    returns = np.random.default_rng(943).uniform(-.03, .03, (trip, 24, 24)).astype("float32")
    def run(pnl, zero):
        def step(i, carry):
            equity, peak, worst = carry
            next_equity = equity + pnl[i]
            next_peak = jnp.maximum(peak, next_equity)
            return next_equity, next_peak, jnp.maximum(worst, next_peak - next_equity)
        return lax.fori_loop(0, trip, step, (zero, zero, zero), unroll=factor)
    return run, (returns, np.zeros((24, 24), np.float32))


def matrix():
    return [(Case(fn.__name__, fn.__name__, "coverage", fn), "panel", 256)
            for fn in (lagged_empirical_cdf, saturating_photodetector, panel_drawdown)]


def freeze(create=False):
    artifact = dict(source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                    policy_sha256=sha256((OUT / "policy.json").read_bytes()).hexdigest(),
                    cases=[(s.name, s.family, v, t) for s,v,t in matrix()], factors=[1,2,4,8],
                    horizons=DIRECT, repeats=3,
                    reason="First two transfer probes had two indexing operations. Exercise one-stream matrix updates.")
    path = DIRECTORY / "matrix.json"
    data = json.dumps(artifact, indent=2) + "\n"
    if create and not path.exists():
        DIRECTORY.mkdir(exist_ok=True)
        with path.open("x") as file:
            file.write(data)
    if path.read_text() != data:
        raise ValueError("coverage or policy changed after freeze")
