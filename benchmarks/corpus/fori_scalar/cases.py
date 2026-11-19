"""Frozen training and transfer families for the long scalar-stream extension."""

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.fori_regret.workloads import Case


def cusum(state, sample):
    return jnp.maximum(0, state + sample - 0.02)


def leaky_minimum(state, sample):
    return jnp.minimum(0.997 * state, sample)


def tracking_error(state, sample):
    return state + 0.25 * (jnp.abs(sample - state) - state)


def inventory(state, sample):
    return jnp.clip(state + sample, 0, 2)


def nonlinear_filter(state, sample):
    return jnp.tanh(0.95 * state + sample)


def build(update, variant, trip, factor):
    samples = np.random.default_rng(583207).normal(0, 0.2, size=trip).astype(np.float32)

    def run(initial, xs):
        return lax.fori_loop(0, trip, lambda i, c: update(c, xs[i]), initial, unroll=factor)

    return run, (np.float32(0), samples)


def settings(split):
    families, trips = {
        "train": ((cusum, leaky_minimum), (2048, 4095, 8192)),
        "test": ((tracking_error, inventory, nonlinear_filter), (3073, 6144)),
    }[split]
    return [(Case(fn.__name__, fn.__name__, split, partial(build, fn)), "scalar", t)
            for fn in families for t in trips]
