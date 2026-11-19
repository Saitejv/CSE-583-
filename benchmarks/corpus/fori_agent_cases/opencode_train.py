"""Reviewed workload drafts from OpenCode."""

import numpy as np
import jax.numpy as jnp
from jax import lax


def order_book_depth(variant, trip, factor):
    """Limit-order-book depth updates across several instruments."""
    rng = np.random.default_rng(1101)
    n_inst, n_levels = (4, 8) if variant == "small" else (8, 16)
    n_upd = 3
    inst = rng.integers(0, n_inst, size=(trip, n_upd), dtype=np.int32)
    level = rng.integers(0, n_levels, size=(trip, n_upd), dtype=np.int32)
    arr = rng.uniform(0.0, 2.0, size=(trip, n_upd)).astype(np.float32)
    can = rng.uniform(0.0, 2.0, size=(trip, n_upd)).astype(np.float32)
    init = rng.uniform(4.0, 10.0, size=(n_inst, n_levels)).astype(np.float32)

    def run(inst, level, arr, can, init):
        def step(i, vol):
            vol = vol.at[inst[i], level[i]].add(arr[i] - can[i])
            return jnp.clip(vol, 0.0, 24.0)

        return lax.fori_loop(0, trip, step, init, unroll=factor)

    return run, (inst, level, arr, can, init)


def halftone_modulator(variant, trip, factor):
    """Bank of second-order error-feedback pulse-density encoders."""
    rng = np.random.default_rng(2202)
    n_enc = 4 if variant == "small" else 8
    t = np.arange(trip, dtype=np.float32)[:, None]
    base = rng.uniform(0.2, 0.8, size=(1, n_enc)).astype(np.float32)
    phase = rng.uniform(0.0, 1.0, size=(1, n_enc)).astype(np.float32)
    x = base + 0.15 * np.sin(2.0 * np.pi * (t / 64.0 + phase))
    x += rng.normal(0.0, 0.02, size=(trip, n_enc))
    x = np.clip(x, 0.05, 0.95).astype(np.float32)
    e1 = rng.uniform(-0.1, 0.1, size=(n_enc,)).astype(np.float32)
    e2 = rng.uniform(-0.1, 0.1, size=(n_enc,)).astype(np.float32)
    out = rng.uniform(0.0, 1.0, size=(n_enc,)).astype(np.float32)

    def run(x, e1, e2, out):
        def step(i, carry):
            e1, e2, out = carry
            v = x[i] + 2.0 * e1 - e2
            y = (v >= 0.5).astype(v.dtype)
            err = jnp.clip(v - y, -2.0, 2.0)
            out = 0.98 * out + 0.02 * y
            return err, e1, out

        return lax.fori_loop(0, trip, step, (e1, e2, out), unroll=factor)

    return run, (x, e1, e2, out)


def fog_transmittance(variant, trip, factor):
    """Front-to-back fog compositing with a new slab every 32 samples."""
    rng = np.random.default_rng(3303)
    h, w = (8, 12) if variant == "small" else (16, 24)
    density = rng.uniform(0.01, 0.2, size=(trip, h, w)).astype(np.float32)
    sample = rng.uniform(0.0, 1.0, size=(trip, h, w)).astype(np.float32)
    t0 = rng.uniform(0.6, 1.0, size=(h, w)).astype(np.float32)
    c0 = np.zeros((h, w), dtype=np.float32)

    def run(density, sample, t0, c0, slab):
        def step(i, carry):
            trans, color = carry
            reset = (i % slab) == 0
            incoming = jnp.where(reset, t0, trans)
            alpha = density[i]
            color = jnp.where(reset, c0, color) + incoming * alpha * sample[i]
            return incoming * (1.0 - alpha), color

        return lax.fori_loop(0, trip, step, (t0, c0), unroll=factor)

    return run, (density, sample, t0, c0, np.int32(32))


def tournament_rating(variant, trip, factor):
    """Paired Elo-style rating updates with repeated opponents."""
    rng = np.random.default_rng(4404)
    n_players = 8 if variant == "small" else 16
    pool = np.arange(n_players, dtype=np.int32).reshape(-1, 2)
    pairs = pool[rng.integers(0, len(pool), size=(trip,))]
    a = pairs[:, 0].astype(np.int32)
    b = pairs[:, 1].astype(np.int32)
    score = rng.integers(0, 2, size=(trip,)).astype(np.float32)
    init = rng.normal(1500.0, 100.0, size=(n_players,)).astype(np.float32)

    def run(a, b, score, init):
        def step(i, ratings):
            ia, ib = a[i], b[i]
            ra, rb = ratings[ia], ratings[ib]
            expected = 1.0 / (1.0 + jnp.exp((rb - ra) / 200.0))
            delta = 12.0 * (score[i] - expected)
            ratings = ratings.at[ia].add(delta).at[ib].add(-delta)
            return jnp.clip(ratings, 800.0, 2200.0)

        return lax.fori_loop(0, trip, step, init, unroll=factor)

    return run, (a, b, score, init)
