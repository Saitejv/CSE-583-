"""Static fori_loop workloads drafted by Anti-Gravity and reviewed for the prediction benchmark."""


import numpy as np
import jax.numpy as jnp
from jax import lax


def agy_saturating_tanh(variant, trip, factor):
    rng = np.random.default_rng(101)
    n = 64 if variant == "small" else 256
    x = rng.standard_normal(n, dtype=np.float32) * 0.5
    b = rng.standard_normal(n, dtype=np.float32) * 0.1

    def fn(x, b):
        def body(i, val):
            return jnp.tanh(0.92 * val + 0.08 * jnp.roll(val, 1) + b)
        return lax.fori_loop(0, trip, body, x, unroll=factor)

    return fn, (x, b)


def agy_smooth_decay(variant, trip, factor):
    rng = np.random.default_rng(102)
    n = 64 if variant == "small" else 256
    x = rng.uniform(0.1, 1.0, size=n).astype(np.float32)
    target = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)

    def fn(x, target):
        def body(i, val):
            diff = target - val
            return val + 0.05 * jnp.sin(diff)
        return lax.fori_loop(0, trip, body, x, unroll=factor)

    return fn, (x, target)


def agy_wave_advection(variant, trip, factor):
    rng = np.random.default_rng(105)
    n = 128 if variant == "small" else 512
    u = rng.standard_normal(n, dtype=np.float32) * 0.5

    def fn(u):
        def body(i, state):
            du = jnp.roll(state, -1) - jnp.roll(state, 1)
            d2u = jnp.roll(state, -1) - 2.0 * state + jnp.roll(state, 1)
            return state - 0.1 * du + 0.02 * d2u
        return lax.fori_loop(0, trip, body, u, unroll=factor)

    return fn, (u,)


def agy_reaction_diffusion(variant, trip, factor):
    rng = np.random.default_rng(106)
    n = 32 if variant == "small" else 128
    u = rng.uniform(0.2, 0.8, size=n).astype(np.float32)
    v = rng.uniform(0.1, 0.4, size=n).astype(np.float32)

    def fn(u, v):
        def body(i, state):
            curr_u, curr_v = state
            lap_u = jnp.roll(curr_u, 1) - 2.0 * curr_u + jnp.roll(curr_u, -1)
            lap_v = jnp.roll(curr_v, 1) - 2.0 * curr_v + jnp.roll(curr_v, -1)
            du = 0.08 * lap_u + curr_u - (curr_u ** 3) / 3.0 - curr_v
            dv = 0.04 * lap_v + 0.05 * (curr_u + 0.7 - 0.8 * curr_v)
            next_u = jnp.clip(curr_u + 0.1 * du, -2.0, 2.0)
            next_v = jnp.clip(curr_v + 0.1 * dv, -2.0, 2.0)
            return (next_u, next_v)
        return lax.fori_loop(0, trip, body, (u, v), unroll=factor)

    return fn, (u, v)


def agy_power_iteration(variant, trip, factor):
    rng = np.random.default_rng(107)
    n = 16 if variant == "small" else 48
    A = rng.standard_normal((n, n), dtype=np.float32)
    A = (A + A.T) / (2.0 * np.sqrt(n).astype(np.float32))
    v = rng.standard_normal(n, dtype=np.float32)
    v = v / (np.linalg.norm(v).astype(np.float32) + 1e-7)

    def fn(A, v):
        def body(i, state):
            w = jnp.dot(A, state)
            norm = jnp.linalg.norm(w) + 1e-7
            return w / norm
        return lax.fori_loop(0, trip, body, v, unroll=factor)

    return fn, (A, v)


def agy_givens_rotation(variant, trip, factor):
    rng = np.random.default_rng(108)
    n = 16 if variant == "small" else 48
    x = rng.standard_normal(n, dtype=np.float32)

    def fn(x):
        theta = jnp.float32(0.015)
        c = jnp.cos(theta)
        s = jnp.sin(theta)

        def body(i, vec):
            idx = i % (n - 1)
            v0 = vec[idx]
            v1 = vec[idx + 1]
            r0 = c * v0 - s * v1
            r1 = s * v0 + c * v1
            return vec.at[idx].set(r0).at[idx + 1].set(r1)
        return lax.fori_loop(0, trip, body, x, unroll=factor)

    return fn, (x,)


def agy_chebyshev_step(variant, trip, factor):
    rng = np.random.default_rng(109)
    n = 16 if variant == "small" else 48
    M = rng.standard_normal((n, n), dtype=np.float32)
    A = (M.T @ M) / np.float32(n) + np.eye(n, dtype=np.float32) * 0.1
    b = rng.standard_normal(n, dtype=np.float32)
    x0 = np.zeros(n, dtype=np.float32)
    x1 = b * 0.05

    def fn(A, b, x0, x1):
        def body(i, state):
            curr, prev = state
            res = b - jnp.dot(A, curr)
            next_x = 0.8 * curr - 0.1 * prev + 0.05 * res
            return (next_x, curr)
        return lax.fori_loop(0, trip, body, (x1, x0), unroll=factor)

    return fn, (A, b, x0, x1)


def agy_cellular_automaton(variant, trip, factor):
    rng = np.random.default_rng(110)
    n = 64 if variant == "small" else 256
    cells = rng.integers(0, 3, size=n, dtype=np.int32)

    def fn(cells):
        def body(i, s):
            left = jnp.roll(s, 1)
            right = jnp.roll(s, -1)
            sum_nbr = left + right
            cond_quench = s == 2
            cond_active = (s == 1) & (sum_nbr >= 2)
            cond_ignite = (s == 0) & (sum_nbr == 1)
            next_s = lax.select(
                cond_quench,
                jnp.zeros_like(s),
                lax.select(
                    cond_active,
                    jnp.full_like(s, 2),
                    lax.select(cond_ignite, jnp.ones_like(s), s)
                )
            )
            return next_s
        return lax.fori_loop(0, trip, body, cells, unroll=factor)

    return fn, (cells,)


def agy_priority_arbiter(variant, trip, factor):
    rng = np.random.default_rng(112)
    n = 32 if variant == "small" else 128
    queue_levels = rng.integers(0, 20, size=n, dtype=np.int32)
    tokens = rng.integers(5, 15, size=n, dtype=np.int32)

    def fn(queue_levels, tokens):
        def body(i, state):
            q, tok = state
            refill = (i % 4) == 0
            tok_refilled = lax.select(refill, tok + 2, tok)
            can_serve = (q > 0) & (tok_refilled > 0)
            served_q = lax.select(can_serve, q - 1, q)
            served_tok = lax.select(can_serve, tok_refilled - 1, tok_refilled)
            high_pressure = served_q > 15
            clamped_tok = lax.select(high_pressure, jnp.maximum(served_tok - 1, 0), served_tok)
            return (served_q, clamped_tok)
        return lax.fori_loop(0, trip, body, (queue_levels, tokens), unroll=factor)

    return fn, (queue_levels, tokens)


WORKLOADS = {
    "agy_saturating_tanh": ("saturating_signal", agy_saturating_tanh),
    "agy_smooth_decay": ("saturating_signal", agy_smooth_decay),
    "agy_wave_advection": ("spatial_update", agy_wave_advection),
    "agy_reaction_diffusion": ("spatial_update", agy_reaction_diffusion),
    "agy_power_iteration": ("linear_algebra", agy_power_iteration),
    "agy_givens_rotation": ("linear_algebra", agy_givens_rotation),
    "agy_chebyshev_step": ("linear_algebra", agy_chebyshev_step),
    "agy_cellular_automaton": ("branch_control", agy_cellular_automaton),
    "agy_priority_arbiter": ("branch_control", agy_priority_arbiter),
}
