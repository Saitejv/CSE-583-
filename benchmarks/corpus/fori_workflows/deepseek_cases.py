"""Reviewed OpenCode DeepSeek workflows with deterministic synthetic observations."""

import jax.numpy as jnp
import numpy as np
from jax import lax


def dc_blocker(variant, trip, factor):
    n = np.arange(trip)
    x = (
        0.30
        + 0.52 * np.sin(0.195 * n + 0.4)
        + 0.17 * np.sin(0.046 * n + 1.25)
        + 0.09 * np.sin(1.087 * n + 2.1)
    )

    def run(stream, r, initial):
        def step(i, state):
            sample = stream[i]
            return jnp.stack((sample, sample - state[0] + r * state[1]))

        return lax.fori_loop(0, trip, step, initial, unroll=factor)

    return run, (x.astype(np.float32), np.float32(0.995), np.zeros(2, np.float32))


def heikin_ashi(variant, trip, factor):
    n = np.arange(trip)
    close = 100 + 0.015 * n + 1.8 * np.sin(n * 0.065) + 0.35 * np.sin(n * 1.19)
    opening = np.r_[close[0], close[:-1]] + 0.18 * np.sin(n * 0.57 + 0.9)
    high = np.maximum(opening, close) + 0.25 + 0.10 * np.abs(np.sin(n * 1.34))
    low = np.minimum(opening, close) - 0.25 - 0.10 * np.abs(np.sin(n * 1.03))

    def run(o, h, l, c, initial):
        def step(i, state):
            return jnp.stack(((state[0] + state[1]) * 0.5, (o[i] + h[i] + l[i] + c[i]) * 0.25))

        return lax.fori_loop(0, trip, step, initial, unroll=factor)

    return run, tuple(np.asarray(a, np.float32) for a in (opening, high, low, close, [opening[0]] * 2))


def battery_ecm(variant, trip, factor):
    # One-second intervals, positive discharge, 3 Ah, R1=.012 ohm, C1=8000 F.
    n = np.arange(trip)
    current = 2 * np.sin(n * 2 * np.pi / 180) + 0.7 * np.sin(n * 2 * np.pi / 540 + 0.4)
    decay = np.float32(np.exp(-1 / (0.012 * 8000)))
    drive = np.float32(0.012 * (1 - decay))

    def run(samples, charge, voltage):
        def step(i, state):
            soc, polarization = state
            amps = samples[i]
            return soc - amps * np.float32(1 / (3 * 3600)), decay * polarization + drive * amps

        return lax.fori_loop(0, trip, step, (charge, voltage), unroll=factor)

    return run, (current.astype(np.float32), np.float32(0.8), np.float32(0))


def holt_trend(variant, trip, factor):
    n = np.arange(trip)
    readings = 10 + 0.1 * n + 0.5 * np.sin(n * 2 * np.pi / 12) + 0.15 * np.sin(n * 2 * np.pi / 30 + 0.7)

    def run(data, level, trend):
        def step(i, state):
            previous, slope = state
            new = 0.3 * data[i] + 0.7 * (previous + slope)
            return new, 0.05 * (new - previous) + 0.95 * slope

        return lax.fori_loop(0, trip, step, (level, trend), unroll=factor)

    return run, (readings.astype(np.float32), np.float32(10), np.float32(0.1))


def rainflow_turning_points(variant, trip, factor):
    trip = int(trip)
    factor = int(factor)
    if trip <= 0 or factor <= 0:
        raise ValueError("trip and factor must be positive")
    pattern = np.array(
        [0.0, 2.0, 2.0, -1.0, 3.5, 3.5, -2.0, 1.0, 1.0, 4.0, -3.0, 0.5, 0.5, -2.5, 2.5, -1.5, 1.5, 0.0],
        dtype=np.float32,
    )
    obs = np.resize(pattern, trip).astype(np.float32)
    init = np.array([obs[0], obs[0], 0.0, 0.0, 0.0], dtype=np.float32)
    if variant in ("turns_range", "range"):

        def pack(state):
            return jnp.stack((state[3], state[4]))
    else:

        def pack(state):
            return state[3]

    def fn(obs_, init_):

        def body(i, state):
            last = state[0]
            prev_turn = state[1]
            direction = state[2]
            turns = state[3]
            span = state[4]
            x = obs_[i]
            same = x == last
            step = jnp.sign(x - last)
            has_dir = direction != 0.0
            turn = ~same & has_dir & (step != direction)
            start = ~same & ~has_dir
            direction = jnp.where(start | turn, step, direction)
            turns = turns + turn.astype(obs_.dtype)
            span = span + jnp.where(turn, jnp.abs(last - prev_turn), 0.0)
            prev_turn = jnp.where(turn, last, prev_turn)
            last = jnp.where(same, last, x)
            return jnp.stack((last, prev_turn, direction, turns, span))

        state = lax.fori_loop(0, trip, body, init_, unroll=factor)
        return pack(state)

    return (fn, (obs, init))


def welford_online_moments(variant, trip, factor):
    trip = int(trip)
    factor = int(factor)
    if trip <= 1 or factor <= 0:
        raise ValueError("trip must exceed 1 and factor must be positive")
    t = np.arange(trip, dtype=np.float32)
    obs = (
        12.5
        + 0.8 * np.sin(2.0 * np.pi * t / np.float32(257.0))
        + 0.25 * np.sin(2.0 * np.pi * t / np.float32(37.0))
        + 0.08 * np.cos(2.0 * np.pi * t / np.float32(11.0))
    )
    obs = np.asarray(obs, dtype=np.float32)
    init = np.array([0.0, 0.0], dtype=np.float32)
    if variant in ("mean_variance", "sample_variance"):

        def pack(state):
            return jnp.stack((state[0], state[1] / jnp.asarray(trip - 1, dtype=state.dtype)))
    elif variant in ("mean_std", "sample_std"):

        def pack(state):
            return jnp.stack((state[0], jnp.sqrt(state[1] / jnp.asarray(trip - 1, dtype=state.dtype))))
    else:

        def pack(state):
            return state

    def fn(obs_, init_):

        def body(i, state):
            mean = state[0]
            m2 = state[1]
            x = obs_[i]
            n = jnp.asarray(i + 1, dtype=obs_.dtype)
            delta = x - mean
            mean = mean + delta / n
            m2 = m2 + delta * (x - mean)
            return jnp.stack((mean, m2))

        return pack(lax.fori_loop(0, trip, body, init_, unroll=factor))

    return (fn, (obs, init))
