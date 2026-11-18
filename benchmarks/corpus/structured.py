"""Structured-memory and numerical corpus bodies.

Every entry is an independent static-trip loop body for the expanded Phase 3
development corpus. Inputs are deterministic float32 arrays, carries stay
bounded through trip 1023, and each loop is a ``lax.scan`` or a static-bound
``lax.fori_loop`` with an explicit unroll factor.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats, scan
from benchmarks.corpus.spec import CorpusSpec, Provenance


def _scan1(step, factor):
    def fn(init, xs):
        return lax.scan(step, init, xs, unroll=factor)[0]

    return fn


_ORIGINAL = Provenance(
    kind="original",
    source="Original implementation for this corpus",
    license="No repository license file; original code",
    adaptation="No adaptation; written independently for the structured corpus.",
)
_JAX_REDUCE_WINDOW = Provenance(
    kind="public-inspired",
    source="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_window.html",
    license="Apache-2.0",
    adaptation="Windowed-reduction semantics reimplemented as a scan from scratch; no code copied.",
)
_JAX_CFD = Provenance(
    kind="public-inspired",
    source="https://github.com/google/jax-cfd",
    license="Apache-2.0",
    adaptation="Finite-difference stencil reimplemented from the project description; no code copied.",
)
_FLASH_ATTENTION = Provenance(
    kind="public-inspired",
    source="https://arxiv.org/abs/2205.14135",
    license="Paper algorithm; FlashAttention repository is BSD-3-Clause",
    adaptation="Online-softmax reduction reimplemented from the paper algorithm; no code copied.",
)
_WELFORD = Provenance(
    kind="public-inspired",
    source="https://github.com/jvf/welford",
    license="BSD-2-Clause",
    adaptation="Welford update reimplemented from the documented algorithm; no code copied.",
)
_SEGMENT_SUM = Provenance(
    kind="public-inspired",
    source="https://docs.jax.dev/en/latest/_autosummary/jax.ops.segment_sum.html",
    license="Apache-2.0",
    adaptation="Segmented-scan semantics reimplemented sequentially from the documentation; no code copied.",
)
_FLAX_LSTM = Provenance(
    kind="public-inspired",
    source="https://github.com/google/flax/blob/main/examples/seq2seq/models.py",
    license="Apache-2.0",
    adaptation="LSTM/GRU cell equations reimplemented without Flax modules; no code copied.",
)
_JAXOPT_BISECTION = Provenance(
    kind="public-inspired",
    source="https://jaxopt.github.io/stable/_autosummary/jaxopt.Bisection.html",
    license="Apache-2.0",
    adaptation="Branchless bracket-update rule reimplemented from the documentation; no code copied.",
)


def _segment_ids(
    rng: np.random.Generator,
    trip: int,
    low: int,
    high: int,
) -> jax.Array:
    ids = np.empty((trip,), dtype=np.int32)
    start = 0
    segment = 0
    while start < trip:
        stop = min(trip, start + int(rng.integers(low, high + 1)))
        ids[start:stop] = segment
        start = stop
        segment += 1
    return jnp.asarray(ids)


def _rolling_mean_shift(variant: str, trip: int, factor: int):
    width = {"w8": 8, "w16": 16}[variant]
    rng = np.random.default_rng(583_400)
    window = floats(rng, width)
    running = jnp.sum(window)
    xs = floats(rng, trip)

    def step(carry, x):
        window, running = carry
        running = running + x - window[0]
        window = jnp.concatenate((window[1:], x[None]))
        return (window, running), None

    def fn(window0, running0, xs_val):
        _, final_running = _scan1(step, factor)(
            (window0, running0), xs_val
        )
        return final_running / jnp.float32(width)

    return built(fn, window, running, xs)


def _rolling_conv1d_fir(variant: str, trip: int, factor: int):
    width = {"k8": 8, "k16": 16}[variant]
    rng = np.random.default_rng(583_401)
    kernel = jnp.asarray(
        rng.normal(scale=1.0 / np.sqrt(width), size=(width,)),
        dtype=jnp.float32,
    )
    history = floats(rng, width)
    last = jnp.asarray(0.0, dtype=jnp.float32)
    xs = floats(rng, trip)

    def fn(history0, last0, kernel_val, xs_val):
        def step_with_kernel(carry, x):
            history, _last = carry
            history = jnp.concatenate((history[1:], x[None]))
            return (history, jnp.sum(kernel_val * history)), None

        _, result = _scan1(step_with_kernel, factor)((history0, last0), xs_val)
        return result

    return built(fn, history, last, kernel, xs)


def _rolling_extreme_max(variant: str, trip: int, factor: int):
    width = {"w16": 16, "w32": 32}[variant]
    rng = np.random.default_rng(583_402)
    window = floats(rng, width)
    maximum = jnp.max(window)
    xs = floats(rng, trip)

    def step(carry, x):
        window, _maximum = carry
        window = jnp.concatenate((window[1:], x[None]))
        return (window, jnp.max(window)), None

    def fn(window0, maximum0, xs_val):
        _, final_maximum = _scan1(step, factor)(
            (window0, maximum0), xs_val
        )
        return final_maximum

    return built(fn, window, maximum, xs)


def _rolling_variance_moments(variant: str, trip: int, factor: int):
    width = {"w12": 12, "w24": 24}[variant]
    rng = np.random.default_rng(583_403)
    window = floats(rng, width)
    total = jnp.sum(window)
    total_sq = jnp.sum(window * window)
    xs = floats(rng, trip)

    def step(carry, x):
        window, total, total_sq = carry
        old = window[0]
        window = jnp.concatenate((window[1:], x[None]))
        total = total + x - old
        total_sq = total_sq + x * x - old * old
        return (window, total, total_sq), None

    def fn(window0, total0, total_sq0, xs_val):
        _, final_total, final_total_sq = _scan1(step, factor)(
            (window0, total0, total_sq0), xs_val
        )
        mean = final_total / jnp.float32(width)
        variance = final_total_sq / jnp.float32(width) - mean * mean
        return jnp.maximum(variance, jnp.asarray(0.0, dtype=jnp.float32))

    return built(fn, window, total, total_sq, xs)


def _rolling_dot_two_streams(variant: str, trip: int, factor: int):
    width = {"w8": 8, "w16": 16}[variant]
    rng = np.random.default_rng(583_404)
    left = floats(rng, width)
    right = floats(rng, width)
    last = jnp.asarray(0.0, dtype=jnp.float32)
    xs = floats(rng, trip)
    ys = floats(rng, trip)

    def step(carry, streams):
        left, right, _last = carry
        x, y = streams
        left = jnp.concatenate((left[1:], x[None]))
        right = jnp.concatenate((right[1:], y[None]))
        return (left, right, jnp.sum(left * right)), None

    def fn(left0, right0, last0, xs_val, ys_val):
        _, _, result = scan(step, factor)(
            (left0, right0, last0), xs_val, ys_val
        )
        return result

    return built(fn, left, right, last, xs, ys)


def _stencil_diffusion_1d_fori(variant: str, trip: int, factor: int):
    size = {"n128": 128, "n256": 256}[variant]
    rng = np.random.default_rng(583_405)
    init = floats(rng, size)

    def body(_index, carry):
        laplacian = jnp.roll(carry, 1) + jnp.roll(carry, -1) - 2.0 * carry
        return carry + jnp.float32(0.2) * laplacian

    def fn(init_val):
        return lax.fori_loop(0, trip, body, init_val, unroll=factor)

    return built(fn, init)


def _stencil_five_point_2d(variant: str, trip: int, factor: int):
    size = {"n16": 16, "n32": 32}[variant]
    rng = np.random.default_rng(583_406)
    init = floats(rng, size, size)

    def body(_index, carry):
        up = jnp.roll(carry, 1, axis=0)
        down = jnp.roll(carry, -1, axis=0)
        left = jnp.roll(carry, 1, axis=1)
        right = jnp.roll(carry, -1, axis=1)
        laplacian = up + down + left + right - 4.0 * carry
        return carry + jnp.float32(0.2) * laplacian

    def fn(init_val):
        return lax.fori_loop(0, trip, body, init_val, unroll=factor)

    return built(fn, init)


def _stencil_nine_point_2d(variant: str, trip: int, factor: int):
    size = {"n16": 16, "n24": 24}[variant]
    rng = np.random.default_rng(583_407)
    init = floats(rng, size, size)

    def body(_index, carry):
        axis_sum = (
            jnp.roll(carry, 1, axis=0)
            + jnp.roll(carry, -1, axis=0)
            + jnp.roll(carry, 1, axis=1)
            + jnp.roll(carry, -1, axis=1)
        )
        diagonal_sum = (
            jnp.roll(carry, (1, 1), axis=(0, 1))
            + jnp.roll(carry, (1, -1), axis=(0, 1))
            + jnp.roll(carry, (-1, 1), axis=(0, 1))
            + jnp.roll(carry, (-1, -1), axis=(0, 1))
        )
        neighborhood = (
            jnp.float32(0.25) * carry
            + jnp.float32(0.125) * axis_sum
            + jnp.float32(0.0625) * diagonal_sum
        )
        return jnp.float32(0.7) * carry + jnp.float32(0.3) * neighborhood

    def fn(init_val):
        return lax.fori_loop(0, trip, body, init_val, unroll=factor)

    return built(fn, init)


def _stencil_anisotropic_2d(variant: str, trip: int, factor: int):
    size = {"n16": 16, "n24": 24}[variant]
    rng = np.random.default_rng(583_408)
    init = floats(rng, size, size)

    def body(_index, carry):
        lap_x = (
            jnp.roll(carry, 1, axis=1)
            + jnp.roll(carry, -1, axis=1)
            - 2.0 * carry
        )
        lap_y = (
            jnp.roll(carry, 1, axis=0)
            + jnp.roll(carry, -1, axis=0)
            - 2.0 * carry
        )
        lap_x2 = (
            jnp.roll(carry, 2, axis=1)
            + jnp.roll(carry, -2, axis=1)
            - 2.0 * carry
        )
        return (
            carry
            + jnp.float32(0.15) * lap_x
            + jnp.float32(0.06) * lap_y
            + jnp.float32(0.04) * lap_x2
        )

    def fn(init_val):
        return lax.fori_loop(0, trip, body, init_val, unroll=factor)

    return built(fn, init)


def _streaming_running_mean(variant: str, trip: int, factor: int):
    width = {"scalar": (), "vec8": (8,)}[variant]
    rng = np.random.default_rng(583_409)
    mean = jnp.zeros(width, dtype=jnp.float32)
    count = jnp.asarray(0.0, dtype=jnp.float32)
    xs = floats(rng, trip, *width)

    def step(carry, x):
        mean, count = carry
        count = count + 1.0
        mean = mean + (x - mean) / count
        return (mean, count), None

    def fn(mean0, count0, xs_val):
        final_mean, _ = _scan1(step, factor)((mean0, count0), xs_val)
        return final_mean

    return built(fn, mean, count, xs)


def _streaming_welford_var(variant: str, trip: int, factor: int):
    width = {"scalar": (), "vec8": (8,)}[variant]
    rng = np.random.default_rng(583_410)
    mean = jnp.zeros(width, dtype=jnp.float32)
    m2 = jnp.zeros(width, dtype=jnp.float32)
    count = jnp.asarray(0.0, dtype=jnp.float32)
    xs = floats(rng, trip, *width)

    def step(carry, x):
        mean, m2, count = carry
        count = count + 1.0
        delta = x - mean
        mean = mean + delta / count
        m2 = m2 + delta * (x - mean)
        return (mean, m2, count), None

    def fn(mean0, m20, count0, xs_val):
        _, final_m2, final_count = _scan1(step, factor)(
            (mean0, m20, count0), xs_val
        )
        return final_m2 / final_count

    return built(fn, mean, m2, count, xs)


def _streaming_decayed_extrema(variant: str, trip: int, factor: int):
    width = {"scalar": (), "vec8": (8,)}[variant]
    rng = np.random.default_rng(583_411)
    high = jnp.zeros(width, dtype=jnp.float32)
    low = jnp.zeros(width, dtype=jnp.float32)
    xs = floats(rng, trip, *width)

    def step(carry, x):
        high, low = carry
        high = jnp.maximum(high * jnp.float32(0.995), x)
        low = jnp.minimum(low * jnp.float32(0.995), x)
        return (high, low), None

    def fn(high0, low0, xs_val):
        final_high, final_low = _scan1(step, factor)((high0, low0), xs_val)
        return final_high - final_low

    return built(fn, high, low, xs)


def _streaming_logsumexp(variant: str, trip: int, factor: int):
    width = {"scalar": (), "vec8": (8,)}[variant]
    rng = np.random.default_rng(583_412)
    maximum = jnp.zeros(width, dtype=jnp.float32)
    total = jnp.zeros(width, dtype=jnp.float32)
    xs = floats(rng, trip, *width)

    def step(carry, x):
        maximum, total = carry
        new_maximum = jnp.maximum(maximum, x)
        total = total * jnp.exp(maximum - new_maximum) + jnp.exp(x - new_maximum)
        return (new_maximum, total), None

    def fn(maximum0, total0, xs_val):
        final_maximum, final_total = _scan1(step, factor)(
            (maximum0, total0), xs_val
        )
        return final_maximum + jnp.log(final_total)

    return built(fn, maximum, total, xs)


_SEGMENT_VARIANTS = {
    "w4_short": (4, 2, 5),
    "w8_long": (8, 8, 20),
}


def _segment_sum_reset(variant: str, trip: int, factor: int):
    width, low, high = _SEGMENT_VARIANTS[variant]
    rng = np.random.default_rng(583_413)
    segments = _segment_ids(rng, trip, low, high)
    xs = floats(rng, trip, width)
    previous = jnp.asarray(-1, dtype=jnp.int32)
    accumulator = jnp.zeros((width,), dtype=jnp.float32)

    def step(carry, streams):
        segment, x = streams
        previous, accumulator = carry
        start = segment != previous
        accumulator = jnp.where(start, x, accumulator + x)
        return (segment, accumulator), None

    def fn(previous0, accumulator0, segments_val, xs_val):
        _, final_accumulator = scan(step, factor)(
            (previous0, accumulator0), segments_val, xs_val
        )
        return final_accumulator

    return built(fn, previous, accumulator, segments, xs)


def _segment_max_reset(variant: str, trip: int, factor: int):
    width, low, high = _SEGMENT_VARIANTS[variant]
    rng = np.random.default_rng(583_414)
    segments = _segment_ids(rng, trip, low, high)
    xs = floats(rng, trip, width)
    previous = jnp.asarray(-1, dtype=jnp.int32)
    accumulator = jnp.full((width,), -1.0, dtype=jnp.float32)

    def step(carry, streams):
        segment, x = streams
        previous, accumulator = carry
        start = segment != previous
        accumulator = jnp.where(start, x, jnp.maximum(accumulator, x))
        return (segment, accumulator), None

    def fn(previous0, accumulator0, segments_val, xs_val):
        _, final_accumulator = scan(step, factor)(
            (previous0, accumulator0), segments_val, xs_val
        )
        return final_accumulator

    return built(fn, previous, accumulator, segments, xs)


def _segment_weighted_mean(variant: str, trip: int, factor: int):
    width, low, high = _SEGMENT_VARIANTS[variant]
    rng = np.random.default_rng(583_415)
    segments = _segment_ids(rng, trip, low, high)
    xs = floats(rng, trip, width)
    weights = jnp.asarray(
        rng.uniform(0.1, 1.0, size=(trip, width)), dtype=jnp.float32
    )
    previous = jnp.asarray(-1, dtype=jnp.int32)
    numerator = jnp.zeros((width,), dtype=jnp.float32)
    denominator = jnp.zeros((width,), dtype=jnp.float32)

    def step(carry, streams):
        segment, x, weight = streams
        previous, numerator, denominator = carry
        start = segment != previous
        numerator = jnp.where(start, x * weight, numerator + x * weight)
        denominator = jnp.where(start, weight, denominator + weight)
        return (segment, numerator, denominator), None

    def fn(previous0, numerator0, denominator0, segments_val, xs_val, weights_val):
        _, final_numerator, final_denominator = scan(step, factor)(
            (previous0, numerator0, denominator0),
            segments_val,
            xs_val,
            weights_val,
        )
        return final_numerator / final_denominator

    return built(fn, previous, numerator, denominator, segments, xs, weights)


def _segment_carry_ema(variant: str, trip: int, factor: int):
    width, low, high = _SEGMENT_VARIANTS[variant]
    rng = np.random.default_rng(583_416)
    segments = _segment_ids(rng, trip, low, high)
    xs = floats(rng, trip, width)
    previous = jnp.asarray(-1, dtype=jnp.int32)
    ema = jnp.zeros((width,), dtype=jnp.float32)

    def step(carry, streams):
        segment, x = streams
        previous, ema = carry
        start = segment != previous
        ema = jnp.where(
            start,
            jnp.float32(0.5) * ema + jnp.float32(0.5) * x,
            jnp.float32(0.9) * ema + jnp.float32(0.1) * x,
        )
        return (segment, ema), None

    def fn(previous0, ema0, segments_val, xs_val):
        _, final_ema = scan(step, factor)(
            (previous0, ema0), segments_val, xs_val
        )
        return final_ema

    return built(fn, previous, ema, segments, xs)


def _stateful_scalar_iir(variant: str, trip: int, factor: int):
    width = {"scalar": (), "vec4": (4,)}[variant]
    rng = np.random.default_rng(583_417)
    init = floats(rng, *width, scale=0.2)
    xs = floats(rng, trip, *width)

    def step(carry, x):
        return jnp.float32(0.9) * carry + jnp.float32(0.1) * x, None

    def fn(init_val, xs_val):
        return _scan1(step, factor)(init_val, xs_val)

    return built(fn, init, xs)


def _stateful_vector_iir(variant: str, trip: int, factor: int):
    width = {"n8": 8, "n16": 16}[variant]
    rng = np.random.default_rng(583_418)
    init = floats(rng, width, scale=0.2)
    xs = floats(rng, trip, width)

    def step(carry, x):
        carry = (
            jnp.float32(0.82) * carry
            + jnp.float32(0.12) * jnp.roll(carry, 1)
            + jnp.float32(0.06) * x
        )
        return carry, None

    def fn(init_val, xs_val):
        return _scan1(step, factor)(init_val, xs_val)

    return built(fn, init, xs)


def _stateful_gru_gate(variant: str, trip: int, factor: int):
    width = {"n4": 4, "n8": 8}[variant]
    rng = np.random.default_rng(583_419)
    scale = 1.0 / np.sqrt(width)
    wz = jnp.asarray(rng.normal(scale=scale, size=(width, width)), dtype=jnp.float32)
    uz = jnp.asarray(rng.normal(scale=scale, size=(width, width)), dtype=jnp.float32)
    wr = jnp.asarray(rng.normal(scale=scale, size=(width, width)), dtype=jnp.float32)
    ur = jnp.asarray(rng.normal(scale=scale, size=(width, width)), dtype=jnp.float32)
    wn = jnp.asarray(rng.normal(scale=scale, size=(width, width)), dtype=jnp.float32)
    un = jnp.asarray(rng.normal(scale=scale, size=(width, width)), dtype=jnp.float32)
    init = floats(rng, width, scale=0.1)
    xs = floats(rng, trip, width)

    def step(carry, x):
        reset = jax.nn.sigmoid(x @ wr + carry @ ur)
        update = jax.nn.sigmoid(x @ wz + carry @ uz)
        candidate = jnp.tanh(x @ wn + (reset * carry) @ un)
        return (1.0 - update) * carry + update * candidate, None

    def fn(init_val, xs_val):
        return _scan1(step, factor)(init_val, xs_val)

    return built(fn, init, xs)


def _stateful_lstm_cell(variant: str, trip: int, factor: int):
    width = {"n4": 4, "n8": 8}[variant]
    rng = np.random.default_rng(583_420)
    scale = 1.0 / np.sqrt(width)
    wi = jnp.asarray(rng.normal(scale=scale, size=(width, width)), dtype=jnp.float32)
    ui = jnp.asarray(rng.normal(scale=scale, size=(width, width)), dtype=jnp.float32)
    wf = jnp.asarray(rng.normal(scale=scale, size=(width, width)), dtype=jnp.float32)
    uf = jnp.asarray(rng.normal(scale=scale, size=(width, width)), dtype=jnp.float32)
    wo = jnp.asarray(rng.normal(scale=scale, size=(width, width)), dtype=jnp.float32)
    uo = jnp.asarray(rng.normal(scale=scale, size=(width, width)), dtype=jnp.float32)
    wg = jnp.asarray(rng.normal(scale=scale, size=(width, width)), dtype=jnp.float32)
    ug = jnp.asarray(rng.normal(scale=scale, size=(width, width)), dtype=jnp.float32)
    cell = jnp.zeros((width,), dtype=jnp.float32)
    hidden = jnp.zeros((width,), dtype=jnp.float32)
    xs = floats(rng, trip, width)

    def step(carry, x):
        cell, hidden = carry
        input_gate = jax.nn.sigmoid(x @ wi + hidden @ ui + 0.5)
        forget_gate = jax.nn.sigmoid(x @ wf + hidden @ uf + 0.5)
        output_gate = jax.nn.sigmoid(x @ wo + hidden @ uo)
        candidate = jnp.tanh(x @ wg + hidden @ ug)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * jnp.tanh(cell)
        return (cell, hidden), None

    def fn(cell0, hidden0, xs_val):
        _, final_hidden = _scan1(step, factor)((cell0, hidden0), xs_val)
        return final_hidden

    return built(fn, cell, hidden, xs)


def _stateful_coupled_oscillator(variant: str, trip: int, factor: int):
    width = {"n4": 4, "n8": 8}[variant]
    rng = np.random.default_rng(583_421)
    position = floats(rng, width, scale=0.2)
    velocity = floats(rng, width, scale=0.2)
    drive = floats(rng, trip, width, scale=0.02)

    def fn(position0, velocity0, drive_val):
        def body(index, carry):
            position, velocity = carry
            laplacian = (
                jnp.roll(position, 1)
                + jnp.roll(position, -1)
                - 2.0 * position
            )
            new_velocity = jnp.float32(0.995) * (
                velocity
                - jnp.float32(0.05) * (position - jnp.float32(0.15) * laplacian)
                + drive_val[index]
            )
            new_position = position + jnp.float32(0.05) * new_velocity
            return (new_position, new_velocity)

        final_position, final_velocity = lax.fori_loop(
            0, trip, body, (position0, velocity0), unroll=factor
        )
        return jnp.stack((final_position, final_velocity))

    return built(fn, position, velocity, drive)


def _control_polynomial_forcing(variant: str, trip: int, factor: int):
    width = {"n512": 512, "n2048": 2048}[variant]
    rng = np.random.default_rng(583_422)
    init = floats(rng, width, scale=0.05)
    xs = floats(rng, trip, width, scale=0.1)

    def step(carry, x):
        carry = (
            jnp.float32(0.75) * carry
            + jnp.float32(0.2) * (x * x + x)
            + jnp.float32(0.05) * carry * x
        )
        return carry, None

    def fn(init_val, xs_val):
        return _scan1(step, factor)(init_val, xs_val)

    return built(fn, init, xs)


def _control_transcendental(variant: str, trip: int, factor: int):
    width = {"n512": 512, "n2048": 2048}[variant]
    rng = np.random.default_rng(583_423)
    init = floats(rng, width, scale=0.1)
    xs = floats(rng, trip, width, scale=0.5)

    def step(carry, x):
        carry = (
            jnp.float32(0.8) * carry
            + jnp.float32(0.25) * jnp.sin(x)
            + jnp.float32(0.1) * jnp.cos(carry)
            + jnp.float32(0.05) * jnp.tanh(carry * x)
        )
        return carry, None

    def fn(init_val, xs_val):
        return _scan1(step, factor)(init_val, xs_val)

    return built(fn, init, xs)


def _control_bisection_bracket(variant: str, trip: int, factor: int):
    shape = {"scalar": (), "vec4": (4,)}[variant]
    rng = np.random.default_rng(583_424)
    target = jnp.float32(0.5) * (floats(rng, *shape) + jnp.float32(1.0))
    low = jnp.zeros(shape, dtype=jnp.float32)
    high = jnp.full(shape, jnp.float32(1.5), dtype=jnp.float32)

    def fn(target_val, low0, high0):
        def step(carry, _):
            low_t, high_t = carry
            mid = jnp.float32(0.5) * (low_t + high_t)
            low_value = low_t * low_t * low_t - target_val
            mid_value = mid * mid * mid - target_val
            same_sign = low_value * mid_value > 0.0
            low_next = jnp.where(same_sign, mid, low_t)
            high_next = jnp.where(same_sign, high_t, mid)
            return (low_next, high_next), None

        low_final, high_final = lax.scan(
            step, (low0, high0), None, length=trip, unroll=factor
        )[0]
        return jnp.float32(0.5) * (low_final + high_final)

    return built(fn, target, low, high)


def _control_lowrank_update(variant: str, trip: int, factor: int):
    width = {"n8": 8, "n16": 16}[variant]
    rng = np.random.default_rng(583_425)
    matrix = floats(rng, width, width, scale=0.1)
    left = floats(rng, trip, width, scale=0.2)
    right = floats(rng, trip, width, scale=0.2)

    def step(carry, streams):
        left_row, right_row = streams
        update = left_row[:, None] * right_row[None, :]
        return jnp.float32(0.95) * carry + jnp.float32(0.05) * update, None

    def fn(matrix0, left_val, right_val):
        return scan(step, factor)(matrix0, left_val, right_val)

    return built(fn, matrix, left, right)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec(
        name="rolling_mean_shift",
        family="rolling_window",
        description="Sliding mean over a shift register with an incremental sum.",
        mechanism="shift-register window plus subtract-old/add-new running sum",
        structural_difference=(
            "Carries the whole window and a scalar sum, so the window shift and "
            "the moment update are separate dependencies; no per-step window reduction."
        ),
        loop_kind="scan",
        shape_variants={
            "w8": "8-sample window with a scalar input stream",
            "w16": "16-sample window with a scalar input stream",
        },
        primary_variant="w8",
        factory=_rolling_mean_shift,
        seed=583_400,
        provenance=_ORIGINAL,
        tier="smoke",
        related_group="rolling_moments",
    ),
    CorpusSpec(
        name="rolling_conv1d_fir",
        family="rolling_window",
        description="Fixed-kernel 1D FIR filter over a shift-register history.",
        mechanism="weighted tap sum over a shifted sample history",
        structural_difference=(
            "Uses a fixed nonuniform kernel and only a history carry; unlike the "
            "sliding mean it has no running sum that can be updated incrementally."
        ),
        loop_kind="scan",
        shape_variants={
            "k8": "8-tap FIR history and kernel",
            "k16": "16-tap FIR history and kernel",
        },
        primary_variant="k8",
        factory=_rolling_conv1d_fir,
        seed=583_401,
        provenance=_JAX_REDUCE_WINDOW,
        tier="smoke",
        related_group="rolling_weighted_windows",
    ),
    CorpusSpec(
        name="rolling_extreme_max",
        family="rolling_window",
        description="Rolling maximum over a shift-register window.",
        mechanism=(
            "Each trip shifts the window, performs a full max reduction inside "
            "the loop body, and carries the resulting window maximum."
        ),
        structural_difference=(
            "The window reduction is a max computed every trip, so old samples "
            "cannot be removed with an incremental subtraction; the loop carries "
            "both the window and its current extremum."
        ),
        loop_kind="scan",
        shape_variants={
            "w16": "16-sample window with a scalar input stream",
            "w32": "32-sample window with a scalar input stream",
        },
        primary_variant="w16",
        factory=_rolling_extreme_max,
        seed=583_402,
        provenance=_ORIGINAL,
        related_group="rolling_extrema",
    ),
    CorpusSpec(
        name="rolling_variance_moments",
        family="rolling_window",
        description="Sliding variance from incrementally updated first and second moments.",
        mechanism="window carry plus running sum and sum-of-squares",
        structural_difference=(
            "Tracks two moment accumulators alongside the window and cancels the "
            "departed sample in both; the final variance adds a subtractive coupling."
        ),
        loop_kind="scan",
        shape_variants={
            "w12": "12-sample window with scalar sum and sum-of-squares state",
            "w24": "24-sample window with scalar sum and sum-of-squares state",
        },
        primary_variant="w12",
        factory=_rolling_variance_moments,
        seed=583_403,
        provenance=_ORIGINAL,
        related_group="rolling_moments",
    ),
    CorpusSpec(
        name="rolling_dot_two_streams",
        family="rolling_window",
        description="Windowed dot product of two synchronized input streams.",
        mechanism="two shift registers with a per-step inner product",
        structural_difference=(
            "Two independent windows advance together and are multiplied pairwise; "
            "there is no fixed kernel and no scalar accumulator across steps."
        ),
        loop_kind="scan",
        shape_variants={
            "w8": "two synchronized 8-sample windows",
            "w16": "two synchronized 16-sample windows",
        },
        primary_variant="w16",
        factory=_rolling_dot_two_streams,
        seed=583_404,
        provenance=_ORIGINAL,
        related_group="rolling_weighted_windows",
    ),
    CorpusSpec(
        name="stencil_diffusion_1d_fori",
        family="stencil",
        description="Periodic 1D three-point diffusion via static-bound fori_loop.",
        mechanism="nearest-neighbor 1D Laplacian update",
        structural_difference=(
            "One-axis dependency with radius one and a loop-counter-driven fori_loop; "
            "the other stencils are 2D grids with larger neighborhoods."
        ),
        loop_kind="fori_loop",
        shape_variants={
            "n128": "128-point periodic 1D grid",
            "n256": "256-point periodic 1D grid",
        },
        primary_variant="n128",
        factory=_stencil_diffusion_1d_fori,
        seed=583_405,
        provenance=_JAX_CFD,
        related_group="stencil_axis_aligned",
    ),
    CorpusSpec(
        name="stencil_five_point_2d",
        family="stencil",
        description="Periodic 2D five-point diffusion stencil.",
        mechanism="axis-neighbor 2D Laplacian with radius-one support",
        structural_difference=(
            "Touches four axis neighbors in two dimensions; unlike the 1D body it "
            "has a two-axis dependency pattern and a larger per-step tensor."
        ),
        loop_kind="fori_loop",
        shape_variants={
            "n16": "16x16 periodic grid",
            "n32": "32x32 periodic grid",
        },
        primary_variant="n16",
        factory=_stencil_five_point_2d,
        seed=583_406,
        provenance=_JAX_CFD,
        tier="smoke",
        related_group="stencil_axis_aligned",
    ),
    CorpusSpec(
        name="stencil_nine_point_2d",
        family="stencil",
        description="Periodic 2D nine-point stencil with diagonal coupling.",
        mechanism="axis plus diagonal neighbor average with a center weight",
        structural_difference=(
            "Adds four diagonal neighbors and a weighted center, giving a denser "
            "dependency fan-in than the five-point body at the same grid size."
        ),
        loop_kind="fori_loop",
        shape_variants={
            "n16": "16x16 periodic grid",
            "n24": "24x24 periodic grid",
        },
        primary_variant="n16",
        factory=_stencil_nine_point_2d,
        seed=583_407,
        provenance=_ORIGINAL,
        related_group="stencil_wide_support",
    ),
    CorpusSpec(
        name="stencil_anisotropic_2d",
        family="stencil",
        description="Anisotropic 2D stencil with a distance-two horizontal term.",
        mechanism="unequal axis weights plus a radius-two x-direction Laplacian",
        structural_difference=(
            "Combines asymmetric axis coefficients with a second-neighbor tap, so "
            "its dependency radius is two on one axis and one on the other."
        ),
        loop_kind="fori_loop",
        shape_variants={
            "n16": "16x16 periodic grid",
            "n24": "24x24 periodic grid",
        },
        primary_variant="n16",
        factory=_stencil_anisotropic_2d,
        seed=583_408,
        provenance=_ORIGINAL,
        related_group="stencil_wide_support",
    ),
    CorpusSpec(
        name="streaming_running_mean",
        family="streaming_reduction",
        description="Whole-stream online mean with a growing sample count.",
        mechanism="incremental mean update with a scalar count carry",
        structural_difference=(
            "No fixed window; the divisor grows with the trip count, so the update "
            "uses division by a carried count instead of a constant window length."
        ),
        loop_kind="scan",
        shape_variants={
            "scalar": "scalar stream and scalar mean state",
            "vec8": "8-channel stream with an 8-element mean vector",
        },
        primary_variant="scalar",
        factory=_streaming_running_mean,
        seed=583_409,
        provenance=_ORIGINAL,
        tier="smoke",
        related_group="streaming_moments",
    ),
    CorpusSpec(
        name="streaming_welford_var",
        family="streaming_reduction",
        description="Online Welford variance with mean, second moment, and count.",
        mechanism="stable single-pass variance update",
        structural_difference=(
            "Carries three coupled scalars or vectors and updates the second moment "
            "from the change in mean; deeper state and a different reduction identity."
        ),
        loop_kind="scan",
        shape_variants={
            "scalar": "scalar stream and scalar moment state",
            "vec8": "8-channel stream with vector moment state",
        },
        primary_variant="scalar",
        factory=_streaming_welford_var,
        seed=583_410,
        provenance=_WELFORD,
        related_group="streaming_moments",
    ),
    CorpusSpec(
        name="streaming_decayed_extrema",
        family="streaming_reduction",
        description="Decaying running maximum and minimum without explicit resets.",
        mechanism="forgetful max/min compare-and-select state",
        structural_difference=(
            "Uses multiplicative decay plus a compare-select instead of additive "
            "updates; the two extrema are independent scalar or small-vector carries."
        ),
        loop_kind="scan",
        shape_variants={
            "scalar": "scalar stream with scalar maximum and minimum",
            "vec8": "8-channel stream with vector extrema",
        },
        primary_variant="scalar",
        factory=_streaming_decayed_extrema,
        seed=583_411,
        provenance=_ORIGINAL,
        related_group="streaming_extrema",
    ),
    CorpusSpec(
        name="streaming_logsumexp",
        family="streaming_reduction",
        description="Online log-sum-exp normalizer with a running maximum.",
        mechanism="max-shifted exponential accumulation",
        structural_difference=(
            "Carries a maximum and a shifted exponential sum that are updated with "
            "two transcendentals; unlike mean and variance it is a normalizer, not a moment."
        ),
        loop_kind="scan",
        shape_variants={
            "scalar": "scalar stream with scalar normalizer state",
            "vec8": "8-channel stream with vector normalizer state",
        },
        primary_variant="scalar",
        factory=_streaming_logsumexp,
        seed=583_412,
        provenance=_FLASH_ATTENTION,
        related_group="streaming_normalizers",
    ),
    CorpusSpec(
        name="segment_sum_reset",
        family="segmented_scan",
        description="Segmented sum that resets at each segment boundary.",
        mechanism="segment-id-gated additive reset",
        structural_difference=(
            "A carried segment id selects between starting a new accumulator and "
            "continuing it; the dependency chain restarts at every boundary."
        ),
        loop_kind="scan",
        shape_variants={
            "w4_short": "4-channel values with 2-5 step segments",
            "w8_long": "8-channel values with 8-20 step segments",
        },
        primary_variant="w4_short",
        factory=_segment_sum_reset,
        seed=583_413,
        provenance=_SEGMENT_SUM,
        tier="smoke",
        related_group="segmented_reset",
    ),
    CorpusSpec(
        name="segment_max_reset",
        family="segmented_scan",
        description="Segmented maximum that resets at each segment boundary.",
        mechanism="segment-id-gated max reset",
        structural_difference=(
            "Shares the segment-boundary select with the segmented sum but reduces "
            "with a compare-select, so the accumulator is non-invertible."
        ),
        loop_kind="scan",
        shape_variants={
            "w4_short": "4-channel values with 2-5 step segments",
            "w8_long": "8-channel values with 8-20 step segments",
        },
        primary_variant="w4_short",
        factory=_segment_max_reset,
        seed=583_414,
        provenance=_ORIGINAL,
        related_group="segmented_reset",
    ),
    CorpusSpec(
        name="segment_weighted_mean",
        family="segmented_scan",
        description="Segmented weighted mean with numerator and denominator resets.",
        mechanism="two segment-local accumulators followed by a ratio",
        structural_difference=(
            "Tracks a weight total alongside the weighted sum and normalizes once; "
            "the state has twice the width and a division at the output."
        ),
        loop_kind="scan",
        shape_variants={
            "w4_short": "4-channel values and weights with 2-5 step segments",
            "w8_long": "8-channel values and weights with 8-20 step segments",
        },
        primary_variant="w4_short",
        factory=_segment_weighted_mean,
        seed=583_415,
        provenance=_ORIGINAL,
        related_group="segmented_reset",
    ),
    CorpusSpec(
        name="segment_carry_ema",
        family="segmented_scan",
        description="Segment-gated exponential average that carries state across boundaries.",
        mechanism="boundary-dependent blend without discarding prior state",
        structural_difference=(
            "Unlike the reset bodies it never zeroes the accumulator: a new segment "
            "uses a different blend coefficient, so state propagates across segments."
        ),
        loop_kind="scan",
        shape_variants={
            "w4_short": "4-channel values with 2-5 step segments",
            "w8_long": "8-channel values with 8-20 step segments",
        },
        primary_variant="w4_short",
        factory=_segment_carry_ema,
        seed=583_416,
        provenance=_ORIGINAL,
        related_group="segmented_carry",
    ),
    CorpusSpec(
        name="stateful_scalar_iir",
        family="stateful_recurrence",
        description="First-order scalar low-pass IIR recurrence normalized to unit DC gain.",
        mechanism="scalar one-pole linear recurrence",
        structural_difference=(
            "A single scalar carry with a short unit-step chain; the companion "
            "vector body widens the carry and adds neighbor coupling."
        ),
        loop_kind="scan",
        shape_variants={
            "scalar": "scalar carry and scalar input stream",
            "vec4": "four independent scalar channels",
        },
        primary_variant="scalar",
        factory=_stateful_scalar_iir,
        seed=583_417,
        provenance=_ORIGINAL,
        tier="smoke",
        related_group="iir_filters",
    ),
    CorpusSpec(
        name="stateful_vector_iir",
        family="stateful_recurrence",
        description="Wide vector IIR with nearest-neighbor state coupling.",
        mechanism="shift-coupled vector one-pole recurrence",
        structural_difference=(
            "The carry is a vector and each component also reads a rolled neighbor, "
            "producing a wide carry with cross-channel dependencies rather than independent channels."
        ),
        loop_kind="scan",
        shape_variants={
            "n8": "8-element coupled vector carry",
            "n16": "16-element coupled vector carry",
        },
        primary_variant="n8",
        factory=_stateful_vector_iir,
        seed=583_418,
        provenance=_ORIGINAL,
        related_group="iir_filters",
    ),
    CorpusSpec(
        name="stateful_gru_gate",
        family="stateful_recurrence",
        description="GRU-like gated recurrence with reset and update gates.",
        mechanism="sigmoid gates over dense input and hidden projections",
        structural_difference=(
            "Adds two multiplicative gates and a tanh candidate around the carry; "
            "the nonlinear gating makes the chain longer than either IIR body."
        ),
        loop_kind="scan",
        shape_variants={
            "n4": "4-element hidden state and input",
            "n8": "8-element hidden state and input",
        },
        primary_variant="n4",
        factory=_stateful_gru_gate,
        seed=583_419,
        provenance=_FLAX_LSTM,
        related_group="gated_recurrences",
    ),
    CorpusSpec(
        name="stateful_lstm_cell",
        family="stateful_recurrence",
        description="LSTM-like cell with input, forget, output, and candidate gates.",
        mechanism="two-state gated cell recurrence",
        structural_difference=(
            "Carries both cell and hidden state with three sigmoid gates; the cell "
            "path is an additional long dependency that the GRU body does not have."
        ),
        loop_kind="scan",
        shape_variants={
            "n4": "4-element cell and hidden state",
            "n8": "8-element cell and hidden state",
        },
        primary_variant="n4",
        factory=_stateful_lstm_cell,
        seed=583_420,
        provenance=_FLAX_LSTM,
        related_group="gated_recurrences",
    ),
    CorpusSpec(
        name="stateful_coupled_oscillator",
        family="stateful_recurrence",
        description="Damped coupled-oscillator ring driven through a fori_loop.",
        mechanism="position-velocity state with neighbor coupling and weak damping",
        structural_difference=(
            "Two wide carries form a second-order system with spatial coupling and "
            "a loop-counter-driven fori_loop; it is the longest dependency chain here."
        ),
        loop_kind="fori_loop",
        shape_variants={
            "n4": "4-oscillator ring",
            "n8": "8-oscillator ring",
        },
        primary_variant="n8",
        factory=_stateful_coupled_oscillator,
        seed=583_421,
        provenance=_ORIGINAL,
        related_group="coupled_dynamics",
    ),
    CorpusSpec(
        name="control_polynomial_forcing",
        family="control_arithmetic",
        description="Elementwise polynomial recurrence with quadratic and bilinear forcing.",
        mechanism="polynomial elementwise map with a decaying carry",
        structural_difference=(
            "Pure elementwise arithmetic with no neighbor, segment, or gate logic; "
            "it isolates polynomial arithmetic from memory-access effects."
        ),
        loop_kind="scan",
        shape_variants={
            "n512": "512-element vector carry and stream",
            "n2048": "2048-element vector carry and stream",
        },
        primary_variant="n512",
        factory=_control_polynomial_forcing,
        seed=583_422,
        provenance=_ORIGINAL,
        tier="smoke",
        related_group="elementwise_control",
    ),
    CorpusSpec(
        name="control_transcendental",
        family="control_arithmetic",
        description="Elementwise recurrence mixing sine, cosine, and tanh.",
        mechanism="bounded transcendental map with a linear carry decay",
        structural_difference=(
            "Adds three transcendental operations per element with no memory "
            "structure, separating transcendental cost from structured dependencies."
        ),
        loop_kind="scan",
        shape_variants={
            "n512": "512-element vector carry and stream",
            "n2048": "2048-element vector carry and stream",
        },
        primary_variant="n512",
        factory=_control_transcendental,
        seed=583_423,
        provenance=_ORIGINAL,
        related_group="elementwise_control",
    ),
    CorpusSpec(
        name="control_bisection_bracket",
        family="control_arithmetic",
        description="Branchless bisection bracket update for a cubic root.",
        mechanism=(
            "Each trip evaluates the cubic residual at the bracket midpoint and "
            "uses a sign comparison plus select to move the low or high endpoint."
        ),
        structural_difference=(
            "Data-dependent control through select rather than dense arithmetic: "
            "the scalar or 4-lane carry holds a bracket, the body performs no "
            "indexed memory access, and the dependency chain is one comparison "
            "and two selects per element."
        ),
        loop_kind="scan",
        shape_variants={
            "scalar": "scalar target and scalar bracket",
            "vec4": "4-element target and bracket",
        },
        primary_variant="vec4",
        factory=_control_bisection_bracket,
        seed=583_424,
        provenance=_JAXOPT_BISECTION,
        related_group="root_finding_control",
    ),
    CorpusSpec(
        name="control_lowrank_update",
        family="control_arithmetic",
        description="Decaying matrix carry updated by rank-one outer products.",
        mechanism="rank-one matrix update with multiplicative contraction",
        structural_difference=(
            "The carry is a small matrix and each step updates it with an outer "
            "product, so there is no matrix-vector contraction and no indexed memory."
        ),
        loop_kind="scan",
        shape_variants={
            "n8": "8x8 matrix carry with 8-element rank-one vectors",
            "n16": "16x16 matrix carry with 16-element rank-one vectors",
        },
        primary_variant="n8",
        factory=_control_lowrank_update,
        seed=583_425,
        provenance=_ORIGINAL,
        related_group="lowrank_control",
    ),
)
