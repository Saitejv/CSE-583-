"""Reviewed Gemini AGY workflows, including two full-history controls."""

import jax
import jax.numpy as jnp
import numpy as np


def alpha_beta_tracker(variant: str, trip: int, factor: int):
    dt_val = np.float32(0.05)
    if variant == "benedict_bordner":
        alpha_val = np.float32(0.35)
        beta_val = np.float32(alpha_val**2 / (2.0 - alpha_val))
    else:
        alpha_val = np.float32(0.4)
        beta_val = np.float32(0.1)
    t = np.arange(trip, dtype=np.float32) * dt_val
    true_position = np.float32(10.0) + np.float32(1.5) * t + np.float32(5.0) * np.sin(np.float32(0.2) * t)
    measurement_noise = np.float32(0.4) * np.cos(np.float32(1.1) * t)
    measurements = (true_position + measurement_noise).astype(np.float32)
    dt = dt_val
    alpha = alpha_val
    beta = beta_val
    init_history = np.zeros((trip, 2), dtype=np.float32)
    init_state = np.zeros((2,), dtype=np.float32)

    def fn(measurements, dt, alpha, beta, init_history, init_state):
        init = (init_history, init_state)

        def body(k, val):
            history, state = val
            pos = state[0]
            vel = state[1]
            pred_pos = pos + dt * vel
            pred_vel = vel
            residual = measurements[k] - pred_pos
            updated_pos = pred_pos + alpha * residual
            updated_vel = pred_vel + beta / dt * residual
            new_state = jnp.array([updated_pos, updated_vel], dtype=jnp.float32)
            new_history = history.at[k].set(new_state)
            return (new_history, new_state)

        final_history, final_state = jax.lax.fori_loop(0, trip, body, init, unroll=factor)
        return (final_history, final_state)

    tuple_of_numpy_args = (measurements, dt, alpha, beta, init_history, init_state)
    return (fn, tuple_of_numpy_args)


def acoustic_resonator_bank(variant: str, trip: int, factor: int):
    vector_width = 8
    fs = np.float32(16000.0)
    if variant == "formant":
        freqs = np.array([270.0, 530.0, 700.0, 1000.0, 1500.0, 2200.0, 2800.0, 3400.0], dtype=np.float32)
        bandwidths = np.array([60.0, 70.0, 80.0, 90.0, 110.0, 130.0, 150.0, 180.0], dtype=np.float32)
    else:
        freqs = np.linspace(200.0, 3200.0, vector_width, dtype=np.float32)
        bandwidths = np.full((vector_width,), 80.0, dtype=np.float32)
    r = np.exp(-np.pi * bandwidths / fs).astype(np.float32)
    theta = (2.0 * np.pi * freqs / fs).astype(np.float32)
    a1 = (2.0 * r * np.cos(theta)).astype(np.float32)
    a2 = (-r * r).astype(np.float32)
    b = (1.0 - r).astype(np.float32)
    n = np.arange(trip, dtype=np.float32)
    excitation = (
        0.6 * np.sin(2.0 * np.pi * 300.0 * n / fs) + 0.4 * np.cos(2.0 * np.pi * 1250.0 * n / fs)
    ).astype(np.float32)
    init_output = np.zeros((trip, vector_width), dtype=np.float32)
    init_y1 = np.zeros((vector_width,), dtype=np.float32)
    init_y2 = np.zeros((vector_width,), dtype=np.float32)

    def fn(excitation, a1, a2, b, init_output, init_y1, init_y2):
        init = (init_output, init_y1, init_y2)

        def body(k, val):
            out_buf, y1, y2 = val
            u = excitation[k]
            y0 = a1 * y1 + a2 * y2 + b * u
            new_out_buf = out_buf.at[k].set(y0)
            return (new_out_buf, y0, y1)

        final_out_buf, final_y1, final_y2 = jax.lax.fori_loop(0, trip, body, init, unroll=factor)
        return (final_out_buf, final_y1, final_y2)

    tuple_of_numpy_args = (excitation, a1, a2, b, init_output, init_y1, init_y2)
    return (fn, tuple_of_numpy_args)


def hysteretic_event_detector(variant="high_alarm", trip=2048, factor=1):
    trip_int = int(trip)
    factor_int = int(factor)

    def fn(signal, t_high, t_low, persist_limit, init_state):

        def body(i, state):
            active = state[0]
            count = state[1]
            total_active = state[2]
            x_i = signal[i]
            new_count_inactive = jnp.where(x_i >= t_high, count + 1.0, 0.0)
            trip_alarm = new_count_inactive >= persist_limit
            active_from_inactive = jnp.where(trip_alarm, 1.0, 0.0)
            count_after_inactive = jnp.where(trip_alarm, 0.0, new_count_inactive)
            clear_alarm = x_i <= t_low
            active_from_active = jnp.where(clear_alarm, 0.0, 1.0)
            count_after_active = jnp.where(clear_alarm, 0.0, count)
            next_active = jnp.where(active > 0.5, active_from_active, active_from_inactive)
            next_count = jnp.where(active > 0.5, count_after_active, count_after_inactive)
            next_total = total_active + next_active
            return jnp.stack([next_active, next_count, next_total])

        return jax.lax.fori_loop(0, trip_int, body, init_state, unroll=factor_int)

    t = np.linspace(0.0, 8.0 * np.pi, trip_int, endpoint=False, dtype=np.float32)
    signal = np.sin(t) + 0.4 * np.sin(3.0 * t) + 0.3 * np.cos(7.0 * t)
    t_high = np.float32(0.85)
    t_low = np.float32(0.15)
    persist_limit = np.float32(12.0)
    init_state = np.zeros(3, dtype=np.float32)
    return (fn, (signal, t_high, t_low, persist_limit, init_state))


def cumulative_trapezoid_integrator(variant="uniform_sampling", trip=2048, factor=1):
    trip_int = int(trip)
    factor_int = int(factor)

    def fn(signal, dt, init_state):

        def body(i, state):
            integral = state[0]
            x_prev = state[1]
            x_curr = signal[i]
            step_area = jnp.where(i == 0, 0.0, 0.5 * dt * (x_curr + x_prev))
            next_integral = integral + step_area
            return jnp.stack([next_integral, x_curr])

        return jax.lax.fori_loop(0, trip_int, body, init_state, unroll=factor_int)

    t = np.linspace(0.0, 10.0, trip_int, endpoint=False, dtype=np.float32)
    signal = (np.sin(2.0 * np.pi * 0.5 * t) + 0.5 * np.cos(2.0 * np.pi * 1.5 * t) + 1.25).astype(np.float32)
    dt = np.float32(10.0 / trip_int)
    init_state = np.zeros(2, dtype=np.float32)
    return (fn, (signal, dt, init_state))


def muskingum(variant, trip, factor):
    """Route measured daily inflow through one illustrative constant-coefficient reach."""
    from pathlib import Path

    flow = np.loadtxt(
        Path(__file__).with_name("huron_daily.csv"), delimiter=",", skiprows=1, usecols=1, dtype=np.float32
    )[-trip - 1 :]

    # K=2 days, X=.20, dt=1 day. Start from equal inflow and outflow.
    def run(inflow, initial):
        def step(i, state):
            previous_in, previous_out = state
            current = inflow[i]
            out = (1 / 21) * current + (9 / 21) * previous_in + (11 / 21) * previous_out
            return current, out

        return jax.lax.fori_loop(0, trip, step, (initial, initial), unroll=factor)

    return run, (flow[1:], flow[0])
