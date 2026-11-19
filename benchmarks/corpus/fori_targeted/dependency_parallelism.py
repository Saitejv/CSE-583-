"""Domain 1: Dependency depth and instruction parallelism workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats
from benchmarks.corpus.spec import CORPUS_TRIP_COUNTS, CorpusSpec, Provenance


# 1. dep_cascaded_biquad (train): Biquad IIR filter bank with carried state
def _cascaded_biquad_factory(variant: str, trip: int, factor: int):
    stages = 4 if variant == "small" else 16
    rng = np.random.default_rng(583_101)
    base_x = floats(rng, 1024, scale=0.5)
    x = base_x[:trip]
    
    # sos shape: (stages, 5) -> [b0, b1, b2, a1, a2]
    sos_np = np.zeros((stages, 5), dtype=np.float32)
    for s in range(stages):
        r = 0.7 + 0.15 * rng.uniform()
        theta = (0.2 + 0.6 * rng.uniform()) * np.pi
        a1 = -2.0 * r * np.cos(theta)
        a2 = r * r
        sos_np[s] = [0.25, 0.5, 0.25, a1, a2]
    sos = jnp.asarray(sos_np, dtype=jnp.float32)
    init_state = jnp.zeros((stages, 2), dtype=jnp.float32)

    def fn(init: jax.Array, x_stream: jax.Array, sos_coeffs: jax.Array):
        def body(i: int, state: jax.Array):
            inp = x_stream[i]
            curr_v = inp
            new_state_rows = []
            # Static unroll over stages to keep exactly one lax.fori_loop
            for s in range(stages):
                b0, b1, b2, a1, a2 = sos_coeffs[s, 0], sos_coeffs[s, 1], sos_coeffs[s, 2], sos_coeffs[s, 3], sos_coeffs[s, 4]
                w1, w2 = state[s, 0], state[s, 1]
                y = b0 * curr_v + w1
                new_w1 = b1 * curr_v - a1 * y + w2
                new_w2 = b2 * curr_v - a2 * y
                new_state_rows.append(jnp.stack([new_w1, new_w2]))
                curr_v = y
            return jnp.stack(new_state_rows, axis=0)

        return lax.fori_loop(0, trip, body, init, unroll=factor)

    return built(fn, init_state, x, sos)


# 2. dep_coupled_oscillator (train): Coupled harmonic oscillator chain with damping
def _coupled_oscillator_factory(variant: str, trip: int, factor: int):
    n = 16 if variant == "small" else 64
    rng = np.random.default_rng(583_102)
    base_forcing = floats(rng, 1024, n, scale=0.1)
    forcing = base_forcing[:trip]
    init_pos = floats(rng, n, scale=0.5)
    init_vel = jnp.zeros((n,), dtype=jnp.float32)
    init_state = jnp.stack([init_pos, init_vel], axis=0)  # (2, n)
    dt = jnp.float32(0.02)
    k_spring = jnp.float32(0.5)
    gamma = jnp.float32(0.05)

    def fn(state: jax.Array, ext_forcing: jax.Array):
        def body(i: int, st: jax.Array):
            pos, vel = st[0], st[1]
            # periodic nearest-neighbor Laplacian
            laplacian = jnp.roll(pos, 1) + jnp.roll(pos, -1) - 2.0 * pos
            accel = k_spring * laplacian - gamma * vel + ext_forcing[i]
            new_vel = vel + dt * accel
            new_pos = pos + dt * new_vel
            return jnp.stack([new_pos, new_vel], axis=0)

        return lax.fori_loop(0, trip, body, state, unroll=factor)

    return built(fn, init_state, forcing)


# 3. dep_dual_accumulator_iir (train): Decoupled dual accumulator recursive filter
def _dual_accumulator_iir_factory(variant: str, trip: int, factor: int):
    channels = 4 if variant == "small" else 16
    rng = np.random.default_rng(583_103)
    base_x = floats(rng, 1024, channels, scale=0.5)
    x = base_x[:trip]
    # State: even_acc (channels,), odd_acc (channels,), prev_even (channels,), prev_odd (channels,)
    init_state = jnp.zeros((channels, 4), dtype=jnp.float32)
    a1 = jnp.float32(0.85)
    a2 = jnp.float32(-0.15)
    b0 = jnp.float32(0.3)

    def fn(state: jax.Array, x_stream: jax.Array):
        def body(i: int, st: jax.Array):
            even_acc = st[:, 0]
            odd_acc = st[:, 1]
            prev_even = st[:, 2]
            prev_odd = st[:, 3]
            inp = x_stream[i]
            # Interleaved even/odd recursive update breaking long serial chain
            new_even = b0 * inp + a1 * odd_acc + a2 * prev_even
            new_odd = b0 * inp + a1 * even_acc + a2 * prev_odd
            return jnp.stack([new_even, new_odd, even_acc, odd_acc], axis=1)

        return lax.fori_loop(0, trip, body, state, unroll=factor)

    return built(fn, init_state, x)


# 4. dep_coupled_ekf_covariance (train): EKF coupled state vector & covariance time-update
def _coupled_ekf_covariance_factory(variant: str, trip: int, factor: int):
    dim = 4 if variant == "small" else 8
    rng = np.random.default_rng(583_104)
    base_u = floats(rng, 1024, dim, scale=0.2)
    u = base_u[:trip]
    init_x = floats(rng, dim, scale=0.5)
    init_p = jnp.eye(dim, dtype=jnp.float32) * 0.1
    # State transition matrix F and process noise Q
    f_mat = jnp.eye(dim, dtype=jnp.float32) * 0.95 + 0.02 * floats(rng, dim, dim, scale=1.0)
    q_mat = jnp.eye(dim, dtype=jnp.float32) * 0.01

    def fn(x_init: jax.Array, p_init: jax.Array, u_stream: jax.Array, f: jax.Array, q: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            x, p = carry
            # State propagation x = F x + u
            x_pred = jnp.dot(f, x) + u_stream[i]
            # Covariance propagation P = F P F^T + Q
            p_pred = jnp.dot(f, jnp.dot(p, f.T)) + q
            return (x_pred, p_pred)

        return lax.fori_loop(0, trip, body, (x_init, p_init), unroll=factor)

    return built(fn, init_x, init_p, u, f_mat, q_mat)


# 5. eval_dep_levinson_durbin (eval): Classic Levinson-Durbin lattice recursion for LPC
def _levinson_durbin_factory(variant: str, trip: int, factor: int):
    order = 8 if variant == "small" else 32
    rng = np.random.default_rng(583_105)
    base_r = floats(rng, 1024, order + 1, scale=0.5)
    # Ensure positive definite autocorrelation r[0] > sum(|r[1:]|)
    base_r = base_r.at[:, 0].set(jnp.abs(base_r[:, 0]) + float(order))
    r_stream = base_r[:trip]
    init_coeffs = jnp.zeros((order + 1,), dtype=jnp.float32).at[0].set(1.0)
    init_error = jnp.float32(1.0)

    def fn(a_init: jax.Array, err_init: jax.Array, r_in: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            a, err = carry
            r = r_in[i]
            # Compute reflection coefficient k_m
            cross = jnp.dot(a[:order], r[1 : order + 1])
            k = -cross / (err + 1e-6)
            k_clamped = jnp.clip(k, -0.99, 0.99)
            new_a = a.at[1 : order + 1].set(
                a[1 : order + 1] + k_clamped * jnp.flip(a[:order])
            )
            new_err = err * (1.0 - k_clamped * k_clamped) + 0.01
            return (new_a, new_err)

        return lax.fori_loop(0, trip, body, (a_init, err_init), unroll=factor)

    return built(fn, init_coeffs, init_error, r_stream)


# 6. eval_dep_savitzky_golay_stream (eval): Real-time streaming Savitzky-Golay polynomial filter
def _savitzky_golay_factory(variant: str, trip: int, factor: int):
    window = 7 if variant == "small" else 31
    rng = np.random.default_rng(583_106)
    base_x = floats(rng, 1024, scale=0.5)
    x = base_x[:trip]
    # S-G polynomial smoothing weights (normalized quadratic polynomial kernel)
    m = (window - 1) // 2
    t = np.arange(-m, m + 1, dtype=np.float32)
    weights_np = 3.0 * (3.0 * m * m + 3.0 * m - 1.0 - 5.0 * t * t) / (4.0 * m * (m * m - 1.0) + 1e-4) if m > 1 else np.ones(window, dtype=np.float32) / window
    weights_np = weights_np / np.sum(weights_np)
    weights = jnp.asarray(weights_np, dtype=jnp.float32)
    init_buffer = jnp.zeros((window,), dtype=jnp.float32)
    init_out = jnp.float32(0.0)

    def fn(buf_init: jax.Array, out_init: jax.Array, x_in: jax.Array, w: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            buf, _ = carry
            new_buf = jnp.roll(buf, 1).at[0].set(x_in[i])
            smoothed = jnp.dot(new_buf, w)
            return (new_buf, smoothed)

        return lax.fori_loop(0, trip, body, (buf_init, out_init), unroll=factor)

    return built(fn, init_buffer, init_out, x, weights)


# 7. eval_dep_carrier_pll_costas (eval): Costas loop carrier phase/frequency recovery
def _carrier_pll_costas_factory(variant: str, trip: int, factor: int):
    channels = 1 if variant == "small" else 4
    rng = np.random.default_rng(583_107)
    base_iq = floats(rng, 1024, channels, 2, scale=0.5)
    iq = base_iq[:trip]
    # Carried state: phase (channels,), freq (channels,), i_arm (channels,), q_arm (channels,)
    init_state = jnp.zeros((channels, 4), dtype=jnp.float32)
    alpha = jnp.float32(0.05)  # phase detector loop gain
    beta = jnp.float32(0.001)  # frequency integrator gain

    def fn(state: jax.Array, iq_stream: jax.Array):
        def body(i: int, st: jax.Array):
            phase = st[:, 0]
            freq = st[:, 1]
            sample = iq_stream[i]
            r_i, r_q = sample[:, 0], sample[:, 1]
            cos_p = jnp.cos(phase)
            sin_p = jnp.sin(phase)
            # Demodulate into I and Q arms
            i_arm = r_i * cos_p + r_q * sin_p
            q_arm = -r_i * sin_p + r_q * cos_p
            # Costas phase error detector: e = sign(I) * Q
            error = jnp.sign(i_arm) * q_arm
            new_freq = freq + beta * error
            new_phase = jnp.remainder(phase + new_freq + alpha * error, 2.0 * np.pi)
            return jnp.stack([new_phase, new_freq, i_arm, q_arm], axis=1)

        return lax.fori_loop(0, trip, body, state, unroll=factor)

    return built(fn, init_state, iq)


# 8. eval_dep_unscented_sigma_prop (eval): UKF sigma-point propagation
def _unscented_sigma_prop_factory(variant: str, trip: int, factor: int):
    dim = 3 if variant == "small" else 6
    num_sigma = 2 * dim + 1
    rng = np.random.default_rng(583_108)
    base_w = floats(rng, 1024, dim, scale=0.1)
    w_stream = base_w[:trip]
    init_mean = floats(rng, dim, scale=0.5)
    init_cov = jnp.eye(dim, dtype=jnp.float32) * 0.2
    wm = jnp.ones((num_sigma,), dtype=jnp.float32) / float(num_sigma)

    def fn(mean_init: jax.Array, cov_init: jax.Array, noise_stream: jax.Array, weights: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            mean, cov = carry
            diag_sqrt = jnp.sqrt(jnp.maximum(jnp.diag(cov), 1e-4))
            delta = jnp.diag(diag_sqrt)
            pts_plus = mean[None, :] + delta
            pts_minus = mean[None, :] - delta
            sigma_pts = jnp.concatenate([mean[None, :], pts_plus, pts_minus], axis=0)
            # Nonlinear propagation: x_next = tanh(x) + noise
            propagated = jnp.tanh(sigma_pts) + noise_stream[i][None, :]
            # Recombine mean and covariance
            new_mean = jnp.sum(weights[:, None] * propagated, axis=0)
            diff = propagated - new_mean[None, :]
            new_cov = jnp.dot(diff.T, diff * weights[:, None]) + jnp.eye(dim, dtype=jnp.float32) * 0.01
            return (new_mean, new_cov)

        return lax.fori_loop(0, trip, body, (mean_init, cov_init), unroll=factor)

    return built(fn, init_mean, init_cov, w_stream, wm)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec(
        name="dep_cascaded_biquad",
        family="dependency_parallelism",
        description="Cascaded Direct-Form II Transposed Biquad Filter Bank with carried state",
        mechanism="carried_state_filter_cascade",
        structural_difference="Static unrolled multi-stage IIR filter with tight feedforward and feedback state carry",
        loop_kind="fori_loop",
        shape_variants={
            "small": "stages=4 (carried state 4x2 floats = 32 B, sos 4x5 = 80 B)",
            "large": "stages=16 (carried state 16x2 floats = 128 B, sos 16x5 = 320 B)",
        },
        primary_variant="small",
        factory=_cascaded_biquad_factory,
        seed=583_101,
        provenance=Provenance(
            kind="original",
            source="Adapted from signal_biquad_cascade historical lineage",
            license="MIT",
            adaptation="Unrolled inner stage loop to strictly maintain single public fori_loop; scaled stages to 4 and 16",
        ),
        related_group="biquad_cascade",
    ),
    CorpusSpec(
        name="dep_coupled_oscillator",
        family="dependency_parallelism",
        description="Coupled harmonic oscillator chain with nearest-neighbor stiffness and damping",
        mechanism="coupled_state_damping_oscillator",
        structural_difference="Coupled vector state with periodic Laplacian spatial communication and external forcing",
        loop_kind="fori_loop",
        shape_variants={
            "small": "N=16 oscillators (state 2x16 floats = 128 B)",
            "large": "N=64 oscillators (state 2x64 floats = 512 B)",
        },
        primary_variant="small",
        factory=_coupled_oscillator_factory,
        seed=583_102,
        provenance=Provenance(
            kind="original",
            source="Adapted from stateful_coupled_oscillator historical lineage",
            license="MIT",
            adaptation="Standardized to dual size variants (16, 64) with deterministic prefix external forcing",
        ),
        related_group="coupled_oscillator",
    ),
    CorpusSpec(
        name="dep_dual_accumulator_iir",
        family="dependency_parallelism",
        description="Decoupled dual-accumulator 2nd-order recursive IIR filter",
        mechanism="decoupled_dual_accumulator_pipeline",
        structural_difference="Interleaved even/odd sample recursive state updates breaking serial dependency chain",
        loop_kind="fori_loop",
        shape_variants={
            "small": "channels=4 (state 4x4 floats = 64 B)",
            "large": "channels=16 (state 16x4 floats = 256 B)",
        },
        primary_variant="small",
        factory=_dual_accumulator_iir_factory,
        seed=583_103,
        provenance=Provenance(
            kind="original",
            source="DSP block interleaved filtering architectures",
            license="MIT",
            adaptation="Original JAX implementation of parallel interleaved IIR recursion",
        ),
        related_group="dual_accumulator_iir",
    ),
    CorpusSpec(
        name="dep_coupled_ekf_covariance",
        family="dependency_parallelism",
        description="Extended Kalman Filter coupled state vector and covariance propagation step",
        mechanism="coupled_matrix_state_covariance",
        structural_difference="Joint vector state propagation coupled with full matrix-matrix covariance update",
        loop_kind="fori_loop",
        shape_variants={
            "small": "dim=4 (x 4 floats = 16 B, P 4x4 floats = 64 B)",
            "large": "dim=8 (x 8 floats = 32 B, P 8x8 floats = 256 B)",
        },
        primary_variant="small",
        factory=_coupled_ekf_covariance_factory,
        seed=583_104,
        provenance=Provenance(
            kind="original",
            source="Kalman, R. E. (1960). A new approach to linear filtering and prediction problems",
            license="MIT",
            adaptation="Self-contained JAX EKF time propagation step with joint vector and matrix carries",
        ),
        related_group="coupled_ekf_covariance",
    ),
    CorpusSpec(
        name="eval_dep_levinson_durbin",
        family="eval_dependency_parallelism",
        description="Levinson-Durbin lattice recursion for Linear Predictive Coding (LPC)",
        mechanism="levinson_durbin_lattice_recursion",
        structural_difference="Three-term order-recursive lattice reflection coefficient and polynomial carry",
        loop_kind="fori_loop",
        shape_variants={
            "small": "order=8 (coeffs 9 floats = 36 B)",
            "large": "order=32 (coeffs 33 floats = 132 B)",
        },
        primary_variant="small",
        factory=_levinson_durbin_factory,
        seed=583_105,
        provenance=Provenance(
            kind="public-inspired",
            source="Makhoul, J. (1975). Linear prediction: A tutorial review. Proc. IEEE 63(4)",
            license="Academic reference / Public domain",
            adaptation="Iterative order recurrence on streaming autocorrelation inputs",
        ),
        related_group="eval_levinson_durbin",
    ),
    CorpusSpec(
        name="eval_dep_savitzky_golay_stream",
        family="eval_dependency_parallelism",
        description="Streaming Savitzky-Golay polynomial smoothing filter with carried shift register",
        mechanism="savitzky_golay_polynomial_shift_register",
        structural_difference="Carried sliding buffer shift-register with dot-product polynomial kernel projection",
        loop_kind="fori_loop",
        shape_variants={
            "small": "window=7 (buffer 7 floats = 28 B)",
            "large": "window=31 (buffer 31 floats = 124 B)",
        },
        primary_variant="small",
        factory=_savitzky_golay_factory,
        seed=583_106,
        provenance=Provenance(
            kind="public-inspired",
            source="Savitzky, A., & Golay, M. J. (1964). Smoothing and differentiation of data. Anal. Chem.",
            license="Academic reference / Public domain",
            adaptation="Real-time streaming convolution filter with carried circular state",
        ),
        related_group="eval_savitzky_golay",
    ),
    CorpusSpec(
        name="eval_dep_carrier_pll_costas",
        family="eval_dependency_parallelism",
        description="Costas loop carrier phase/frequency recovery with coupled I/Q phase detector",
        mechanism="costas_loop_phase_recovery",
        structural_difference="Coupled non-linear trigonometric phase detector and 2nd-order loop filter integrator",
        loop_kind="fori_loop",
        shape_variants={
            "small": "channels=1 (state 4 floats = 16 B)",
            "large": "channels=4 (state 16 floats = 64 B)",
        },
        primary_variant="small",
        factory=_carrier_pll_costas_factory,
        seed=583_107,
        provenance=Provenance(
            kind="public-inspired",
            source="Costas, J. P. (1956). Synchronous communications. Proc. IRE 44(12)",
            license="IEEE standard reference",
            adaptation="Digital Costas loop tracking carrier phase and frequency offsets",
        ),
        related_group="eval_carrier_pll_costas",
    ),
    CorpusSpec(
        name="eval_dep_unscented_sigma_prop",
        family="eval_dependency_parallelism",
        description="Unscented Kalman Filter (UKF) sigma-point nonlinear propagation",
        mechanism="unscented_sigma_point_propagation",
        structural_difference="Deterministic sigma point set expansion, nonlinear transfer, and covariance recombination",
        loop_kind="fori_loop",
        shape_variants={
            "small": "dim=3 (7 sigma points, mean 12 B, cov 36 B)",
            "large": "dim=6 (13 sigma points, mean 24 B, cov 144 B)",
        },
        primary_variant="small",
        factory=_unscented_sigma_prop_factory,
        seed=583_108,
        provenance=Provenance(
            kind="public-inspired",
            source="Julier, S. J., & Uhlmann, J. K. (1997). New extension of the Kalman filter to nonlinear systems. Proc. SPIE",
            license="Academic reference",
            adaptation="Scaled unscented transform propagation step with mean and covariance carry",
        ),
        related_group="eval_unscented_sigma_prop",
    ),
)
