"""Signal processing independent fori_loop workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats
from benchmarks.corpus.spec import CORPUS_TRIP_COUNTS, CorpusSpec, Provenance

# 1. sig_cascaded_biquad: Direct-Form II Transposed Biquad Filter Bank
def _biquad_factory(variant: str, trip: int, factor: int):
    stages = {"stages2": 2, "stages4": 4, "stages8": 8}[variant]
    rng = np.random.default_rng(583_801)
    x = floats(rng, trip, scale=0.5)
    # sos shape: (stages, 5) -> [b0, b1, b2, a1, a2]
    sos_np = np.zeros((stages, 5), dtype=np.float32)
    for s in range(stages):
        r = 0.7 + 0.2 * rng.uniform()
        theta = rng.uniform(0.1, 0.9) * np.pi
        a1 = -2.0 * r * np.cos(theta)
        a2 = r * r
        sos_np[s] = [0.2, 0.4, 0.2, a1, a2]
    sos = jnp.asarray(sos_np, dtype=jnp.float32)
    init_state = jnp.zeros((stages, 2), dtype=jnp.float32)

    def fn(init: jax.Array, x_stream: jax.Array, sos_coeffs: jax.Array):
        def body(i: int, state: jax.Array):
            inp = x_stream[i]
            def stage_step(s: int, val_and_st: tuple[jax.Array, jax.Array]):
                v, st = val_and_st
                b0, b1, b2, a1, a2 = sos_coeffs[s]
                w1, w2 = st[s, 0], st[s, 1]
                y = b0 * v + w1
                new_w1 = b1 * v - a1 * y + w2
                new_w2 = b2 * v - a2 * y
                new_st = st.at[s, 0].set(new_w1).at[s, 1].set(new_w2)
                return (y, new_st)
            _, next_state = lax.fori_loop(0, stages, stage_step, (inp, state))
            return next_state

        return lax.fori_loop(0, trip, body, init, unroll=factor)

    return built(fn, init_state, x, sos)


# 2. sig_goertzel: Goertzel DFT Single-Bin Resonator
def _goertzel_factory(variant: str, trip: int, factor: int):
    width = {"scalar": 1, "vec4": 4}[variant]
    rng = np.random.default_rng(583_802)
    x = floats(rng, trip, width, scale=0.5)
    k = np.arange(1, width + 1, dtype=np.float32)
    coeff = jnp.asarray(2.0 * np.cos(2.0 * np.pi * k / max(trip, 64)), dtype=jnp.float32)
    init_s1 = jnp.zeros((width,), dtype=jnp.float32)
    init_s2 = jnp.zeros((width,), dtype=jnp.float32)

    def fn(s1: jax.Array, s2: jax.Array, x_stream: jax.Array, c: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            s_prev, s_prev2 = carry
            s = x_stream[i] + c * s_prev - s_prev2
            return (s, s_prev)

        return lax.fori_loop(0, trip, body, (s1, s2), unroll=factor)

    return built(fn, init_s1, init_s2, x, coeff)


# 3. sig_lms_adaptive: Normalized LMS Adaptive Filter
def _lms_factory(variant: str, trip: int, factor: int):
    taps = {"taps8": 8, "taps16": 16, "taps32": 32}[variant]
    rng = np.random.default_rng(583_803)
    ref = floats(rng, trip, scale=0.5)
    desired = floats(rng, trip, scale=0.5)
    init_weights = jnp.zeros((taps,), dtype=jnp.float32)
    init_hist = jnp.zeros((taps,), dtype=jnp.float32)
    mu = jnp.float32(0.01)

    def fn(w_init: jax.Array, h_init: jax.Array, ref_in: jax.Array, des_in: jax.Array, step_size: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            w, h = carry
            new_h = jnp.roll(h, 1).at[0].set(ref_in[i])
            y = jnp.dot(w, new_h)
            err = des_in[i] - y
            norm = jnp.dot(new_h, new_h) + jnp.float32(1e-4)
            new_w = w + (step_size / norm) * err * new_h
            return (new_w, new_h)

        return lax.fori_loop(0, trip, body, (w_init, h_init), unroll=factor)

    return built(fn, init_weights, init_hist, ref, desired, mu)


# 4. sig_polyphase_resample: Polyphase FIR Fractional Rate Interpolator
def _polyphase_factory(variant: str, trip: int, factor: int):
    phases, taps = {"p4t8": (4, 8), "p8t8": (8, 8)}[variant]
    rng = np.random.default_rng(583_804)
    x = floats(rng, trip, scale=0.5)
    filter_bank = floats(rng, phases, taps, scale=0.1)
    init_history = jnp.zeros((taps,), dtype=jnp.float32)

    def fn(h_init: jax.Array, x_stream: jax.Array, bank: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            h, total = carry
            phase = i % phases
            new_h = jnp.roll(h, 1).at[0].set(x_stream[i])
            out_sample = jnp.dot(bank[phase], new_h)
            return (new_h, total + out_sample)

        return lax.fori_loop(0, trip, body, (h_init, jnp.float32(0.0)), unroll=factor)

    return built(fn, init_history, x, filter_bank)


# 5. sig_hilbert_analytic: Dual Allpass Analytic Signal Generator [fresh_eval]
def _hilbert_factory(variant: str, trip: int, factor: int):
    order = {"order4": 4, "order8": 8}[variant]
    rng = np.random.default_rng(583_805)
    x = floats(rng, trip, scale=0.5)
    a0 = floats(rng, order, scale=0.4)
    a1 = floats(rng, order, scale=0.4)
    init_s0 = jnp.zeros((order,), dtype=jnp.float32)
    init_s1 = jnp.zeros((order,), dtype=jnp.float32)

    def fn(s0_init: jax.Array, s1_init: jax.Array, x_stream: jax.Array, c0: jax.Array, c1: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            s0, s1 = carry
            inp = x_stream[i]
            # Branch 0 allpass cascade
            def step_branch(k: int, st_and_val: tuple[jax.Array, jax.Array, jax.Array]):
                st, val, c = st_and_val
                y = c[k] * val + st[k]
                new_st = val - c[k] * y
                return (st.at[k].set(new_st), y, c)
            new_s0, _, _ = lax.fori_loop(0, order, step_branch, (s0, inp, c0))
            new_s1, _, _ = lax.fori_loop(0, order, step_branch, (s1, inp, c1))
            return (new_s0, new_s1)

        return lax.fori_loop(0, trip, body, (s0_init, s1_init), unroll=factor)

    return built(fn, init_s0, init_s1, x, a0, a1)


# 6. sig_pll_carrier: Costas Loop Carrier Phase Recovery [fresh_eval]
def _pll_factory(variant: str, trip: int, factor: int):
    channels = {"ch1": 1, "ch4": 4}[variant]
    rng = np.random.default_rng(583_806)
    iq = floats(rng, trip, channels, 2, scale=0.5)
    alpha = jnp.float32(0.05)
    beta = jnp.float32(0.001)
    init_phi = jnp.zeros((channels,), dtype=jnp.float32)
    init_freq = jnp.zeros((channels,), dtype=jnp.float32)

    def fn(phi_init: jax.Array, freq_init: jax.Array, iq_stream: jax.Array, a: jax.Array, b: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            phi, freq = carry
            i_in = iq_stream[i, :, 0]
            q_in = iq_stream[i, :, 1]
            i_demod = i_in * jnp.cos(phi) + q_in * jnp.sin(phi)
            q_demod = -i_in * jnp.sin(phi) + q_in * jnp.cos(phi)
            phase_error = i_demod * q_demod
            new_freq = freq + b * phase_error
            new_phi = phi + new_freq + a * phase_error
            return (new_phi, new_freq)

        return lax.fori_loop(0, trip, body, (phi_init, freq_init), unroll=factor)

    return built(fn, init_phi, init_freq, iq, alpha, beta)


# 7. sig_overlap_add_step: Overlap-Add Spectral Framing Buffer
def _overlap_add_factory(variant: str, trip: int, factor: int):
    hop, win_size = {"h32w64": (32, 64), "h64w128": (64, 128)}[variant]
    rng = np.random.default_rng(583_807)
    frames = floats(rng, trip, win_size, scale=0.2)
    window = jnp.asarray(np.hanning(win_size), dtype=jnp.float32)
    init_overlap = jnp.zeros((win_size - hop,), dtype=jnp.float32)

    def fn(ov_init: jax.Array, frame_stream: jax.Array, win: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            overlap, total = carry
            reconstructed = frame_stream[i] * win
            out_chunk = reconstructed[:hop] + jnp.pad(overlap[:min(hop, overlap.size)], (0, max(0, hop - overlap.size)))
            remaining = reconstructed[hop:]
            prev_tail = overlap[hop:] if overlap.size > hop else jnp.zeros((0,), dtype=jnp.float32)
            new_overlap = remaining + jnp.pad(prev_tail, (0, max(0, remaining.size - prev_tail.size)))
            return (new_overlap, total + jnp.sum(out_chunk))

        return lax.fori_loop(0, trip, body, (ov_init, jnp.float32(0.0)), unroll=factor)

    return built(fn, init_overlap, frames, window)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec("sig_cascaded_biquad", "signal_processing", "Direct-Form II Transposed Biquad Filter Bank",
               "signal_biquad_cascade", "Series cascade of S second-order stages feeding sequentially",
               "fori_loop", {"stages2": "2 stages", "stages4": "4 stages", "stages8": "8 stages"}, "stages4",
               _biquad_factory, 583_801, Provenance("public-inspired", "Oppenheim & Schafer; scipy.signal.sosfilt", "BSD-3-Clause", "Direct pure JAX implementation"),
               tier="full", related_group="sig_biquad"),
    CorpusSpec("sig_goertzel", "signal_processing", "Goertzel DFT Single-Bin Resonator",
               "signal_spectral_detection", "Second-order recurrence evaluating single-bin discrete Fourier energy",
               "fori_loop", {"scalar": "1 bin", "vec4": "4 bins"}, "scalar",
               _goertzel_factory, 583_802, Provenance("public-inspired", "Goertzel (1958) Am. Math. Monthly", "Public Domain", "Standard Goertzel recurrence"),
               tier="smoke", related_group="sig_spectral"),
    CorpusSpec("sig_lms_adaptive", "signal_processing", "Normalized LMS Adaptive Filter",
               "signal_adaptive_filter", "Dynamic filter coefficient update per sample based on instantaneous error",
               "fori_loop", {"taps8": "8 taps", "taps16": "16 taps", "taps32": "32 taps"}, "taps16",
               _lms_factory, 583_803, Provenance("public-inspired", "Widrow & Hoff (1960)", "Public Domain", "Normalized LMS stateful recurrence"),
               tier="full", related_group="sig_adaptive"),
    CorpusSpec("sig_polyphase_resample", "signal_processing", "Polyphase FIR Fractional Rate Interpolator",
               "signal_polyphase_commutation", "Commutator phase cycling selecting bank coefficients conditionally",
               "fori_loop", {"p4t8": "4 phases 8 taps", "p8t8": "8 phases 8 taps"}, "p4t8",
               _polyphase_factory, 583_804, Provenance("public-inspired", "Crochiere & Rabiner (1983)", "Public Domain", "Polyphase filter bank interpolation"),
               tier="full", related_group="sig_polyphase"),
    CorpusSpec("sig_hilbert_analytic", "signal_processing", "Dual Allpass Analytic Signal Generator",
               "signal_analytic_phase", "Dual parallel allpass branches maintaining orthogonal pi/2 phase difference",
               "fori_loop", {"order4": "order 4", "order8": "order 8"}, "order4",
               _hilbert_factory, 583_805, Provenance("public-inspired", "Vaidyanathan (1993)", "Public Domain", "Dual allpass Hilbert transformer"),
               tier="full", related_group="sig_analytic_eval"),
    CorpusSpec("sig_pll_carrier", "signal_processing", "Costas Loop Carrier Phase Recovery",
               "signal_carrier_tracking", "Nonlinear IQ phase detector with PI loop filter updating carrier phase",
               "fori_loop", {"ch1": "1 channel", "ch4": "4 channels"}, "ch1",
               _pll_factory, 583_806, Provenance("public-inspired", "Costas (1956) Proc. IRE", "Public Domain", "Costas loop phase tracker"),
               tier="full", related_group="sig_carrier_eval"),
    CorpusSpec("sig_overlap_add_step", "signal_processing", "Overlap-Add Spectral Framing Buffer",
               "signal_overlap_synthesis", "Overlapping boundary segments accumulate arithmetically in circular buffer",
               "fori_loop", {"h32w64": "hop 32 win 64", "h64w128": "hop 64 win 128"}, "h32w64",
               _overlap_add_factory, 583_807, Provenance("public-inspired", "Allen (1977) IEEE Trans. ASSP", "Public Domain", "Overlap-add reconstruction buffer"),
               tier="full", related_group="sig_overlap"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
