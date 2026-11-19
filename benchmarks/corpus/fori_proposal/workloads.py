"""Proposal-aligned coverage for unchanged native fori_loop bodies."""

from dataclasses import dataclass
from itertools import product

import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.fori_expansion import registry as expansion
from benchmarks.corpus.fori_regret.workloads import Case, cases
from benchmarks.corpus.fori_targeted import registry as targeted

TRIPS = (255, 1024)
SIZES = ("small", "large")


def slew_limited_actuator(variant, trip, factor):
    shape = () if variant == "small" else (8,)
    commands = np.random.default_rng(583_701).uniform(-1, 1, (trip, *shape)).astype(np.float32)
    state = np.zeros(shape, np.float32)

    def run(initial, stream):
        def step(i, current):
            # Limit motion per control tick, with position carried between ticks.
            return current + jnp.clip(stream[i] - current, -.125, .125)

        return lax.fori_loop(0, trip, step, initial, unroll=factor)

    return run, (state, commands)


def compensated_charge(variant, trip, factor):
    shape = () if variant == "small" else (4,)
    samples = np.random.default_rng(583_702).uniform(-.5, .5, (trip, *shape)).astype(np.float32)
    zero = np.zeros(shape, np.float32)

    def run(initial, current):
        def step(i, carry):
            previous, total, error = carry
            next_sample = current[i]
            corrected = (previous + next_sample) * .5 - error
            updated = total + corrected
            return next_sample, updated, (updated - total) - corrected

        return lax.fori_loop(0, trip, step, (initial, initial, initial), unroll=factor)

    return run, (zero, samples)


def lane_error_moments(variant, trip, factor):
    lanes = 4 if variant == "small" else 32
    rng = np.random.default_rng(583_703)
    observations = rng.uniform(-.25, .25, (trip, lanes)).astype(np.float32)
    reference = rng.uniform(-.1, .1, lanes).astype(np.float32)
    zero = np.zeros(lanes, np.float32)

    def run(initial, target, stream):
        def step(i, carry):
            bias, energy = carry
            error = stream[i] - target
            return .9375 * bias + error, .96875 * energy + error * error

        return lax.fori_loop(0, trip, step, (initial, initial), unroll=factor)

    return run, (zero, reference, observations)


def packed_word_digest(variant, trip, factor):
    lanes = () if variant == "small" else (8,)
    values = np.random.default_rng(583_704).integers(0, 2**32, (trip, *lanes), dtype=np.uint32)
    seed = np.full(lanes, 2166136261, dtype=np.uint32)

    def run(initial, words):
        def step(i, digest):
            rotated = (digest << 5) | (digest >> 27)
            return (rotated ^ words[i]) * jnp.uint32(16777619)

        return lax.fori_loop(0, trip, step, initial, unroll=factor)

    return run, (seed, values)


def driven_state_space(variant, trip, factor):
    width = 12 if variant == "small" else 64
    rng = np.random.default_rng(583_705)
    transition = rng.uniform(-.25 / width, .25 / width, (width, width)).astype(np.float32)
    transition += .7 * np.eye(width, dtype=np.float32)
    projection = rng.uniform(-.3, .3, (width, 4)).astype(np.float32)
    controls = rng.uniform(-1, 1, (trip, 4)).astype(np.float32)
    initial = np.zeros(width, np.float32)

    def run(state, dynamics, inputs, commands):
        def step(i, current):
            return dynamics @ current + inputs @ commands[i]

        return lax.fori_loop(0, trip, step, state, unroll=factor)

    return run, (initial, transition, projection, controls)


def staggered_charge_deposit(variant, trip, factor):
    bins, samples = (64, 4) if variant == "small" else (512, 16)
    rng = np.random.default_rng(583_706)
    # Pair each source cell with its right neighbor using disjoint pairs per step.
    positions = np.stack([rng.choice(bins // 2, samples, replace=False) for _ in range(trip)])
    positions = np.int32(2 * positions)
    fractions = rng.uniform(0, 1, positions.shape).astype(np.float32)
    charges = rng.uniform(-.25, .25, positions.shape).astype(np.float32)

    def run(grid, cells, offsets, charge):
        def step(i, density):
            left, fraction, q = cells[i], offsets[i], charge[i]
            density = density.at[left].add(q * (1 - fraction), unique_indices=True)
            return density.at[left + 1].add(q * fraction, unique_indices=True)

        return lax.fori_loop(0, trip, step, grid, unroll=factor)

    return run, (np.zeros(bins, np.float32), positions, fractions, charges)


def streaming_cross_spectrum(variant, trip, factor):
    samples, channels = (4, 8) if variant == "small" else (16, 24)
    rng = np.random.default_rng(583_707)
    left = rng.uniform(-.25, .25, (trip, samples, channels)).astype(np.float32)
    right = rng.uniform(-.25, .25, (trip, samples, channels)).astype(np.float32)
    zero = np.zeros((channels, channels), np.float32)

    def run(initial, receiver_a, receiver_b):
        def step(i, cross):
            a, b = receiver_a[i], receiver_b[i]
            return .984375 * cross + a.T @ b

        return lax.fori_loop(0, trip, step, initial, unroll=factor)

    return run, (zero, left, right)


def acoustic_transmission_line(variant, trip, factor):
    cells = 24 if variant == "small" else 128
    rng = np.random.default_rng(583_708)
    reflection = rng.uniform(-.08, .08, cells).astype(np.float32)
    excitation = rng.uniform(-.03, .03, (trip, 2)).astype(np.float32)
    initial = np.zeros(cells, np.float32)

    def run(zero, impedance, drive):
        def step(i, waves):
            right, left = waves
            incoming_right = jnp.roll(right, 1)
            incoming_left = jnp.roll(left, -1)
            reflected = impedance * (incoming_left - incoming_right)
            return (
                .97 * (incoming_right + reflected).at[0].add(drive[i, 0]),
                .97 * (incoming_left - reflected).at[-1].add(drive[i, 1]),
            )

        return lax.fori_loop(0, trip, step, (zero, zero), unroll=factor)

    return run, (initial, reflection, excitation)


FRESH = (
    Case("slew_limited_actuator", "scalar_control", "fresh", slew_limited_actuator),
    Case("compensated_charge", "compensated_integration", "fresh", compensated_charge),
    Case("lane_error_moments", "lane_wise_arithmetic", "fresh", lane_error_moments),
    Case("packed_word_digest", "integer_word_mixing", "fresh", packed_word_digest),
    Case("driven_state_space", "state_space_matrix_vector", "fresh", driven_state_space),
    Case("staggered_charge_deposit", "one_dimensional_indexed_deposit", "fresh", staggered_charge_deposit),
    Case("streaming_cross_spectrum", "streamed_cross_products", "fresh", streaming_cross_spectrum),
    Case("acoustic_transmission_line", "one_dimensional_acoustic_scattering", "fresh", acoustic_transmission_line),
)


@dataclass(frozen=True)
class Configuration:
    spec: object
    variant: str
    trip: int
    cohort: str
    live: bool


def settings():
    fresh = [Configuration(spec, size, trip, "fresh", size == "small" or trip == 255)
             for spec, size, trip in product(FRESH, SIZES, TRIPS)]
    controls = [
        Configuration(expansion.get_spec(name), size, trip, "proposal_control", True)
        for name, size, trip in (
            ("conv_scalar_recurrence", "scalar", 1024),
            ("conv_elementwise_sine", "n256", 255),
            ("conv_reduction_vector", "n256", 255),
            ("conv_dense_matvec", "n32", 255),
        )
    ]
    historical = [
        Configuration(targeted.get_spec(name), size, trip, "historical_control", True)
        for name, size, trip in (
            ("eval_write_waveguide_mesh", "large", 255),
            ("dense_gram_accum", "large", 1024),
        )
    ]
    stress = [Configuration(spec, size, trip, "regression_control", True)
              for spec, (size, trip) in product(cases("holdout").values(), (("small", 1023),))]
    return fresh + controls + historical + stress
