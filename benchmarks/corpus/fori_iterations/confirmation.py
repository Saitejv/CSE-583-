"""Frozen confirmation bodies for the five optimization cycles."""

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.fori_iterations.storage import OUT
from benchmarks.corpus.fori_regret.acquire import REPEATS, measure, run_workers
from benchmarks.corpus.fori_regret.workloads import Case

SETTINGS = (("small", 1024), ("large", 255))
HORIZONS = (1, 300, 5_000, 20_000)
MANIFEST = OUT / "confirmation_matrix.json"


def phase_transport(variant, trip, factor):
    shape = () if variant == "small" else (12,)
    phase = np.random.default_rng(583_821).uniform(-3, 3, (trip, *shape)).astype(np.float32)
    zero = np.zeros(shape, np.float32)

    def run(z, values):
        def update(i, carry):
            previous, displacement = carry
            new = values[i]
            delta = new - previous
            delta -= jnp.where(delta > 3.1416, 6.2832, 0.)
            delta += jnp.where(delta < -3.1416, 6.2832, 0.)
            return new, displacement * .997 + delta
        return lax.fori_loop(0, trip, update, (z, z), unroll=factor)

    return run, (zero, phase)


def predictive_sensor_tracker(variant, trip, factor):
    shape = () if variant == "small" else (8,)
    values = np.random.default_rng(583_822).normal(0, .2, (trip, *shape)).astype(np.float32)
    zero = np.zeros(shape, np.float32)

    def run(z, observations):
        def update(i, carry):
            estimate, trend = carry
            prediction = estimate + trend
            residual = observations[i] - prediction
            return prediction + .5 * residual, .8 * trend + .125 * residual
        return lax.fori_loop(0, trip, update, (z, z), unroll=factor)

    return run, (zero, values)


def dual_envelope_meter(variant, trip, factor):
    width = 8 if variant == "small" else 48
    values = np.random.default_rng(583_823).uniform(-1, 1, (trip, width)).astype(np.float32)
    zero = np.zeros(width, np.float32)

    def run(z, signal):
        def update(i, carry):
            peak, floor = carry
            return jnp.maximum(.95 * peak, signal[i]), jnp.minimum(.99 * floor, signal[i])
        return lax.fori_loop(0, trip, update, (z, z), unroll=factor)

    return run, (zero, values)


def coupled_chemical_surrogate(variant, trip, factor):
    width = 12 if variant == "small" else 48
    rng = np.random.default_rng(583_824)
    coupling = rng.uniform(-.7 / width, .7 / width, (width, width)).astype(np.float32)
    forcing = rng.uniform(-.25, .25, (trip, width)).astype(np.float32)
    initial = np.zeros(width, np.float32)

    def run(x, matrix, drive):
        def update(i, carry):
            projected = matrix @ carry
            return .7 * carry + jnp.tanh(projected + drive[i])
        return lax.fori_loop(0, trip, update, x, unroll=factor)

    return run, (initial, coupling, forcing)


def index_expiry_counters(variant, trip, factor):
    width = 64 if variant == "small" else 512
    rng = np.random.default_rng(583_825)
    indices = rng.integers(0, width, trip, dtype=np.int32)
    values = rng.uniform(.05, .5, trip).astype(np.float32)
    zeros = np.zeros(width, np.float32)
    expiry = np.zeros(width, np.int32)

    def run(data, ttl, keys, events):
        def update(i, carry):
            values, expiry = carry
            address = keys[i]
            active = i < expiry[address]
            values = values.at[address].set(jnp.where(active, .92 * values[address], 0.) + events[i])
            return values, expiry.at[address].set(i + 5 + (address & 7))
        return lax.fori_loop(0, trip, update, (data, ttl), unroll=factor)

    return run, (zeros, expiry, indices, values)


def adaptive_spike_recovery(variant, trip, factor):
    width = 16 if variant == "small" else 128
    rng = np.random.default_rng(583_826)
    forcing = rng.uniform(-.05, .22, (trip, width)).astype(np.float32)
    zero = np.zeros(width, np.float32)
    counts = np.zeros(width, np.int32)

    def run(z, n, stimulus):
        def update(i, carry):
            membrane, threshold, spikes = carry
            drive = .94 * membrane + stimulus[i]
            fired = drive > (.8 + threshold)
            return (jnp.where(fired, .05, drive), .97 * threshold + .1 * fired,
                    spikes + fired.astype(jnp.int32))
        return lax.fori_loop(0, trip, update, (z, z, n), unroll=factor)

    return run, (zero, counts, forcing)


CASES = (
    Case("phase_transport", "wrapped_phase_transport", "confirmation", phase_transport),
    Case("predictive_sensor_tracker", "alpha_beta_tracking", "confirmation", predictive_sensor_tracker),
    Case("dual_envelope_meter", "dual_envelope_detection", "confirmation", dual_envelope_meter),
    Case("coupled_chemical_surrogate", "dense_nonlinear_projection", "confirmation", coupled_chemical_surrogate),
    Case("index_expiry_counters", "indexed_expiry_cache", "confirmation", index_expiry_counters),
    Case("adaptive_spike_recovery", "spiking_threshold_recovery", "confirmation", adaptive_spike_recovery),
)


def matrix():
    return [(case, size, trip) for case, (size, trip) in product(CASES, SETTINGS)]


def freeze(create=False):
    value = {"source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
             "cases": [{"workload": c.name, "family": c.family, "variant": size, "trip_count": trip}
                       for c, size, trip in matrix()],
             "horizons": list(HORIZONS), "repeats": REPEATS}
    if create and not MANIFEST.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open("x") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
    if json.loads(MANIFEST.read_text()) != value:
        raise ValueError("confirmation matrix changed after the freeze")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "sweep"))
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    freeze(create=args.stage == "freeze")
    if args.worker is not None:
        print(json.dumps(measure(matrix(), args.worker)))
    elif args.stage == "sweep":
        run_workers(__spec__.name, ["sweep"], REPEATS, OUT / "confirmation_native.json")


if __name__ == "__main__":
    main()
