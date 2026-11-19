"""Reviewed OpenCode evaluation drafts. Host streams are fixed across factors."""

import jax.numpy as jnp
import numpy as np
from jax import lax


def queue_workload(variant, trip, factor):
    """Lindley waiting-time recursion at independent service desks."""
    rng = np.random.default_rng(5201)
    desks = 4 if variant == "small" else 32
    interarrivals = rng.uniform(.6, 1.5, (trip, desks)).astype(np.float32)
    services = rng.uniform(.5, 1.5, (trip, desks)).astype(np.float32)
    initial = np.zeros(desks, np.float32)

    def run(wait, arrivals, durations):
        def step(i, carry):
            return jnp.maximum(0., carry + durations[i] - arrivals[i])
        return lax.fori_loop(0, trip, step, wait, unroll=factor)

    return run, (initial, interarrivals, services)


def warranty_ladder(variant, trip, factor):
    """Move surviving warranty exposures between age cohorts as new contracts arrive."""
    rng = np.random.default_rng(5202)
    products, ages = (4, 12) if variant == "small" else (24, 32)
    survival = rng.uniform(.9, .999, (products, ages - 1)).astype(np.float32)
    contracts = rng.uniform(1., 30., (trip, products)).astype(np.float32)
    initial = rng.uniform(0., 10., (products, ages)).astype(np.float32)

    def run(exposure, survival, new_contracts):
        def step(i, carry):
            survivors = carry[:, :-1] * survival
            return jnp.concatenate((new_contracts[i, :, None], survivors), axis=1)
        return lax.fori_loop(0, trip, step, exposure, unroll=factor)

    return run, (initial, survival, contracts)


def bitplane_motion(variant, trip, factor):
    """Smooth blockwise motion evidence from Hamming distances of successive bit windows."""
    rng = np.random.default_rng(5203)
    blocks = 8 if variant == "small" else 64
    samples = rng.integers(2, size=(trip, blocks), dtype=np.uint32)
    signature = rng.integers(2**32, size=blocks, dtype=np.uint32)
    evidence = np.zeros(blocks, np.float32)

    def run(initial, evidence, bit_stream):
        def step(i, carry):
            signature, recent = carry
            updated = (signature << np.uint32(1)) | bit_stream[i]
            distance = lax.population_count(updated ^ signature).astype(jnp.float32)
            return updated, .95 * recent + .05 * distance
        return lax.fori_loop(0, trip, step, (initial, evidence), unroll=factor)

    return run, (signature, evidence, samples)
