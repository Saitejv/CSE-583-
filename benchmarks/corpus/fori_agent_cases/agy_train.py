"""Reviewed workload drafts from the Antigravity CLI."""

import jax
import jax.numpy as jnp
import numpy as np


def tile_stipple(variant, trip, factor):
    """Ordered probability stippling with accumulated tile coverage error feedback."""
    h, w = (16, 16) if variant == "small" else (64, 64)
    rng = np.random.default_rng(1001)
    threshold_matrix = rng.uniform(0.0, 1.0, size=(h, w)).astype(np.float32)
    initial = np.zeros((h, w), dtype=np.float32)
    exposures = rng.uniform(0.1, 0.9, size=(trip,)).astype(np.float32)

    def run(initial, threshold_matrix, exposures):
        def step(i, carry):
            r = (i // w) % h
            c = i % w
            val = exposures[i] + carry[r, c]
            dot = jnp.where(val >= threshold_matrix[r, c], jnp.float32(1.0), jnp.float32(0.0))
            err = jnp.clip(val - dot, jnp.float32(-1.0), jnp.float32(1.0))
            carry = carry.at[r, c].set(err * jnp.float32(0.1))
            carry = carry.at[r, (c + 1) % w].add(err * jnp.float32(0.45))
            carry = carry.at[(r + 1) % h, c].add(err * jnp.float32(0.45))
            return carry

        return jax.lax.fori_loop(0, trip, step, initial, unroll=factor)

    return run, (initial, threshold_matrix, exposures)


def soft_kmeans(variant, trip, factor):
    """Online soft k-means clustering with streaming sample assignments."""
    k, d = (4, 8) if variant == "small" else (16, 32)
    rng = np.random.default_rng(1002)
    initial_centroids = rng.normal(0.0, 1.0, size=(k, d)).astype(np.float32)
    stream_points = rng.normal(0.0, 1.0, size=(trip, d)).astype(np.float32)

    def run(initial_centroids, stream_points):
        def step(i, carry):
            x = stream_points[i]
            diff = carry - x
            dist_sq = jnp.sum(diff * diff, axis=-1)
            weights = jax.nn.softmax(-dist_sq)
            delta = weights[:, None] * (x - carry)
            return carry + jnp.float32(0.05) * delta

        return jax.lax.fori_loop(0, trip, step, initial_centroids, unroll=factor)

    return run, (initial_centroids, stream_points)


def traffic_signal_debt(variant, trip, factor):
    """Cyclic traffic signal scheduling with queue pressure and phase debt."""
    num_lanes = 4 if variant == "small" else 16
    rng = np.random.default_rng(1003)
    init_phase = np.int32(0)
    init_queues = np.zeros(num_lanes, dtype=np.float32)
    initial = (init_phase, init_queues)
    arrivals = rng.poisson(0.95 / num_lanes, size=(trip, num_lanes)).astype(np.float32)
    service_rates = np.full(num_lanes, 1.2, dtype=np.float32)

    def run(initial, arrivals, service_rates):
        def step(i, carry):
            phase, queues = carry
            q = queues + arrivals[i]
            served = jnp.minimum(q[phase], service_rates[phase])
            q = q.at[phase].add(-served)
            next_cand = (phase + 1) % num_lanes
            should_switch = (q[phase] < jnp.float32(1.0)) | (q[next_cand] > q[phase] + jnp.float32(2.0))
            new_phase = jnp.where(should_switch, next_cand, phase)
            q = jnp.clip(q, jnp.float32(0.0), jnp.float32(50.0))
            return (new_phase, q)

        return jax.lax.fori_loop(0, trip, step, initial, unroll=factor)

    return run, (initial, arrivals, service_rates)
