"""Reviewed workload drafts from the Antigravity CLI."""

from jax import lax
import jax.numpy as jnp
import numpy as np


def inventory_lost_sales(variant, trip, factor):
    """Store by SKU panel inventory tracking with replenishment, sales, and lost demand."""
    rng = np.random.default_rng(101)
    stores, skus = (8, 16) if variant == "small" else (16, 32)
    init_inv = rng.uniform(5.0, 15.0, size=(stores, skus)).astype(np.float32)
    init_sales = np.zeros((stores, skus), dtype=np.float32)
    init_lost = np.zeros((stores, skus), dtype=np.float32)
    replenishments = rng.uniform(1.0, 6.0, size=(trip, stores, skus)).astype(np.float32)
    demand = rng.uniform(1.0, 8.0, size=(trip, stores, skus)).astype(np.float32)

    def run(inv, sales, lost, replenishments, demand):
        def step(i, carry):
            cur_inv, cur_sales, cur_lost = carry
            stock = cur_inv + replenishments[i]
            dem = demand[i]
            realized = jnp.minimum(stock, dem)
            next_inv = stock - realized
            next_sales = cur_sales + realized
            next_lost = cur_lost + (dem - realized)
            return (next_inv, next_sales, next_lost)

        return lax.fori_loop(0, trip, step, (inv, sales, lost), unroll=factor)

    return run, (init_inv, init_sales, init_lost, replenishments, demand)


def flood_duration(variant, trip, factor):
    """Environmental sensor duration and run-length statistics across threshold grid."""
    rng = np.random.default_rng(303)
    sensors, num_thresh = (16, 8) if variant == "small" else (64, 16)
    thresholds = np.linspace(2.0, 8.0, num=num_thresh, dtype=np.float32)
    stream = rng.uniform(0.0, 10.0, size=(trip, sensors)).astype(np.float32)
    init_active = np.zeros((sensors, num_thresh), dtype=np.int32)
    init_completed = np.zeros((sensors, num_thresh), dtype=np.int32)

    def run(active_runs, completed_runs, stream, thresholds):
        def step(i, carry):
            cur_active, cur_completed = carry
            exceeded = stream[i, :, None] >= thresholds
            run_ended = (~exceeded) & (cur_active > 0)
            next_completed = cur_completed + run_ended.astype(jnp.int32)
            next_active = jnp.where(exceeded, cur_active + 1, 0)
            return (next_active, next_completed)

        return lax.fori_loop(0, trip, step, (active_runs, completed_runs), unroll=factor)

    return run, (init_active, init_completed, stream, thresholds)
