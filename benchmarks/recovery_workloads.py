"""Distinct memory-access bodies for the separately frozen recovery split."""

import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.workloads import WorkloadSpec, _Built, _shape_json
from jax_unroll.recovery_split import CONFIRMATION, NEW_DEV, TRIPS, VARIANTS


class Inputs:
    def __init__(self, variant, seed):
        self.width = int(variant[1:])
        self.rng = np.random.default_rng(seed)

    def floats(self, *shape):
        return jnp.asarray(self.rng.uniform(-0.5, 0.5, size=shape), dtype=jnp.float32)

    def indices(self, high, *shape):
        return jnp.asarray(self.rng.integers(high, size=shape, dtype=np.int32))


def _built(fn, *args):
    return _Built(fn, args, _shape_json(**{f"arg{index}": arg.shape for index, arg in enumerate(args)}))


def _scan(step, factor):
    def fn(init, *streams):
        return lax.scan(step, init, streams, unroll=factor)[0]
    return fn


def _window_read(variant, trip, factor):
    data = Inputs(variant, 583_101)

    def fn(init, table, starts):
        def step(carry, start):
            return carry * 0.875 + jnp.sum(lax.dynamic_slice_in_dim(table, start, 8)), None
        return lax.scan(step, init, starts, unroll=factor)[0]

    return _built(fn, data.floats(), data.floats(data.width), data.indices(data.width - 7, trip))


def _window_replace(variant, trip, factor):
    data = Inputs(variant, 583_102)

    def step(carry, streams):
        start, patch = streams
        return lax.dynamic_update_slice_in_dim(carry, patch, start, 0), None

    return _built(_scan(step, factor), data.floats(data.width), data.indices(data.width - 7, trip), data.floats(trip, 8))


def _gather_lookup(variant, trip, factor):
    data = Inputs(variant, 583_103)

    def fn(init, table, indices, weights):
        def step(carry, streams):
            index, weight = streams
            return carry + jnp.sum(table[index] * weight), None
        return lax.scan(step, init, (indices, weights), unroll=factor)[0]

    return _built(fn, data.floats(), data.floats(data.width), data.indices(data.width, trip, 8), data.floats(trip, 8))


def _gather_rows(variant, trip, factor):
    data = Inputs(variant, 583_104)

    def fn(init, table, indices):
        def step(carry, index):
            return jnp.maximum(carry * 0.875, jnp.sum(table[index], axis=0)), None
        return lax.scan(step, init, indices, unroll=factor)[0]

    return _built(fn, data.floats(16), data.floats(data.width, 16), data.indices(data.width, trip, 4))


def _scatter_accumulate(variant, trip, factor):
    data = Inputs(variant, 583_105)

    def step(carry, streams):
        indices, updates = streams
        return carry.at[indices].add(updates), None

    return _built(_scan(step, factor), data.floats(data.width), data.indices(data.width, trip, 12), data.floats(trip, 12))


def _scatter_blend(variant, trip, factor):
    data = Inputs(variant, 583_106)
    indices = jnp.asarray(np.stack([data.rng.choice(data.width, 8, replace=False) for _ in range(trip)]), dtype=jnp.int32)

    def step(carry, streams):
        index, scale = streams
        patch = jnp.tanh(carry[index] * scale + 0.125)
        return carry.at[index].set(patch, unique_indices=True), None

    return _built(_scan(step, factor), data.floats(data.width), indices, data.floats(trip, 8))


def _strided_mix(variant, trip, factor):
    data = Inputs(variant, 583_201)

    def fn(init, table, starts):
        def step(carry, start):
            offset = start + 3 * jnp.arange(6, dtype=jnp.int32)
            mixed = table[offset] * table[offset + 1]
            return 0.625 * carry + jnp.sum(mixed), None
        return lax.scan(step, init, starts, unroll=factor)[0]

    return _built(fn, data.floats(), data.floats(data.width), data.indices(data.width - 16, trip))


def _bucket_max(variant, trip, factor):
    data = Inputs(variant, 583_202)

    def step(carry, streams):
        keys, values = streams
        buckets = jnp.bitwise_and(keys, carry.size - 1)
        return carry.at[buckets].max(values), None

    return _built(_scan(step, factor), data.floats(data.width), data.indices(4 * data.width, trip, 12), data.floats(trip, 12))


def _masked_rows(variant, trip, factor):
    data = Inputs(variant, 583_203)

    def fn(init, table, left, right, mask):
        def step(carry, streams):
            first, second, choose_first = streams
            rows = jnp.where(choose_first[:, None], table[first], table[second])
            return 0.75 * carry + jnp.sum(rows, axis=0), None
        return lax.scan(step, init, (left, right, mask), unroll=factor)[0]

    return _built(fn, data.floats(8), data.floats(data.width, 8), data.indices(data.width, trip, 6),
                  data.indices(data.width, trip, 6), data.indices(2, trip, 6).astype(jnp.bool_))


def _ring_feedback(variant, trip, factor):
    data = Inputs(variant, 583_204)

    def fn(init, drive):
        def step(index, carry):
            start = (index % (carry.size // 4)) * 4
            prev = lax.dynamic_slice_in_dim(carry, start, 4)
            patch = 0.5 * prev + 0.25 * jnp.roll(prev, 1) + drive[index]
            return lax.dynamic_update_slice_in_dim(carry, patch, start, 0)
        return lax.fori_loop(0, trip, step, init, unroll=factor)

    return _built(fn, data.floats(data.width), data.floats(trip, 4))


FACTORIES = {
    "window_read": _window_read, "window_replace": _window_replace,
    "gather_lookup": _gather_lookup, "gather_rows": _gather_rows,
    "scatter_accumulate": _scatter_accumulate, "scatter_blend": _scatter_blend,
    "strided_mix": _strided_mix, "bucket_max": _bucket_max,
    "masked_rows": _masked_rows, "ring_feedback": _ring_feedback,
}
RECOVERY_WORKLOADS = {
    name: WorkloadSpec(
        name, family, name.replace("_", " "), "fori_loop" if name == "ring_feedback" else "scan",
        {variant: f"{variant[1:]} element address space" for variant in VARIANTS},
        VARIANTS[0], FACTORIES[name], trip_counts=TRIPS,
    )
    for name, family in (NEW_DEV | CONFIRMATION).items()
}
