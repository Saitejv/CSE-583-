"""Indirect-memory corpus bodies: gathers, scatters, locality, dependences.

Every body is an independent fixed-trip-count loop. Static tables and index
streams are workload arguments so the loop body closes over runtime values
rather than baking table data into the executable. Each factory is
deterministic for a given ``(variant, trip, factor)`` and uses one seed from
the corpus block 583_300..583_399.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import (
    built,
    floats,
    indices,
    permutation,
)
from benchmarks.corpus.spec import CORPUS_TRIP_COUNTS, CorpusSpec, Provenance

_ORIGINAL = Provenance(
    kind="original",
    source="Original implementation for this corpus",
    license="No repository license file; original code",
    adaptation="Independent body written for this corpus; no external code reused.",
)
_JAX_GATHER = Provenance(
    kind="public-inspired",
    source="https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html",
    license="Apache-2.0",
    adaptation="Documented gather semantics reimplemented from scratch; no code copied.",
)
_JAX_SCATTER = Provenance(
    kind="public-inspired",
    source="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html",
    license="Apache-2.0",
    adaptation="Documented scatter semantics reimplemented from scratch; no code copied.",
)
_JAX_CFD = Provenance(
    kind="public-inspired",
    source="https://github.com/google/jax-cfd",
    license="Apache-2.0",
    adaptation="Grid-access pattern reimplemented from the project description; no code copied.",
)
_JRAPH = Provenance(
    kind="public-inspired",
    source="https://github.com/google-deepmind/jraph/blob/master/jraph/_src/models.py",
    license="Apache-2.0",
    adaptation="Dependent gather traversal reimplemented from the message-passing description; no code copied.",
)


def _bools(rng: np.random.Generator, *shape: int):
    return jnp.asarray(rng.random(shape) < 0.5)


# ---------------------------------------------------------------------------
# indirect_gather
# ---------------------------------------------------------------------------


def _gather_scalar_table(variant: str, trip: int, factor: int):
    size = {"n64": 64, "n512": 512, "n2048": 2048}[variant]
    rng = np.random.default_rng(583_300)
    table = floats(rng, size)
    index = indices(rng, size, trip)
    gain = floats(rng, trip, scale=0.25)

    def fn(init, table, index, gain):
        def step(carry, streams):
            index_t, gain_t = streams
            return carry * 0.9 + table[index_t] * gain_t, None

        return lax.scan(step, init, (index, gain), unroll=factor)[0]

    return built(fn, jnp.float32(0.0), table, index, gain)


def _gather_row_pool(variant: str, trip: int, factor: int):
    rows = {"rows64": 64, "rows512": 512}[variant]
    width = 16
    rng = np.random.default_rng(583_301)
    table = floats(rng, rows, width)
    index = indices(rng, rows, trip)
    gain = floats(rng, trip, 8, scale=0.25)

    def fn(init, table, index, gain):
        def step(carry, streams):
            row_index, gain_t = streams
            row = table[row_index]
            pooled = row[:8] + row[8:]
            return carry * 0.875 + jnp.tanh(pooled) * gain_t, None

        return lax.scan(step, init, (index, gain), unroll=factor)[0]

    return built(fn, jnp.zeros((8,), jnp.float32), table, index, gain)


def _gather_multi_table(variant: str, trip: int, factor: int):
    rows = {"t64": 64, "t512": 512}[variant]
    width = 4
    rng = np.random.default_rng(583_302)
    left_table = floats(rng, rows, width)
    right_table = floats(rng, rows, width)
    left_index = indices(rng, rows, trip)
    right_index = indices(rng, rows, trip)
    choose_left = _bools(rng, trip)
    gain = floats(rng, trip, width, scale=0.25)

    def fn(init, left_table, right_table, left_index, right_index, choose_left, gain):
        def step(carry, streams):
            left_t, right_t, choose_t, gain_t = streams
            row = jnp.where(
                choose_t, left_table[left_t], right_table[right_t]
            )
            return carry * 0.875 + jnp.tanh(row) * gain_t, None

        return lax.scan(
            step,
            init,
            (left_index, right_index, choose_left, gain),
            unroll=factor,
        )[0]

    return built(
        fn,
        jnp.zeros((width,), jnp.float32),
        left_table,
        right_table,
        left_index,
        right_index,
        choose_left,
        gain,
    )


def _gather_duplicate_index(variant: str, trip: int, factor: int):
    rows = {"d128": 128, "d1024": 1024}[variant]
    width = 4
    copies = 4
    rng = np.random.default_rng(583_303)
    table = floats(rng, rows, width)
    base = rng.integers(rows, size=(trip, copies), dtype=np.int32)
    duplicate_index = jnp.asarray(np.concatenate([base, base], axis=1))
    gain = floats(rng, trip, width, scale=0.25)

    def fn(init, table, duplicate_index, gain):
        def step(carry, streams):
            index_t, gain_t = streams
            values = table[index_t]
            paired = values[:copies] + values[copies:]
            return carry * 0.9 + jnp.sum(paired * gain_t), None

        return lax.scan(step, init, (duplicate_index, gain), unroll=factor)[0]

    return built(fn, jnp.float32(0.0), table, duplicate_index, gain)


def _gather_sorted_blocks(variant: str, trip: int, factor: int):
    size = {"b256": 256, "b2048": 2048}[variant]
    block = 8
    rng = np.random.default_rng(583_304)
    table = floats(rng, size)
    starts = jnp.asarray(
        np.sort(rng.integers(size - block + 1, size=trip, dtype=np.int32))
    )
    pooled_width = block // 2
    gain = floats(rng, trip, pooled_width, scale=0.25)
    offsets = jnp.arange(block, dtype=jnp.int32)

    def fn(init, table, starts, gain):
        def step(carry, streams):
            start_t, gain_t = streams
            values = table[start_t + offsets]
            pooled = values[:pooled_width] + values[pooled_width:]
            return carry * 0.625 + jnp.tanh(pooled) * gain_t, None

        return lax.scan(step, init, (starts, gain), unroll=factor)[0]

    return built(
        fn, jnp.zeros((pooled_width,), jnp.float32), table, starts, gain
    )


def _gather_masked_window(variant: str, trip: int, factor: int):
    size = {"m128": 128, "m1024": 1024}[variant]
    lanes = 8
    rng = np.random.default_rng(583_305)
    table = floats(rng, size)
    index = indices(rng, size, trip, lanes)
    mask = _bools(rng, trip, lanes)
    gain = floats(rng, trip, lanes, scale=0.25)

    def fn(init, table, index, mask, gain):
        def step(carry, streams):
            index_t, mask_t, gain_t = streams
            safe_index = jnp.where(mask_t, index_t, size)
            values = jnp.take(
                table, safe_index, axis=0, mode="fill", fill_value=0.0
            )
            return carry * 0.9 + jnp.sum(values * gain_t), None

        return lax.scan(step, init, (index, mask, gain), unroll=factor)[0]

    return built(fn, jnp.float32(0.0), table, index, mask, gain)


# ---------------------------------------------------------------------------
# indirect_scatter
# ---------------------------------------------------------------------------


def _scatter_unique_rows(variant: str, trip: int, factor: int):
    rows = {"r128": 128, "r512": 512}[variant]
    width = 8
    lanes = 4
    rng = np.random.default_rng(583_306)
    init = floats(rng, rows, width, scale=0.25)
    updates = floats(rng, trip, lanes, width, scale=0.25)
    unique_index = np.stack(
        [rng.choice(rows, lanes, replace=False) for _ in range(trip)]
    ).astype(np.int32)
    unique_index = jnp.asarray(unique_index)

    def fn(init, unique_index, updates):
        def step(carry, streams):
            index_t, update_t = streams
            current = carry[index_t]
            updated = jnp.tanh(current + update_t)
            return carry.at[index_t].set(updated, unique_indices=True), None

        return lax.scan(step, init, (unique_index, updates), unroll=factor)[0]

    return built(fn, init, unique_index, updates)


def _scatter_collision_add(variant: str, trip: int, factor: int):
    grid = {"g16x64": (16, 64), "g32x128": (32, 128)}[variant]
    rows, cols = grid
    lanes = 12
    rng = np.random.default_rng(583_307)
    init = floats(rng, rows, cols, scale=0.25)
    row_index = indices(rng, rows, trip, lanes)
    col_index = indices(rng, cols, trip, lanes)
    updates = floats(rng, trip, lanes, scale=0.05)

    def fn(init, row_index, col_index, updates):
        def step(carry, streams):
            row_t, col_t, update_t = streams
            return (carry * 0.99).at[row_t, col_t].add(update_t), None

        return lax.scan(step, init, (row_index, col_index, updates), unroll=factor)[0]

    return built(fn, init, row_index, col_index, updates)


def _scatter_min_update(variant: str, trip: int, factor: int):
    size = {"s256": 256, "s1024": 1024}[variant]
    lanes = 12
    rng = np.random.default_rng(583_308)
    init = floats(rng, size) + 0.5
    index = indices(rng, size, trip, lanes)
    updates = floats(rng, trip, lanes)

    def fn(init, index, updates):
        def body(i, carry):
            return carry.at[index[i]].min(updates[i])

        return lax.fori_loop(0, trip, body, init, unroll=factor)

    return built(fn, init, index, updates)


def _scatter_two_tables(variant: str, trip: int, factor: int):
    size = {"t256": 256, "t1024": 1024}[variant]
    lanes = 8
    rng = np.random.default_rng(583_309)
    left = floats(rng, size, scale=0.25)
    right = floats(rng, size, scale=0.25)
    updates = floats(rng, trip, lanes, scale=0.25)
    mask = _bools(rng, trip, lanes)
    unique_index = np.stack(
        [rng.choice(size, lanes, replace=False) for _ in range(trip)]
    ).astype(np.int32)
    unique_index = jnp.asarray(unique_index)

    def fn(left, right, unique_index, updates, mask):
        def step(carry, streams):
            left_carry, right_carry = carry
            index_t, update_t, mask_t = streams
            left_current = left_carry[index_t]
            right_current = right_carry[index_t]
            candidate = jnp.tanh(0.5 * (left_current + right_current) + update_t)
            left_written = jnp.where(mask_t, candidate, left_current)
            right_written = jnp.where(mask_t, right_current, candidate)
            left_next = left_carry.at[index_t].set(
                left_written, unique_indices=True
            )
            right_next = right_carry.at[index_t].set(
                right_written, unique_indices=True
            )
            return (left_next, right_next), None

        return lax.scan(
            step, (left, right), (unique_index, updates, mask), unroll=factor
        )[0]

    return built(fn, left, right, unique_index, updates, mask)


def _scatter_permutation(variant: str, trip: int, factor: int):
    size = {"p128": 128, "p512": 512}[variant]
    block = 8
    rng = np.random.default_rng(583_310)
    init = floats(rng, size, scale=0.25)
    drive = floats(rng, trip, scale=0.1)
    block_perms = np.stack(
        [rng.permutation(block) for _ in range(size // block)]
    )
    perm = (
        np.arange(size // block)[:, None] * block + block_perms
    ).reshape(-1)
    perm = jnp.asarray(perm.astype(np.int32))

    def fn(init, perm, drive):
        def step(carry, drive_t):
            updated = jnp.tanh(carry + drive_t)
            return carry.at[perm].set(updated, unique_indices=True), None

        return lax.scan(step, init, drive, unroll=factor)[0]

    return built(fn, init, perm, drive)


# ---------------------------------------------------------------------------
# locality_access
# ---------------------------------------------------------------------------


def _locality_stream_contiguous(variant: str, trip: int, factor: int):
    size = {"n256": 256, "n2048": 2048}[variant]
    window = 8
    rng = np.random.default_rng(583_311)
    table = floats(rng, size)
    starts = (np.arange(trip) * window) % (size - 2 * window + 1)
    starts = jnp.asarray(starts.astype(np.int32))

    def fn(init, table, starts):
        def step(carry, start):
            left = lax.dynamic_slice_in_dim(table, start, window)
            right = lax.dynamic_slice_in_dim(table, start + window, window)
            contrast = jnp.tanh(jnp.sum(left) - jnp.sum(right))
            return carry * 0.75 + contrast * 0.25, None

        return lax.scan(step, init, starts, unroll=factor)[0]

    return built(fn, jnp.float32(0.0), table, starts)


def _locality_stride_sample(variant: str, trip: int, factor: int):
    size = {"n256": 256, "n2048": 2048}[variant]
    stride = 5
    window = 8
    rng = np.random.default_rng(583_312)
    table = floats(rng, size)
    starts = indices(rng, size - stride * (window - 1), trip)
    gain = floats(rng, trip, window, scale=0.25)
    offsets = stride * jnp.arange(window, dtype=jnp.int32)

    def fn(init, table, starts, gain):
        def step(carry, streams):
            start_t, gain_t = streams
            values = table[start_t + offsets]
            return carry * 0.625 + jnp.tanh(values) * gain_t, None

        return lax.scan(step, init, (starts, gain), unroll=factor)[0]

    return built(fn, jnp.zeros((window,), jnp.float32), table, starts, gain)


def _locality_random_window(variant: str, trip: int, factor: int):
    size = {"n256": 256, "n2048": 2048}[variant]
    window = 8
    rng = np.random.default_rng(583_313)
    table = floats(rng, size)
    starts = indices(rng, size - window + 1, trip)
    gain = floats(rng, trip, window, scale=0.25)

    def fn(init, table, starts, gain):
        def step(carry, streams):
            start_t, gain_t = streams
            values = lax.dynamic_slice_in_dim(table, start_t, window)
            return carry * 0.625 + jnp.tanh(values) * gain_t, None

        return lax.scan(step, init, (starts, gain), unroll=factor)[0]

    return built(fn, jnp.zeros((window,), jnp.float32), table, starts, gain)


def _locality_block_reuse(variant: str, trip: int, factor: int):
    hot, width = {"hot4x16": (4, 16), "hot8x32": (8, 32)}[variant]
    rng = np.random.default_rng(583_314)
    init = floats(rng, hot, width, scale=0.25)
    blocks = indices(rng, hot, trip)
    drive = floats(rng, trip, width, scale=0.1)

    def fn(init, blocks, drive):
        def body(i, carry):
            block = blocks[i]
            row = carry[block]
            return carry.at[block].set(
                jnp.tanh(row + drive[i]), unique_indices=True
            )

        return lax.fori_loop(0, trip, body, init, unroll=factor)

    return built(fn, init, blocks, drive)


def _locality_permutation_read(variant: str, trip: int, factor: int):
    size = {"n128": 128, "n512": 512}[variant]
    rng = np.random.default_rng(583_315)
    table = floats(rng, size)
    perm = permutation(rng, size)
    weights = jnp.asarray(
        rng.uniform(-1.0, 1.0, size=size) / np.sqrt(size), dtype=jnp.float32
    )

    def fn(init, table, perm, weights):
        def step(carry, _):
            values = table[perm]
            return carry * 0.875 + jnp.sum(values * weights), None

        return lax.scan(step, init, None, length=trip, unroll=factor)[0]

    return built(fn, jnp.float32(0.0), table, perm, weights)


# ---------------------------------------------------------------------------
# dependent_indexing
# ---------------------------------------------------------------------------


def _dep_pointer_chase(variant: str, trip: int, factor: int):
    size = {"n256": 256, "n1024": 1024}[variant]
    rng = np.random.default_rng(583_316)
    table = floats(rng, size)

    def fn(index0, acc0, table):
        def body(_i, state):
            index, acc = state
            value = table[index]
            offset = jnp.floor(jnp.abs(value) * 8.0).astype(jnp.int32)
            next_index = jnp.mod(index + 1 + offset, table.shape[0])
            return next_index, acc * 0.9 + value

        return lax.fori_loop(0, trip, body, (index0, acc0), unroll=factor)

    return built(fn, jnp.int32(0), jnp.float32(0.0), table)


def _dep_parallel_chains(variant: str, trip: int, factor: int):
    size = {"n256": 256, "n1024": 1024}[variant]
    lanes = 8
    rng = np.random.default_rng(583_317)
    table = floats(rng, size)
    drive = floats(rng, trip, lanes, scale=0.05)

    def fn(index0, acc0, table, drive):
        def step(state, drive_t):
            index, acc = state
            values = table[index]
            offsets = jnp.floor(jnp.abs(values) * 16.0).astype(jnp.int32)
            next_index = jnp.mod(index + 1 + offsets, table.shape[0])
            next_acc = acc * 0.9 + values + drive_t
            return (next_index, next_acc), None

        _, acc = lax.scan(step, (index0, acc0), drive, unroll=factor)[0]
        return acc

    return built(
        fn,
        jnp.arange(lanes, dtype=jnp.int32),
        jnp.zeros((lanes,), jnp.float32),
        table,
        drive,
    )


def _dep_two_level_indirection(variant: str, trip: int, factor: int):
    first_size, second_size = {
        "small": (64, 128),
        "large": (512, 1024),
    }[variant]
    rng = np.random.default_rng(583_318)
    first_table = floats(rng, first_size)
    second_table = floats(rng, second_size)

    def fn(index0, acc0, first_table, second_table):
        def body(_i, state):
            index, acc = state
            probe = first_table[index]
            second = jnp.mod(
                jnp.floor(jnp.abs(probe) * (2.0 * second_size)).astype(jnp.int32),
                second_size,
            )
            value = second_table[second]
            next_index = jnp.mod(index + 1 + second, first_size)
            return next_index, acc * 0.875 + value

        return lax.fori_loop(0, trip, body, (index0, acc0), unroll=factor)

    return built(
        fn,
        jnp.int32(0),
        jnp.float32(0.0),
        first_table,
        second_table,
    )


def _dep_linked_list(variant: str, trip: int, factor: int):
    size = {"n256": 256, "n1024": 1024}[variant]
    rng = np.random.default_rng(583_319)
    payload = floats(rng, size)
    order = rng.permutation(size).astype(np.int32)
    next_values = np.empty(size, dtype=np.int32)
    next_values[order] = np.roll(order, -1)
    next_table = jnp.asarray(next_values)

    def fn(index0, acc0, payload, next_table):
        def body(_i, state):
            index, acc = state
            value = payload[index]
            return next_table[index], acc * 0.9 + value

        return lax.fori_loop(0, trip, body, (index0, acc0), unroll=factor)

    return built(fn, jnp.int32(0), jnp.float32(0.0), payload, next_table)


def _dep_bucket_stateful(variant: str, trip: int, factor: int):
    bins = {"b16": 16, "b64": 64}[variant]
    width = 8
    rng = np.random.default_rng(583_320)
    buckets = floats(rng, bins, width)

    def fn(bucket0, acc0, buckets):
        def body(i, state):
            bucket, acc = state
            row = buckets[bucket]
            probe = jnp.sum(row) * 0.5
            delta = jnp.mod(
                jnp.floor(jnp.abs(probe) * 8.0).astype(jnp.int32), 4
            )
            next_bucket = jnp.mod(
                bucket + 1 + delta + jnp.mod(i, 3), buckets.shape[0]
            )
            return next_bucket, acc * 0.9 + jnp.tanh(row)

        return lax.fori_loop(0, trip, body, (bucket0, acc0), unroll=factor)

    return built(
        fn,
        jnp.int32(0),
        jnp.zeros((width,), jnp.float32),
        buckets,
    )


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec(
        name="gather_scalar_table",
        family="indirect_gather",
        description=(
            "Scalar table lookup with data-dependent 1-D indices feeding a "
            "scalar recurrence."
        ),
        mechanism=(
            "One random scalar load from a 1-D table per trip; "
            "carry = 0.9*carry + value*gain."
        ),
        structural_difference=(
            "Single scalar payload and scalar carry; the address stream is an "
            "unconstrained random 1-D index with no vector lanes or reduction."
        ),
        loop_kind="scan",
        shape_variants={
            "n64": "64-entry scalar table",
            "n512": "512-entry scalar table",
            "n2048": "2048-entry scalar table",
        },
        primary_variant="n512",
        factory=_gather_scalar_table,
        seed=583_300,
        provenance=_JAX_GATHER,
        tier="smoke",
        related_group="random_scalar_gather",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="gather_row_pool",
        family="indirect_gather",
        description=(
            "Row gather with in-row pooling feeding an 8-lane recurrence."
        ),
        mechanism=(
            "Each trip loads a 16-wide row at a random row index, folds it to "
            "8 lanes by pairwise addition, and applies a tanh-scaled vector "
            "update."
        ),
        structural_difference=(
            "Two-dimensional table with a row gather plus partial reduction; "
            "vector carry of 8 lanes instead of a scalar, and random row "
            "addresses."
        ),
        loop_kind="scan",
        shape_variants={
            "rows64": "64 rows x 16 features",
            "rows512": "512 rows x 16 features",
        },
        primary_variant="rows512",
        factory=_gather_row_pool,
        seed=583_301,
        provenance=_JAX_GATHER,
        related_group="random_row_gather",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="gather_multi_table",
        family="indirect_gather",
        description="Two-table gather selected per trip by a predicate.",
        mechanism=(
            "Loads rows at independent indices from two tables and selects one "
            "with a Boolean stream before a vector update."
        ),
        structural_difference=(
            "Two distinct table bases and two indexed reads per trip gated by "
            "a data-dependent selector; the loaded rows are not combined "
            "arithmetically."
        ),
        loop_kind="scan",
        shape_variants={
            "t64": "two 64-row x 4-feature tables",
            "t512": "two 512-row x 4-feature tables",
        },
        primary_variant="t512",
        factory=_gather_multi_table,
        seed=583_302,
        provenance=_ORIGINAL,
        related_group="select_gather",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="gather_duplicate_index",
        family="indirect_gather",
        description=(
            "Duplicate-row gather with replicated row addresses and a paired "
            "weighted reduction."
        ),
        mechanism=(
            "Each trip gathers 8 rows whose 4 base row indices each appear "
            "twice, adds the replicated row pairs, then reduces with per-row "
            "gains into a scalar carry."
        ),
        structural_difference=(
            "Within-trip 2-way row-address duplication stresses replicated 2D "
            "loads and combines duplicate payloads before reduction, unlike a "
            "one-dimensional element gather."
        ),
        loop_kind="scan",
        shape_variants={
            "d128": "128x4 table, 8 row loads with 2-way row duplicates",
            "d1024": "1024x4 table, 8 row loads with 2-way row duplicates",
        },
        primary_variant="d1024",
        factory=_gather_duplicate_index,
        seed=583_303,
        provenance=_ORIGINAL,
        related_group="collision_memory",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="gather_sorted_blocks",
        family="indirect_gather",
        description="Sorted contiguous block gather with pairwise pooling.",
        mechanism=(
            "Monotone sorted block starts select 8 consecutive elements per "
            "trip, pool adjacent pairs, and drive a 4-lane recurrence."
        ),
        structural_difference=(
            "Addresses advance in sorted order with static intra-block "
            "offsets and pairwise pooling; sequential spatial locality with "
            "no collisions, unlike the random scalar and row gathers."
        ),
        loop_kind="scan",
        shape_variants={
            "b256": "256-entry table, 8-wide sorted blocks",
            "b2048": "2048-entry table, 8-wide sorted blocks",
        },
        primary_variant="b2048",
        factory=_gather_sorted_blocks,
        seed=583_304,
        provenance=_ORIGINAL,
        related_group="ordered_gather",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="gather_masked_window",
        family="indirect_gather",
        description="Masked vector gather with out-of-bounds fill lanes.",
        mechanism=(
            "Eight random table loads per trip are attempted; masked lanes "
            "redirect to an out-of-bounds sentinel and return fill value zero "
            "before the weighted sum enters a scalar carry."
        ),
        structural_difference=(
            "The mask changes the address set itself: disabled lanes issue "
            "out-of-bounds fill gathers instead of in-bounds loads whose "
            "payload is later zeroed."
        ),
        loop_kind="scan",
        shape_variants={
            "m128": "128-entry table with 8 masked loads",
            "m1024": "1024-entry table with 8 masked loads",
        },
        primary_variant="m1024",
        factory=_gather_masked_window,
        seed=583_305,
        provenance=_ORIGINAL,
        related_group="masked_memory",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="scatter_unique_rows",
        family="indirect_scatter",
        description="Read-modify-write of unique table rows.",
        mechanism=(
            "Each trip reads 4 distinct rows, applies tanh(current+update) to "
            "an 8-wide row vector, and writes the rows back with a "
            "unique-index set."
        ),
        structural_difference=(
            "Row-granularity unique-index scatter with 2D read-modify-write "
            "payloads, contrasting with element-granularity sparse updates."
        ),
        loop_kind="scan",
        shape_variants={
            "r128": "128 rows of width 8, 4 unique row writes",
            "r512": "512 rows of width 8, 4 unique row writes",
        },
        primary_variant="r512",
        factory=_scatter_unique_rows,
        seed=583_306,
        provenance=_JAX_SCATTER,
        tier="smoke",
        related_group="unique_update",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="scatter_collision_add",
        family="indirect_scatter",
        description="Colliding additive scatter into a 2D grid.",
        mechanism=(
            "Twelve random (row, column) index pairs per trip may repeat; "
            "updates are added into the addressed cells after a 0.99 decay of "
            "the whole grid."
        ),
        structural_difference=(
            "Two-dimensional scatter-add with row/column collisions, rather "
            "than a one-dimensional duplicate-index accumulator."
        ),
        loop_kind="scan",
        shape_variants={
            "g16x64": "16x64 grid, 12 colliding 2D adds",
            "g32x128": "32x128 grid, 12 colliding 2D adds",
        },
        primary_variant="g32x128",
        factory=_scatter_collision_add,
        seed=583_307,
        provenance=_JAX_SCATTER,
        tier="smoke",
        related_group="collision_memory",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="scatter_min_update",
        family="indirect_scatter",
        description="Colliding min-reduction scatter.",
        mechanism=(
            "Twelve random indices per trip apply elementwise minimum updates "
            "to the addressed slots."
        ),
        structural_difference=(
            "Monotone min reduction with duplicate indices resolves repeated "
            "addresses by reduction rather than addition or overwrite; the "
            "fori_loop body indexes its streams with the counter."
        ),
        loop_kind="fori_loop",
        shape_variants={
            "s256": "256 slots, 12 colliding min updates",
            "s1024": "1024 slots, 12 colliding min updates",
        },
        primary_variant="s1024",
        factory=_scatter_min_update,
        seed=583_308,
        provenance=_ORIGINAL,
        related_group="collision_memory",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="scatter_two_tables",
        family="indirect_scatter",
        description="Predicate-directed read-modify-write across two tables.",
        mechanism=(
            "Eight unique indices per trip read both tables; a Boolean lane "
            "sends the blended tanh update to the left table when true and "
            "to the right table when false, while both tables are read and "
            "written each trip."
        ),
        structural_difference=(
            "Two simultaneous table gathers and two unique scatters per trip "
            "with predicate-directed destinations, instead of one table and "
            "one masked payload write."
        ),
        loop_kind="scan",
        shape_variants={
            "t256": "two 256-slot tables, 8 predicate-directed writes",
            "t1024": "two 1024-slot tables, 8 predicate-directed writes",
        },
        primary_variant="t1024",
        factory=_scatter_two_tables,
        seed=583_309,
        provenance=_ORIGINAL,
        related_group="split_table_memory",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="scatter_permutation",
        family="indirect_scatter",
        description="Dense block-local permutation scatter.",
        mechanism=(
            "Every trip moves every element to a fixed permuted position "
            "after a tanh update driven by a scalar stream."
        ),
        structural_difference=(
            "Full-width permutation scatter writes each slot exactly once at "
            "static unique addresses, in contrast to sparse data-dependent "
            "uniqueness."
        ),
        loop_kind="scan",
        shape_variants={
            "p128": "128-element block-local permutation",
            "p512": "512-element block-local permutation",
        },
        primary_variant="p512",
        factory=_scatter_permutation,
        seed=583_310,
        provenance=_ORIGINAL,
        related_group="permutation_memory",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="locality_stream_contiguous",
        family="locality_access",
        description="Sequential streaming read of adjacent window pairs.",
        mechanism=(
            "Each trip reads two adjacent eight-wide windows, compares their "
            "sums with tanh, and advances the start by one window width."
        ),
        structural_difference=(
            "Two adjacent contiguous windows per trip create a sequential "
            "contrast reduction, unlike a single random-window sum."
        ),
        loop_kind="scan",
        shape_variants={
            "n256": "256-element table, adjacent 8-wide window pairs",
            "n2048": "2048-element table, adjacent 8-wide window pairs",
        },
        primary_variant="n2048",
        factory=_locality_stream_contiguous,
        seed=583_311,
        provenance=_JAX_CFD,
        tier="smoke",
        related_group="sequential_locality",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="locality_stride_sample",
        family="locality_access",
        description="Fixed-stride sampled read.",
        mechanism=(
            "Each trip samples 8 elements spaced five apart from a random "
            "base and updates a vector carry."
        ),
        structural_difference=(
            "A constant nonzero stride inside each sample lowers cache-line "
            "use even though the sample pattern is fixed; distinct from "
            "contiguous windows."
        ),
        loop_kind="scan",
        shape_variants={
            "n256": "256-element table, stride-5 samples",
            "n2048": "2048-element table, stride-5 samples",
        },
        primary_variant="n2048",
        factory=_locality_stride_sample,
        seed=583_312,
        provenance=_ORIGINAL,
        related_group="strided_locality",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="locality_random_window",
        family="locality_access",
        description="Random contiguous window read.",
        mechanism=(
            "An eight-wide contiguous window is loaded from a random base per "
            "trip and drives a vector recurrence."
        ),
        structural_difference=(
            "Contiguous windows improve within-window locality, but random "
            "bases remove the sequential streaming reuse of the streaming "
            "body and differ from strided sampling."
        ),
        loop_kind="scan",
        shape_variants={
            "n256": "256-element table, random 8-wide windows",
            "n2048": "2048-element table, random 8-wide windows",
        },
        primary_variant="n2048",
        factory=_locality_random_window,
        seed=583_313,
        provenance=_ORIGINAL,
        related_group="random_window_locality",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="locality_block_reuse",
        family="locality_access",
        description="Hot-row reuse via repeated reads and writes.",
        mechanism=(
            "A tiny working set of rows is revisited by counter-indexed reads "
            "and in-place tanh updates."
        ),
        structural_difference=(
            "High temporal reuse of a few contiguous rows with read-modify-"
            "write; addresses stay in a small hot set instead of sweeping the "
            "table."
        ),
        loop_kind="fori_loop",
        shape_variants={
            "hot4x16": "4 hot rows x 16 features",
            "hot8x32": "8 hot rows x 32 features",
        },
        primary_variant="hot8x32",
        factory=_locality_block_reuse,
        seed=583_314,
        provenance=_ORIGINAL,
        related_group="reuse_locality",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="locality_permutation_read",
        family="locality_access",
        description="Full-table permutation read.",
        mechanism=(
            "Every trip reads the entire table through a fixed permutation "
            "and reduces it with static weights."
        ),
        structural_difference=(
            "All elements are consumed each trip for maximum reuse across "
            "trips, but in scrambled order rather than in windows or blocks."
        ),
        loop_kind="scan",
        shape_variants={
            "n128": "128-element permuted read",
            "n512": "512-element permuted read",
        },
        primary_variant="n512",
        factory=_locality_permutation_read,
        seed=583_315,
        provenance=_ORIGINAL,
        related_group="permutation_memory",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="dep_pointer_chase",
        family="dependent_indexing",
        description="Scalar pointer chase with payload-derived addresses.",
        mechanism=(
            "The next index is the previous index advanced by one plus a "
            "bucket of the previous loaded value; a scalar accumulator decays "
            "and adds the payload."
        ),
        structural_difference=(
            "Address transition depends on the loaded float payload, forming "
            "a single dependent chain with no index stream."
        ),
        loop_kind="fori_loop",
        shape_variants={
            "n256": "256-entry payload table",
            "n1024": "1024-entry payload table",
        },
        primary_variant="n1024",
        factory=_dep_pointer_chase,
        seed=583_316,
        provenance=_JRAPH,
        tier="smoke",
        related_group="dependent_chain",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="dep_parallel_chains",
        family="dependent_indexing",
        description="Eight parallel dependent pointer chains.",
        mechanism=(
            "Eight lane pointers each advance using their own loaded value "
            "while an eight-lane accumulator integrates the payload."
        ),
        structural_difference=(
            "Vector pointer state and vector carry run eight independent "
            "payload-derived chains per trip instead of one scalar chain."
        ),
        loop_kind="scan",
        shape_variants={
            "n256": "256-entry table, 8 parallel chains",
            "n1024": "1024-entry table, 8 parallel chains",
        },
        primary_variant="n1024",
        factory=_dep_parallel_chains,
        seed=583_317,
        provenance=_ORIGINAL,
        related_group="dependent_chain",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="dep_two_level_indirection",
        family="dependent_indexing",
        description="Two-level dependent indirection.",
        mechanism=(
            "A first load yields the address for a second load; the chain "
            "index combines both levels and a scalar accumulator integrates "
            "the payload."
        ),
        structural_difference=(
            "Load-to-address dependency inside one trip gives two dependent "
            "loads, not just a payload-derived index update."
        ),
        loop_kind="fori_loop",
        shape_variants={
            "small": "64-entry first level, 128-entry second level",
            "large": "512-entry first level, 1024-entry second level",
        },
        primary_variant="large",
        factory=_dep_two_level_indirection,
        seed=583_318,
        provenance=_ORIGINAL,
        related_group="dependent_chain",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="dep_linked_list",
        family="dependent_indexing",
        description="Linked-list pointer chase over a static next-pointer table.",
        mechanism=(
            "An integer next-pointer table walks a fixed cycle while payload "
            "loads feed a decaying scalar accumulator."
        ),
        structural_difference=(
            "Address transitions come from an indirection table rather than "
            "payload arithmetic or buckets, giving pointer-chasing traversal."
        ),
        loop_kind="fori_loop",
        shape_variants={
            "n256": "256-node list",
            "n1024": "1024-node list",
        },
        primary_variant="n1024",
        factory=_dep_linked_list,
        seed=583_319,
        provenance=_ORIGINAL,
        related_group="dependent_chain",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
    CorpusSpec(
        name="dep_bucket_stateful",
        family="dependent_indexing",
        description="Counter-assisted bucket traversal.",
        mechanism=(
            "Rows are read from a bucket table; the next bucket depends on the "
            "previous row statistic and the loop counter, while a vector "
            "accumulator integrates tanh(row)."
        ),
        structural_difference=(
            "Loop state (the counter) and a row statistic both steer a "
            "two-dimensional bucket traversal with vector carry; there is no "
            "scalar pointer chain."
        ),
        loop_kind="fori_loop",
        shape_variants={
            "b16": "16 buckets x 8 features",
            "b64": "64 buckets x 8 features",
        },
        primary_variant="b64",
        factory=_dep_bucket_stateful,
        seed=583_320,
        provenance=_ORIGINAL,
        related_group="stateful_dependent",
        trip_counts=CORPUS_TRIP_COUNTS,
    ),
)
