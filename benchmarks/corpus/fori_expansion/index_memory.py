"""Index and memory access independent fori_loop workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats, indices
from benchmarks.corpus.spec import CorpusSpec, Provenance

# 1. idx_permutation_cycle_walk: Permutation Orbit Cycle Walk with Visited Mask [fresh_eval]
def _perm_cycle_factory(variant: str, trip: int, factor: int):
    n = {"n64": 64, "n128": 128, "n256": 256}[variant]
    rng = np.random.default_rng(583_871)
    perm_table = jnp.asarray(rng.permutation(n), dtype=jnp.int32)
    visited_init = jnp.zeros((n,), dtype=jnp.bool_)
    init_curr = jnp.int32(0)

    def fn(curr_init: jax.Array, vis_init: jax.Array, perm: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            curr, vis, cycle_count = carry
            next_node = perm[curr]
            already_vis = vis[curr]
            new_vis = vis.at[curr].set(True)
            # If already visited, jump to next unvisited or reset
            jump_node = (curr + 1) % n
            next_curr = jnp.where(already_vis, jump_node, next_node)
            new_cycles = cycle_count + jnp.where(already_vis, 1, 0)
            return (next_curr, new_vis, new_cycles)

        return lax.fori_loop(0, trip, body, (curr_init, vis_init, jnp.int32(0)), unroll=factor)

    return built(fn, init_curr, visited_init, perm_table)


# 2. idx_double_indirect_gather: Two-Level B-Tree Columnar Leaf Retrieval
def _double_indirect_factory(variant: str, trip: int, factor: int):
    n_leaves, d = {"l64d8": (64, 8), "l128d8": (128, 8), "l256d16": (256, 16)}[variant]
    rng = np.random.default_rng(583_872)
    leaves = floats(rng, n_leaves, d, scale=1.0)
    pointers = indices(rng, n_leaves, trip)
    query_offsets = indices(rng, trip, trip)
    init_accum = jnp.zeros((d,), dtype=jnp.float32)

    def fn(accum_init: jax.Array, leaf_table: jax.Array, ptrs: jax.Array, queries: jax.Array):
        def body(i: int, carry: jax.Array):
            ptr_idx = queries[i]
            leaf_idx = ptrs[ptr_idx]
            retrieved = leaf_table[leaf_idx]
            return carry * 0.95 + retrieved

        return lax.fori_loop(0, trip, body, accum_init, unroll=factor)

    return built(fn, init_accum, leaves, pointers, query_offsets)


# 3. idx_gather_transform_scatter: Particle-in-Cell Gather-Weight-Scatter Pipeline
def _pic_pipeline_factory(variant: str, trip: int, factor: int):
    grid = {"g16": 16, "g32": 32, "g64": 64}[variant]
    rng = np.random.default_rng(583_873)
    particles_x = jnp.asarray(rng.uniform(1.0, grid - 2.0, size=trip), dtype=jnp.float32)
    particles_y = jnp.asarray(rng.uniform(1.0, grid - 2.0, size=trip), dtype=jnp.float32)
    charges = floats(rng, trip, scale=1.0)
    init_grid = jnp.zeros((grid, grid), dtype=jnp.float32)

    def fn(grid_init: jax.Array, px: jax.Array, py: jax.Array, q_arr: jax.Array):
        def body(i: int, carry: jax.Array):
            x = px[i]
            y = py[i]
            q = q_arr[i]
            x0 = jnp.int32(x)
            y0 = jnp.int32(y)
            fx = x - x0.astype(jnp.float32)
            fy = y - y0.astype(jnp.float32)
            w00 = (1.0 - fx) * (1.0 - fy) * q
            w10 = fx * (1.0 - fy) * q
            w01 = (1.0 - fx) * fy * q
            w11 = fx * fy * q
            # Scatter 4 bilinear weights to grid
            new_carry = carry.at[x0, y0].add(w00).at[x0 + 1, y0].add(w10).at[x0, y0 + 1].add(w01).at[x0 + 1, y0 + 1].add(w11)
            return new_carry

        return lax.fori_loop(0, trip, body, grid_init, unroll=factor)

    return built(fn, init_grid, particles_x, particles_y, charges)


# 4. idx_overlapping_block_update: 2D Overlapping Patch Stitching with Blending
def _overlapping_patch_factory(variant: str, trip: int, factor: int):
    canvas_dim, p_dim = {"c32p8": (32, 8), "c64p8": (64, 8), "c64p16": (64, 16)}[variant]
    rng = np.random.default_rng(583_874)
    patches = floats(rng, trip, p_dim, p_dim, scale=1.0)
    origins_x = indices(rng, canvas_dim - p_dim, trip)
    origins_y = indices(rng, canvas_dim - p_dim, trip)
    blend_w = jnp.float32(0.5)
    init_canvas = jnp.zeros((canvas_dim, canvas_dim), dtype=jnp.float32)

    def fn(canvas_init: jax.Array, patch_stream: jax.Array, ox: jax.Array, oy: jax.Array, w_blend: jax.Array):
        def body(i: int, carry: jax.Array):
            patch = patch_stream[i]
            x0 = ox[i]
            y0 = oy[i]
            # Dynamic slice and blend
            existing = lax.dynamic_slice(carry, (x0, y0), (p_dim, p_dim))
            blended = existing * (1.0 - w_blend) + patch * w_blend
            return lax.dynamic_update_slice(carry, blended, (x0, y0))

        return lax.fori_loop(0, trip, body, canvas_init, unroll=factor)

    return built(fn, init_canvas, patches, origins_x, origins_y, blend_w)


# 5. idx_compressed_sparse_lookup: In-Loop Binary Search over Sorted Index Arrays
def _binary_search_factory(variant: str, trip: int, factor: int):
    n_keys = {"k64": 64, "k128": 128, "k256": 256}[variant]
    rng = np.random.default_rng(583_875)
    sorted_keys = jnp.asarray(np.sort(rng.uniform(0.0, 100.0, size=n_keys)), dtype=jnp.float32)
    values = floats(rng, n_keys, scale=1.0)
    queries = jnp.asarray(rng.uniform(0.0, 100.0, size=trip), dtype=jnp.float32)
    init_sum = jnp.float32(0.0)

    def fn(sum_init: jax.Array, keys: jax.Array, vals: jax.Array, q_stream: jax.Array):
        n_bits = int(np.ceil(np.log2(n_keys)))
        def body(i: int, carry: jax.Array):
            q = q_stream[i]
            # Binary search step via unrolled comparison
            def bs_step(b: int, range_lo: jax.Array):
                step = n_keys >> (b + 1)
                mid = range_lo + step
                mid_clamped = jnp.minimum(n_keys - 1, mid)
                cond = keys[mid_clamped] <= q
                return jnp.where(cond, mid, range_lo)

            final_idx = lax.fori_loop(0, n_bits, bs_step, jnp.int32(0))
            return carry + vals[final_idx]

        return lax.fori_loop(0, trip, body, sum_init, unroll=factor)

    return built(fn, init_sum, sorted_keys, values, queries)


# 6. idx_strided_transpose_copy: Multi-Dimensional Strided Coordinate Remapping
def _strided_transpose_factory(variant: str, trip: int, factor: int):
    tile_size = {"t8": 8, "t16": 16, "t32": 32}[variant]
    rng = np.random.default_rng(583_876)
    source_mat = floats(rng, tile_size * 2, tile_size * 2, scale=1.0)
    origin_coords = indices(rng, tile_size, trip, 2)
    init_tile = jnp.zeros((tile_size, tile_size), dtype=jnp.float32)

    def fn(tile_init: jax.Array, src: jax.Array, origins: jax.Array):
        def body(i: int, carry: jax.Array):
            ox, oy = origins[i, 0], origins[i, 1]
            sub = lax.dynamic_slice(src, (ox, oy), (tile_size, tile_size))
            return carry * 0.9 + sub.T

        return lax.fori_loop(0, trip, body, tile_init, unroll=factor)

    return built(fn, init_tile, source_mat, origin_coords)


# 7. idx_circular_fifo_buffer: Decoupled Circular FIFO Queue Buffer [fresh_eval]
def _circular_fifo_factory(variant: str, trip: int, factor: int):
    cap, d = {"c16d4": (16, 4), "c32d4": (32, 4), "c32d8": (32, 8)}[variant]
    rng = np.random.default_rng(583_877)
    items_in = floats(rng, trip, d, scale=1.0)
    # Commands: 1 = push, 0 = pop
    commands = jnp.asarray(rng.choice([0, 1], size=trip, p=[0.3, 0.7]), dtype=jnp.int32)
    init_buf = jnp.zeros((cap, d), dtype=jnp.float32)
    init_ptrs = jnp.zeros((3,), dtype=jnp.int32) # [head, tail, count]

    def fn(buf_init: jax.Array, ptrs_init: jax.Array, items: jax.Array, cmds: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            buf, ptrs, accum = carry
            head, tail, count = ptrs[0], ptrs[1], ptrs[2]
            cmd = cmds[i]
            # Push logic
            can_push = (cmd == 1) & (count < cap)
            new_buf = jnp.where(can_push, buf.at[tail].set(items[i]), buf)
            new_tail = (tail + jnp.where(can_push, 1, 0)) % cap
            # Pop logic
            can_pop = (cmd == 0) & (count > 0)
            popped_item = buf[head]
            new_head = (head + jnp.where(can_pop, 1, 0)) % cap
            new_count = count + jnp.where(can_push, 1, 0) - jnp.where(can_pop, 1, 0)
            new_ptrs = jnp.stack([new_head, new_tail, new_count])
            new_accum = accum + jnp.where(can_pop, popped_item, 0.0)
            return (new_buf, new_ptrs, new_accum)

        return lax.fori_loop(0, trip, body, (buf_init, ptrs_init, jnp.zeros((d,), dtype=jnp.float32)), unroll=factor)

    return built(fn, init_buf, init_ptrs, items_in, commands)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec("idx_permutation_cycle_walk", "index_memory", "Permutation Orbit Cycle Walk with Visited Mask",
               "index_permutation_orbit", "In-loop cycle detection tracking orbit closures and unvisited jumps",
               "fori_loop", {"n64": "64 elements", "n128": "128 elements", "n256": "256 elements"}, "n128",
               _perm_cycle_factory, 583_871, Provenance("public-inspired", "Knuth (1997) TAOCP Vol 1", "Public Domain", "Permutation orbit cycle walk"),
               tier="full", related_group="idx_orbit_eval"),
    CorpusSpec("idx_double_indirect_gather", "index_memory", "Two-Level B-Tree Columnar Leaf Retrieval",
               "index_nested_pointer", "Multi-lane vector record retrieval via pointer indirection",
               "fori_loop", {"l64d8": "64 leaves dim 8", "l128d8": "128 leaves dim 8", "l256d16": "256 leaves dim 16"}, "l128d8",
               _double_indirect_factory, 583_872, Provenance("original", "Columnar nested pointer lookup", "Internal", "Two-level indirect gather"),
               tier="full", related_group="idx_nested_ptr"),
    CorpusSpec("idx_gather_transform_scatter", "index_memory", "Particle-in-Cell Gather-Weight-Scatter Pipeline",
               "index_pic_pipeline", "Bilinear interpolation scattering weighted fractions to adjacent grid cells",
               "fori_loop", {"g16": "grid 16", "g32": "grid 32", "g64": "grid 64"}, "g32",
               _pic_pipeline_factory, 583_873, Provenance("public-inspired", "Birdsall & Langdon (1985) Plasma Physics", "Public Domain", "PIC bilinear charge deposition"),
               tier="full", related_group="idx_pic"),
    CorpusSpec("idx_overlapping_block_update", "index_memory", "2D Overlapping Patch Stitching with Partition of Unity Blending",
               "index_patch_blend", "2D weighted blend accumulation rather than 1D non-overlapping overwrites",
               "fori_loop", {"c32p8": "canvas 32 patch 8", "c64p8": "canvas 64 patch 8", "c64p16": "canvas 64 patch 16"}, "c64p8",
               _overlapping_patch_factory, 583_874, Provenance("original", "Image patch blending engine", "Internal", "Overlapping 2D patch blending"),
               tier="smoke", related_group="idx_patch_blend"),
    CorpusSpec("idx_compressed_sparse_lookup", "index_memory", "In-Loop Binary Search over Sorted Index Arrays",
               "index_binary_search", "Discrete binary search probing memory-backed sorted keys",
               "fori_loop", {"k64": "64 keys", "k128": "128 keys", "k256": "256 keys"}, "k128",
               _binary_search_factory, 583_875, Provenance("original", "Columnar binary search probe", "Internal", "In-loop binary search lookup"),
               tier="full", related_group="idx_bsearch"),
    CorpusSpec("idx_strided_transpose_copy", "index_memory", "Multi-Dimensional Strided Coordinate Remapping",
               "index_strided_transpose", "Dynamic slice and transposed memory striding kernels",
               "fori_loop", {"t8": "tile 8", "t16": "tile 16", "t32": "tile 32"}, "t16",
               _strided_transpose_factory, 583_876, Provenance("original", "Tensor transpose kernel", "Internal", "Strided transpose copy kernel"),
               tier="full", related_group="idx_transpose"),
    CorpusSpec("idx_circular_fifo_buffer", "index_memory", "Decoupled Circular FIFO Queue Buffer",
               "index_circular_fifo", "Decouples read and write pointers with wrap-around modulo arithmetic",
               "fori_loop", {"c16d4": "cap 16 dim 4", "c32d4": "cap 32 dim 4", "c32d8": "cap 32 dim 8"}, "c32d4",
               _circular_fifo_factory, 583_877, Provenance("original", "Streaming circular FIFO queue", "Internal", "Circular FIFO queue state transition"),
               tier="full", related_group="idx_fifo_eval"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
