"""Domain 2: Locality and indexing workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats, indices
from benchmarks.corpus.spec import CORPUS_TRIP_COUNTS, CorpusSpec, Provenance


# 1. loc_strided_transpose (train): Strided 2D matrix transpose copy with coordinate offset
def _strided_transpose_factory(variant: str, trip: int, factor: int):
    dim = 32 if variant == "small" else 64
    rng = np.random.default_rng(583_201)
    base_src = floats(rng, dim, dim, scale=1.0)
    # Coordinate indexing stream over trips
    row_idx = (np.arange(1024, dtype=np.int32) * 7) % dim
    col_idx = (np.arange(1024, dtype=np.int32) * 13) % dim
    rows = jnp.asarray(row_idx[:trip])
    cols = jnp.asarray(col_idx[:trip])
    init_dst = jnp.zeros((dim, dim), dtype=jnp.float32)

    def fn(dst: jax.Array, src: jax.Array, r_idx: jax.Array, c_idx: jax.Array):
        def body(i: int, target: jax.Array):
            r = r_idx[i]
            c = c_idx[i]
            val = src[r, c]
            # Transposed store target[c, r] = val
            return target.at[c, r].set(val)

        return lax.fori_loop(0, trip, body, dst, unroll=factor)

    return built(fn, init_dst, base_src, rows, cols)


# 2. loc_double_indirect (train): Two-level pointer dereferencing table lookup A[B[i]]
def _double_indirect_factory(variant: str, trip: int, factor: int):
    tbl_size = 256 if variant == "small" else 1024
    rng = np.random.default_rng(583_202)
    ptr1_np = rng.integers(0, tbl_size, size=tbl_size, dtype=np.int32)
    ptr2_np = rng.integers(0, tbl_size, size=tbl_size, dtype=np.int32)
    payload_np = rng.uniform(-1.0, 1.0, size=tbl_size).astype(np.float32)
    ptr1 = jnp.asarray(ptr1_np)
    ptr2 = jnp.asarray(ptr2_np)
    payload = jnp.asarray(payload_np)

    base_idx = (np.arange(1024, dtype=np.int32) * 17) % tbl_size
    query_indices = jnp.asarray(base_idx[:trip])
    init_acc = jnp.float32(0.0)

    def fn(acc_init: jax.Array, p1: jax.Array, p2: jax.Array, data: jax.Array, q_idx: jax.Array):
        def body(i: int, acc: jax.Array):
            k0 = q_idx[i]
            k1 = p1[k0]
            k2 = p2[k1]
            val = data[k2]
            return acc * 0.99 + val

        return lax.fori_loop(0, trip, body, acc_init, unroll=factor)

    return built(fn, init_acc, ptr1, ptr2, payload, query_indices)


# 3. loc_morton_z_order_lookup (train): Morton Z-order curve coordinate mapping to 2D grid
def _morton_z_order_factory(variant: str, trip: int, factor: int):
    dim = 32 if variant == "small" else 128
    rng = np.random.default_rng(583_203)
    grid = floats(rng, dim, dim, scale=1.0)
    # Sequence of 1D Morton codes
    mask_bits = 5 if variant == "small" else 7
    max_code = 1 << (2 * mask_bits)
    base_codes = (np.arange(1024, dtype=np.int32) * 31) % max_code
    codes = jnp.asarray(base_codes[:trip])
    init_sum = jnp.float32(0.0)

    def fn(acc_init: jax.Array, texture: jax.Array, morton_codes: jax.Array):
        def body(i: int, acc: jax.Array):
            c = morton_codes[i]
            # Bit demultiplexing to reconstruct 2D (x, y)
            # x gets even bits, y gets odd bits
            x = (c & 1) | ((c >> 1) & 2) | ((c >> 2) & 4) | ((c >> 3) & 8) | ((c >> 4) & 16)
            y = ((c >> 1) & 1) | ((c >> 2) & 2) | ((c >> 3) & 4) | ((c >> 4) & 8) | ((c >> 5) & 16)
            x_clamp = jnp.remainder(x, dim)
            y_clamp = jnp.remainder(y, dim)
            val = texture[y_clamp, x_clamp]
            return acc * 0.98 + val

        return lax.fori_loop(0, trip, body, acc_init, unroll=factor)

    return built(fn, init_sum, grid, codes)


# 4. loc_runlength_compressed_scan (train): Scan over RLE compressed segments
def _runlength_compressed_factory(variant: str, trip: int, factor: int):
    segments = 64 if variant == "small" else 256
    rng = np.random.default_rng(583_204)
    counts = jnp.asarray(rng.integers(1, 16, size=segments, dtype=np.int32))
    values = floats(rng, segments, scale=1.0)
    offsets = jnp.cumsum(counts)
    total_len = int(np.sum(counts))
    base_queries = (np.arange(1024, dtype=np.int32) * 23) % total_len
    queries = jnp.asarray(base_queries[:trip])
    init_acc = jnp.float32(0.0)

    def fn(acc_init: jax.Array, cum_offsets: jax.Array, vals: jax.Array, q_stream: jax.Array):
        def body(i: int, acc: jax.Array):
            pos = q_stream[i]
            # Branchless upper bound index search over compressed segment offsets
            seg_idx = jnp.sum(cum_offsets <= pos)
            seg_clamped = jnp.clip(seg_idx, 0, segments - 1)
            val = vals[seg_clamped]
            return acc * 0.95 + val

        return lax.fori_loop(0, trip, body, acc_init, unroll=factor)

    return built(fn, init_acc, offsets, values, queries)


# 5. eval_loc_btree_search_step (eval): Implicit Eytzinger layout B-tree branchless search
def _btree_search_factory(variant: str, trip: int, factor: int):
    depth = 6 if variant == "small" else 10
    num_nodes = (1 << depth) - 1
    # Sorted unique keys placed in 1-based Eytzinger array
    sorted_keys = np.linspace(-10.0, 10.0, num_nodes, dtype=np.float32)
    eytzinger_keys = np.zeros(num_nodes + 1, dtype=np.float32)
    
    # Fill Eytzinger in-order layout
    def fill(idx: int, it):
        if idx <= num_nodes:
            fill(2 * idx, it)
            eytzinger_keys[idx] = next(it)
            fill(2 * idx + 1, it)

    fill(1, iter(sorted_keys))
    tree_keys = jnp.asarray(eytzinger_keys)

    rng = np.random.default_rng(583_205)
    base_queries = floats(rng, 1024, scale=12.0)
    queries = base_queries[:trip]
    init_indices = jnp.int32(1)

    def fn(idx_init: jax.Array, tree: jax.Array, q_in: jax.Array):
        def body(i: int, curr_node: jax.Array):
            q = q_in[i]
            # Branchless traversal step: if q > tree[node], node = 2*node + 1 else 2*node
            node_key = tree[curr_node]
            go_right = q > node_key
            next_node = curr_node * 2 + jnp.where(go_right, 1, 0)
            # Wrap around if out of bounds to keep infinite stream traversing
            wrapped_node = jnp.where(next_node <= num_nodes, next_node, 1)
            return wrapped_node

        return lax.fori_loop(0, trip, body, idx_init, unroll=factor)

    return built(fn, init_indices, tree_keys, queries)


# 6. eval_loc_bresenham_voxel_walk (eval): 3D Amanatides-Woo fast voxel ray traversal
def _bresenham_voxel_factory(variant: str, trip: int, factor: int):
    dim = 32 if variant == "small" else 64
    rng = np.random.default_rng(583_206)
    # 3D grid data: density values
    density = floats(rng, dim, dim, dim, scale=1.0)
    
    # Ray origin and direction
    ray_pos = jnp.array([2.0, 3.0, 1.0], dtype=jnp.float32)
    ray_dir = jnp.array([0.707, 0.5, 0.5], dtype=jnp.float32)
    
    # State: voxel_coords (3,), t_max (3,), t_delta (3,), step (3,), accumulated_density (scalar)
    step = jnp.sign(ray_dir).astype(jnp.int32)
    t_delta = jnp.abs(1.0 / (ray_dir + 1e-6))
    t_max = t_delta * 0.5
    voxel = jnp.floor(ray_pos).astype(jnp.int32)
    init_acc = jnp.float32(0.0)

    def fn(vox_init: jax.Array, tm_init: jax.Array, acc_init: jax.Array, grid: jax.Array, t_d: jax.Array, stp: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            vox, t_m, acc = carry
            min_axis = jnp.argmin(t_m)
            new_vox = vox.at[min_axis].add(stp[min_axis])
            new_vox_wrap = jnp.remainder(new_vox, dim)
            new_t_m = t_m.at[min_axis].add(t_d[min_axis])
            val = grid[new_vox_wrap[0], new_vox_wrap[1], new_vox_wrap[2]]
            new_acc = acc * 0.99 + val
            return (new_vox_wrap, new_t_m, new_acc)

        return lax.fori_loop(0, trip, body, (vox_init, tm_init, acc_init), unroll=factor)

    return built(fn, voxel, t_max, init_acc, density, t_delta, step)


# 7. eval_loc_quadtree_point_query (eval): Quadtree hierarchical point search
def _quadtree_point_query_factory(variant: str, trip: int, factor: int):
    depth = 5 if variant == "small" else 7
    total_nodes = (4 ** depth - 1) // 3
    rng = np.random.default_rng(583_207)
    # Quadtree centers (X, Y) and values
    node_centers = floats(rng, total_nodes, 2, scale=10.0)
    node_values = floats(rng, total_nodes, scale=1.0)
    
    base_points = floats(rng, 1024, 2, scale=10.0)
    query_points = base_points[:trip]
    init_acc = jnp.float32(0.0)

    def fn(acc_init: jax.Array, centers: jax.Array, vals: jax.Array, q_pts: jax.Array):
        def body(i: int, acc: jax.Array):
            pt = q_pts[i]
            # Step down one quadtree level branchlessly from current root
            node_idx = 0
            # Quadtree quadrant index: bit0 = (x >= cx), bit1 = (y >= cy)
            c = centers[node_idx]
            quadrant = jnp.where(pt[0] >= c[0], 1, 0) + jnp.where(pt[1] >= c[1], 2, 0)
            child_idx = 4 * node_idx + 1 + quadrant
            child_clamped = jnp.clip(child_idx, 0, total_nodes - 1)
            val = vals[child_clamped]
            return acc * 0.97 + val

        return lax.fori_loop(0, trip, body, acc_init, unroll=factor)

    return built(fn, init_acc, node_centers, node_values, query_points)


# 8. eval_loc_hilbert_curve_gather (eval): 2D Hilbert space-filling curve gather
def _hilbert_curve_gather_factory(variant: str, trip: int, factor: int):
    order = 4 if variant == "small" else 6
    dim = 1 << order
    rng = np.random.default_rng(583_208)
    table = floats(rng, dim, dim, scale=1.0)
    
    max_t = dim * dim
    base_t = (np.arange(1024, dtype=np.int32) * 41) % max_t
    t_stream = jnp.asarray(base_t[:trip])
    init_acc = jnp.float32(0.0)

    def fn(acc_init: jax.Array, data: jax.Array, t_indices: jax.Array):
        def body(i: int, acc: jax.Array):
            t = t_indices[i]
            # Convert 1D Hilbert index t to 2D (x, y) coordinates branchlessly
            # Unrolled bits for order
            x = 0
            y = 0
            s = 1
            for b in range(order):
                rx = (t >> 1) & 1
                ry = (t ^ rx) & 1
                # Rotate coordinates
                new_x = jnp.where(ry == 0, jnp.where(rx == 1, s - 1 - x, y), x) + s * rx
                new_y = jnp.where(ry == 0, jnp.where(rx == 1, s - 1 - y, x), y) + s * ry
                x = new_x
                y = new_y
                t = t >> 2
                s = s << 1
            x_clamp = jnp.clip(x, 0, dim - 1)
            y_clamp = jnp.clip(y, 0, dim - 1)
            val = data[y_clamp, x_clamp]
            return acc * 0.99 + val

        return lax.fori_loop(0, trip, body, acc_init, unroll=factor)

    return built(fn, init_acc, table, t_stream)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec(
        name="loc_strided_transpose",
        family="locality_indexing",
        description="Strided 2D matrix transpose copy with coordinate offset indexing",
        mechanism="strided_matrix_coordinate_indexing",
        structural_difference="2D matrix coordinate offset calculations and strided scattered transpose writes",
        loop_kind="fori_loop",
        shape_variants={
            "small": "tile=32x32 (4 KB)",
            "large": "tile=64x64 (16 KB)",
        },
        primary_variant="small",
        factory=_strided_transpose_factory,
        seed=583_201,
        provenance=Provenance(
            kind="original",
            source="Adapted from idx_strided_transpose_copy historical lineage",
            license="MIT",
            adaptation="Expanded to dual tile sizes (32, 64) with deterministic coordinate generator",
        ),
        related_group="strided_transpose",
    ),
    CorpusSpec(
        name="loc_double_indirect",
        family="locality_indexing",
        description="Two-level pointer dereferencing table lookup A[B[i]]",
        mechanism="nested_pointer_indirection",
        structural_difference="Coupled sequential two-level memory dereference before payload load",
        loop_kind="fori_loop",
        shape_variants={
            "small": "table_size=256 (1 KB)",
            "large": "table_size=1024 (4 KB)",
        },
        primary_variant="small",
        factory=_double_indirect_factory,
        seed=583_202,
        provenance=Provenance(
            kind="original",
            source="Adapted from idx_double_indirect_gather historical lineage",
            license="MIT",
            adaptation="Standardized table dimensions and deterministic query prefix",
        ),
        related_group="double_indirect_gather",
    ),
    CorpusSpec(
        name="loc_morton_z_order_lookup",
        family="locality_indexing",
        description="Morton Z-order curve bit-interleaving coordinate mapping to 2D texture grid",
        mechanism="morton_z_order_bit_demux",
        structural_difference="Bit-level interleave arithmetic to compute non-linear locality mapping",
        loop_kind="fori_loop",
        shape_variants={
            "small": "grid=32x32 (4 KB)",
            "large": "grid=128x128 (64 KB)",
        },
        primary_variant="small",
        factory=_morton_z_order_factory,
        seed=583_203,
        provenance=Provenance(
            kind="original",
            source="Morton, G. M. (1966). A computer Oriented Geodetic Data Base; and a New Technique in File Sequencing",
            license="MIT",
            adaptation="Z-order bit manipulation and 2D spatial texture lookup",
        ),
        related_group="morton_z_order",
    ),
    CorpusSpec(
        name="loc_runlength_compressed_scan",
        family="locality_indexing",
        description="Scan over run-length encoded compressed segments with cumulative index search",
        mechanism="rle_compressed_index_lookup",
        structural_difference="Cumulative offset thresholding over compressed runs to dereference payload values",
        loop_kind="fori_loop",
        shape_variants={
            "small": "segments=64 (512 B)",
            "large": "segments=256 (2 KB)",
        },
        primary_variant="small",
        factory=_runlength_compressed_factory,
        seed=583_204,
        provenance=Provenance(
            kind="original",
            source="Columnar compressed database indexing algorithms",
            license="MIT",
            adaptation="Branchless RLE offset search with vector thresholding",
        ),
        related_group="runlength_compressed_scan",
    ),
    CorpusSpec(
        name="eval_loc_btree_search_step",
        family="eval_locality_indexing",
        description="Implicit cache-oblivious Eytzinger layout B-tree branchless search step",
        mechanism="cache_oblivious_eytzinger_search",
        structural_difference="Implicit binary tree indexing: left=2k, right=2k+1 in flat 1-based array",
        loop_kind="fori_loop",
        shape_variants={
            "small": "depth=6 (63 nodes, 252 B)",
            "large": "depth=10 (1023 nodes, 4092 B)",
        },
        primary_variant="small",
        factory=_btree_search_factory,
        seed=583_205,
        provenance=Provenance(
            kind="public-inspired",
            source="Brodal, G. S., et al. (2012). Cache-oblivious searching on UMA architectures. ACM TALG",
            license="Academic reference",
            adaptation="Branchless Eytzinger array layout search iteration",
        ),
        related_group="eval_btree_search",
    ),
    CorpusSpec(
        name="eval_loc_bresenham_voxel_walk",
        family="eval_locality_indexing",
        description="3D Amanatides-Woo fast voxel ray traversal grid marching step",
        mechanism="amanatides_woo_voxel_traversal",
        structural_difference="3D ray marching with parametric t-max axis selection and discrete voxel updating",
        loop_kind="fori_loop",
        shape_variants={
            "small": "grid=32x32x32 (128 KB)",
            "large": "grid=64x64x64 (1024 KB)",
        },
        primary_variant="small",
        factory=_bresenham_voxel_factory,
        seed=583_206,
        provenance=Provenance(
            kind="public-inspired",
            source="Amanatides, J., & Woo, A. (1987). A fast voxel traversal algorithm for ray tracing. Eurographics",
            license="Public domain / academic",
            adaptation="3D grid DDA ray marcher with axis-minimum stepping",
        ),
        related_group="eval_bresenham_voxel",
    ),
    CorpusSpec(
        name="eval_loc_quadtree_point_query",
        family="eval_locality_indexing",
        description="Quadtree hierarchical spatial point lookup descending quadrants",
        mechanism="quadtree_spatial_point_query",
        structural_difference="Hierarchical 2D quadrant index computation: child = 4*k + 1 + quad",
        loop_kind="fori_loop",
        shape_variants={
            "small": "depth=5 (341 nodes, 4 KB)",
            "large": "depth=7 (5461 nodes, 65 KB)",
        },
        primary_variant="small",
        factory=_quadtree_point_query_factory,
        seed=583_207,
        provenance=Provenance(
            kind="public-inspired",
            source="Finkel, R. A., & Bentley, J. L. (1974). Quad trees: A data structure for retrieval. Acta Inf.",
            license="Academic reference",
            adaptation="Branchless quadrant descent for spatial coordinate queries",
        ),
        related_group="eval_quadtree_point_query",
    ),
    CorpusSpec(
        name="eval_loc_hilbert_curve_gather",
        family="eval_locality_indexing",
        description="2D Hilbert space-filling curve index calculation and locality gather",
        mechanism="hilbert_space_filling_curve_mapping",
        structural_difference="Recursive quadrant rotation and bit-shifting mapping 1D index to 2D coordinates",
        loop_kind="fori_loop",
        shape_variants={
            "small": "order=4 (grid 16x16 = 1 KB)",
            "large": "order=6 (grid 64x64 = 16 KB)",
        },
        primary_variant="small",
        factory=_hilbert_curve_gather_factory,
        seed=583_208,
        provenance=Provenance(
            kind="public-inspired",
            source="Hilbert, D. (1891). Ueber die stetige Abbildung einer Linie auf ein Flachenstuck. Math. Annalen",
            license="Public domain",
            adaptation="Self-contained bit-twiddling 1D to 2D Hilbert curve coordinate transformer",
        ),
        related_group="eval_hilbert_curve_gather",
    ),
)
