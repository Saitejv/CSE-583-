"""Domain 3: Writes and state size workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats, indices
from benchmarks.corpus.spec import CORPUS_TRIP_COUNTS, CorpusSpec, Provenance


# 1. write_patch_blend (train): Overlapping 2D patch update blending into canvas
def _patch_blend_factory(variant: str, trip: int, factor: int):
    dim = 32 if variant == "small" else 64
    patch_dim = 8 if variant == "small" else 16
    rng = np.random.default_rng(583_301)
    init_canvas = jnp.zeros((dim, dim), dtype=jnp.float32)
    
    # Pre-generate stream of patch positions (top-left corners)
    max_offset = dim - patch_dim
    base_rows = (np.arange(1024, dtype=np.int32) * 5) % max_offset
    base_cols = (np.arange(1024, dtype=np.int32) * 11) % max_offset
    row_starts = jnp.asarray(base_rows[:trip])
    col_starts = jnp.asarray(base_cols[:trip])
    
    # Patch values stream: shape (trip, patch_dim, patch_dim)
    base_patches = floats(rng, 1024, patch_dim, patch_dim, scale=0.2)
    patches = base_patches[:trip]

    def fn(canvas: jax.Array, p_stream: jax.Array, r_idx: jax.Array, c_idx: jax.Array):
        def body(i: int, target: jax.Array):
            r = r_idx[i]
            c = c_idx[i]
            p = p_stream[i]
            # Dynamic slice read, blend, dynamic slice update
            curr_patch = lax.dynamic_slice(target, (r, c), (patch_dim, patch_dim))
            blended = curr_patch * 0.8 + p
            return lax.dynamic_update_slice(target, blended, (r, c))

        return lax.fori_loop(0, trip, body, canvas, unroll=factor)

    return built(fn, init_canvas, patches, row_starts, col_starts)


# 2. write_scatter_accumulate (train): Dynamic update slice / scatter add into carried array
def _scatter_accumulate_factory(variant: str, trip: int, factor: int):
    dim = 64 if variant == "small" else 256
    rng = np.random.default_rng(583_302)
    init_acc = jnp.zeros((dim,), dtype=jnp.float32)
    
    base_indices = (np.arange(1024, dtype=np.int32) * 19) % dim
    target_indices = jnp.asarray(base_indices[:trip])
    base_vals = floats(rng, 1024, scale=0.5)
    vals = base_vals[:trip]

    def fn(acc: jax.Array, v_stream: jax.Array, idx_stream: jax.Array):
        def body(i: int, target: jax.Array):
            idx = idx_stream[i]
            val = v_stream[i]
            # In-place add into addressed slot
            return target.at[idx].add(val)

        return lax.fori_loop(0, trip, body, acc, unroll=factor)

    return built(fn, init_acc, vals, target_indices)


# 3. write_multitap_delay_ring (train): Circular delay ring with 4 multi-tap reads and write
def _multitap_delay_ring_factory(variant: str, trip: int, factor: int, *, blocked=False):
    ring_size = 256 if variant == "small" else 2048
    rng = np.random.default_rng(583_303)
    base_x = floats(rng, 1024, scale=0.5)
    x_in = base_x[:trip]
    init_ring = jnp.zeros((ring_size,), dtype=jnp.float32)
    
    # 4 tap offsets
    tap1 = ring_size // 8
    tap2 = ring_size // 4
    tap3 = (ring_size * 3) // 8
    tap4 = ring_size // 2

    # Carried state: (ring_buffer, write_head, out_sample)
    init_head = jnp.int32(0)
    init_sample = jnp.float32(0.0)

    def fn(ring_init: jax.Array, head_init: jax.Array, sample_init: jax.Array, x_stream: jax.Array):
        if (blocked and ring_size % (8 * factor) == 0
                and ring_init.shape == (ring_size,) and x_stream.shape == (trip,)):
            blocks, tail = divmod(trip, factor)
            # Align the write head to block zero, even when resuming from a
            # nonzero head. All four taps and the wrap point align to blocks.
            aligned = jnp.roll(ring_init, -head_init)

            def update(ring, start, inputs):
                width = inputs.size

                def read(tap):
                    return lax.dynamic_slice_in_dim(
                        ring, (start - tap) % ring_size, width, allow_negative_indices=False,
                    )

                # No tap can see a write from its block, so the writes can
                # share one slice update without changing the recurrence.
                tap_sum = 0.4 * read(tap1) + 0.3 * read(tap2) + 0.2 * read(tap3) + 0.1 * read(tap4)
                values = inputs + 0.3 * tap_sum
                ring = lax.dynamic_update_slice_in_dim(
                    ring, values, start % ring_size, 0, allow_negative_indices=False,
                )
                return ring, tap_sum[-1]

            def run_block(block, carry):
                ring, _ = carry
                start = block * factor
                inputs = lax.dynamic_slice_in_dim(x_stream, start, factor, allow_negative_indices=False)
                return update(ring, start, inputs)

            ring, sample = aligned, sample_init
            if blocks:
                ring, sample = lax.fori_loop(0, blocks, run_block, (ring, sample))
            if tail:
                start = blocks * factor
                ring, sample = update(ring, start, x_stream[start:trip])
            return jnp.roll(ring, head_init), (head_init + trip) % ring_size, sample

        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            ring, head, _ = carry
            inp = x_stream[i]
            # Multi-tap read
            idx1 = (head - tap1) % ring_size
            idx2 = (head - tap2) % ring_size
            idx3 = (head - tap3) % ring_size
            idx4 = (head - tap4) % ring_size
            tap_sum = (
                0.4 * ring[idx1]
                + 0.3 * ring[idx2]
                + 0.2 * ring[idx3]
                + 0.1 * ring[idx4]
            )
            # Write new sample with feedback
            new_val = inp + 0.3 * tap_sum
            new_ring = ring.at[head].set(new_val)
            next_head = (head + 1) % ring_size
            return (new_ring, next_head, tap_sum)

        return lax.fori_loop(0, trip, body, (ring_init, head_init, sample_init), unroll=factor)

    return built(fn, init_ring, init_head, init_sample, x_in)


# 4. write_tiled_stencil_accum (train): 2D grid accumulator receiving overlapping stencil writes
def _tiled_stencil_accum_factory(variant: str, trip: int, factor: int):
    dim = 16 if variant == "small" else 48
    rng = np.random.default_rng(583_304)
    init_grid = jnp.zeros((dim, dim), dtype=jnp.float32)
    
    # 3x3 stencil impulse kernel
    kernel = jnp.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]], dtype=jnp.float32)
    
    max_pos = dim - 3
    base_rows = (np.arange(1024, dtype=np.int32) * 7) % max_pos
    base_cols = (np.arange(1024, dtype=np.int32) * 13) % max_pos
    row_starts = jnp.asarray(base_rows[:trip])
    col_starts = jnp.asarray(base_cols[:trip])
    base_amplitudes = floats(rng, 1024, scale=1.0)
    amplitudes = base_amplitudes[:trip]

    def fn(grid: jax.Array, k: jax.Array, r_idx: jax.Array, c_idx: jax.Array, amps: jax.Array):
        def body(i: int, target: jax.Array):
            r = r_idx[i]
            c = c_idx[i]
            amp = amps[i]
            # Read local 3x3 window, add weighted impulse, dynamic update back
            window = lax.dynamic_slice(target, (r, c), (3, 3))
            updated_window = window * 0.95 + amp * k
            return lax.dynamic_update_slice(target, updated_window, (r, c))

        return lax.fori_loop(0, trip, body, grid, unroll=factor)

    return built(fn, init_grid, kernel, row_starts, col_starts, amplitudes)


# 5. eval_write_waveguide_mesh (eval): 2D digital waveguide mesh scattering junction state
def _waveguide_pairs(state, excitations, factor):
    center = state.shape[-1] // 2
    decay = np.float32(0.99)
    decay_squared = np.float32(float(decay) ** 2)
    directions = ((1, 0), (-1, 0), (1, 1), (-1, 1))

    def pressure(waves):
        north, south, east, west = waves
        return 0.5 * (north + south + east + west)

    def inject(grid, excitation):
        return grid.at[center, center].add(excitation)

    def scatter_and_propagate(waves, driven_pressure):
        north, south, east, west = waves
        return tuple(
            jnp.roll(driven_pressure - wave, shift, axis) * decay
            for wave, (shift, axis) in zip((south, north, west, east), directions, strict=True)
        )

    def advance_pressure(previous, current, excitation_delta):
        # Static slices of a one-cell periodic halo supply the four neighbors.
        rows = jnp.concatenate((current[-1:], current, current[:1]), axis=0)
        cols = jnp.concatenate((current[:, -1:], current, current[:, :1]), axis=1)
        neighbors = rows[:-2] + rows[2:] + cols[:, :-2] + cols[:, 2:]
        return inject((0.5 * decay) * neighbors - decay_squared * previous, excitation_delta)

    initial_even = inject(pressure(state), excitations[0])
    first_waves = scatter_and_propagate(state, initial_even)
    initial_odd = inject(pressure(first_waves), excitations[1])

    def advance_pair(carry, forcing):
        even, odd, even_sum, odd_sum = carry
        # Opposite directions undo their shifts after two steps. The wave
        # correction for each pair is one unshifted and one shifted pressure.
        even_sum = decay_squared * (even_sum - even)
        odd_sum = decay_squared * odd_sum + decay * odd
        next_even = advance_pressure(even, odd, forcing[0])
        next_odd = advance_pressure(odd, next_even, forcing[1])
        return (next_even, next_odd, even_sum, odd_sum), None

    pairs, tail = divmod(excitations.size, 2)
    # The pressure recurrence includes the new excitation and subtracts the
    # damped excitation from two steps earlier. Its final lookahead is unused.
    forcing = jnp.pad(excitations, (0, 2))[2:] - decay_squared * excitations
    forcing = forcing[:2 * pairs].reshape(pairs, 2)
    zeros = jnp.zeros_like(initial_even)
    (_, _, even_sum, odd_sum), _ = lax.scan(
        advance_pair, (initial_even, initial_odd, zeros, zeros),
        forcing, unroll=factor // 2,
    )
    shifted = jnp.stack([jnp.roll(odd_sum, shift, axis) for shift, axis in directions])
    waves = state * np.float32(float(decay) ** (2 * pairs)) + even_sum + shifted
    if tail:
        waves = jnp.stack(scatter_and_propagate(waves, inject(pressure(waves), excitations[-1])))
    return waves


def _waveguide_mesh_factory(variant: str, trip: int, factor: int, *, paired=False):
    dim = 16 if variant == "small" else 48
    rng = np.random.default_rng(583_305)
    # State: 4 directional wave buffers (North, South, East, West) shape (4, dim, dim)
    init_state = jnp.zeros((4, dim, dim), dtype=jnp.float32)
    
    base_excitations = floats(rng, 1024, scale=1.0)
    excitations = base_excitations[:trip]
    center_r = dim // 2
    center_c = dim // 2

    def fn(state: jax.Array, ex_stream: jax.Array):
        if (paired and factor % 2 == 0 and trip >= 2 and ex_stream.shape == (trip,)
                and state.shape == (4, dim, dim) and state.dtype == ex_stream.dtype == jnp.float32):
            return _waveguide_pairs(state, ex_stream, factor)

        def body(i: int, st: jax.Array):
            # 4 directional incoming waves
            w_n, w_s, w_e, w_w = st[0], st[1], st[2], st[3]
            # Junction junction pressure = 0.5 * (N + S + E + W)
            p = 0.5 * (w_n + w_s + w_e + w_w)
            # Inject center excitation
            p = p.at[center_r, center_c].add(ex_stream[i])
            # Scattering update: Outgoing = Pressure - Incoming
            out_n = p - w_n
            out_s = p - w_s
            out_e = p - w_e
            out_w = p - w_w
            # Delay line propagation: shift along respective spatial direction with boundary damping
            next_n = jnp.roll(out_s, 1, axis=0) * 0.99
            next_s = jnp.roll(out_n, -1, axis=0) * 0.99
            next_e = jnp.roll(out_w, 1, axis=1) * 0.99
            next_w = jnp.roll(out_e, -1, axis=1) * 0.99
            return jnp.stack([next_n, next_s, next_e, next_w], axis=0)

        return lax.fori_loop(0, trip, body, state, unroll=factor)

    return built(fn, init_state, excitations)


# 6. eval_write_pic_charge_deposit (eval): PIC bilinear charge scatter deposition into 2D grid
def _pic_charge_deposit_factory(variant: str, trip: int, factor: int, *, blocked=False):
    dim = 16 if variant == "small" else 64
    rng = np.random.default_rng(583_306)
    init_grid = jnp.zeros((dim, dim), dtype=jnp.float32)
    
    # Streaming particle continuous coordinates (x, y) and charge q
    base_pos = rng.uniform(1.0, float(dim - 2), size=(1024, 2)).astype(np.float32)
    pos_stream = jnp.asarray(base_pos[:trip])
    base_charges = floats(rng, 1024, scale=1.0)
    charges = base_charges[:trip]

    def fn(grid: jax.Array, positions: jax.Array, qs: jax.Array):
        def deposit(position, q):
            x, y = position[0], position[1]
            ix, iy = jnp.floor(x).astype(jnp.int32), jnp.floor(y).astype(jnp.int32)
            fx, fy = x - jnp.floor(x), y - jnp.floor(y)
            # Cloud-In-Cell bilinear weights
            w00 = (1.0 - fx) * (1.0 - fy) * q
            w10 = fx * (1.0 - fy) * q
            w01 = (1.0 - fx) * fy * q
            w11 = fx * fy * q
            return ix, iy, (w00, w10, w01, w11)

        width = min(factor, trip)
        if (blocked and width > 1 and positions.shape == (trip, 2) and qs.shape == (trip,)
                and grid.dtype == positions.dtype == qs.dtype == jnp.float32):
            ix, iy, weights = jax.vmap(deposit)(positions, qs)
            rows = jnp.stack((iy, iy, iy + 1, iy + 1), axis=1)
            cols = jnp.stack((ix, ix + 1, ix, ix + 1), axis=1)
            updates = jnp.stack(weights, axis=1)
            blocks, tail = divmod(trip, width)

            def block_update(target, block_rows, block_cols, charges, count):
                # Each update experiences the damping of the steps after it
                # in the block. Scatter-add also handles repeated grid cells.
                decay = float(np.float32(0.999))
                powers = jnp.asarray(decay ** np.arange(count - 1, -1, -1), dtype=target.dtype)
                damped = target * jnp.asarray(decay ** count, dtype=target.dtype)
                return damped.at[block_rows.ravel(), block_cols.ravel()].add(
                    (charges * powers[:, None]).ravel(),
                )

            def advance(target, block):
                return block_update(target, *block, width), None

            batched = tuple(x[:blocks * width].reshape(blocks, width, 4) for x in (rows, cols, updates))
            grid = lax.scan(advance, grid, batched)[0]
            if tail:
                grid = block_update(grid, rows[-tail:], cols[-tail:], updates[-tail:], tail)
            return grid

        def body(i: int, target: jax.Array):
            ix, iy, (w00, w10, w01, w11) = deposit(positions[i], qs[i])
            # Scatter updates to 4 adjacent nodes with damping
            g = target * 0.999
            g = g.at[iy, ix].add(w00)
            g = g.at[iy, ix + 1].add(w10)
            g = g.at[iy + 1, ix].add(w01)
            g = g.at[iy + 1, ix + 1].add(w11)
            return g

        return lax.fori_loop(0, trip, body, grid, unroll=factor)

    return built(fn, init_grid, pos_stream, charges)


# 7. eval_write_pingpong_frame_swap (eval): Dual-buffer ping-pong state swap
def _pingpong_frame_swap_factory(variant: str, trip: int, factor: int):
    dim = 64 if variant == "small" else 256
    rng = np.random.default_rng(583_307)
    buf_a = floats(rng, dim, scale=1.0)
    buf_b = jnp.zeros((dim,), dtype=jnp.float32)
    init_active = jnp.int32(0)

    def fn(buf_a_init: jax.Array, buf_b_init: jax.Array, act_init: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            a, b, active = carry
            # If active == 0, read A, blur into B; else read B, blur into A
            src = jnp.where(active == 0, a, b)
            # 3-point smoothing blur
            blurred = 0.25 * jnp.roll(src, 1) + 0.5 * src + 0.25 * jnp.roll(src, -1)
            new_a = jnp.where(active == 0, a, blurred)
            new_b = jnp.where(active == 0, blurred, b)
            new_active = 1 - active
            return (new_a, new_b, new_active)

        return lax.fori_loop(0, trip, body, (buf_a_init, buf_b_init, act_init), unroll=factor)

    return built(fn, buf_a, buf_b, init_active)


# 8. eval_write_circular_sliding_hist (eval): Carried circular buffer updating rolling histogram
def _circular_sliding_hist_factory(variant: str, trip: int, factor: int):
    win = 32 if variant == "small" else 128
    buckets = 16 if variant == "small" else 64
    rng = np.random.default_rng(583_308)
    base_x = rng.integers(0, buckets, size=1024, dtype=np.int32)
    x_in = jnp.asarray(base_x[:trip])
    
    init_circ = jnp.zeros((win,), dtype=jnp.int32)
    init_hist = jnp.zeros((buckets,), dtype=jnp.float32)
    init_head = jnp.int32(0)

    def fn(circ_init: jax.Array, hist_init: jax.Array, head_init: jax.Array, stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            circ, hist, head = carry
            new_val = stream[i]
            old_val = circ[head]
            # Increment incoming bucket, decrement outgoing bucket
            new_hist = hist.at[new_val].add(1.0).at[old_val].add(-1.0)
            # Store in circular buffer
            new_circ = circ.at[head].set(new_val)
            next_head = (head + 1) % win
            return (new_circ, new_hist, next_head)

        return lax.fori_loop(0, trip, body, (circ_init, hist_init, head_init), unroll=factor)

    return built(fn, init_circ, init_hist, init_head, x_in)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec(
        name="write_patch_blend",
        family="writes_state",
        description="Overlapping 2D patch update blending weighted patches into canvas",
        mechanism="dynamic_slice_patch_blend",
        structural_difference="Dynamic multi-dimensional slice load, blend arithmetic, and dynamic slice writeback",
        loop_kind="fori_loop",
        shape_variants={
            "small": "canvas=32x32, patch=8x8 (4 KB)",
            "large": "canvas=64x64, patch=16x16 (16 KB)",
        },
        primary_variant="small",
        factory=_patch_blend_factory,
        seed=583_301,
        provenance=Provenance(
            kind="original",
            source="Adapted from idx_overlapping_block_update historical lineage",
            license="MIT",
            adaptation="Standardized dual canvas sizes (32, 64) with deterministic coordinate offsets",
        ),
        related_group="patch_blend",
    ),
    CorpusSpec(
        name="write_scatter_accumulate",
        family="writes_state",
        description="Dynamic update scatter addition into carried accumulator array",
        mechanism="carried_array_scatter_accumulation",
        structural_difference="In-place index-directed mutation and addition into mutable carried buffer",
        loop_kind="fori_loop",
        shape_variants={
            "small": "dim=64 (256 B)",
            "large": "dim=256 (1 KB)",
        },
        primary_variant="small",
        factory=_scatter_accumulate_factory,
        seed=583_302,
        provenance=Provenance(
            kind="original",
            source="Adapted from conv_scatter_accumulate historical lineage",
            license="MIT",
            adaptation="Expanded to dual accumulator dimensions with deterministic query stream",
        ),
        related_group="scatter_accumulate",
    ),
    CorpusSpec(
        name="write_multitap_delay_ring",
        family="writes_state",
        description="Circular delay line ring buffer with 4 multi-tap read positions and pointer update",
        mechanism="circular_delay_ring_pointer",
        structural_difference="Large live ring buffer state mutated at advancing circular head with multi-tap feedback",
        loop_kind="fori_loop",
        shape_variants={
            "small": "ring=256 floats (1 KB)",
            "large": "ring=2048 floats (8 KB)",
        },
        primary_variant="small",
        factory=_multitap_delay_ring_factory,
        seed=583_303,
        provenance=Provenance(
            kind="original",
            source="Digital audio effects reverberation delay networks",
            license="MIT",
            adaptation="JAX circular ring buffer with multi-tap reads and write pointer tracking",
        ),
        related_group="multitap_delay_ring",
    ),
    CorpusSpec(
        name="write_tiled_stencil_accum",
        family="writes_state",
        description="2D grid accumulator receiving overlapping 3x3 stencil impulse writes",
        mechanism="tiled_stencil_impulse_scatter",
        structural_difference="Overlapping spatial 3x3 kernel accumulation into carried 2D grid buffer",
        loop_kind="fori_loop",
        shape_variants={
            "small": "grid=16x16 (1 KB)",
            "large": "grid=48x48 (9.2 KB)",
        },
        primary_variant="small",
        factory=_tiled_stencil_accum_factory,
        seed=583_304,
        provenance=Provenance(
            kind="original",
            source="Finite difference impulse response simulation",
            license="MIT",
            adaptation="Dynamic slice 3x3 stencil accumulator over moving spatial coordinates",
        ),
        related_group="tiled_stencil_accum",
    ),
    CorpusSpec(
        name="eval_write_waveguide_mesh",
        family="eval_writes_state",
        description="2D digital waveguide mesh scattering junction directional buffer propagation",
        mechanism="waveguide_scattering_mesh_propagation",
        structural_difference="Directional 4-way wave buffer scattering and circular directional rolling shift",
        loop_kind="fori_loop",
        shape_variants={
            "small": "mesh=16x16 (1 KB)",
            "large": "mesh=48x48 (9.2 KB)",
        },
        primary_variant="small",
        factory=_waveguide_mesh_factory,
        seed=583_305,
        provenance=Provenance(
            kind="public-inspired",
            source="Van Duyne, S. A., & Smith, J. O. (1993). Physical modeling with 2-D digital waveguide mesh. ICMC",
            license="Academic reference / CCRMA",
            adaptation="4-layer directional wave buffer scattering with center spatial excitation",
        ),
        related_group="eval_waveguide_mesh",
    ),
    CorpusSpec(
        name="eval_write_pic_charge_deposit",
        family="eval_writes_state",
        description="Particle-In-Cell bilinear charge scatter deposition into carried 2D grid",
        mechanism="particle_in_cell_charge_deposition",
        structural_difference="Bilinear continuous-coordinate weight projection and 4-point nodal grid accumulation",
        loop_kind="fori_loop",
        shape_variants={
            "small": "grid=16x16 (1 KB)",
            "large": "grid=64x64 (16 KB)",
        },
        primary_variant="small",
        factory=_pic_charge_deposit_factory,
        seed=583_306,
        provenance=Provenance(
            kind="public-inspired",
            source="Birdsall, C. K., & Langdon, A. B. (2004). Plasma physics via computer simulation. CRC Press",
            license="Academic standard reference",
            adaptation="Cloud-in-cell charge scatter deposition with continuous particle motion",
        ),
        related_group="eval_pic_charge_deposit",
    ),
    CorpusSpec(
        name="eval_write_pingpong_frame_swap",
        family="eval_writes_state",
        description="Dual-buffer ping-pong state swap with alternating blur updates",
        mechanism="dual_buffer_pingpong_frame_swap",
        structural_difference="Alternating active buffer pointer and in-place directional stencil writeback",
        loop_kind="fori_loop",
        shape_variants={
            "small": "2 buffers of 64 floats (512 B)",
            "large": "2 buffers of 256 floats (2 KB)",
        },
        primary_variant="small",
        factory=_pingpong_frame_swap_factory,
        seed=583_307,
        provenance=Provenance(
            kind="public-inspired",
            source="Foley, J. D., et al. (1996). Computer graphics: principles and practice. Addison-Wesley",
            license="Graphics engineering standard",
            adaptation="Branchless dual-buffer ping-pong state alternation with spatial diffusion",
        ),
        related_group="eval_pingpong_frame_swap",
    ),
    CorpusSpec(
        name="eval_write_circular_sliding_hist",
        family="eval_writes_state",
        description="Carried circular buffer updating rolling histogram bucket counts",
        mechanism="sliding_window_histogram_update",
        structural_difference="Simultaneous circular write pointer update and dual bucket increment/decrement",
        loop_kind="fori_loop",
        shape_variants={
            "small": "window=32, buckets=16 (192 B)",
            "large": "window=128, buckets=64 (768 B)",
        },
        primary_variant="small",
        factory=_circular_sliding_hist_factory,
        seed=583_308,
        provenance=Provenance(
            kind="public-inspired",
            source="Huang, T., Yang, G., & Tang, G. (1979). A fast 2D median filtering algorithm. IEEE ASSP",
            license="Academic reference",
            adaptation="Online sliding window histogram maintenance with circular buffer memory",
        ),
        related_group="eval_circular_sliding_hist",
    ),
)
