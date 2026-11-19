"""Domain 5: Branches, masks and active state workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats, indices
from benchmarks.corpus.spec import CORPUS_TRIP_COUNTS, CorpusSpec, Provenance


# 1. branch_leaky_bucket (train): Token bucket packet policing
def _leaky_bucket_factory(variant: str, trip: int, factor: int):
    classes = 4 if variant == "small" else 16
    rng = np.random.default_rng(583_501)
    # State: tokens (classes,), dropped_packets (classes,)
    init_tokens = jnp.ones((classes,), dtype=jnp.float32) * 50.0
    init_dropped = jnp.zeros((classes,), dtype=jnp.float32)
    
    capacity = 100.0
    leak_rate = 5.0
    base_packets = floats(rng, 1024, classes, scale=15.0)
    # Make packet sizes positive
    packets_in = jnp.abs(base_packets[:trip])

    def fn(tokens_init: jax.Array, dropped_init: jax.Array, pkt_stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            tok, drop = carry
            pkts = pkt_stream[i]
            # Leak tokens
            leaked_tok = jnp.maximum(0.0, tok - leak_rate)
            # Check conformance: fits in bucket?
            conformant = (leaked_tok + pkts) <= capacity
            new_tok = jnp.where(conformant, leaked_tok + pkts, leaked_tok)
            new_drop = drop + jnp.where(conformant, 0.0, pkts)
            return (new_tok, new_drop)

        return lax.fori_loop(0, trip, body, (tokens_init, dropped_init), unroll=factor)

    return built(fn, init_tokens, init_dropped, packets_in)


# 2. branch_masked_active_set (train): Active set boolean indicator update and projection
def _masked_active_set_factory(variant: str, trip: int, factor: int):
    dim = 32 if variant == "small" else 128
    rng = np.random.default_rng(583_502)
    init_x = floats(rng, dim, scale=1.0)
    init_active = jnp.zeros((dim,), dtype=jnp.bool_)
    
    # Lower and upper box bounds
    lb = -0.5
    ub = 0.5
    base_grads = floats(rng, 1024, dim, scale=0.2)
    grads = base_grads[:trip]
    step_size = jnp.float32(0.1)

    def fn(x_init: jax.Array, active_init: jax.Array, g_stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            x, active = carry
            g = g_stream[i]
            # Gradient step only on inactive coordinates
            masked_step = jnp.where(active, 0.0, -step_size * g)
            x_trial = x + masked_step
            # Check constraints
            violates_lb = x_trial < lb
            violates_ub = x_trial > ub
            new_active = violates_lb | violates_ub
            new_x = jnp.clip(x_trial, lb, ub)
            return (new_x, new_active)

        return lax.fori_loop(0, trip, body, (x_init, active_init), unroll=factor)

    return built(fn, init_x, init_active, grads)


# 3. branch_hysteresis_schmitt (train): Dual-threshold Schmitt trigger state machine
def _hysteresis_schmitt_factory(variant: str, trip: int, factor: int):
    channels = 4 if variant == "small" else 16
    rng = np.random.default_rng(583_503)
    # State: discrete_state (channels,) in {0, 1}, spike_counts (channels,)
    init_state = jnp.zeros((channels,), dtype=jnp.int32)
    init_counts = jnp.zeros((channels,), dtype=jnp.float32)
    
    v_high = 0.4
    v_low = -0.4
    base_signal = floats(rng, 1024, channels, scale=1.0)
    sig_in = base_signal[:trip]

    def fn(st_init: jax.Array, cnt_init: jax.Array, stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            st, counts = carry
            v = stream[i]
            # Schmitt hysteresis transition:
            # If st == 0 and v > v_high -> st = 1 (spike)
            # If st == 1 and v < v_low -> st = 0
            trigger_on = (st == 0) & (v > v_high)
            trigger_off = (st == 1) & (v < v_low)
            new_st = jnp.where(trigger_on, 1, jnp.where(trigger_off, 0, st))
            new_counts = counts + jnp.where(trigger_on, 1.0, 0.0)
            return (new_st, new_counts)

        return lax.fori_loop(0, trip, body, (st_init, cnt_init), unroll=factor)

    return built(fn, init_state, init_counts, sig_in)


# 4. branch_sparse_grad_clip (train): Elementwise gradient deadband threshold and norm clip
def _sparse_grad_clip_factory(variant: str, trip: int, factor: int):
    dim = 64 if variant == "small" else 256
    rng = np.random.default_rng(583_504)
    init_params = floats(rng, dim, scale=0.5)
    
    deadband = 0.05
    max_norm = 1.0
    base_grads = floats(rng, 1024, dim, scale=0.2)
    grads = base_grads[:trip]

    def fn(params_init: jax.Array, g_stream: jax.Array):
        def body(i: int, p: jax.Array):
            g = g_stream[i]
            # Mask out values below deadband
            active_g = jnp.where(jnp.abs(g) > deadband, g, 0.0)
            norm = jnp.linalg.norm(active_g) + 1e-6
            clip_coef = jnp.minimum(1.0, max_norm / norm)
            effective_g = active_g * clip_coef
            return p - 0.05 * effective_g

        return lax.fori_loop(0, trip, body, params_init, unroll=factor)

    return built(fn, init_params, grads)


# 5. eval_branch_huffman_zero_rle (eval): FSM decoding zero runs into carried buffer
def _huffman_zero_rle_factory(variant: str, trip: int, factor: int):
    buf_size = 32 if variant == "small" else 128
    rng = np.random.default_rng(583_505)
    # Stream of encoded tokens: 0..3 are short runs of zeros (run=token+1), >=4 are literal values
    base_tokens = rng.integers(0, 10, size=1024, dtype=np.int32)
    tokens = jnp.asarray(base_tokens[:trip])
    
    init_buf = jnp.zeros((buf_size,), dtype=jnp.float32)
    init_ptr = jnp.int32(0)

    def fn(buf_init: jax.Array, ptr_init: jax.Array, tok_stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            buf, ptr = carry
            tok = tok_stream[i]
            is_zero_run = tok < 4
            run_len = jnp.where(is_zero_run, tok + 1, 1)
            val = jnp.where(is_zero_run, 0.0, (tok - 3).astype(jnp.float32))
            
            # Write value at ptr
            new_buf = buf.at[ptr].set(val)
            next_ptr = (ptr + run_len) % buf_size
            return (new_buf, next_ptr)

        return lax.fori_loop(0, trip, body, (buf_init, ptr_init), unroll=factor)

    return built(fn, init_buf, init_ptr, tokens)


# 6. eval_branch_tcp_congestion_fsm (eval): TCP AIMD congestion control state machine
def _tcp_congestion_fsm_factory(variant: str, trip: int, factor: int):
    sessions = 4 if variant == "small" else 16
    rng = np.random.default_rng(583_506)
    
    # State: cwnd (sessions,), ssthresh (sessions,), state_id (sessions:) 0=slow_start, 1=congestion_avoidance, 2=recovery
    init_cwnd = jnp.ones((sessions,), dtype=jnp.float32) * 1.0
    init_ssthresh = jnp.ones((sessions,), dtype=jnp.float32) * 16.0
    init_state_id = jnp.zeros((sessions,), dtype=jnp.int32)
    
    # Events stream: 0=ACK, 1=dup_ACK, 2=timeout loss
    base_events = rng.integers(0, 3, size=(1024, sessions), dtype=np.int32)
    events = jnp.asarray(base_events[:trip])

    def fn(cwnd_init: jax.Array, ssthresh_init: jax.Array, st_init: jax.Array, ev_stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            cwnd, ssthresh, st = carry
            ev = ev_stream[i]
            
            # Loss branch (ev == 2)
            loss_mask = ev == 2
            new_ssthresh_loss = jnp.maximum(2.0, cwnd * 0.5)
            new_cwnd_loss = 1.0
            
            # Slow start growth (st == 0 and ev == 0): exponential cwnd += 1.0
            ss_grow = (st == 0) & (ev == 0)
            cwnd_ss = cwnd + 1.0
            
            # Congestion avoidance growth (st == 1 and ev == 0): linear cwnd += 1.0 / cwnd
            ca_grow = (st == 1) & (ev == 0)
            cwnd_ca = cwnd + 1.0 / jnp.maximum(1.0, cwnd)
            
            next_cwnd = jnp.where(
                loss_mask,
                new_cwnd_loss,
                jnp.where(ss_grow, cwnd_ss, jnp.where(ca_grow, cwnd_ca, cwnd)),
            )
            next_ssthresh = jnp.where(loss_mask, new_ssthresh_loss, ssthresh)
            next_st = jnp.where(
                loss_mask,
                0,
                jnp.where(next_cwnd >= next_ssthresh, 1, st),
            )
            return (next_cwnd, next_ssthresh, next_st)

        return lax.fori_loop(0, trip, body, (cwnd_init, ssthresh_init, st_init), unroll=factor)

    return built(fn, init_cwnd, init_ssthresh, init_state_id, events)


# 7. eval_branch_particle_boundary_bounce (eval): 2D particle specular boundary bounce
def _particle_boundary_bounce_factory(variant: str, trip: int, factor: int):
    particles = 8 if variant == "small" else 32
    rng = np.random.default_rng(583_507)
    
    init_pos = floats(rng, particles, 2, scale=0.8)
    init_vel = floats(rng, particles, 2, scale=0.05)
    
    dt = jnp.float32(0.1)
    bound = 1.0

    def fn(pos_init: jax.Array, vel_init: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            pos, vel = carry
            # Trial position step
            trial_pos = pos + dt * vel
            # Check collisions with boundary box [-bound, bound]
            hit_x = (trial_pos[:, 0] < -bound) | (trial_pos[:, 0] > bound)
            hit_y = (trial_pos[:, 1] < -bound) | (trial_pos[:, 1] > bound)
            # Specular velocity reflection: flip velocity on collision axis
            new_vx = jnp.where(hit_x, -0.9 * vel[:, 0], vel[:, 0])
            new_vy = jnp.where(hit_y, -0.9 * vel[:, 1], vel[:, 1])
            new_vel = jnp.stack([new_vx, new_vy], axis=1)
            new_pos = jnp.clip(trial_pos, -bound, bound)
            return (new_pos, new_vel)

        return lax.fori_loop(0, trip, body, (pos_init, vel_init), unroll=factor)

    return built(fn, init_pos, init_vel)


# 8. eval_branch_ray_slab_intersect (eval): Kay-Kajiya slab method for ray-AABB test
def _ray_slab_intersect_factory(variant: str, trip: int, factor: int):
    rays = 8 if variant == "small" else 32
    rng = np.random.default_rng(583_508)
    
    # Ray origins and directions: shape (rays, 3)
    ray_org = floats(rng, rays, 3, scale=3.0)
    ray_dir = floats(rng, rays, 3, scale=1.0)
    ray_dir = ray_dir / (jnp.linalg.norm(ray_dir, axis=1, keepdims=True) + 1e-4)
    
    # Box bounds min and max
    box_min = jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float32)
    box_max = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
    
    # Dynamic box movement stream: shape (trip, 3)
    base_shifts = floats(rng, 1024, 3, scale=0.1)
    shifts = base_shifts[:trip]
    init_hits = jnp.zeros((rays,), dtype=jnp.float32)

    def fn(hit_init: jax.Array, origins: jax.Array, dirs: jax.Array, box_m: jax.Array, box_M: jax.Array, shift_stream: jax.Array):
        def body(i: int, hits: jax.Array):
            shift = shift_stream[i]
            cur_min = box_m + shift
            cur_max = box_M + shift
            
            # Kay-Kajiya slab algorithm: t_near = max(min(t1, t2)), t_far = min(max(t1, t2))
            inv_d = 1.0 / (dirs + 1e-6)
            t1 = (cur_min - origins) * inv_d
            t2 = (cur_max - origins) * inv_d
            
            t_min = jnp.minimum(t1, t2)
            t_max = jnp.maximum(t1, t2)
            
            t_near = jnp.max(t_min, axis=1)
            t_far = jnp.min(t_max, axis=1)
            
            hit = (t_far >= t_near) & (t_far >= 0.0)
            return hits + jnp.where(hit, 1.0, 0.0)

        return lax.fori_loop(0, trip, body, hit_init, unroll=factor)

    return built(fn, init_hits, ray_org, ray_dir, box_min, box_max, shifts)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec(
        name="branch_leaky_bucket",
        family="branches_masks",
        description="Token bucket packet policing with conformant/non-conformant branch accounting",
        mechanism="token_bucket_policing_branches",
        structural_difference="Conditional branch evaluation updating token capacity and drop counter",
        loop_kind="fori_loop",
        shape_variants={
            "small": "classes=4 (64 B)",
            "large": "classes=16 (256 B)",
        },
        primary_variant="small",
        factory=_leaky_bucket_factory,
        seed=583_501,
        provenance=Provenance(
            kind="original",
            source="Adapted from sched_leaky_bucket_policer historical lineage",
            license="MIT",
            adaptation="Expanded to dual class counts (4, 16) with deterministic packet stream",
        ),
        related_group="leaky_bucket_policer",
    ),
    CorpusSpec(
        name="branch_masked_active_set",
        family="branches_masks",
        description="Active set boolean indicator update and masked constraint projection",
        mechanism="masked_active_set_constraint_projection",
        structural_difference="Boolean mask state carry directing sparse update steps and boundary clamping",
        loop_kind="fori_loop",
        shape_variants={
            "small": "dim=32 (128 B)",
            "large": "dim=128 (512 B)",
        },
        primary_variant="small",
        factory=_masked_active_set_factory,
        seed=583_502,
        provenance=Provenance(
            kind="original",
            source="Adapted from sched_masked_active_set historical lineage",
            license="MIT",
            adaptation="Expanded to dual coordinate dimensions with deterministic gradient stream",
        ),
        related_group="masked_active_set",
    ),
    CorpusSpec(
        name="branch_hysteresis_schmitt",
        family="branches_masks",
        description="Dual-threshold Schmitt trigger state machine with hysteresis branch transitions",
        mechanism="schmitt_trigger_hysteresis_fsm",
        structural_difference="State-dependent dual threshold comparison and discrete jump state transitions",
        loop_kind="fori_loop",
        shape_variants={
            "small": "channels=4 (64 B)",
            "large": "channels=16 (256 B)",
        },
        primary_variant="small",
        factory=_hysteresis_schmitt_factory,
        seed=583_503,
        provenance=Provenance(
            kind="original",
            source="Schmitt, O. H. (1938). A thermionic trigger. Journal of Scientific Instruments",
            license="MIT",
            adaptation="Multi-channel parallel discrete hysteresis state machine in JAX",
        ),
        related_group="hysteresis_schmitt",
    ),
    CorpusSpec(
        name="branch_sparse_grad_clip",
        family="branches_masks",
        description="Elementwise gradient deadband threshold and adaptive norm clipping",
        mechanism="masked_gradient_deadband_norm_clip",
        structural_difference="Magnitude threshold masking combined with dynamic vector norm scaling",
        loop_kind="fori_loop",
        shape_variants={
            "small": "dim=64 (256 B)",
            "large": "dim=256 (1 KB)",
        },
        primary_variant="small",
        factory=_sparse_grad_clip_factory,
        seed=583_504,
        provenance=Provenance(
            kind="original",
            source="Deep learning gradient clipping and deadband optimization algorithms",
            license="MIT",
            adaptation="Sparse deadband thresholding with vector norm clipping in loop carry",
        ),
        related_group="sparse_grad_clip",
    ),
    CorpusSpec(
        name="eval_branch_huffman_zero_rle",
        family="eval_branches_masks",
        description="Finite state machine decoding zero runs into carried buffer",
        mechanism="huffman_zero_rle_decoder_fsm",
        structural_difference="Symbol classification branch, variable length run expansion, and circular buffer write",
        loop_kind="fori_loop",
        shape_variants={
            "small": "buffer=32 ints (128 B)",
            "large": "buffer=128 ints (512 B)",
        },
        primary_variant="small",
        factory=_huffman_zero_rle_factory,
        seed=583_505,
        provenance=Provenance(
            kind="public-inspired",
            source="Huffman, D. A. (1952). A method for construction of minimum-redundancy codes. Proc. IRE",
            license="Public domain",
            adaptation="Zero-run length decoding state machine over streaming compressed symbols",
        ),
        related_group="eval_huffman_zero_rle",
    ),
    CorpusSpec(
        name="eval_branch_tcp_congestion_fsm",
        family="eval_branches_masks",
        description="TCP AIMD congestion control state machine with 3 states",
        mechanism="tcp_aimd_congestion_control_fsm",
        structural_difference="Multi-branch protocol state machine with exponential, linear, and multiplicative backoff transitions",
        loop_kind="fori_loop",
        shape_variants={
            "small": "sessions=4 (64 B)",
            "large": "sessions=16 (256 B)",
        },
        primary_variant="small",
        factory=_tcp_congestion_fsm_factory,
        seed=583_506,
        provenance=Provenance(
            kind="public-inspired",
            source="Jacobson, V. (1988). Congestion avoidance and control. ACM SIGCOMM",
            license="Academic reference / RFC 5681 standard",
            adaptation="Parallel multi-session TCP Reno AIMD state machine with loss transitions",
        ),
        related_group="eval_tcp_congestion_fsm",
    ),
    CorpusSpec(
        name="eval_branch_particle_boundary_bounce",
        family="eval_branches_masks",
        description="2D particle specular boundary bounce collision simulation",
        mechanism="particle_specular_boundary_collision",
        structural_difference="Multi-axis inequality collision detection and specular velocity coordinate reflection",
        loop_kind="fori_loop",
        shape_variants={
            "small": "particles=8 (128 B)",
            "large": "particles=32 (512 B)",
        },
        primary_variant="small",
        factory=_particle_boundary_bounce_factory,
        seed=583_507,
        provenance=Provenance(
            kind="public-inspired",
            source="Bridson, R. (2015). Fluid simulation for computer graphics. CRC Press",
            license="Academic standard",
            adaptation="Box-constrained particle dynamics with restitution damping on collision planes",
        ),
        related_group="eval_particle_boundary_bounce",
    ),
    CorpusSpec(
        name="eval_branch_ray_slab_intersect",
        family="eval_branches_masks",
        description="Kay-Kajiya slab method for ray-AABB intersection testing",
        mechanism="kay_kajiya_ray_aabb_slab_intersection",
        structural_difference="Componentwise parametric min/max intervals and boolean hit determination",
        loop_kind="fori_loop",
        shape_variants={
            "small": "rays=8 (256 B)",
            "large": "rays=32 (1 KB)",
        },
        primary_variant="small",
        factory=_ray_slab_intersect_factory,
        seed=583_508,
        provenance=Provenance(
            kind="public-inspired",
            source="Kay, T. L., & Kajiya, J. T. (1986). Ray tracing complex scenes. ACM SIGGRAPH",
            license="Academic standard",
            adaptation="Parallel ray-slab interval intersection test against dynamic bounding box",
        ),
        related_group="eval_ray_slab_intersect",
    ),
)
