"""Scheduling and control flow independent fori_loop workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats, indices
from benchmarks.corpus.spec import CorpusSpec, Provenance

# 1. sched_priority_queue_step: Fixed-Capacity Array Binary Min-Heap Priority Queue
def _heap_queue_factory(variant: str, trip: int, factor: int):
    cap = {"cap16": 16, "cap32": 32, "cap64": 64}[variant]
    rng = np.random.default_rng(583_891)
    # Operations: 1 = push, 0 = pop-min
    ops = jnp.asarray(rng.choice([0, 1], size=trip, p=[0.4, 0.6]), dtype=jnp.int32)
    keys_in = floats(rng, trip, scale=10.0)
    init_heap = jnp.full((cap,), 1e6, dtype=jnp.float32)
    init_sz = jnp.int32(0)

    def fn(heap_init: jax.Array, sz_init: jax.Array, op_stream: jax.Array, k_stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            heap, sz, total_popped = carry
            op = op_stream[i]
            k = k_stream[i]
            # Push logic: append at sz, sift-up
            can_push = (op == 1) & (sz < cap)
            heap_pushed = heap.at[sz].set(k)
            # Sift-up unrolled for tree depth log2(cap)
            depth = int(np.ceil(np.log2(cap)))
            def sift_up_step(d: int, st: tuple[jax.Array, jax.Array]):
                h, curr_idx = st
                parent = (curr_idx - 1) // 2
                should_swap = (curr_idx > 0) & (h[curr_idx] < h[parent])
                val_curr = h[curr_idx]
                val_parent = h[parent]
                h_swapped = h.at[curr_idx].set(jnp.where(should_swap, val_parent, val_curr)).at[parent].set(jnp.where(should_swap, val_curr, val_parent))
                new_idx = jnp.where(should_swap, parent, curr_idx)
                return (h_swapped, new_idx)
            heap_after_push, _ = lax.fori_loop(0, depth, sift_up_step, (heap_pushed, sz))

            # Pop logic: extract root, replace with last element, sift-down
            can_pop = (op == 0) & (sz > 0)
            popped_val = heap[0]
            last_val = heap[jnp.maximum(0, sz - 1)]
            heap_popped = heap.at[0].set(last_val).at[jnp.maximum(0, sz - 1)].set(1e6)
            def sift_down_step(d: int, st: tuple[jax.Array, jax.Array]):
                h, curr_idx = st
                left = 2 * curr_idx + 1
                right = 2 * curr_idx + 2
                has_left = left < sz
                has_right = right < sz
                val_left = jnp.where(has_left, h[left], 1e6)
                val_right = jnp.where(has_right, h[right], 1e6)
                min_child = jnp.where(val_left < val_right, left, right)
                min_val = jnp.minimum(val_left, val_right)
                should_swap = (min_child < sz) & (min_val < h[curr_idx])
                val_curr = h[curr_idx]
                h_swapped = h.at[curr_idx].set(jnp.where(should_swap, min_val, val_curr)).at[min_child].set(jnp.where(should_swap, val_curr, min_val))
                new_idx = jnp.where(should_swap, min_child, curr_idx)
                return (h_swapped, new_idx)
            heap_after_pop, _ = lax.fori_loop(0, depth, sift_down_step, (heap_popped, jnp.int32(0)))

            final_heap = jnp.where(can_push, heap_after_push, jnp.where(can_pop, heap_after_pop, heap))
            final_sz = sz + jnp.where(can_push, 1, 0) - jnp.where(can_pop, 1, 0)
            final_accum = total_popped + jnp.where(can_pop, popped_val, 0.0)
            return (final_heap, final_sz, final_accum)

        return lax.fori_loop(0, trip, body, (heap_init, sz_init, jnp.float32(0.0)), unroll=factor)

    return built(fn, init_heap, init_sz, ops, keys_in)


# 2. sched_fsm_traffic_controller: Hierarchical Finite State Machine Traffic Controller [fresh_eval]
def _traffic_fsm_factory(variant: str, trip: int, factor: int):
    num_dirs = {"dir2": 2, "dir4": 4}[variant]
    rng = np.random.default_rng(583_892)
    # Sensor triggers per direction
    sensors = jnp.asarray(rng.choice([0, 1], size=(trip, num_dirs), p=[0.7, 0.3]), dtype=jnp.int32)
    init_state = jnp.zeros((3,), dtype=jnp.int32) # [active_dir, timer, yellow_flag]
    init_lights = jnp.zeros((num_dirs,), dtype=jnp.int32).at[0].set(2) # 2 = Green, 1 = Yellow, 0 = Red

    def fn(st_init: jax.Array, lights_init: jax.Array, sens_stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            st, lights = carry
            active_dir, timer, yellow = st[0], st[1], st[2]
            sens = sens_stream[i]
            # State transitions: Green countdown (min 5 ticks), then Yellow (2 ticks), then Next Green
            time_expired = timer <= 0
            demand_other = jnp.sum(sens) > sens[active_dir]

            # Trigger yellow transition
            start_yellow = (yellow == 0) & time_expired & demand_other
            new_yellow = jnp.where(start_yellow, 1, jnp.where(time_expired & (yellow == 1), 0, yellow))
            next_dir = (active_dir + jnp.where(time_expired & (yellow == 1), 1, 0)) % num_dirs
            new_timer = jnp.where(start_yellow, 2, jnp.where(time_expired & (yellow == 1), 5, timer - 1))
            new_st = jnp.stack([next_dir, new_timer, new_yellow])

            # Output lights encoding
            new_lights = jnp.zeros((num_dirs,), dtype=jnp.int32)
            light_color = jnp.where(new_yellow == 1, 1, 2)
            new_lights = new_lights.at[next_dir].set(light_color)
            return (new_st, new_lights)

        return lax.fori_loop(0, trip, body, (st_init, lights_init), unroll=factor)

    return built(fn, init_state, init_lights, sensors)


# 3. sched_masked_active_set: Active-Set Quadratic Programming Constraint Selection
def _active_set_factory(variant: str, trip: int, factor: int):
    dim, m_cons = {"d8m8": (8, 8), "d16m16": (16, 16), "d32m32": (32, 32)}[variant]
    rng = np.random.default_rng(583_893)
    A_cons = floats(rng, m_cons, dim, scale=0.5)
    b_cons = floats(rng, m_cons, scale=1.0)
    g_stream = floats(rng, trip, dim, scale=0.5)
    init_x = jnp.zeros((dim,), dtype=jnp.float32)
    init_active = jnp.zeros((m_cons,), dtype=jnp.bool_)

    def fn(x_init: jax.Array, act_init: jax.Array, A: jax.Array, b: jax.Array, grads: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            x, active = carry
            g = grads[i]
            # Take gradient step
            x_cand = x - 0.05 * g
            # Check constraints: A x_cand - b <= 0
            viol = A @ x_cand - b
            most_viol_idx = jnp.argmax(viol)
            has_viol = viol[most_viol_idx] > 0.0
            # Add most violated constraint to active set
            new_active = jnp.where(has_viol, active.at[most_viol_idx].set(True), active)
            # Project onto violated constraint hyperplane
            a_v = A[most_viol_idx]
            proj_step = viol[most_viol_idx] / (jnp.dot(a_v, a_v) + 1e-6) * a_v
            new_x = jnp.where(has_viol, x_cand - proj_step, x_cand)
            return (new_x, new_active)

        return lax.fori_loop(0, trip, body, (x_init, act_init), unroll=factor)

    return built(fn, init_x, init_active, A_cons, b_cons, g_stream)


# 4. sched_leaky_bucket_policer: Dual-Rate Token Bucket Traffic Policer (RFC 2698)
def _leaky_bucket_factory(variant: str, trip: int, factor: int):
    cir, pir = {"low": (100.0, 200.0), "high": (500.0, 1000.0)}[variant]
    rng = np.random.default_rng(583_894)
    arrival_times = jnp.asarray(np.sort(rng.uniform(0.0, 10.0, size=trip)), dtype=jnp.float32)
    packet_sizes = jnp.asarray(rng.integers(40, 1500, size=trip), dtype=jnp.float32)
    init_c_tokens = jnp.float32(5000.0)
    init_p_tokens = jnp.float32(10000.0)
    init_st = (init_c_tokens, init_p_tokens, jnp.float32(0.0)) # [C, P, last_t]

    def fn(c_init: jax.Array, p_init: jax.Array, times: jax.Array, sizes: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array]):
            c_tok, p_tok, last_t, passed_bytes = carry
            t = times[i]
            b = sizes[i]
            dt = jnp.maximum(0.0, t - last_t)
            # Token replenishment
            c_repl = jnp.minimum(5000.0, c_tok + dt * cir)
            p_repl = jnp.minimum(10000.0, p_tok + dt * pir)
            # 3-color marking: Red if p_repl < b; Yellow if c_repl < b <= p_repl; Green otherwise
            is_red = p_repl < b
            is_yellow = (p_repl >= b) & (c_repl < b)
            is_green = (p_repl >= b) & (c_repl >= b)
            new_p = jnp.where(~is_red, p_repl - b, p_repl)
            new_c = jnp.where(is_green, c_repl - b, c_repl)
            new_passed = passed_bytes + jnp.where(is_green, b, 0.0)
            return (new_c, new_p, t, new_passed)

        return lax.fori_loop(0, trip, body, (c_init, p_init, jnp.float32(0.0), jnp.float32(0.0)), unroll=factor)

    return built(fn, init_c_tokens, init_p_tokens, arrival_times, packet_sizes)


# 5. sched_round_robin_arbiter: Deficit Weighted Round-Robin Task Arbiter
def _round_robin_factory(variant: str, trip: int, factor: int):
    n_queues = {"q4": 4, "q8": 8, "q16": 16}[variant]
    rng = np.random.default_rng(583_895)
    packet_demands = jnp.asarray(rng.integers(1, 10, size=(trip, n_queues)), dtype=jnp.float32)
    weights = jnp.asarray(rng.integers(1, 4, size=n_queues), dtype=jnp.float32)
    init_deficit = jnp.zeros((n_queues,), dtype=jnp.float32)
    init_ptr = jnp.int32(0)

    def fn(def_init: jax.Array, ptr_init: jax.Array, demands: jax.Array, w_quanta: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            deficit, ptr, served_total = carry
            demand = demands[i]
            # Advance round-robin pointer and replenish deficit
            next_ptr = (ptr + 1) % n_queues
            def_repl = deficit.at[next_ptr].add(w_quanta[next_ptr])
            # Service queue if deficit >= demand
            can_serve = def_repl[next_ptr] >= demand[next_ptr]
            amt_served = jnp.where(can_serve, demand[next_ptr], 0.0)
            new_deficit = def_repl.at[next_ptr].add(-amt_served)
            return (new_deficit, next_ptr, served_total + amt_served)

        return lax.fori_loop(0, trip, body, (def_init, ptr_init, jnp.float32(0.0)), unroll=factor)

    return built(fn, init_deficit, init_ptr, packet_demands, weights)


# 6. sched_pid_anti_windup: Discrete PID Controller with Conditional Anti-Windup Clamping
def _pid_anti_windup_factory(variant: str, trip: int, factor: int):
    channels = {"ch1": 1, "ch4": 4, "ch8": 8}[variant]
    rng = np.random.default_rng(583_896)
    setpoints = floats(rng, trip, channels, scale=5.0)
    plant_outputs = floats(rng, trip, channels, scale=5.0)
    Kp = jnp.float32(1.5)
    Ki = jnp.float32(0.2)
    Kd = jnp.float32(0.05)
    limit = jnp.float32(10.0)
    init_integ = jnp.zeros((channels,), dtype=jnp.float32)
    init_prev_err = jnp.zeros((channels,), dtype=jnp.float32)

    def fn(integ_init: jax.Array, err_init: jax.Array, sp: jax.Array, pv: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            integral, prev_err, total_output = carry
            error = sp[i] - pv[i]
            # Proportional & derivative terms
            P = Kp * error
            D = Kd * (error - prev_err)
            # Anti-windup conditional clamping: only integrate if output un-saturated
            u_unsat = P + Ki * (integral + error) + D
            saturated = jnp.abs(u_unsat) > limit
            same_sign = (u_unsat * error) > 0.0
            clamp_integ = saturated & same_sign
            new_integral = jnp.where(clamp_integ, integral, integral + error)
            u_sat = jnp.clip(P + Ki * new_integral + D, -limit, limit)
            return (new_integral, error, total_output + jnp.sum(u_sat))

        return lax.fori_loop(0, trip, body, (integ_init, err_init, jnp.float32(0.0)), unroll=factor)

    return built(fn, init_integ, init_prev_err, setpoints, plant_outputs)


# 7. sched_branching_tree_walker: Static-Bound Decision Tree Feature Traversal
def _tree_walker_factory(variant: str, trip: int, factor: int):
    depth, num_features = {"d4f8": (4, 8), "d6f16": (6, 16)}[variant]
    num_nodes = 2 ** depth - 1
    rng = np.random.default_rng(583_897)
    sample_features = floats(rng, trip, num_features, scale=2.0)
    # Tree topology
    split_features = indices(rng, num_features, num_nodes)
    split_thresholds = floats(rng, num_nodes, scale=0.5)
    leaf_values = floats(rng, 2 ** depth, scale=5.0)
    init_accum = jnp.float32(0.0)

    def fn(accum_init: jax.Array, x_samples: jax.Array, f_splits: jax.Array, thresholds: jax.Array, leaves: jax.Array):
        def body(i: int, carry: jax.Array):
            x = x_samples[i]
            # Traverse depth levels down binary tree
            def tree_level(d: int, node_idx: jax.Array):
                feat = f_splits[node_idx]
                th = thresholds[node_idx]
                go_left = x[feat] <= th
                return jnp.where(go_left, 2 * node_idx + 1, 2 * node_idx + 2)

            leaf_node = lax.fori_loop(0, depth, tree_level, jnp.int32(0))
            pred = leaves[leaf_node - num_nodes]
            return carry + pred

        return lax.fori_loop(0, trip, body, accum_init, unroll=factor)

    return built(fn, init_accum, sample_features, split_features, split_thresholds, leaf_values)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec("sched_priority_queue_step", "scheduling_control", "Fixed-Capacity Array Binary Min-Heap Priority Queue",
               "sched_priority_queue", "Tree-structured index stepping with conditional sift-up and sift-down",
               "fori_loop", {"cap16": "cap 16", "cap32": "cap 32", "cap64": "cap 64"}, "cap32",
               _heap_queue_factory, 583_891, Provenance("public-inspired", "Williams (1964) Comm. ACM; heapq", "Public Domain", "Binary min-heap priority queue"),
               tier="full", related_group="sched_heap"),
    CorpusSpec("sched_fsm_traffic_controller", "scheduling_control", "Hierarchical Finite State Machine Traffic Controller",
               "sched_traffic_fsm", "Multi-way discrete state transition table with timer guards and override branches",
               "fori_loop", {"dir2": "2 directions", "dir4": "4 directions"}, "dir4",
               _traffic_fsm_factory, 583_892, Provenance("original", "Digital traffic light controller", "Internal", "Hierarchical state machine traffic arbiter"),
               tier="full", related_group="sched_fsm_eval"),
    CorpusSpec("sched_masked_active_set", "scheduling_control", "Active-Set Quadratic Programming Constraint Selection",
               "sched_active_set", "Dynamically activates most-violated constraint with hyperplane projection",
               "fori_loop", {"d8m8": "dim 8 cons 8", "d16m16": "dim 16 cons 16", "d32m32": "dim 32 cons 32"}, "d16m16",
               _active_set_factory, 583_893, Provenance("public-inspired", "Nocedal & Wright (2006) Numerical Optimization", "Public Domain", "Active set quadratic programming selection"),
               tier="smoke", related_group="sched_active_set"),
    CorpusSpec("sched_leaky_bucket_policer", "scheduling_control", "Dual-Rate Token Bucket Traffic Policer RFC 2698",
               "sched_token_bucket", "Continuous token replenishment bounded by capacities evaluating 3-color marking",
               "fori_loop", {"low": "CIR 100 PIR 200", "high": "CIR 500 PIR 1000"}, "low",
               _leaky_bucket_factory, 583_894, Provenance("public-inspired", "Heinanen & Guerin (1999) IETF RFC 2698", "Public Domain", "Dual rate token bucket traffic policer"),
               tier="full", related_group="sched_policer"),
    CorpusSpec("sched_round_robin_arbiter", "scheduling_control", "Deficit Weighted Round-Robin Task Arbiter",
               "sched_deficit_rr", "Multi-queue credit-based service allocation advancing pointer and replenishing deficit",
               "fori_loop", {"q4": "4 queues", "q8": "8 queues", "q16": "16 queues"}, "q8",
               _round_robin_factory, 583_895, Provenance("public-inspired", "Katevenis et al. (1991) IEEE JSAC", "Public Domain", "Deficit weighted round-robin scheduler"),
               tier="full", related_group="sched_drr"),
    CorpusSpec("sched_pid_anti_windup", "scheduling_control", "Discrete PID Controller with Conditional Anti-Windup Clamping",
               "sched_pid_control", "Nonlinear conditional clamping on integrator state preventing actuator saturation",
               "fori_loop", {"ch1": "1 channel", "ch4": "4 channels", "ch8": "8 channels"}, "ch4",
               _pid_anti_windup_factory, 583_896, Provenance("public-inspired", "Astrom & Hagglund (2006) Advanced PID Control", "Public Domain", "PID anti-windup controller"),
               tier="full", related_group="sched_pid"),
    CorpusSpec("sched_branching_tree_walker", "scheduling_control", "Static-Bound Decision Tree Feature Traversal",
               "sched_tree_traversal", "Branching pointer traversal guided by runtime feature evaluations",
               "fori_loop", {"d4f8": "depth 4 features 8", "d6f16": "depth 6 features 16"}, "d4f8",
               _tree_walker_factory, 583_897, Provenance("public-inspired", "Breiman et al. (1984) CART; scikit-learn", "BSD-3-Clause", "Decision tree traversal kernel"),
               tier="full", related_group="sched_tree"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
