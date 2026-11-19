"""Dynamic programming independent fori_loop workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats, indices
from benchmarks.corpus.spec import CorpusSpec, Provenance

# 1. dp_banded_smith_waterman: Banded Smith-Waterman Sequence Alignment [fresh_eval]
def _smith_waterman_factory(variant: str, trip: int, factor: int):
    band = {"band16": 16, "band32": 32, "band64": 64}[variant]
    rng = np.random.default_rng(583_841)
    scores = floats(rng, trip, band, scale=2.0)
    gap = jnp.float32(-1.0)
    init_diag_prev = jnp.zeros((band,), dtype=jnp.float32)
    init_diag_curr = jnp.zeros((band,), dtype=jnp.float32)

    def fn(d_prev: jax.Array, d_curr: jax.Array, sim_stream: jax.Array, gap_pen: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            prev2, prev1 = carry
            sim = sim_stream[i]
            # 3-way banded recurrence
            diag_score = prev2 + sim
            up_score = prev1 + gap_pen
            left_score = jnp.roll(prev1, 1) + gap_pen
            new_curr = jnp.maximum(0.0, jnp.maximum(diag_score, jnp.maximum(up_score, left_score)))
            return (prev1, new_curr)

        return lax.fori_loop(0, trip, body, (d_prev, d_curr), unroll=factor)

    return built(fn, init_diag_prev, init_diag_curr, scores, gap)


# 2. dp_viterbi_max_plus: Viterbi Trellis Decoding in Max-Plus Semiring
def _viterbi_factory(variant: str, trip: int, factor: int):
    states = {"states8": 8, "states16": 16, "states32": 32}[variant]
    rng = np.random.default_rng(583_842)
    log_emissions = floats(rng, trip, states, scale=2.0)
    log_trans = floats(rng, states, states, scale=1.0)
    init_probs = jnp.zeros((states,), dtype=jnp.float32)

    def fn(p_init: jax.Array, emis: jax.Array, trans: jax.Array):
        def body(i: int, carry: jax.Array):
            # max-plus tropical matrix multiplication: carry[k] + trans[k, j]
            scores = carry[:, None] + trans
            best_prev = jnp.max(scores, axis=0)
            return best_prev + emis[i]

        return lax.fori_loop(0, trip, body, p_init, unroll=factor)

    return built(fn, init_probs, log_emissions, log_trans)


# 3. dp_edit_distance_row: Rolling Wagner-Fischer Levenshtein Edit Distance
def _edit_dist_factory(variant: str, trip: int, factor: int):
    target_len = {"t32": 32, "t64": 64, "t128": 128}[variant]
    rng = np.random.default_rng(583_843)
    query_tokens = indices(rng, 10, trip)
    target_tokens = indices(rng, 10, target_len)
    init_row = jnp.arange(target_len + 1, dtype=jnp.float32)

    def fn(row_init: jax.Array, q_tokens: jax.Array, t_tokens: jax.Array):
        def body(i: int, carry: jax.Array):
            q_char = q_tokens[i]
            # Carry is row of length target_len + 1
            # For each position j in 1..target_len:
            def inner(j: int, new_row: jax.Array):
                cost = jnp.where(q_char == t_tokens[j - 1], 0.0, 1.0)
                sub = carry[j - 1] + cost
                ins = new_row[j - 1] + 1.0
                delete = carry[j] + 1.0
                return new_row.at[j].set(jnp.minimum(sub, jnp.minimum(ins, delete)))

            first_elem = jnp.asarray(i + 1, dtype=jnp.float32)
            temp = jnp.zeros((target_len + 1,), dtype=jnp.float32).at[0].set(first_elem)
            updated = lax.fori_loop(1, target_len + 1, inner, temp)
            return updated

        return lax.fori_loop(0, trip, body, row_init, unroll=factor)

    return built(fn, init_row, query_tokens, target_tokens)


# 4. dp_knapsack_bounded: Bounded 0/1 Knapsack Dynamic Programming [fresh_eval]
def _knapsack_factory(variant: str, trip: int, factor: int):
    cap = {"cap64": 64, "cap128": 128, "cap256": 256}[variant]
    rng = np.random.default_rng(583_844)
    item_weights = indices(rng, 16, trip) + 1
    item_values = floats(rng, trip, scale=5.0) + 1.0
    init_dp = jnp.zeros((cap + 1,), dtype=jnp.float32)

    def fn(dp_init: jax.Array, w_arr: jax.Array, v_arr: jax.Array):
        def body(i: int, carry: jax.Array):
            w = w_arr[i]
            v = v_arr[i]
            # Reverse-capacity sweep logic via rolled vector op
            shifted = jnp.roll(carry, w) + v
            # Valid mask: indices >= w
            mask = jnp.arange(cap + 1) >= w
            cand = jnp.where(mask, shifted, carry)
            return jnp.maximum(carry, cand)

        return lax.fori_loop(0, trip, body, dp_init, unroll=factor)

    return built(fn, init_dp, item_weights, item_values)


# 5. dp_longest_common_subsequence: Rolling Longest Common Subsequence Length Vector
def _lcs_factory(variant: str, trip: int, factor: int):
    target_len = {"t32": 32, "t64": 64, "t128": 128}[variant]
    rng = np.random.default_rng(583_845)
    seq1 = indices(rng, 8, trip)
    seq2 = indices(rng, 8, target_len)
    init_row = jnp.zeros((target_len + 1,), dtype=jnp.float32)

    def fn(row_init: jax.Array, s1: jax.Array, s2: jax.Array):
        def body(i: int, carry: jax.Array):
            c1 = s1[i]
            def inner(j: int, new_row: jax.Array):
                c2 = s2[j - 1]
                match_val = carry[j - 1] + 1.0
                nomatch_val = jnp.maximum(carry[j], new_row[j - 1])
                val = jnp.where(c1 == c2, match_val, nomatch_val)
                return new_row.at[j].set(val)

            temp = jnp.zeros((target_len + 1,), dtype=jnp.float32)
            updated = lax.fori_loop(1, target_len + 1, inner, temp)
            return updated

        return lax.fori_loop(0, trip, body, row_init, unroll=factor)

    return built(fn, init_row, seq1, seq2)


# 6. dp_matrix_chain_tile: Matrix Chain Optimal Substructure Tile Sweep
def _matrix_chain_factory(variant: str, trip: int, factor: int):
    n = {"n16": 16, "n32": 32}[variant]
    rng = np.random.default_rng(583_846)
    dims = jnp.asarray(rng.integers(5, 30, size=n + 1), dtype=jnp.float32)
    init_m = jnp.zeros((n, n), dtype=jnp.float32)

    def fn(m_init: jax.Array, d: jax.Array):
        def body(i: int, carry: jax.Array):
            # Chain length l from 2 to n
            l = (i % (n - 1)) + 2
            def row_step(j: int, m_table: jax.Array):
                k = j + l - 1
                valid = k < n
                k_idx = jnp.minimum(n - 1, k)
                # Compute cost for split point p
                def split_step(p: int, best: jax.Array):
                    cost = m_table[j, p] + m_table[p + 1, k_idx] + d[j] * d[p + 1] * d[k_idx + 1]
                    return jnp.minimum(best, cost)
                best_cost = lax.fori_loop(j, k_idx, split_step, jnp.float32(1e9))
                new_val = jnp.where(valid, best_cost, m_table[j, k_idx])
                return m_table.at[j, k_idx].set(new_val)
            return lax.fori_loop(0, n - 1, row_step, carry)

        return lax.fori_loop(0, trip, body, m_init, unroll=factor)

    return built(fn, init_m, dims)


# 7. dp_dtw_banded_step: Banded Dynamic Time Warping with Sakoe-Chiba Constraint
def _dtw_factory(variant: str, trip: int, factor: int):
    band, feat_dim = {"b16d4": (16, 4), "b32d4": (32, 4), "b64d8": (64, 8)}[variant]
    rng = np.random.default_rng(583_847)
    X = floats(rng, trip, feat_dim, scale=1.0)
    Y_band = floats(rng, band, feat_dim, scale=1.0)
    init_costs = jnp.full((band,), 1e5, dtype=jnp.float32).at[0].set(0.0)

    def fn(c_init: jax.Array, x_stream: jax.Array, y_template: jax.Array):
        def body(i: int, carry: jax.Array):
            x = x_stream[i]
            # Local distance across band
            local_dists = jnp.sum((y_template - x) ** 2, axis=-1)
            # 3-way predecessor minimum: match (carry[j-1]), insert (carry[j]), delete (new[j-1])
            def inner(j: int, new_c: jax.Array):
                pred_diag = jnp.where(j > 0, carry[j - 1], 1e5)
                pred_up = carry[j]
                pred_left = jnp.where(j > 0, new_c[j - 1], 1e5)
                m = jnp.minimum(pred_diag, jnp.minimum(pred_up, pred_left))
                return new_c.at[j].set(m + local_dists[j])

            temp = jnp.zeros((band,), dtype=jnp.float32)
            updated = lax.fori_loop(0, band, inner, temp)
            return updated

        return lax.fori_loop(0, trip, body, c_init, unroll=factor)

    return built(fn, init_costs, X, Y_band)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec("dp_banded_smith_waterman", "dynamic_programming", "Banded Smith-Waterman Sequence Alignment Wavefront",
               "dp_banded_alignment", "2D wavefront antidiagonal recurrence bounded below by zero",
               "fori_loop", {"band16": "band 16", "band32": "band 32", "band64": "band 64"}, "band32",
               _smith_waterman_factory, 583_841, Provenance("public-inspired", "Smith & Waterman (1981); Durbin et al.", "Public Domain", "Banded local alignment dynamic programming"),
               tier="full", related_group="dp_align_eval"),
    CorpusSpec("dp_viterbi_max_plus", "dynamic_programming", "Viterbi Trellis Decoding in Max-Plus Semiring",
               "dp_viterbi_trellis", "Max-plus tropical algebra state transition recurrence",
               "fori_loop", {"states8": "8 states", "states16": "16 states", "states32": "32 states"}, "states16",
               _viterbi_factory, 583_842, Provenance("public-inspired", "Viterbi (1967) IEEE Trans. Inf. Theory", "Public Domain", "Viterbi max-plus semiring trellis decoding"),
               tier="smoke", related_group="dp_viterbi"),
    CorpusSpec("dp_edit_distance_row", "dynamic_programming", "Rolling Wagner-Fischer Levenshtein Edit Distance",
               "dp_edit_distance", "1D DP wavefront evaluating insertion, deletion, and substitution",
               "fori_loop", {"t32": "target 32", "t64": "target 64", "t128": "target 128"}, "t64",
               _edit_dist_factory, 583_843, Provenance("public-inspired", "Wagner & Fischer (1974) J. ACM", "Public Domain", "Levenshtein edit distance row sweep"),
               tier="full", related_group="dp_edit_dist"),
    CorpusSpec("dp_knapsack_bounded", "dynamic_programming", "Bounded 0/1 Knapsack Dynamic Programming Row Transition",
               "dp_knapsack", "Reverse-capacity sweep evaluating 0/1 item consumption",
               "fori_loop", {"cap64": "cap 64", "cap128": "cap 128", "cap256": "cap 256"}, "cap128",
               _knapsack_factory, 583_844, Provenance("public-inspired", "Bellman (1957); CLRS Chapter 16", "Public Domain", "0/1 knapsack dynamic programming"),
               tier="full", related_group="dp_knapsack_eval"),
    CorpusSpec("dp_longest_common_subsequence", "dynamic_programming", "Rolling Longest Common Subsequence Length Vector",
               "dp_lcs", "Equality match logic incrementing diagonal value vs horizontal/vertical maximum",
               "fori_loop", {"t32": "target 32", "t64": "target 64", "t128": "target 128"}, "t64",
               _lcs_factory, 583_845, Provenance("public-inspired", "Hirschberg (1975) Comm. ACM", "Public Domain", "LCS dynamic programming row sweep"),
               tier="full", related_group="dp_lcs"),
    CorpusSpec("dp_matrix_chain_tile", "dynamic_programming", "Matrix Chain Optimal Substructure Tile Sweep",
               "dp_matrix_chain", "Subproblem split-point minimization across split coordinates",
               "fori_loop", {"n16": "dim 16", "n32": "dim 32"}, "n16",
               _matrix_chain_factory, 583_846, Provenance("public-inspired", "Cormen et al. (2009) CLRS", "Public Domain", "Matrix chain multiplication DP table"),
               tier="full", related_group="dp_matrix_chain"),
    CorpusSpec("dp_dtw_banded_step", "dynamic_programming", "Banded Dynamic Time Warping with Sakoe-Chiba Constraint",
               "dp_dtw_banded", "Continuous Euclidean metric with 3-way predecessor minimum",
               "fori_loop", {"b16d4": "band 16 dim 4", "b32d4": "band 32 dim 4", "b64d8": "band 64 dim 8"}, "b32d4",
               _dtw_factory, 583_847, Provenance("public-inspired", "Sakoe & Chiba (1978) IEEE Trans. ASSP; fastdtw", "Public Domain", "Dynamic time warping banded step"),
               tier="full", related_group="dp_dtw"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
