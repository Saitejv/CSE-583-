"""Reviewed workloads drafted with agy and Gemini 3.8 Flash."""

import jax.numpy as jnp
import numpy as np
from jax import lax


def make_nb_streaming_loglik(variant="small", trip=256, factor=1):
    num_models, feat_dim = (8, 4) if variant == "small" else (32, 8)
    rng = np.random.default_rng(1337)

    y_counts = rng.poisson(lam=5.0, size=(trip,)).astype(np.float32)
    features = rng.normal(size=(trip, feat_dim)).astype(np.float32) * 0.2
    weights = rng.normal(size=(num_models, feat_dim)).astype(np.float32) * 0.3
    dispersion = rng.uniform(1.0, 5.0, size=(num_models,)).astype(np.float32)

    j_y = jnp.array(y_counts)
    j_features = jnp.array(features)
    j_weights = jnp.array(weights)
    j_dispersion = jnp.array(dispersion)

    init_ll_acc = jnp.zeros((num_models,), dtype=jnp.float32)
    init_latent_bias = jnp.zeros((feat_dim,), dtype=jnp.float32)
    init_carry = (init_ll_acc, init_latent_bias)

    def run(init_c, y_stream, x_stream, w_bank, r_bank):
        def body(i, carry):
            ll_acc, latent_bias = carry
            y_i = y_stream[i]
            x_i = x_stream[i] + 0.05 * latent_bias

            # Predictor eta and bounded mean parameter
            eta = jnp.dot(w_bank, x_i)
            mu = jnp.exp(jnp.clip(eta, -6.0, 6.0))

            # Negative-binomial log likelihood using log1p and lgamma:
            # log P(y|r, mu) = lgamma(y + r) - lgamma(r) - lgamma(y + 1)
            #                  - r * log1p(mu / r) + y * (log(mu) - log(r) - log1p(mu / r))
            ratio = mu / r_bank
            term_gamma = (
                lax.lgamma(y_i + r_bank) - lax.lgamma(r_bank) - lax.lgamma(y_i + 1.0)
            )
            log1p_term = jnp.log1p(ratio)
            log_ratio = jnp.log(jnp.maximum(ratio, 1e-7))
            step_ll = (
                term_gamma - r_bank * log1p_term + y_i * (log_ratio - log1p_term)
            )

            next_ll_acc = ll_acc + step_ll

            # Normalized carry recurrence to steer subsequent state
            next_latent_bias = jnp.tanh(
                jnp.dot(w_bank.T, step_ll) / jnp.float32(num_models)
            )

            return (next_ll_acc, next_latent_bias)

        return lax.fori_loop(0, trip, body, init_c, unroll=factor)

    return run, (init_carry, j_y, j_features, j_weights, j_dispersion)


def piecewise_spectral_equalizer_factory(variant="small", trip=256, factor=1):
    num_bands = 16 if variant == "small" else 48
    rng = np.random.default_rng(1337)

    # Input frequency band spectra and dynamic target envelopes (float32)
    spectra_np = rng.uniform(0.1, 1.5, size=(trip, num_bands)).astype(
        np.float32
    )
    targets_np = rng.uniform(0.2, 1.2, size=(trip, num_bands)).astype(
        np.float32
    )
    weights_np = rng.uniform(0.5, 2.0, size=(trip, num_bands)).astype(
        np.float32
    )

    spectra = jnp.asarray(spectra_np)
    targets = jnp.asarray(targets_np)
    weights = jnp.asarray(weights_np)

    init_gain = jnp.ones((num_bands,), dtype=jnp.float32)
    init_accum_error = jnp.array(0.0, dtype=jnp.float32)
    init_carry = (init_gain, init_accum_error)

    def run(spectra_in, targets_in, weights_in, carry_init):
        def body_fn(i, carry):
            gain, accum_err = carry
            x_i = spectra_in[i]
            t_i = targets_in[i]
            w_i = weights_in[i]

            # Nonlinear gain compression preventing algebraic elimination
            activated = jnp.tanh(gain * x_i)
            band_error = (activated - t_i) * w_i

            # Smooth carry update with bounded leaky contraction
            gain_grad = band_error * x_i * (1.0 - activated * activated)
            gain_next = 0.94 * gain - 0.04 * jnp.sin(gain_grad) + 0.02
            accum_next = (
                accum_err + jnp.sum(band_error * band_error) * (1.0 / num_bands)
            )

            return (gain_next, accum_next)

        return lax.fori_loop(0, trip, body_fn, carry_init, unroll=factor)

    args = (spectra, targets, weights, init_carry)
    return run, args


def packet_reorder_scoreboard_factory(variant="small", trip=256, factor=1):
    num_words = 4 if variant == "small" else 16
    window_bits = num_words * 32
    rng = np.random.default_rng(2024)

    # Ingress packet stream: relative offsets from current expected sequence
    # and associated payload checksum contributions
    offsets_np = rng.integers(0, window_bits, size=(trip,)).astype(np.int32)
    payload_np = rng.uniform(0.5, 4.0, size=(trip,)).astype(np.float32)

    offsets = jnp.asarray(offsets_np)
    payloads = jnp.asarray(payload_np)

    init_bitmap = jnp.zeros((num_words,), dtype=jnp.uint32)
    init_retired = jnp.array(0, dtype=jnp.int32)
    init_checksum = jnp.array(1.0, dtype=jnp.float32)
    init_carry = (init_bitmap, init_retired, init_checksum)

    def run(offsets_in, payloads_in, carry_init):
        def body_fn(i, carry):
            bitmap, retired, checksum = carry
            pkt_offset = offsets_in[i]
            pkt_data = payloads_in[i]

            # Determine target bitmap word and bit position
            word_idx = (pkt_offset >> 5) % num_words
            bit_mask = jnp.uint32(1) << (pkt_offset & 31).astype(jnp.uint32)

            # Mark packet arrived
            curr_word = bitmap[word_idx]
            updated_word = curr_word | bit_mask
            bitmap_marked = bitmap.at[word_idx].set(updated_word)

            # Carry-dependent bitmap select: inspect base word of current reorder window
            base_word_idx = (retired >> 5) % num_words
            inspect_word = bitmap_marked[base_word_idx]

            # Associative clearing: clear lowest set bit (x & (x - 1))
            cleared_word = inspect_word & (inspect_word - jnp.uint32(1))
            bit_popped = inspect_word ^ cleared_word
            bitmap_next = bitmap_marked.at[base_word_idx].set(cleared_word)

            # State feedback: conditional progression of retired counter and checksum
            retired_step = jnp.where(inspect_word != jnp.uint32(0), 1, 0)
            retired_next = retired + retired_step
            checksum_next = (
                checksum * 0.999
                + pkt_data * (1.0 + jnp.float32(bit_popped & jnp.uint32(255)))
            )

            return (bitmap_next, retired_next, checksum_next)

        return lax.fori_loop(0, trip, body_fn, carry_init, unroll=factor)

    args = (offsets, payloads, init_carry)
    return run, args


def compressed_trie_walk_factory(variant="small", trip=256, factor=1):
    num_nodes = 256 if variant == "small" else 1024
    num_symbols = 4

    rng = np.random.default_rng(42)
    # Transitions table: positive integer index for valid child, -1 for miss
    transitions = rng.integers(0, num_nodes, size=(num_nodes, num_symbols), dtype=np.int32)
    transitions[rng.random(transitions.shape) < .25] = -1
    # Failure transition fallback per node
    failure_links = rng.integers(0, max(1, num_nodes // 4), size=(num_nodes,), dtype=np.int32)
    node_weights = rng.uniform(0.05, 1.0, size=(num_nodes,)).astype(np.float32)
    symbols = rng.integers(0, num_symbols, size=(trip,), dtype=np.int32)

    j_symbols = jnp.array(symbols)
    j_transitions = jnp.array(transitions)
    j_failure_links = jnp.array(failure_links)
    j_node_weights = jnp.array(node_weights)

    args = (j_symbols, j_transitions, j_failure_links, j_node_weights)

    def run(syms, trans, fails, weights):
        def body(i, carry):
            node, accum = carry
            sym = syms[i]
            cand = trans[node, sym]
            fail = fails[node]
            next_node = jnp.where(cand >= 0, cand, fail)
            w = weights[next_node]
            new_accum = accum * 0.999 + w * 0.001
            return (next_node, new_accum)

        init = (jnp.int32(0), jnp.float32(0.0))
        return lax.fori_loop(0, trip, body, init, unroll=factor)

    return run, args


def packed_prefix_rank_factory(variant="small", trip=256, factor=1):
    num_lanes = 8 if variant == "small" else 32

    rng = np.random.default_rng(2024)
    bit_blocks = rng.integers(0, 0x7FFFFFFF, size=(trip, num_lanes), dtype=np.int32)
    super_ranks = rng.integers(0, 32, size=(trip, num_lanes), dtype=np.int32)

    j_bit_blocks = jnp.array(bit_blocks)
    j_super_ranks = jnp.array(super_ranks)

    args = (j_bit_blocks, j_super_ranks)

    def run(blocks, super_r):
        def popcount32(v):
            v = v - ((v >> 1) & 0x55555555)
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
            v = (v + (v >> 4)) & 0x0F0F0F0F
            v = v + (v >> 8)
            v = v + (v >> 16)
            return v & 0x3F

        def body(i, carry):
            pos, checksum = carry
            w = blocks[i]
            shift = pos & 31
            # Dynamic low-order mask dependent on carried bit position
            mask = (1 << shift) - 1
            masked_w = w & mask
            rank = popcount32(masked_w)

            s = super_r[i]
            # Carry update with dependent integer recurrence
            next_pos = (pos + rank * 3 + s) & 0x7FFFFFFF
            lane_mean = jnp.mean(rank.astype(jnp.float32))
            next_checksum = checksum * 0.999 + lane_mean * 0.001

            return (next_pos, next_checksum)

        init = (jnp.zeros((num_lanes,), dtype=jnp.int32), jnp.float32(0.0))
        return lax.fori_loop(0, trip, body, init, unroll=factor)

    return run, args
