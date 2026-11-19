"""Streaming and compression independent fori_loop workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats, indices
from benchmarks.corpus.spec import CorpusSpec, Provenance

# 1. stream_count_min_sketch: Count-Min Sketch Frequency Estimator
def _count_min_factory(variant: str, trip: int, factor: int):
    depth, width = {"d4w32": (4, 32), "d4w64": (4, 64), "d8w64": (8, 64)}[variant]
    rng = np.random.default_rng(583_881)
    keys = indices(rng, 1000, trip)
    # Hash parameters: a and b for depth universal hash functions: (depth,)
    hash_a = jnp.asarray(rng.integers(1, 10000, size=depth), dtype=jnp.int32)
    hash_b = jnp.asarray(rng.integers(0, 10000, size=depth), dtype=jnp.int32)
    init_table = jnp.zeros((depth, width), dtype=jnp.float32)

    def fn(table_init: jax.Array, key_stream: jax.Array, a_coeffs: jax.Array, b_coeffs: jax.Array):
        def body(i: int, carry: jax.Array):
            k = key_stream[i]
            # Universal hashing: h_d(k) = (a * k + b) % width
            buckets = ((a_coeffs * k + b_coeffs) % width)
            def update_row(d: int, t: jax.Array):
                b = buckets[d]
                return t.at[d, b].add(1.0)
            return lax.fori_loop(0, depth, update_row, carry)

        return lax.fori_loop(0, trip, body, table_init, unroll=factor)

    return built(fn, init_table, keys, hash_a, hash_b)


# 2. stream_run_length_encode: Streaming Run-Length Encoder with Preallocated Buffer
def _rle_factory(variant: str, trip: int, factor: int):
    cap = {"cap32": 32, "cap64": 64, "cap128": 128}[variant]
    rng = np.random.default_rng(583_882)
    # Generate tokens with repeated values (runs)
    symbols = np.repeat(rng.integers(0, 8, size=trip // 2 + 1), 2)[:trip]
    token_stream = jnp.asarray(symbols, dtype=jnp.int32)
    init_syms = jnp.zeros((cap,), dtype=jnp.int32)
    init_lens = jnp.zeros((cap,), dtype=jnp.int32)
    init_state = jnp.array([0, -1, 0], dtype=jnp.int32) # [out_idx, curr_sym, curr_len]

    def fn(syms: jax.Array, lens: jax.Array, st: jax.Array, tokens: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            s_buf, l_buf, state = carry
            out_idx, curr_sym, curr_len = state[0], state[1], state[2]
            tok = tokens[i]
            same = (tok == curr_sym)
            first = (i == 0)
            # Flush when token changes
            flush = (~same) & (~first) & (out_idx < cap)
            s_flushed = jnp.where(flush, s_buf.at[out_idx].set(curr_sym), s_buf)
            l_flushed = jnp.where(flush, l_buf.at[out_idx].set(curr_len), l_buf)
            new_idx = out_idx + jnp.where(flush, 1, 0)
            new_sym = tok
            new_len = jnp.where(same, curr_len + 1, 1)
            new_state = jnp.stack([new_idx, new_sym, new_len])
            return (s_flushed, l_flushed, new_state)

        return lax.fori_loop(0, trip, body, (syms, lens, st), unroll=factor)

    return built(fn, init_syms, init_lens, init_state, token_stream)


# 3. stream_bit_pack_u8_to_u32: Bit-Packing Sub-Byte Quantized Stream Compressor
def _bit_pack_factory(variant: str, trip: int, factor: int):
    n_words = {"w16": 16, "w32": 32, "w64": 64}[variant]
    rng = np.random.default_rng(583_883)
    symbols_4bit = indices(rng, 16, trip)
    init_words = jnp.zeros((n_words,), dtype=jnp.int32)
    init_st = jnp.zeros((3,), dtype=jnp.int32) # [accum_bits, bit_count, word_idx]

    def fn(words: jax.Array, st: jax.Array, syms: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            w_buf, state = carry
            accum, bit_count, w_idx = state[0], state[1], state[2]
            sym = syms[i] & 0x0F
            new_accum = accum | (sym << bit_count)
            new_count = bit_count + 4
            flush = (new_count >= 32) & (w_idx < n_words)
            new_w_buf = jnp.where(flush, w_buf.at[w_idx].set(new_accum), w_buf)
            final_idx = w_idx + jnp.where(flush, 1, 0)
            final_accum = jnp.where(flush, 0, new_accum)
            final_count = jnp.where(flush, 0, new_count)
            return (new_w_buf, jnp.stack([final_accum, final_count, final_idx]))

        return lax.fori_loop(0, trip, body, (words, st), unroll=factor)

    return built(fn, init_words, init_st, symbols_4bit)


# 4. stream_hyperloglog_step: HyperLogLog Distinct Cardinality Sketch [fresh_eval]
def _hll_factory(variant: str, trip: int, factor: int):
    num_registers = {"m16": 16, "m32": 32, "m64": 64}[variant]
    rng = np.random.default_rng(583_884)
    hash_stream = jnp.asarray(rng.integers(0, 2**30, size=trip), dtype=jnp.int32)
    init_regs = jnp.zeros((num_registers,), dtype=jnp.int32)

    def fn(regs_init: jax.Array, hashes: jax.Array):
        def body(i: int, carry: jax.Array):
            h = hashes[i]
            reg_idx = h % num_registers
            w = h // num_registers
            # Count leading zeros / trailing pattern: rank = clz(w) + 1
            # In integer arithmetic: count consecutive trailing zeros
            rank = jnp.minimum(31, 31 - lax.clz(w | 1))
            return carry.at[reg_idx].max(rank)

        return lax.fori_loop(0, trip, body, regs_init, unroll=factor)

    return built(fn, init_regs, hash_stream)


# 5. stream_lzw_bounded_dict: Bounded LZW Sliding Dictionary Token Matcher
def _lzw_factory(variant: str, trip: int, factor: int):
    dict_size = {"d32": 32, "d64": 64, "d128": 128}[variant]
    rng = np.random.default_rng(583_885)
    byte_stream = indices(rng, 8, trip)
    init_keys = jnp.zeros((dict_size,), dtype=jnp.int32)
    init_prefixes = jnp.zeros((dict_size,), dtype=jnp.int32)
    init_st = jnp.zeros((2,), dtype=jnp.int32) # [size, curr_prefix]

    def fn(keys: jax.Array, prefixes: jax.Array, st: jax.Array, stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            d_keys, d_pref, state = carry
            sz, prefix = state[0], state[1]
            c = stream[i]
            pair_hash = ((prefix * 31 + c) % dict_size)
            hit = (d_keys[pair_hash] == c) & (d_pref[pair_hash] == prefix) & (sz > 0)
            # On hit: extend prefix; on miss: add to dictionary if room
            new_prefix = jnp.where(hit, pair_hash, c)
            can_add = (~hit) & (sz < dict_size)
            new_keys = jnp.where(can_add, d_keys.at[pair_hash].set(c), d_keys)
            new_pref = jnp.where(can_add, d_pref.at[pair_hash].set(prefix), d_pref)
            new_sz = sz + jnp.where(can_add, 1, 0)
            return (new_keys, new_pref, jnp.stack([new_sz, new_prefix]))

        return lax.fori_loop(0, trip, body, (keys, prefixes, st), unroll=factor)

    return built(fn, init_keys, init_prefixes, init_st, byte_stream)


# 6. stream_exponential_decay_histogram: Time-Decayed Continuous Streaming Histogram
def _decay_hist_factory(variant: str, trip: int, factor: int):
    bins = {"b8": 8, "b16": 16, "b32": 32}[variant]
    rng = np.random.default_rng(583_886)
    timestamps = jnp.asarray(np.sort(rng.uniform(0.0, 10.0, size=trip)), dtype=jnp.float32)
    bin_indices = indices(rng, bins, trip)
    decay_rate = jnp.float32(0.1)
    init_hist = jnp.zeros((bins,), dtype=jnp.float32)

    def fn(hist_init: jax.Array, times: jax.Array, b_idx: jax.Array, rate: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            hist, last_t = carry
            t = times[i]
            dt = jnp.maximum(0.0, t - last_t)
            decayed = hist * jnp.exp(-rate * dt)
            updated = decayed.at[b_idx[i]].add(1.0)
            return (updated, t)

        return lax.fori_loop(0, trip, body, (hist_init, jnp.float32(0.0)), unroll=factor)

    return built(fn, init_hist, timestamps, bin_indices, decay_rate)


# 7. stream_reservoir_sampling: Streaming Reservoir Sampling with LCG State
def _reservoir_factory(variant: str, trip: int, factor: int):
    k_res, d = {"k4d4": (4, 4), "k8d4": (8, 4), "k8d8": (8, 8)}[variant]
    rng = np.random.default_rng(583_887)
    items = floats(rng, trip, d, scale=1.0)
    init_res = jnp.zeros((k_res, d), dtype=jnp.float32)
    init_rng = jnp.int32(123456789)

    def fn(res_init: jax.Array, rng_init: jax.Array, item_stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            res, rng_st = carry
            # Advance LCG: state = (state * 1103515245 + 12345) & 0x7fffffff
            next_rng = (rng_st * 1103515245 + 12345) & 0x7FFFFFFF
            u = (next_rng % 1000).astype(jnp.float32) / 1000.0
            # Replace item in reservoir if u < k / (i + 1)
            prob = jnp.minimum(1.0, float(k_res) / (i + 1.0))
            replace_slot = next_rng % k_res
            should_replace = u < prob
            new_res = jnp.where(should_replace, res.at[replace_slot].set(item_stream[i]), res)
            return (new_res, next_rng)

        return lax.fori_loop(0, trip, body, (res_init, rng_init), unroll=factor)

    return built(fn, init_res, init_rng, items)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec("stream_count_min_sketch", "streaming_compression", "Count-Min Sketch Frequency Estimator",
               "stream_count_min", "Evaluates independent universal hash functions and increments depth counters",
               "fori_loop", {"d4w32": "depth 4 width 32", "d4w64": "depth 4 width 64", "d8w64": "depth 8 width 64"}, "d4w64",
               _count_min_factory, 583_881, Provenance("public-inspired", "Cormode & Muthukrishnan (2005) J. Algorithms", "Public Domain", "Count-Min sketch update"),
               tier="full", related_group="stream_cms"),
    CorpusSpec("stream_run_length_encode", "streaming_compression", "Streaming Run-Length Encoder with Preallocated Buffer",
               "stream_rle", "Dynamic run tracking committing (symbol, count) upon symbol change",
               "fori_loop", {"cap32": "cap 32", "cap64": "cap 64", "cap128": "cap 128"}, "cap64",
               _rle_factory, 583_882, Provenance("original", "Lossless RLE compression buffer", "Internal", "Streaming run-length encoder"),
               tier="full", related_group="stream_rle"),
    CorpusSpec("stream_bit_pack_u8_to_u32", "streaming_compression", "Bit-Packing Sub-Byte Quantized Stream Compressor",
               "stream_bit_packing", "Bitwise shift and mask operations flushing to 32-bit word buffer",
               "fori_loop", {"w16": "16 words", "w32": "32 words", "w64": "64 words"}, "w32",
               _bit_pack_factory, 583_883, Provenance("original", "Quantized weight bit packing", "Internal", "Sub-byte bit packer"),
               tier="smoke", related_group="stream_bit_pack"),
    CorpusSpec("stream_hyperloglog_step", "streaming_compression", "HyperLogLog Distinct Cardinality Sketch",
               "stream_hll_cardinality", "Leading zero counter updating harmonic mean bucket registers",
               "fori_loop", {"m16": "16 registers", "m32": "32 registers", "m64": "64 registers"}, "m32",
               _hll_factory, 583_884, Provenance("public-inspired", "Flajolet et al. (2007) AOFA", "Public Domain", "HyperLogLog register update"),
               tier="full", related_group="stream_hll_eval"),
    CorpusSpec("stream_lzw_bounded_dict", "streaming_compression", "Bounded LZW Sliding Dictionary Token Matcher",
               "stream_lzw_dictionary", "Hash-probes prefix table and emits token upon dictionary miss",
               "fori_loop", {"d32": "32 dict entries", "d64": "64 dict entries", "d128": "128 dict entries"}, "d64",
               _lzw_factory, 583_885, Provenance("public-inspired", "Welch (1984) IEEE Computer; Zlib", "Public Domain", "Bounded LZW dictionary update"),
               tier="full", related_group="stream_lzw"),
    CorpusSpec("stream_exponential_decay_histogram", "streaming_compression", "Time-Decayed Continuous Streaming Histogram",
               "stream_decay_hist", "Exponentially decays histogram vector before incrementing observed bin",
               "fori_loop", {"b8": "8 bins", "b16": "16 bins", "b32": "32 bins"}, "b16",
               _decay_hist_factory, 583_886, Provenance("original", "Time-decayed telemetry monitoring", "Internal", "Decayed streaming histogram"),
               tier="full", related_group="stream_decay_hist"),
    CorpusSpec("stream_reservoir_sampling", "streaming_compression", "Streaming Reservoir Sampling with Linear Congruential State",
               "stream_reservoir", "Item replaces random reservoir slot with dynamic probability K / t",
               "fori_loop", {"k4d4": "res 4 dim 4", "k8d4": "res 8 dim 4", "k8d8": "res 8 dim 8"}, "k8d4",
               _reservoir_factory, 583_887, Provenance("public-inspired", "Vitter (1985) ACM Trans. Math. Softw.", "Public Domain", "Streaming reservoir sampling"),
               tier="full", related_group="stream_reservoir"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
