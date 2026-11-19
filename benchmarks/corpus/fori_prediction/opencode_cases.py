"""Static fori_loop workloads drafted by OpenCode and reviewed for the prediction benchmark."""

import numpy as np
import jax.numpy as jnp
from jax import lax


def oc_ti_codebook_decode(variant, trip, factor):
    n, t = (4096, 257) if variant == "small" else (65536, 1021)
    rng = np.random.default_rng(101)
    codes = rng.integers(0, t, size=n, dtype=np.int32)
    table = rng.standard_normal(t).astype(np.float32)

    def fn(codes, table):
        n = codes.shape[0]
        initial = jnp.float32(0.0)

        def body(i, acc):
            return acc + table[codes[i % n]]

        return lax.fori_loop(0, trip, body, initial, unroll=factor)

    return fn, (codes, table)


def oc_ti_expert_route(variant, trip, factor):
    n, e, d = (2048, 32, 64) if variant == "small" else (8192, 64, 128)
    rng = np.random.default_rng(102)
    logits = rng.standard_normal((n, e)).astype(np.float32)
    table = (0.05 * rng.standard_normal((e, d))).astype(np.float32)

    def fn(logits, table):
        n = logits.shape[0]
        d = table.shape[1]
        initial = jnp.zeros((d,), jnp.float32)

        def body(i, acc):
            row = logits[i % n]
            weights = jnp.exp(row - jnp.max(row))
            weights = weights / jnp.sum(weights)
            return acc + weights @ table

        return lax.fori_loop(0, trip, body, initial, unroll=factor)

    return fn, (logits, table)


def oc_ti_embedding_bag(variant, trip, factor):
    n, t, d = (4096, 512, 64) if variant == "small" else (32768, 2048, 128)
    rng = np.random.default_rng(103)
    ids = rng.integers(0, t, size=n, dtype=np.int32)
    table = (0.1 * rng.standard_normal((t, d))).astype(np.float32)

    def fn(ids, table):
        n = ids.shape[0]
        d = table.shape[1]
        initial = jnp.zeros((d,), jnp.float32)

        def body(i, acc):
            return acc + table[ids[i % n]]

        return lax.fori_loop(0, trip, body, initial, unroll=factor)

    return fn, (ids, table)


def oc_rec_iir_lowpass(variant, trip, factor):
    n = 4096 if variant == "small" else 32768
    rng = np.random.default_rng(104)
    x = rng.standard_normal(n).astype(np.float32)
    r = rng.random(2)
    coeff = np.array(
        [0.85 + 0.1 * r[0], 0.02 + 0.05 * r[1]], dtype=np.float32
    )

    def fn(x, coeff):
        n = x.shape[0]
        initial = jnp.float32(0.0)

        def body(i, y):
            return coeff[0] * y + coeff[1] * x[i % n]

        return lax.fori_loop(0, trip, body, initial, unroll=factor)

    return fn, (x, coeff)


def oc_rec_spring_damper(variant, trip, factor):
    n = 4096 if variant == "small" else 32768
    rng = np.random.default_rng(105)
    force = rng.standard_normal(n).astype(np.float32)
    r = rng.random(3)
    params = np.array(
        [
            0.004 + 0.004 * r[0],
            0.5 + 1.0 * r[1],
            0.1 + 0.2 * r[2],
        ],
        dtype=np.float32,
    )

    def fn(force, params):
        n = force.shape[0]
        initial = jnp.zeros((2,), jnp.float32)

        def body(i, state):
            p, v = state
            dt, k, c = params[0], params[1], params[2]
            v2 = v + dt * (force[i % n] - k * p - c * v)
            p2 = p + dt * v2
            return jnp.stack([p2, v2])

        return lax.fori_loop(0, trip, body, initial, unroll=factor)

    return fn, (force, params)


def oc_wc_token_dispatch(variant, trip, factor):
    n, e, d = (4096, 16, 64) if variant == "small" else (16384, 32, 128)
    rng = np.random.default_rng(108)
    dest = rng.integers(0, e, size=n, dtype=np.int32)
    tokens = rng.standard_normal((n, d)).astype(np.float32)

    def fn(dest, tokens):
        n = dest.shape[0]
        initial = jnp.zeros((e, tokens.shape[1]), jnp.float32)

        def body(i, acc):
            return acc.at[dest[i % n]].add(tokens[i % n])

        return lax.fori_loop(0, trip, body, initial, unroll=factor)

    return fn, (dest, tokens)


def oc_wc_scatter_max(variant, trip, factor):
    n, b = (8192, 128) if variant == "small" else (32768, 512)
    rng = np.random.default_rng(109)
    idx = rng.integers(0, b, size=n, dtype=np.int32)
    vals = rng.random(n).astype(np.float32)

    def fn(idx, vals):
        n = idx.shape[0]
        initial = jnp.zeros((b,), jnp.float32)

        def body(i, acc):
            return acc.at[idx[i % n]].max(vals[i % n])

        return lax.fori_loop(0, trip, body, initial, unroll=factor)

    return fn, (idx, vals)


def oc_nr_streaming_topk(variant, trip, factor):
    n, k = (4096, 8) if variant == "small" else (32768, 16)
    rng = np.random.default_rng(110)
    x = rng.standard_normal(n).astype(np.float32)

    def fn(x):
        n = x.shape[0]
        initial = jnp.full((k,), -1e9, jnp.float32)

        def body(i, top):
            candidate = jnp.concatenate(
                [top, jnp.reshape(x[i % n], (1,))]
            )
            return jnp.sort(candidate)[-k:]

        return lax.fori_loop(0, trip, body, initial, unroll=factor)

    return fn, (x,)


def oc_nr_row_reduce(variant, trip, factor):
    n, d = (2048, 64) if variant == "small" else (8192, 128)
    rng = np.random.default_rng(111)
    x = rng.standard_normal((n, d)).astype(np.float32)

    def fn(x):
        n = x.shape[0]
        initial = jnp.float32(0.0)

        def body(i, acc):
            row = x[i % n]
            return acc + jnp.sum(row * row)

        return lax.fori_loop(0, trip, body, initial, unroll=factor)

    return fn, (x,)


def oc_nr_stream_moments(variant, trip, factor):
    n = 4096 if variant == "small" else 32768
    rng = np.random.default_rng(112)
    x = rng.standard_normal(n).astype(np.float32)

    def fn(x):
        n = x.shape[0]
        initial = (jnp.float32(0.0), jnp.float32(0.0))

        def body(i, state):
            mean, m2 = state
            value = x[i % n]
            delta = value - mean
            mean2 = mean + delta / (i + 1).astype(jnp.float32)
            m2_2 = m2 + delta * (value - mean2)
            return mean2, m2_2

        return lax.fori_loop(0, trip, body, initial, unroll=factor)

    return fn, (x,)


WORKLOADS = {
    "oc_ti_codebook_decode": ("table_index", oc_ti_codebook_decode),
    "oc_ti_expert_route": ("table_index", oc_ti_expert_route),
    "oc_ti_embedding_bag": ("table_index", oc_ti_embedding_bag),
    "oc_rec_iir_lowpass": ("recurrence", oc_rec_iir_lowpass),
    "oc_rec_spring_damper": ("recurrence", oc_rec_spring_damper),
    "oc_wc_token_dispatch": ("write_conflict", oc_wc_token_dispatch),
    "oc_wc_scatter_max": ("write_conflict", oc_wc_scatter_max),
    "oc_nr_streaming_topk": ("numeric_reduce", oc_nr_streaming_topk),
    "oc_nr_row_reduce": ("numeric_reduce", oc_nr_row_reduce),
    "oc_nr_stream_moments": ("numeric_reduce", oc_nr_stream_moments),
}
