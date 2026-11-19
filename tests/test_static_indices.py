"""Semantic tests for the experimental static index block."""
import jax
import jax.numpy as jnp
from jax import lax

from jax_unroll.features import extract_loop_body
from jax_unroll.static_indices import INDEX_COLUMNS, index_features


def test_kinds_include_add_and_min_and_assignment():
    def body(c, x):
        c = c.at[x].add(1.)
        c = c.at[x].min(2.)
        c = c.at[x].set(3.)
        return c, c[x]
    loop = extract_loop_body(body_fn=body, body_args=(jnp.zeros(8), jnp.array([2, 3])), body_num_xs=1)
    features = index_features(loop)
    assert tuple(features) == INDEX_COLUMNS
    assert features['index_gather_count'] == 1
    assert features['index_scatter_count'] == 1
    assert features['index_scatter_add_count'] == 1
    assert features['index_scatter_min_count'] == 1
    assert features['index_from_scanned'] == 4
    assert features['index_from_carry'] == features['index_from_constant'] == 0


def test_counter_constant_and_carry_sources():
    def body(i, c):
        return c.at[i % 8].add(c[jnp.asarray(2)])
    loop = extract_loop_body(body_fn=body, body_args=(jnp.int32(0), jnp.zeros(8)), loop_kind='fori_loop')
    f = index_features(loop)
    # Literal scalar indexing may use static slice, not gather; scatter sees counter.
    assert f['index_from_counter'] == 1
    assert f['index_from_carry'] == 0

    def carried(c):
        idx, values = c
        return idx + 1, values.at[idx % 8].add(1.)
    loop = extract_loop_body(body_fn=carried, body_args=((jnp.int32(0), jnp.zeros(8)),))
    assert index_features(loop)['index_from_carry'] == 1

    def invariant(table, c):
        return c.at[table].add(1.)
    loop = extract_loop_body(body_fn=invariant, body_args=(jnp.array([1, 2]), jnp.zeros(8)), body_num_consts=1)
    assert index_features(loop)['index_from_constant'] == 1


def test_nested_jit_propagates_sources_and_mixed_sources():
    @jax.jit
    def read(values, idx):
        return values[idx]
    def body(c, x):
        i, total, values = c
        return (i + 1, total + jnp.sum(read(values, (i + x) % 8)), values), None
    loop = extract_loop_body(body_fn=body, body_args=((jnp.int32(0), jnp.float32(0), jnp.arange(8.)), jnp.array([2, 3])), body_num_xs=1)
    f = index_features(loop)
    assert f['index_gather_count'] == 1
    assert f['index_from_carry'] == f['index_from_scanned'] == 1
    assert f['index_mixed_sources_count'] == 1
    assert f['index_opaque_call_count'] == 0


def test_runtime_locality_and_collisions_are_invisible():
    def fn(c, indices):
        return lax.scan(lambda c, x: (c.at[x].add(1.), None), c, indices)[0]
    clustered = jnp.zeros((8, 4), dtype=jnp.int32)
    dispersed = jnp.arange(32, dtype=jnp.int32).reshape(8, 4) % 16
    a = extract_loop_body(fn, (jnp.zeros(16), clustered), loop_kind='scan')
    b = extract_loop_body(fn, (jnp.zeros(16), dispersed), loop_kind='scan')
    assert str(a.jaxpr) == str(b.jaxpr)
    assert index_features(a) == index_features(b)
