import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_unroll.library_kernels import add_reduceat_rows, subtract_reduce_rows


@pytest.mark.parametrize("shape", [(257, 1), (512, 8), (5, 3), (128, 20), (512, 1)])
def test_subtraction_keeps_left_to_right_rounding(shape):
    data = jnp.asarray(np.random.default_rng(17).normal(size=shape), dtype=jnp.float32)
    np.testing.assert_array_equal(subtract_reduce_rows(data), jnp.subtract.reduce(data, axis=0))


@pytest.mark.parametrize("starts", [[0, 12, 19, 201], [2, 2, 1, 255], [10, -1, 0, 12], [256, 1, 10, 0]])
def test_segmented_sum_matches_jax(starts):
    data = jnp.linspace(-1, 2, 256)[:, None]
    indices = jnp.array(starts, dtype=jnp.int32)
    np.testing.assert_array_equal(add_reduceat_rows(data, indices), jnp.add.reduceat(data, indices))


def test_segmented_sum_falls_back_for_wide_rows():
    data = jnp.arange(512).reshape(256, 2)
    indices = jnp.array([1, 2, 41])
    np.testing.assert_array_equal(add_reduceat_rows(data, indices), jnp.add.reduceat(data, indices))


@pytest.mark.parametrize("name,factor", [("subtract", 8), ("reduceat", 4)])
def test_native_graph_changes_only_at_unroll_parameter(name, factor):
    if name == "subtract":
        native, optimized = jnp.subtract.reduce, subtract_reduce_rows
        args = (jnp.ones((512, 8), dtype=jnp.float32),)
    else:
        native, optimized = jnp.add.reduceat, add_reduceat_rows
        args = (jnp.ones((256, 1), dtype=jnp.float32), jnp.array([0, 8, 16, 32], dtype=jnp.int32))
    original = str(jax.make_jaxpr(native)(*args))
    actual = str(jax.make_jaxpr(optimized.__wrapped__)(*args))
    assert original.count("unroll=1\n") == 1
    assert actual == original.replace("unroll=1\n", f"unroll={factor}\n")


def test_unconfirmed_signature_keeps_native_graph():
    args = (jnp.ones((512, 1), dtype=jnp.float32),)
    assert str(jax.make_jaxpr(subtract_reduce_rows.__wrapped__)(*args)) == str(
        jax.make_jaxpr(jnp.subtract.reduce)(*args)
    )
