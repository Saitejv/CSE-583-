import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_unroll.auto_loop import auto_fori_loop, precompute_fori_loop
from jax_unroll.model import LogCosts
from jax_unroll.schema import FACTORS, FEATURE_VERSION
from jax_unroll.selector import DecisionEngine


def step(i, carry):
    return {"values": (carry["values"][0] * 0.5 + i, carry["values"][1] + i)}


@pytest.mark.parametrize("options,factor", [
    ({}, 1),
    ({"force_factor": 3}, 3),
    ({"force_factor": 4, "disable": True}, 1),
    ({"unroll": True}, 17),
])
def test_static_pytree_and_unroll_only(options, factor):
    def native(carry):
        return jax.lax.fori_loop(-3, 14, step, carry)

    def wrapped(carry):
        return auto_fori_loop(-3, 14, step, carry, **options)

    carry = {"values": (jnp.arange(3, dtype=jnp.float32), jnp.int32(2))}
    expected = jax.jit(native)(carry)
    actual = jax.jit(wrapped)(carry)
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    jax.tree.map(np.testing.assert_array_equal, actual, expected)
    graph = str(jax.make_jaxpr(native)(carry))
    assert str(jax.make_jaxpr(wrapped)(carry)) == graph.replace("unroll=1\n", f"unroll={factor}\n")


def test_dynamic_and_empty_loops_use_native():
    def dynamic(stop, value):
        return auto_fori_loop(2, stop, lambda i, x: x + i, value, force_factor=8)

    assert jax.jit(dynamic)(6, 0) == 14
    graph = jax.make_jaxpr(dynamic)(6, 0).jaxpr
    assert [eq.primitive.name for eq in graph.eqns] == ["while"]
    assert auto_fori_loop(4, 2, lambda i, x: x + i, 7, force_factor=8) == 7


def test_static_reverse_mode_and_closure_over_jit_input():
    @jax.jit
    def run(values):
        return auto_fori_loop(0, values.size, lambda i, acc: acc + values[i] ** 2, 0.,
                              force_factor=2)

    values = jnp.arange(7, dtype=jnp.float32)
    np.testing.assert_array_equal(jax.grad(run)(values), 2 * values)


def test_override_validation():
    with pytest.raises(ValueError, match="positive integer"):
        auto_fori_loop(0, 4, step, None, force_factor=0)
    with pytest.raises(ValueError, match="only one"):
        auto_fori_loop(0, 4, step, None, force_factor=2, unroll=True)


def test_engine_reuses_decision_and_outer_jit_specialization():
    class Model:
        version = "test"
        backend = jax.default_backend()
        feature_version = FEATURE_VERSION
        jax_versions = (jax.__version__, jax.lib.__version__)
        predictions = 0

        def supports(self, features):
            return features["trip_count"] == 16

        def predict_candidates(self, features):
            self.predictions += 1
            return {f: LogCosts(0., 0. if f == 4 else 1.) for f in FACTORS}

    model = Model()
    engine = DecisionEngine(model)
    traces = []

    @jax.jit
    def run(values):
        traces.append(values.shape)
        return auto_fori_loop(0, 16, lambda i, acc: acc + values, values, engine=engine, B=5, R=100)

    values = jnp.ones((2,), dtype=jnp.float32)
    np.testing.assert_array_equal(run(values), 17 * values)
    np.testing.assert_array_equal(run(2 * values), 34 * values)
    assert traces == [(2,)]
    assert model.predictions == 1
    graph = str(jax.make_jaxpr(run.__wrapped__)(values))
    assert "unroll=4" in graph
    assert model.predictions == 1

    graph = str(jax.make_jaxpr(
        lambda x: auto_fori_loop(0, 5, lambda i, carry: carry + i, x, engine=engine)
    )(values))
    assert "unroll=1" in graph
    assert model.predictions == 1


def test_throughput_policy_uses_the_same_body_and_respects_budget():
    class Model:
        version = "test-throughput"
        backend = jax.default_backend()
        feature_version = FEATURE_VERSION
        jax_versions = (jax.__version__, jax.lib.__version__)

        def supports(self, features):
            return True

        def predict_candidates(self, features):
            return {f: LogCosts(3, -1) if f == 4 else LogCosts(0, 0) for f in FACTORS}

    engine = DecisionEngine(Model())
    carry = {"values": (jnp.ones(2), jnp.int32(2))}
    native = str(jax.make_jaxpr(lambda x: jax.lax.fori_loop(-3, 14, step, x))(carry))
    for options, expected in [({}, 1), ({"throughput_only": True}, 4),
                              ({"throughput_only": True, "B": 5}, 1)]:
        def run(x):
            return auto_fori_loop(-3, 14, step, x, engine=engine, **options)

        actual = str(jax.make_jaxpr(run)(carry))
        assert actual == native.replace("unroll=1\n", f"unroll={expected}\n")


@pytest.mark.parametrize("limit", [0, 1, 1024])
def test_precomputed_pytree_with_remainder_and_reverse_mode(limit):
    def run(values, start):
        def prepare(i):
            return {"value": values[i + 3], "index": i}

        def update(carry, data):
            return carry * 0.5 + data["value"] ** 2 + data["index"]

        return precompute_fori_loop(-3, 4, prepare, update, start, unroll=4, max_precompute_bytes=limit)

    def reference(values, start):
        return jax.lax.fori_loop(-3, 4, lambda i, c: c * 0.5 + values[i + 3] ** 2 + i, start)

    values = jnp.arange(7, dtype=jnp.float32)
    np.testing.assert_array_equal(jax.jit(run)(values, 0.), reference(values, 0.))
    np.testing.assert_array_equal(jax.grad(run)(values, 0.), jax.grad(reference)(values, 0.))


def test_precompute_respects_memory_cap_and_empty_range():
    def run(values, start, stop, limit):
        return precompute_fori_loop(
            start, stop, lambda i: values[i], jnp.add, 0.,
            max_precompute_bytes=limit,
        )

    values = jnp.arange(8, dtype=jnp.float32)
    buffered = jax.make_jaxpr(lambda x: run(x, 0, 8, 32))(values)
    fallback = jax.make_jaxpr(lambda x: run(x, 0, 8, 31))(values)
    assert len(buffered.jaxpr.eqns[-1].params["ft_in"].unpack()[2]) == 1
    assert len(fallback.jaxpr.eqns[-1].params["ft_in"].unpack()[2]) == 0
    assert run(values, 4, 1, 32) == 0
    with pytest.raises(ValueError, match="must be nonnegative"):
        run(values, 0, 8, -1)
