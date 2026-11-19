from concurrent.futures import ThreadPoolExecutor
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.corpus.fori_iterations.confirmation import freeze, matrix
from jax_unroll.auto_loop import auto_fori_loop
from jax_unroll.features import extract_features
from jax_unroll.model import LogCosts
from jax_unroll.schema import FEATURE_VERSION
from jax_unroll.selector import DecisionEngine, SavingsBand, SavingsPolicy


def forecast(cold, warm):
    return LogCosts(math.log(cold), math.log(warm))


PREDICTIONS = {
    1: forecast(10, 1_000), 2: forecast(14, 500), 4: forecast(100, 100), 8: forecast(10, 990),
}


class Model:
    version = "lazy-test"
    backend = jax.default_backend()
    feature_version = FEATURE_VERSION
    jax_versions = jax.__version__, jax.lib.__version__

    def supports(self, features):
        return True

    def predict_candidates(self, features):
        return PREDICTIONS


def test_savings_policy_accounts_for_uncertainty_cold_budget_and_calls():
    rule = SavingsPolicy((SavingsBand(15, .40, 1, 1), SavingsBand(None, .10, 1, 1)), 2, 3)
    assert rule.choose(PREDICTIONS, R=24) == 1
    assert rule.choose(PREDICTIONS, R=27) == 2
    assert rule.choose(PREDICTIONS, R=1_000, B=16.99) == 1
    assert rule.choose(PREDICTIONS, R=1_000, B=17) == 2
    assert rule.choose(PREDICTIONS, R=1_000, B=103) == 4
    assert rule.choose({**PREDICTIONS, 2: LogCosts(math.nan, 0)}, R=1_000) == 1
    cheap = {1: forecast(10, 10), 2: forecast(10, 7), 4: forecast(100, 7), 8: forecast(10, 10)}
    assert rule.choose(cheap, R=100_000) == 1
    with pytest.raises(ValueError, match="increasing"):
        SavingsPolicy((SavingsBand(15, .1, 0, 0), SavingsBand(10, .1, 0, 0), SavingsBand(None, .1, 0, 0)), 1)


def test_lazy_wrapper_does_not_load_until_a_prediction_is_useful():
    loads = []
    model = Model()

    def load():
        loads.append("loaded")
        return model

    rule = SavingsPolicy((SavingsBand(None, .1, 1, 1),), 2, 3)
    engine = DecisionEngine(model_loader=load, payoff=rule)
    x = jnp.arange(32, dtype=jnp.float32)

    def run(value, *, calls, budget=None):
        return auto_fori_loop(0, value.size, lambda i, carry: .5 * carry + value[i], jnp.float32(0),
                              engine=engine, R=calls, B=budget)

    short_graph = str(jax.make_jaxpr(lambda z: run(z, calls=1))(x))
    small_budget_graph = str(jax.make_jaxpr(lambda z: run(z, calls=1_000, budget=2))(x))
    assert not loads and engine.model is None
    assert small_budget_graph == short_graph and "unroll=1" in short_graph
    long = lambda z: run(z, calls=27)
    assert str(jax.make_jaxpr(long)(x)) == short_graph.replace("unroll=1\n", "unroll=2\n")
    np.testing.assert_allclose(jax.jit(long)(x), jax.jit(lambda z: run(z, calls=1))(x), rtol=1e-6)
    assert loads == ["loaded"]


def test_lazy_loader_retries_failures_and_shares_one_success():
    calls = []
    model = Model()

    def load():
        calls.append(None)
        if len(calls) == 1:
            raise OSError("temporary file error")
        return model

    engine = DecisionEngine(model_loader=load)
    with pytest.raises(OSError, match="temporary file"):
        engine._require_model()
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(value is model for value in pool.map(lambda _: engine._require_model(), range(8)))
    assert len(calls) == 2


def test_features_from_a_closed_jaxpr_match_direct_extraction():
    def run(x):
        return jax.lax.fori_loop(0, 32, lambda i, c: .75 * c + x[i], jnp.float32(0))
    x = jnp.ones(32, dtype=jnp.float32)
    options = {"trip_count": 32, "loop_kind": "fori_loop", "factor": 1}
    direct = extract_features(run, (x,), **options)
    captured = extract_features(closed_jaxpr=jax.make_jaxpr(run)(x), **options)
    assert {k: v for k, v in direct.items() if k != "extraction_ms"} == {
        k: v for k, v in captured.items() if k != "extraction_ms"
    }


def test_confirmation_is_frozen_at_whole_family_level():
    freeze()
    rows = matrix()
    assert len(rows) == 12
    assert len({case.family for case, _, _ in rows}) == 6
