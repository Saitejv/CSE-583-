from dataclasses import replace
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_unroll import selector
from jax_unroll.auto_loop import auto_fori_loop
from jax_unroll.model import LogCosts
from jax_unroll.schema import FEATURE_VERSION
from jax_unroll.selector import DecisionCache, DecisionEngine, PayoffPolicy


@pytest.fixture
def predictions():
    return {factor: LogCosts(math.log(cold), math.log(warm))
            for factor, (cold, warm) in {1: (10, 1000), 2: (14, 500), 4: (100, 100), 8: (10, 990)}.items()}


@pytest.fixture
def payoff():
    return PayoffPolicy(math.log(1.10), math.log(1.25), 1, 1, 2, model_load_ms=3)


def test_payoff_charges_remaining_calls_and_overhead(payoff, predictions):
    # Factor 2 saves at least 0.36 ms per warm call and has 10 ms of extra setup.
    assert payoff.choose(predictions, R=28) == 1
    assert payoff.choose(predictions, R=29) == 2
    assert replace(payoff, model_load_ms=0).choose(predictions, R=21) == 2
    assert payoff.choose(predictions, R=10**400) == 4


def test_cold_budget_counts_decision_but_excludes_shared_model_load(payoff, predictions):
    assert payoff.choose(predictions, R=1_000, B=16.99) == 1
    assert payoff.choose(predictions, R=1_000, B=17) == 2
    assert payoff.choose(predictions, R=1_000, B=102.99) == 2
    assert payoff.choose(predictions, R=1_000, B=103) == 4


def test_uncertain_incomplete_and_invalid_forecasts_keep_factor_one(payoff, predictions):
    assert replace(payoff, warm_effect_margin=math.log(11)).choose(predictions, R=1_000) == 1
    assert payoff.choose({1: predictions[1]}, R=1_000) == 1
    assert payoff.choose({**predictions, 2: LogCosts(math.nan, 0)}, R=1_000) == 1
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(payoff, extra_cold_ms=-1)


def test_wrapper_horizons_budget_and_policy_cache(payoff, predictions, monkeypatch, tmp_path):
    class Model:
        version = "test-payoff"
        backend = jax.default_backend()
        feature_version = FEATURE_VERSION
        jax_versions = jax.__version__, jax.lib.__version__
        evaluations = 0

        def supports(self, row):
            return True

        def predict_candidates(self, row):
            self.evaluations += 1
            return predictions

    args = jnp.arange(32, dtype=jnp.float32)
    model = Model()
    path = tmp_path / "payoff-decisions.json"
    engine = DecisionEngine(model, DecisionCache(path), payoff=payoff)

    def body(i, carry):
        return carry * .75 + args[i]

    initial = jnp.float32(0)
    native = lambda: jax.lax.fori_loop(0, args.size, body, initial, unroll=1)
    graph = str(jax.make_jaxpr(native)())

    def wrapped(engine, **options):
        return lambda: auto_fori_loop(0, args.size, body, initial, engine=engine, **options)

    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            pytest.fail("single-call payoff fallback extracted features")

        patch.setattr(selector, "extract_features", forbidden)
        assert str(jax.make_jaxpr(wrapped(engine, R=1))()) == graph
    assert model.evaluations == 0

    for options, factor in [({"R": 29}, 2), ({"R": 29, "B": 16}, 1),
                            ({"R": 1, "throughput_only": True}, 4)]:
        run = wrapped(engine, **options)
        assert str(jax.make_jaxpr(run)()) == graph.replace("unroll=1\n", f"unroll={factor}\n")
        np.testing.assert_allclose(jax.jit(run)(), jax.jit(native)(), rtol=1e-6)
    assert model.evaluations == 3

    # Changing a calibrated margin cannot replay decisions persisted by another policy.
    strict = DecisionEngine(model, DecisionCache(path), payoff=replace(payoff, warm_effect_margin=3))
    assert str(jax.make_jaxpr(wrapped(strict, R=29))()) == graph
    assert model.evaluations == 4
