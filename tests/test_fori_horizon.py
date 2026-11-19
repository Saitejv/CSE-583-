import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.corpus.fori_horizon.replay import total
from jax_unroll.auto_loop import auto_fori_loop
from jax_unroll.model import LogCosts
from jax_unroll.schema import FEATURE_VERSION
from jax_unroll.selector import DecisionEngine, HorizonPolicy


RULE = HorizonPolicy(min_calls=1800, decision_ms=3, matrix_min_calls=1000, matrix_bytes=2048,
                     max_index_ops=1, matrix_factor=2)
PRICES = {f: LogCosts(math.log(c), math.log(w)) for f, c, w in
          ((1, 12, 8), (2, 16, 10), (4, 80, 6), (8, 200, 4))}
FEATURES = dict(carry_bytes=2048, eq_dot=0, eq_reduce=0, eq_gather_scatter=1, carry_shapes="[[16,32]]")


@pytest.mark.parametrize("calls,budget,expected", [
    (1, None, 1), (999, None, 1), (1000, 19, 2), (1000, 18.99, 1),
    (1800, 18.99, 1), (1800, 19, 2),
])
def test_matrix_crossover_and_inclusive_cold_budget(calls, budget, expected):
    assert RULE.choose(PRICES, features=FEATURES, R=calls, B=budget) == expected


@pytest.mark.parametrize("change", [
    {"carry_shapes":"[[512]]"}, {"eq_dot":1}, {"eq_reduce":1}, {"eq_gather_scatter":2},
    {"carry_bytes":2047},
])
def test_outside_guard_uses_feasible_warm_incumbent(change):
    features = FEATURES | change
    assert RULE.choose(PRICES, features=features, R=1000) == 1
    assert RULE.choose(PRICES, features=features, R=1800, B=125) == 4
    assert RULE.choose(PRICES, features=features, R=10**400) == 8
    assert RULE.choose({**PRICES, 8:LogCosts(0, math.nan)}, features=features, R=1800) == 1


def test_short_and_budget_bypass_before_load_or_trace(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("bypass performed feature extraction or model loading")
    monkeypatch.setattr("jax_unroll.selector.extract_features", forbidden)
    engine = DecisionEngine(model_loader=forbidden, payoff=RULE)
    for shape, h, budget in [((32,32),1,None), ((512,),1200,None), ((2,2),1500,None),
                              ((32,32),1800,3)]:
        def run(carry):
            return auto_fori_loop(0, 16, lambda i, x: jax.tree.map(lambda a: a + i, x),
                                  carry, engine=engine, R=h, B=budget)
        assert "unroll=1\n" in str(jax.make_jaxpr(run)({"pair":(jnp.ones(shape), 2)}))


def test_actual_wrapper_uses_only_unroll_and_shared_loader():
    class Model:
        backend, feature_version = "cpu", FEATURE_VERSION
        jax_versions = (jax.__version__, jax.lib.__version__)
        version = "test-matrix"
        def supports(self, features):
            return True
        def predict_candidates(self, features):
            return PRICES

    loads = []
    def load():
        loads.append(1)
        return Model()
    engine = DecisionEngine(model_loader=load, payoff=RULE)
    x = jnp.zeros((16, 32), dtype=jnp.float32)
    def body(i, x):
        return x + i
    def wrapped(x):
        return auto_fori_loop(0, 16, body, x, R=1000, B=125, engine=engine)
    def native(x):
        return jax.lax.fori_loop(0, 16, body, x)
    graph = str(jax.make_jaxpr(native)(x))
    assert str(jax.make_jaxpr(wrapped)(x)) == graph.replace("unroll=1\n","unroll=2\n")
    np.testing.assert_array_equal(jax.jit(wrapped)(x), jax.jit(native)(x))
    assert loads == [1]


def test_first_call_is_charged_once_and_oracle_depends_on_horizon():
    cold, warm = np.array([10, 12, 30, 60]), np.array([20, 10, 8, 7])
    np.testing.assert_array_equal(total(cold, warm, 1), cold)
    assert [total(cold, warm, h).argmin() for h in (1, 300, 20000, 50000)] == [0, 1, 2, 3]


def test_simple_crossover_never_overrides_and_rejects_invalid_parameters():
    rule = HorizonPolicy(1800, 3)
    assert rule.choose(PRICES, features={}, B=125, R=1800) == 4
    assert rule.choose(PRICES, features={}, R=1799) == 1
    for kwargs in (dict(min_calls=1), dict(decision_ms=math.nan), dict(matrix_min_calls=2000),
                   dict(matrix_bytes=-1), dict(matrix_factor=3), dict(max_index_ops=-1)):
        with pytest.raises(ValueError):
            replace(RULE, **kwargs)


def test_repeat_accounting_charges_each_load_once():
    from benchmarks.corpus.fori_horizon.live import ARMS
    from benchmarks.corpus.fori_horizon.report import aggregate

    labels = [dict(workload=k, variant="s", trip_count=16, cohort="mixed", opportunity=False,
                   oracle={"1":1}, replay_ms={"1":[1., 2., 3., 4.]}) for k in ("a", "b")]
    blocks = []
    for arm in ARMS:
        for repeat in range(3):
            inline = 4. if arm == "candidate" else 0.
            setup = 4. if arm == "incumbent" else 0.
            rows = [dict(**{k:r[k] for k in ("workload","variant","trip_count")}, calls=1,
                         factor=1, direct_ms=10 + (inline if i == 0 else 0),
                         inline_load_ms=inline if i == 0 else 0.) for i, r in enumerate(labels)]
            blocks.append(dict(arm=arm, calls=1, repeat=repeat, rows=rows, setup_ms=setup))
    result = aggregate(blocks, labels, horizons=(1,))
    for arm in ARMS:
        expected = .012 if arm in ("candidate", "incumbent") else .010
        np.testing.assert_allclose(result[result.arm == arm].direct_s, expected)
    with pytest.raises(ValueError, match="missing"):
        aggregate(blocks[:-1], labels, horizons=(1,))
