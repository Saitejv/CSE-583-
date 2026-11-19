import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.corpus.fori_application import applications, transfer
from jax_unroll.auto_loop import auto_fori_loop
from jax_unroll.compact_fori import application_engine


def test_bundled_policy_and_model_match_frozen_fit(monkeypatch, tmp_path):
    fit = json.loads(Path("results/fori_application/fit.json").read_text())
    # A deployed application can start in any working directory.
    monkeypatch.chdir(tmp_path)
    engine = application_engine()
    assert engine.model is None and asdict(engine.payoff) == fit["policy"]

    def run(data, gain, x, calls, budget):
        def update(i, acc):
            return 0.8 * acc + data[i] * gain[i]

        return auto_fori_loop(0, data.shape[0], update, x, B=budget, R=calls, engine=engine)

    data = jnp.ones((24, 32, 32), jnp.float32)
    x = jnp.zeros((32, 32), jnp.float32)

    def graph(h, b):
        return str(jax.make_jaxpr(lambda a, g, x: run(a, g, x, h, b))(data, data, x))

    short = graph(300, 125)
    assert "unroll=1\n" in short and engine.model is None
    assert graph(5000, 0) == short and engine.model is None
    assert graph(5000, 125) == short.replace("unroll=1\n", "unroll=2\n")
    scalar_fit = json.loads((Path(__file__).resolve().parents[1] / "results/fori_scalar/fit.json").read_text())
    assert scalar_fit["base"] == fit["model_version"]
    assert engine.model.version == scalar_fit["model"]


def test_real_data_kernel_references():
    for spec, size, trip in transfer.settings():
        build = spec.build(trip, 2, size)
        args = build.args
        actual = jax.device_get(jax.jit(build.fn)(*jax.device_put(args)))
        if spec.name == "weighted_covariance":
            x, w = args[:2]
            mean = np.average(x, axis=0, weights=w)
            expect = mean, np.cov(x.T, aweights=w, ddof=0)
        elif spec.name == "recursive_least_squares":
            x, y = args[:2]
            x = x.astype(np.float64)
            precision = np.eye(x.shape[1]) + x.T @ x
            expect = np.linalg.solve(precision, x.T @ y), np.linalg.inv(precision)
        elif spec.name == "terrain_interpolation":
            sx, sy, heights, x, y = args[:5]
            d = (x[None, None, :] - sx[:, None, None]) ** 2 + (y[None, :, None] - sy[:, None, None]) ** 2
            expect = heights[d.argmin(axis=0)]
        else:
            signal, returns, buy, leave, held, pnl = args
            for s, r in zip(signal, returns, strict=True):
                current = (s > buy[:, None]) | (held & (s >= leave))
                pnl = pnl + r * current - np.float32(0.001) * (current != held)
                held = current
            expect = held, pnl
        jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5), actual, expect)


def test_upstream_segment_envelopes_equal_public_jax():
    specs = applications.settings()
    spec, size, trip = next(row for row in specs if row[0].name == "class_pixel_peaks")
    b = spec.build(trip, 4, size)
    args = jax.device_put(b.args)
    np.testing.assert_array_equal(jax.jit(b.fn)(*args), jnp.maximum.reduceat(*args, axis=0))
