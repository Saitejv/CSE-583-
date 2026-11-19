"""Predeclared independent CPU confirmation for horizon-aware selection."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.fori_regret.acquire import REPEATS, measure, run_workers
from benchmarks.corpus.fori_regret.workloads import Case
from jax_unroll.schema import FACTORS

OUT = Path("results/fori_horizon")
HORIZONS = (1, 300, 5000, 20000)
SETTINGS = (("small", 1024), ("large", 255))


def schmitt_edges(size, trip, factor):
    w = 1 if size == "small" else 16
    samples = np.random.default_rng(901).normal(size=(trip, w)).astype("float32")

    def run(data, latched, count):
        def step(i, carry):
            high, edges = carry
            next_high = (high & (data[i] > -.4)) | (data[i] > .7)
            return next_high, edges + (next_high & ~high)
        return lax.fori_loop(0, trip, step, (latched, count), unroll=factor)
    return run, (samples, np.zeros(w, bool), np.zeros(w, np.int32))


def polynomial_value_derivative(size, trip, factor):
    w = 1 if size == "small" else 12
    rng = np.random.default_rng(902)
    coefficients = rng.uniform(-.2, .2, trip).astype("float32")
    x = np.linspace(.7, .99, w, dtype="float32")

    def run(coeff, point, zero):
        def step(i, carry):
            val, deriv = carry
            return val * point + coeff[i], deriv * point + val
        return lax.fori_loop(0, trip, step, (zero, zero), unroll=factor)
    return run, (coefficients, x, np.zeros_like(x))


def reservoir_balance(size, trip, factor):
    w = 4 if size == "small" else 32
    rain = np.random.default_rng(903).uniform(0, .2, (trip, w)).astype("float32")

    def run(initial, rainfall):
        def step(i, carry):
            storage, discharge = carry
            available = .98 * storage + rainfall[i]
            overflow = jnp.maximum(available - 1., 0.)
            return available - overflow, discharge + overflow
        return lax.fori_loop(0, trip, step, (initial, initial), unroll=factor)
    return run, (np.zeros(w, np.float32), rain)


def triangle_area(size, trip, factor):
    w = 4 if size == "small" else 32
    vertices = np.random.default_rng(904).uniform(-1, 1, (trip, w, 2)).astype("float32")

    def run(points, zero):
        def step(i, carry):
            prev, area = carry
            curr = points[i]
            cross = prev[:, 0] * curr[:, 1] - prev[:, 1] * curr[:, 0]
            return curr, area + cross
        return lax.fori_loop(0, trip, step, (points[-1], zero), unroll=factor)
    return run, (vertices, np.zeros(w, np.float32))


def emission_forward(size, trip, factor):
    w = 10 if size == "small" else 40
    rng = np.random.default_rng(905)
    trans = rng.uniform(.1, 1., (w, w)).astype("float32")
    emissions = rng.uniform(.5, 1., (trip, w)).astype("float32")

    def run(matrix, observations, initial):
        def step(i, prob):
            weights = (matrix @ prob) * observations[i]
            return weights / jnp.sum(weights)
        return lax.fori_loop(0, trip, step, initial, unroll=factor)
    return run, (trans / trans.sum(axis=0), emissions, np.full(w, 1 / w, np.float32))


def union_labels(size, trip, factor):
    w = 96 if size == "small" else 768
    pairs = np.random.default_rng(906).integers(w, size=(trip, 2), dtype=np.int32)

    def run(edges, initial):
        def step(i, labels):
            a, b = edges[i]
            value = jnp.minimum(labels[a], labels[b])
            return labels.at[a].set(value).at[b].set(value)
        return lax.fori_loop(0, trip, step, initial, unroll=factor)
    return run, (pairs, np.arange(w, dtype=np.int32))


CASES = tuple(Case(fn.__name__, fn.__name__, "confirmation", fn) for fn in (
    schmitt_edges, polynomial_value_derivative, reservoir_balance, triangle_area,
    emission_forward, union_labels,
))


def matrix():
    return [(c, size, trip) for c, (size, trip) in product(CASES, SETTINGS)]


def freeze(create=False):
    value = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 cases=[dict(workload=c.name, family=c.family, variant=s, trip_count=t)
                        for c, s, t in matrix()], horizons=HORIZONS, factors=FACTORS,
                 repeats=REPEATS, opportunity="warm median paired gain >=1.05 and win in every repeat",
                 budget_ms=125)
    path = OUT / "matrix.json"
    data = json.dumps(value, indent=2) + "\n"
    if create and not path.exists():
        OUT.mkdir(exist_ok=True, parents=True)
        with path.open("x") as stream:
            stream.write(data)
    if path.read_text() != data:
        raise ValueError("frozen confirmation changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "sweep"])
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    freeze(create=args.stage == "freeze")
    if args.worker is not None:
        print(json.dumps(measure(matrix(), args.worker)))
    elif args.stage == "sweep":
        run_workers(__spec__.name, ["sweep"], REPEATS, OUT / "native.json")
