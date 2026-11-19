"""Mechanism probes added after historical diagnosis, before fitting or timing."""

import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.fori_horizon.cohort import OUT, freeze, matrix
from benchmarks.corpus.fori_regret.acquire import REPEATS, measure, run_workers
from benchmarks.corpus.fori_regret.workloads import Case
from benchmarks.corpus.fori_targeted.registry import get_spec


def synaptic_eligibility(size, trip, factor):
    rng = np.random.default_rng(913)
    pre = rng.uniform(-.3, .3, (trip, 24)).astype("float32")
    post = rng.uniform(-.3, .3, (trip, 32)).astype("float32")

    def run(weights, inputs, errors):
        def step(i, state):
            return .98 * state + inputs[i, :, None] * errors[i, None, :]
        return lax.fori_loop(0, trip, step, weights, unroll=factor)
    return run, (np.zeros((24, 32), np.float32), pre, post)


def manhattan_voronoi(size, trip, factor):
    rng = np.random.default_rng(914)
    points = rng.uniform(0, 30, (trip, 2)).astype("float32")
    x = np.arange(32, dtype=np.float32)
    y = np.arange(28, dtype=np.float32)

    def run(sites, x, y, initial):
        def step(i, closest):
            distance = jnp.abs(x - sites[i, 0])[:, None] + jnp.abs(y - sites[i, 1])[None, :]
            return jnp.minimum(closest, distance)
        return lax.fori_loop(0, trip, step, initial, unroll=factor)
    return run, (points, x, y, np.full((32, 28), 100., np.float32))


def settings():
    probes = [Case(fn.__name__, fn.__name__, "probe", fn)
              for fn in (synaptic_eligibility, manhattan_voronoi)]
    return [(spec, "large", 256) for spec in probes] + [(get_spec("dense_gram_accum"), "large", 1024)]


def freeze_all(create=False):
    freeze()
    rows = [(s.name, s.family, v, t) for s, v, t in settings()]
    text = json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           cases=rows, cohorts=["transfer", "transfer", "historical"]),
                      indent=2) + "\n"
    path = OUT / "transfer_matrix.json"
    if create and not path.exists():
        with path.open("x") as stream:
            stream.write(text)
    if path.read_text() != text:
        raise ValueError("frozen probes changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze","sweep"])
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    freeze_all(create=args.stage == "freeze")
    if args.worker is not None:
        print(json.dumps(measure(matrix() + settings(), args.worker)))
    elif args.stage == "sweep":
        run_workers(__spec__.name, ["sweep"], REPEATS, OUT / "sweep.json")
