"""The four 583 scan scenarios, with their source-size inputs and shared step bodies."""

import argparse
import json
from functools import lru_cache, partial
from hashlib import sha256
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.corpus.fori_application.protocol import ARMS, HORIZONS, OUT
from benchmarks.corpus.fori_regret.acquire import REPEATS, measure, run_workers
from benchmarks.corpus.fori_regret.workloads import Case
from jax_unroll.schema import FACTORS


def scalar(x, inputs, *, unroll, loop=jax.lax.scan):
    def step(carry, value):
        return 0.999 * carry + value, None

    return loop(step, x, inputs, unroll=unroll)[0]


def elementwise(x, drive, *, unroll, loop=jax.lax.scan):
    def step(carry, _):
        return jnp.sin(1.01 * carry + drive), None

    return loop(step, x, None, length=128, unroll=unroll)[0]


def reduction(x, drive, *, unroll, loop=jax.lax.scan):
    def step(carry, _):
        return 0.99 * carry + 0.01 * jnp.tanh(jnp.mean(carry) + drive), None

    return loop(step, x, None, length=128, unroll=unroll)[0]


def matrix(x, weight, bias, *, unroll, loop=jax.lax.scan):
    def step(carry, _):
        return jnp.tanh(carry @ weight + bias), None

    return loop(step, x, None, length=64, unroll=unroll)[0]


@lru_cache(maxsize=1)
def make_workloads():
    rng = np.random.default_rng(20261103)

    def normal(shape, scale=1.0):
        return jax.device_put((rng.standard_normal(shape) * scale).astype(np.float32)).block_until_ready()

    return [
        ("Scalar recurrence", scalar, (normal(()), normal((8192,), 0.01)), "8192 steps; scalar carry"),
        (
            "Elementwise sine",
            elementwise,
            (normal((4096,)), normal((4096,), 0.1)),
            "128 steps; 4096-element vector",
        ),
        (
            "Reduction + vector update",
            reduction,
            (normal((8192,)), normal((8192,), 0.1)),
            "128 steps; 8192-element mean and update",
        ),
        (
            "Dense matrix recurrence",
            matrix,
            (normal((128, 128), 0.1), normal((128, 128), 0.9 / np.sqrt(128)), normal((128,), 0.1)),
            "64 steps; 128x128 matmul + tanh",
        ),
    ]


def fori(step, init, xs, *, length=None, unroll):
    """Use the final carry of these source scans; none produces per-step outputs."""
    trip = len(xs) if length is None else length

    def body(i, carry):
        return step(carry, None if xs is None else xs[i])[0]

    return jax.lax.fori_loop(0, trip, body, init, unroll=unroll), None


TRIPS = (8192, 128, 128, 64)


def build(fn, inputs, expected_trip, variant, trip, factor):
    if variant != "source" or trip != expected_trip:
        raise ValueError("expected the original dimensions")
    return partial(fn, loop=fori, unroll=factor), inputs


def settings():
    return [
        (
            Case(fn.__name__, fn.__name__, "external", partial(build, fn, inputs, trip), atol=1e-6),
            "source",
            trip,
        )
        for (_, fn, inputs, _), trip in zip(make_workloads(), TRIPS, strict=True)
    ]


def check(create=False):
    manifest = dict(
        source="583/benchmark_unroll.py",
        source_sha256="1ef9606b92cd1baaf294dbcd66fdb12ada2472570c23dceec4ce5f0ce063edb3",
        port_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        fit_sha256=sha256((OUT / "fit.json").read_bytes()).hexdigest(),
        cases=[
            dict(
                name=fn.__name__,
                description=description,
                trip_count=trip,
                inputs=[
                    dict(
                        shape=x.shape, dtype=str(x.dtype), sha256=sha256(np.asarray(x).tobytes()).hexdigest()
                    )
                    for x in inputs
                ],
            )
            for (_, fn, inputs, description), trip in zip(make_workloads(), TRIPS, strict=True)
        ],
        factors=FACTORS,
        calls=(*HORIZONS, 20_000),
        arms=ARMS,
        repeats=REPEATS,
    )
    path = OUT / "scan_ports_manifest.json"
    encoded = json.dumps(manifest, indent=2) + "\n"
    if create and not path.exists():
        with path.open("x") as f:
            f.write(encoded)
    if path.read_text() != encoded:
        raise ValueError("scenario source, policy or inputs changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "sweep"))
    parser.add_argument("--worker", type=int)
    opts = parser.parse_args()
    check(create=opts.stage == "freeze")
    if opts.worker is not None:
        print(json.dumps(measure(settings(), opts.worker)))
    elif opts.stage == "sweep":
        run_workers(__spec__.name, ["sweep"], REPEATS, OUT / "scan_ports_runs.json")


if __name__ == "__main__":
    main()
