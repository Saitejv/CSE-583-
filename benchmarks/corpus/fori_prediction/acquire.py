"""Collect interleaved factor timings on the generated, unchanged fori_loop bodies."""

from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from time import perf_counter_ns

import jax

from benchmarks.corpus.fori_prediction.workloads import CASES, HOLDOUT_FAMILIES
from benchmarks.corpus.fori_targeted.specialization import check_output
from benchmarks.sweep import _warm_backend
from jax_unroll.autotune import _times
from jax_unroll.features import extract_features
from jax_unroll.schema import FACTORS
from jax_unroll.fori_v2 import loop_features

OUT = Path("results/fori_prediction")
SIZES = ("small", "large")
TRIPS = (127, 256, 511, 512)
CONFIRM_TRIPS = (767, 768)


def cases(split):
    return {
        name: spec for name, spec in CASES.items()
        if (spec.family in HOLDOUT_FAMILIES) == (split != "train")
    }


def measure(split, repeat):
    jax.config.update("jax_enable_compilation_cache", False)
    _warm_backend()
    rng = random.Random(58332 + repeat)
    settings = list(product(cases(split).values(), SIZES, CONFIRM_TRIPS if split == "confirm" else TRIPS))
    rng.shuffle(settings)
    rows = []
    for spec, size, trip in settings:
        native = spec.build(trip, 1, size)
        args = jax.block_until_ready(jax.device_put(native.args))
        workloads = {1: native, **{f: spec.build(trip, f, size) for f in FACTORS[1:]}}
        graphs = {f: str(jax.make_jaxpr(w.fn)(*args)) for f, w in workloads.items()}
        original = graphs[1]
        if original.count("unroll=1\n") != 1 or any(
            graph != original.replace("unroll=1\n", f"unroll={f}\n")
            for f, graph in graphs.items()
        ):
            raise ValueError(f"{spec.name}: Jaxpr changed beyond the unroll parameter")
        features = extract_features(
            native.fn, args, loop_kind="fori_loop", trip_count=trip, factor=1,
            feature_block=loop_features,
        )

        order = list(FACTORS)
        rng.shuffle(order)
        compiled, cold, outputs = {}, {}, {}
        for factor in order:
            jax.clear_caches()
            start = perf_counter_ns()
            compiled[factor] = jax.jit(workloads[factor].fn).lower(*args).compile()
            outputs[factor] = jax.block_until_ready(compiled[factor](*args))
            cold[factor] = (perf_counter_ns() - start) / 1e6
        for output in outputs.values():
            check_output(output, outputs[1], spec)
        calls = min(160, max(6, int(1000 / _times(compiled[1], args, 5))))
        timings = {f: [] for f in order}
        for _ in range(7):
            rng.shuffle(order)
            for factor in order:
                timings[factor].append(_times(compiled[factor], args, calls))
        rows.append({
            "workload": spec.name, "family": spec.family, "variant": size,
            "trip_count": trip, "repeat": repeat, "features": features,
            "cold_ms": [cold[f] for f in FACTORS],
            "warm_us": [statistics.median(timings[f]) for f in FACTORS],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "holdout", "confirm"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(measure(args.split, args.worker)))
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.split}_runs.json"
    rows = []
    for repeat in range(args.repeats):
        print(f"{args.split} repeat {repeat + 1}/{args.repeats}", file=sys.stderr, flush=True)
        output = subprocess.check_output(
            [sys.executable, "-m", __spec__.name, args.split, "--worker", str(repeat)],
            text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"},
        )
        rows.extend(json.loads(output))
        path.write_text(json.dumps(rows, separators=(",", ":")) + "\n")
    print(f"{len(rows)} configurations × process repeats", flush=True)


if __name__ == "__main__":
    main()
