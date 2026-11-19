"""Sweep native factors on the frozen regret-targeted development or holdout split."""

from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import platform
import random
from statistics import median
import subprocess
import sys

import jax
import jaxlib
import numpy as np

from benchmarks.corpus.fori_regret.workloads import cases
from benchmarks.corpus.fori_targeted.auto_fori import compile_and_run
from benchmarks.corpus.fori_targeted.specialization import check_output
from benchmarks.sweep import _warm_backend
from jax_unroll.autotune import _times
from jax_unroll.features import extract_features
from jax_unroll.schema import FACTORS

OUT = Path("results/fori_regret")
SIZES = ("small", "large")
TRIPS = (255, 256, 1023, 1024)
REPEATS = 3


def assert_same_graph(fn, args, native_graph, factor):
    graph = str(jax.make_jaxpr(fn)(*args))
    if native_graph.count("unroll=1\n") != 1 or graph != native_graph.replace(
        "unroll=1\n", f"unroll={factor}\n"
    ):
        raise ValueError("expected exactly one fori_loop with only the unroll parameter changed")


def worker(split, repeat):
    return measure(list(product(cases(split).values(), SIZES, TRIPS)), repeat)


def measure(settings, repeat):
    """Measure a fixed list of (spec, variant, trip) settings with paired factors."""
    jax.config.update("jax_enable_compilation_cache", False)
    _warm_backend()
    rng = random.Random(583_404 + repeat)
    settings = list(settings)
    rng.shuffle(settings)
    rows = []
    for spec, size, trip in settings:
        builds = {factor: spec.build(trip, factor, size) for factor in FACTORS}
        native = builds[1]
        args = jax.block_until_ready(jax.device_put(native.args))
        graph = str(jax.make_jaxpr(native.fn)(*args))
        for factor, build in builds.items():
            jax.tree.map(np.testing.assert_array_equal, build.args, native.args)
            assert_same_graph(build.fn, args, graph, factor)
        features = extract_features(native.fn, args, trip_count=trip, factor=1, loop_kind="fori_loop")

        order = list(FACTORS)
        rng.shuffle(order)
        compiled, cold, outputs = {}, {}, {}
        for factor in order:
            jax.clear_caches()
            compiled[factor], outputs[factor], cold[factor] = compile_and_run(builds[factor].fn, args)
        for output in outputs.values():
            check_output(output, outputs[1], spec)
        calls = min(160, max(6, int(1_500 / _times(compiled[1], args, 5))))
        warm = {factor: [] for factor in FACTORS}
        for _ in range(9):
            rng.shuffle(order)
            for factor in order:
                warm[factor].append(_times(compiled[factor], args, calls))
        rows.append({
            "workload": spec.name, "family": spec.family, "variant": size, "trip_count": trip,
            "repeat": repeat, "features": features, "calls_per_sample": calls,
            "cold_ms": [cold[f] for f in FACTORS],
            "warm_samples_us": [warm[f] for f in FACTORS],
        })
    return rows


def host():
    return {
        "machine": platform.machine(), "platform": platform.platform(),
        "processor": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform == "darwin" else platform.processor(),
        "jax": jax.__version__, "jaxlib": jaxlib.__version__, "backend": "cpu",
    }


def run_workers(module, arguments, repeats, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for repeat in range(repeats):
        print(f"{path.stem} process {repeat + 1}/{repeats}", file=sys.stderr, flush=True)
        cmd = [sys.executable, "-m", module, *arguments, "--worker", str(repeat)]
        runs.append(json.loads(subprocess.check_output(
            cmd, text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"},
        )))
        path.write_text(json.dumps({"host": host(), "runs": runs}, separators=(",", ":")) + "\n")
    return runs


def read(split):
    import pandas as pd

    data = json.loads((OUT / f"{split}_runs.json").read_text())
    keys = list(product(cases(split), SIZES, TRIPS))
    lookup = {(r["workload"], r["variant"], r["trip_count"], r["repeat"]): r
              for run in data["runs"] for r in run}
    expected = {(*key, repeat) for key in keys for repeat in range(REPEATS)}
    if len(lookup) != sum(map(len, data["runs"])) or set(lookup) != expected:
        raise ValueError(f"incomplete or duplicate {split} timing matrix")
    frame, labels = [], []
    for key in sorted(keys):
        group = [lookup[(*key, repeat)] for repeat in range(REPEATS)]
        row = group[0]
        if len({r["features"]["body_hash"] for r in group}) != 1:
            raise ValueError(f"inconsistent body hash for {key}")
        frame.append({
            **row["features"], "workload": row["workload"], "family": row["family"],
            "variant": row["variant"], "source": "regret",
        })
        labels.append(np.median([
            np.column_stack((r["cold_ms"], list(map(median, r["warm_samples_us"]))))
            for r in group
        ], axis=0))
    costs = np.asarray(labels)
    if costs.shape != (len(keys), len(FACTORS), 2) or not (np.isfinite(costs) & (costs > 0)).all():
        raise ValueError(f"invalid measured costs in {split}")
    return pd.DataFrame(frame), costs


def main():
    from benchmarks.corpus.fori_regret.freeze import check

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "holdout"))
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.split, args.worker)))
        return
    check()
    run_workers(__spec__.name, [args.split], REPEATS, OUT / f"{args.split}_runs.json")


if __name__ == "__main__":
    main()
