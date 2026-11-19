"""Measure the public auto_fori_loop wrapper on unchanged targeted corpus bodies."""

from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import random
from statistics import geometric_mean, median
import subprocess
import sys
from time import perf_counter_ns
from unittest.mock import patch

import jax

from benchmarks.corpus.fori_targeted.registry import EVAL_SPECS
from benchmarks.corpus.fori_targeted.specialization import SIZES, TRIPS, check_output
from benchmarks.phase4 import TimedEngine
from benchmarks.sweep import _warm_backend
from jax_unroll.auto_loop import auto_fori_loop
from jax_unroll.autotune import _times


MODEL = Path("results/fori_predictor/model.joblib")
NATIVE_FORI = jax.lax.fori_loop


def through_auto_fori(fn, engine, *, throughput_only=True, R=1, B=None, wrapper=auto_fori_loop):
    """Route one native loop through the wrapper while tracing a corpus function."""

    def replace(lower, upper, body, initial, *, unroll=None):
        if unroll != 1:
            raise ValueError("expected an unmodified factor-1 workload")
        with patch.object(jax.lax, "fori_loop", NATIVE_FORI):
            return wrapper(lower, upper, body, initial, engine=engine,
                           throughput_only=throughput_only, R=R, B=B)

    def run(*args):
        with patch.object(jax.lax, "fori_loop", replace):
            return fn(*args)

    return run


def compile_and_run(fn, args):
    start = perf_counter_ns()
    kernel = jax.jit(fn).lower(*args).compile()
    output = jax.block_until_ready(kernel(*args))
    return kernel, output, (perf_counter_ns() - start) / 1e6


def worker(repeat):
    jax.config.update("jax_enable_compilation_cache", False)
    _warm_backend()
    start = perf_counter_ns()
    from jax_unroll.fori_model import ForiCostPredictor

    engine = TimedEngine(ForiCostPredictor.load(MODEL))
    load_ms = (perf_counter_ns() - start) / 1e6
    rng = random.Random(583 + repeat)
    settings = list(product(EVAL_SPECS, SIZES, TRIPS))
    rng.shuffle(settings)
    rows = []
    for spec, size, trip in settings:
        workload = spec.build(trip, 1, size)
        args = jax.block_until_ready(workload.args)
        automatic = through_auto_fori(workload.fn, engine)
        variants = {"native": workload.fn, "auto": automatic}
        order = list(variants)
        rng.shuffle(order)
        kernels, outputs, first = {}, {}, {}
        for key in order:
            jax.clear_caches()
            kernels[key], outputs[key], first[key] = compile_and_run(variants[key], args)
        decision_ms = engine.decision_ms
        decision = engine.decision
        if decision.fallback_reason:
            raise ValueError(f"{workload.name}: {decision.fallback_reason}")

        # The native and wrapper traces must differ solely in the unroll param.
        native = str(jax.make_jaxpr(workload.fn)(*args))
        automatic_graph = str(jax.make_jaxpr(automatic)(*args))
        if native.count("unroll=1\n") != 1 or automatic_graph != native.replace(
            "unroll=1\n", f"unroll={decision.factor}\n"
        ):
            raise AssertionError(f"{workload.name}: expected only the unroll factor to change")
        check_output(outputs["auto"], outputs["native"], spec)

        calls = min(128, max(5, int(1500 / _times(kernels["native"], args, 3))))
        samples = {key: [] for key in order}
        for _ in range(9):
            rng.shuffle(order)
            for key in order:
                samples[key].append(_times(kernels[key], args, calls))
        rows.append({
            "workload": workload.name, "variant": size, "trip_count": trip, "repeat": repeat,
            "factor": decision.factor, "decision_ms": decision_ms, "first_ms": first,
            "warm_us": {key: median(times) for key, times in samples.items()},
        })
    return {"load_ms": load_ms, "rows": rows}


def summarize(runs):
    rows = [row for run in runs for row in run["rows"]]
    # At factor 1 the verified program is identical. Use one warm measurement
    # for those pairs to keep dispatch noise out of the estimated policy effect.
    warm_baseline = [r["warm_us"]["native"] for r in rows]
    warm_selected = [r["warm_us"]["native" if r["factor"] == 1 else "auto"] for r in rows]
    ratios = [native / auto for native, auto in zip(warm_baseline, warm_selected, strict=True)]
    cold_native = sum(r["first_ms"]["native"] for r in rows)
    cold_auto = sum(r["first_ms"]["auto"] for r in rows) + sum(run["load_ms"] for run in runs)
    native_ms = sum(warm_baseline) / 1000
    auto_ms = sum(warm_selected) / 1000
    horizons = {}
    for calls in (1, 10, 100, 1000, 10_000):
        horizons[calls] = (cold_native + (calls - 1) * native_ms) / (cold_auto + (calls - 1) * auto_ms)
    return {
        "configurations": len(rows) // len(runs),
        "process_repeats": len(runs),
        "pooled_warm_speedup": sum(warm_baseline) / sum(warm_selected),
        "geometric_mean_warm_speedup": geometric_mean(ratios),
        "worst_warm_slowdown": 1 / min(ratios),
        "warm_wins_over_3_percent": sum(r > 1 / .97 for r in ratios),
        "warm_losses_over_3_percent": sum(r < 1 / 1.03 for r in ratios),
        "median_decision_ms": median(r["decision_ms"] for r in rows),
        "pooled_end_to_end_speedup": horizons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/raw/auto_fori.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.worker)))
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    runs = []
    for repeat in range(args.repeats):
        print(f"Fresh CPU process {repeat + 1}/{args.repeats}", file=sys.stderr, flush=True)
        output = subprocess.check_output(
            [sys.executable, "-m", __spec__.name, "--worker", str(repeat)], text=True,
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
        )
        runs.append(json.loads(output))
    summary = summarize(runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "runs": runs}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
