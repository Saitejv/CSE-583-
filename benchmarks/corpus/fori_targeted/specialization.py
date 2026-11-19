"""Measure opt-in fori specializations against every native factor."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
from itertools import product
from pathlib import Path
from time import perf_counter_ns

import jax
import jaxlib
import numpy as np

from benchmarks.corpus.fori_targeted.registry import (
    SPECIALIZED_NAMES,
    build_specialized_workload,
    get_spec,
)
from benchmarks.sweep import _warm_backend
from jax_unroll.autotune import _times
from jax_unroll.schema import FACTORS

SIZES = ("small", "large")
TRIPS = (255, 256, 1023, 1024)
NATIVE = tuple(f"native_{factor}" for factor in FACTORS)
SPECIALIZED = "specialized"
SPECIALIZED_FACTOR_1 = "specialized_factor_1"


def alternate_inputs(name, args, repeat):
    """Change the runtime inputs while retaining the original workload shapes."""
    rng = np.random.default_rng(583_700 + repeat)

    def varied(arg):
        return rng.uniform(-0.5, 0.5, arg.shape).astype(arg.dtype)

    if name == "write_multitap_delay_ring":
        ring, head, sample, stream = args
        return varied(ring), np.asarray(ring.size - 3 - repeat, dtype=head.dtype), sample, varied(stream)
    if name == "eval_app_ct_backprojection_step":
        image, sino, cos, sin, gx, gy = args
        angles = rng.uniform(0, np.pi, cos.shape)
        return varied(image), varied(sino), np.cos(angles).astype(cos.dtype), np.sin(angles).astype(sin.dtype), gx, gy
    if name == "eval_dense_horner_matrix_poly":
        initial, matrix, coeffs = args
        return varied(initial), varied(matrix) / matrix.shape[0], varied(coeffs)
    if name == "eval_write_pic_charge_deposit":
        grid, positions, charges = args
        coords = rng.uniform(1, grid.shape[0] - 2, positions.shape).astype(positions.dtype)
        return varied(grid), coords, varied(charges)
    return tuple(varied(arg) for arg in args)


def check_output(actual, expected, spec):
    if jax.tree.structure(actual) != jax.tree.structure(expected):
        raise ValueError(f"{spec.name}: mismatched output structure")
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        if np.issubdtype(want.dtype, np.inexact):
            np.testing.assert_allclose(got, want, rtol=spec.rtol, atol=spec.atol, equal_nan=False)
            if not np.isfinite(got).all():
                raise ValueError(f"{spec.name}: nonfinite output")
        else:
            np.testing.assert_array_equal(got, want)


def measure(name, repeat):
    jax.config.update("jax_enable_compilation_cache", False)
    _warm_backend()
    spec = get_spec(name)
    rng = random.Random(583 + repeat)
    configs = list(product(SIZES, TRIPS))
    rng.shuffle(configs)
    for size, trip in configs:
        workloads = {label: spec.build(trip, factor, size) for label, factor in zip(NATIVE, FACTORS, strict=True)}
        baseline = workloads[NATIVE[0]]
        optimized = build_specialized_workload(name, trip, size)
        workloads[SPECIALIZED] = optimized
        workloads[SPECIALIZED_FACTOR_1] = build_specialized_workload(name, trip, size, factor=1)
        original = jax.block_until_ready(baseline.args)
        alternate = jax.block_until_ready(jax.device_put(alternate_inputs(name, original, repeat)))
        jax.clear_caches()

        labels = list(workloads)
        rng.shuffle(labels)
        compiled = {}
        first_ms = {}
        for label in labels:
            start = perf_counter_ns()
            kernel = jax.jit(workloads[label].fn).lower(*original).compile()
            jax.block_until_ready(kernel(*original))
            first_ms[label] = (perf_counter_ns() - start) / 1e6
            compiled[label] = kernel

        for args in (original, alternate):
            expected = jax.block_until_ready(compiled[NATIVE[0]](*args))
            for label in labels:
                check_output(jax.block_until_ready(compiled[label](*args)), expected, spec)

        calls = min(128, max(4, int(3000 / _times(compiled[NATIVE[0]], alternate, 3))))
        samples = {label: [] for label in labels}
        for _ in range(9):
            rng.shuffle(labels)
            for label in labels:
                samples[label].append(_times(compiled[label], alternate, calls))
        yield {
            "workload": name,
            "variant": size,
            "trip": trip,
            "repeat": repeat,
            "factor": optimized.factor,
            "calls_per_block": calls,
            "first_ms": first_ms,
            "warm_us": {label: statistics.median(blocks) for label, blocks in samples.items()},
        }


def summarize(rows):
    summary = {}
    for name in SPECIALIZED_NAMES:
        cases = [row for row in rows if row["workload"] == name]
        if not cases:
            continue
        baseline_us = [row["warm_us"][NATIVE[0]] for row in cases]
        best_native_us = [min(row["warm_us"][label] for label in NATIVE) for row in cases]
        specialized_us = [row["warm_us"][SPECIALIZED] for row in cases]
        specialized_factor_1_us = [row["warm_us"][SPECIALIZED_FACTOR_1] for row in cases]
        summary[name] = {
            "paired_warm_speedup_vs_factor_1": statistics.geometric_mean(
                native / fast for native, fast in zip(baseline_us, specialized_us, strict=True)
            ),
            "paired_warm_speedup_vs_fastest_native": statistics.geometric_mean(
                native / fast for native, fast in zip(best_native_us, specialized_us, strict=True)
            ),
            "paired_warm_speedup_vs_specialized_factor_1": statistics.geometric_mean(
                rolled / fast for rolled, fast in zip(specialized_factor_1_us, specialized_us, strict=True)
            ),
            "min_paired_warm_speedup_vs_factor_1": min(
                native / fast for native, fast in zip(baseline_us, specialized_us, strict=True)
            ),
            "pooled_warm_speedup_vs_factor_1": sum(baseline_us) / sum(specialized_us),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/raw/fori_specialization.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workload", choices=SPECIALIZED_NAMES, action="append")
    parser.add_argument("--worker", nargs=2, metavar=("WORKLOAD", "REPEAT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        name, repeat = args.worker
        print(json.dumps(list(measure(name, int(repeat)))))
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    tasks = list(product(args.workload or SPECIALIZED_NAMES, range(args.repeats)))
    random.Random(583).shuffle(tasks)
    rows = []
    for name, repeat in tasks:
        print(f"{name}, repeat {repeat + 1}/{args.repeats}", file=sys.stderr, flush=True)
        result = subprocess.check_output(
            [sys.executable, "-m", __spec__.name, "--worker", name, str(repeat)],
            text=True,
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
        )
        rows.extend(json.loads(result))
    summary = summarize(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "machine": platform.machine(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "backend": "cpu",
        "summary": summary,
        "runs": rows,
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
