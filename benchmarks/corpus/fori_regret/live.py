"""Measure complete invocation horizons through the public auto_fori_loop wrapper."""

from __future__ import annotations

import argparse
from itertools import product
import json
import math
import os
import random
from statistics import geometric_mean, median
import subprocess
import sys
from time import perf_counter_ns

import jax

from benchmarks.corpus.fori_regret.acquire import OUT, REPEATS, SIZES, TRIPS, assert_same_graph, host
from benchmarks.corpus.fori_regret.calibrate import CALIBRATION, SETTINGS, load_model
from benchmarks.corpus.fori_regret.freeze import check
from benchmarks.corpus.fori_regret.workloads import cases
from benchmarks.corpus.fori_targeted.auto_fori import through_auto_fori
from benchmarks.corpus.fori_targeted.specialization import check_output
from benchmarks.phase4 import TimedEngine
from benchmarks.sweep import _warm_backend
from jax_unroll.selector import PayoffPolicy

ARMS = ("native", "incumbent", "refit", "guarded_incumbent", "guarded_refit")
HORIZONS = (1, 300, 5_000)
EXTRAPOLATED = (20_000, 100_000)
COLD_BUDGET_MS = 125


def measure_calls(fn, args, horizon):
    """Time compilation and every synchronized invocation of one executable."""
    jax.clear_caches()
    start = perf_counter_ns()
    lowered = jax.jit(fn).lower(*args)
    lowered_at = perf_counter_ns()
    kernel = lowered.compile()
    compiled_at = perf_counter_ns()
    output = jax.block_until_ready(kernel(*args))
    first_at = perf_counter_ns()
    for _ in range(horizon - 1):
        jax.block_until_ready(kernel(*args))
    end = perf_counter_ns() if horizon > 1 else first_at
    return output, {
        "trace_lower_ms": (lowered_at - start) / 1e6,
        "compile_ms": (compiled_at - lowered_at) / 1e6,
        "first_exec_ms": (first_at - compiled_at) / 1e6,
        "first_ms": (first_at - start) / 1e6,
        "warm_ms": (end - first_at) / 1e6,
        "direct_ms": (end - start) / 1e6,
    }


def worker(arm, repeat, all_configs=False):
    jax.config.update("jax_enable_compilation_cache", False)
    _warm_backend()
    guarded = arm.startswith("guarded_")
    engine, load_ms = None, 0.
    if arm != "native":
        start = perf_counter_ns()
        name = arm.removeprefix("guarded_")
        model = load_model(name)
        policy = None
        if guarded:
            entry = json.loads(CALIBRATION.read_text())["models"][name]
            if model.version != entry["version"]:
                raise ValueError(f"calibration does not match the {name} artifact")
            policy = PayoffPolicy(**entry["policy"])
        engine = TimedEngine(model, payoff=policy)
        load_ms = (perf_counter_ns() - start) / 1e6
    settings = list(product(SIZES, TRIPS)) if all_configs else SETTINGS
    configs = list(product(cases("holdout").values(), settings, HORIZONS))
    random.Random(583_448 + repeat).shuffle(configs)
    rows, checks = [], []
    for spec, (size, trip), horizon in configs:
        native = spec.build(trip, 1, size)
        args = jax.block_until_ready(jax.device_put(native.args))
        fn = native.fn
        if engine is not None:
            # Keep each timed specialization cold, including its decision cache.
            engine = TimedEngine(engine.model, payoff=engine.payoff)
            fn = through_auto_fori(native.fn, engine, throughput_only=not guarded, R=horizon,
                                  B=COLD_BUDGET_MS if guarded else None)
        output, timing = measure_calls(fn, args, horizon)
        decision = engine.decision if engine is not None else None
        factor = decision.factor if decision is not None else 1
        fallback = "insufficient_calls" if guarded and horizon == 1 else None
        if decision is not None and (decision.fallback_reason != fallback or decision.cache_hit):
            raise ValueError(f"unexpected decision for {arm}, {spec.name}: {decision}")
        rows.append({
            "workload": spec.name, "family": spec.family, "variant": size, "trip_count": trip,
            "calls": horizon, "factor": factor, "decision_ms": engine.decision_ms if engine else 0.,
            **timing,
        })
        checks.append((spec, native.fn, fn, args, output, factor))

    # Keep correctness checks and reference compilation outside the timed run.
    for spec, native, fn, args, output, factor in checks:
        graph = str(jax.make_jaxpr(native)(*args))
        assert_same_graph(fn, args, graph, factor)
        expected = jax.block_until_ready(jax.jit(native)(*args)) if arm != "native" else output
        check_output(output, expected, spec)
    return {"load_ms": load_ms, "rows": rows}


def summarize(runs):
    indexed = {}
    for run in runs:
        for arm, data in run.items():
            for row in data["rows"]:
                key = row["workload"], row["variant"], row["trip_count"], row["calls"]
                indexed.setdefault((arm, key), []).append((row, data["load_ms"]))
    keys = sorted(key for arm, key in indexed if arm == "native")
    if set(indexed) != set(product(ARMS, keys)) or any(len(rows) != len(runs) for rows in indexed.values()):
        raise ValueError("incomplete wrapper timing matrix")
    if any(len({row["factor"] for row, _ in rows}) != 1 for rows in indexed.values()):
        raise ValueError("selector chose different factors across identical process repeats")
    counts = {calls: sum(key[-1] == calls for key in keys) for calls in HORIZONS}

    def samples(arm, key):
        return indexed[arm, key]

    def totals(arm, key):
        return [row["direct_ms"] + load / counts[key[-1]] for row, load in samples(arm, key)]

    def rates(arm, key):
        return [
            (row if row["factor"] != 1 else native)["warm_ms"] / (key[-1] - 1)
            for (row, _), (native, _) in zip(samples(arm, key), samples("native", key), strict=True)
        ]

    summary = {"direct": {}, "extrapolated": {}, "break_even": {}}
    for calls in HORIZONS:
        setting = [key for key in keys if key[-1] == calls]
        native_ms = sum(sum(totals("native", key)) for key in setting)
        incumbent_ms = sum(sum(totals("incumbent", key)) for key in setting)
        summary["direct"][calls] = {}
        for arm in ARMS:
            pooled = sum(sum(totals(arm, key)) for key in setting)
            by_workload = {}
            ratios = []
            for key in setting:
                ratio = median(a / n for a, n in zip(totals(arm, key), totals("native", key), strict=True))
                ratios.append(ratio)
                by_workload.setdefault(key[0], []).append(ratio)
            regressions = [ratio for ratio in ratios if ratio > 1.03]
            summary["direct"][calls][arm] = {
                "pooled_ms_per_process": pooled / len(runs),
                "pooled_speedup_vs_native": native_ms / pooled,
                "pooled_speedup_vs_incumbent": incumbent_ms / pooled,
                "workload_speedup_vs_native": 1 / geometric_mean(map(geometric_mean, by_workload.values())),
                "slowdowns": sum(ratio > 1 for ratio in ratios),
                "slowdowns_over_3pct": len(regressions),
                "median_slowdown_over_3pct": median(regressions) if regressions else None,
                "worst_ratio_vs_native": max(ratios),
                "factors": {f: sum(samples(arm, key)[0][0]["factor"] == f for key in setting)
                            for f in (1, 2, 4, 8)},
                "median_decision_ms": median(row["decision_ms"] for key in setting
                                             for row, _ in samples(arm, key)),
            }
    long = HORIZONS[-1]
    setting = [key for key in keys if key[-1] == long]
    for arm in ARMS:
        first_deltas, savings, config_break_even = [], [], []
        for key in setting:
            pairs = zip(samples(arm, key), samples("native", key), strict=True)
            deltas = [a["first_ms"] + load / len(setting) - n["first_ms"] for (a, load), (n, _) in pairs]
            gains = [n - a for n, a in zip(rates("native", key), rates(arm, key), strict=True)]
            first_deltas.extend(deltas)
            savings.extend(gains)
            count = break_even_calls(median(deltas), median(gains))
            if count is not None:
                config_break_even.append({"workload": key[0], "size": key[1], "trip": key[2], "calls": count})
        summary["break_even"][arm] = {
            "pooled_calls": break_even_calls(sum(first_deltas), sum(savings)),
            "configurations_with_break_even": config_break_even,
        }
    for calls in EXTRAPOLATED:
        projected = {
            arm: sum(sum(totals(arm, key)) + (calls - long) * sum(rates(arm, key)) for key in setting)
            for arm in ARMS
        }
        summary["extrapolated"][calls] = {
            arm: {"pooled_ms_per_process": value / len(runs),
                  "pooled_speedup_vs_native": projected["native"] / value,
                  "pooled_speedup_vs_incumbent": projected["incumbent"] / value}
            for arm, value in projected.items()
        }
    return summary


def break_even_calls(setup_ms, saving_ms):
    # A durable crossover requires positive warm savings.
    if saving_ms <= 0:
        return None
    if setup_ms < 0:
        return 1
    return math.floor(setup_ms / saving_ms) + 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--all-configs", action="store_true", help="cover every frozen holdout configuration")
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.arm, args.worker, args.all_configs)))
        return
    check()
    stem = "live_all" if args.all_configs else "live"
    runs = []
    for repeat in range(REPEATS):
        order = list(ARMS)
        random.Random(583_484 + repeat).shuffle(order)
        run = {}
        for arm in order:
            print(f"wrapper process {repeat + 1}/{REPEATS} {arm}", file=sys.stderr, flush=True)
            cmd = [sys.executable, "-m", __spec__.name, "--worker", str(repeat), "--arm", arm]
            if args.all_configs:
                cmd.append("--all-configs")
            run[arm] = json.loads(subprocess.check_output(
                cmd, text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"},
            ))
        runs.append(run)
        payload = {"host": host(), "cold_budget_ms": COLD_BUDGET_MS, "runs": runs}
        (OUT / f"{stem}_runs.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    summary = summarize(runs)
    data = json.dumps(summary, indent=2) + "\n"
    (OUT / f"{stem}_summary.json").write_text(data)
    print(data, flush=True)


if __name__ == "__main__":
    main()
