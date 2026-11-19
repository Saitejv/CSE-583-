"""Compare directly executed short, medium and long horizons on the frozen matrix."""

import argparse
import csv
from itertools import product
import json
import os
import random
from statistics import geometric_mean, median
import subprocess
import sys
from time import perf_counter_ns

import jax
import jaxlib

from benchmarks.corpus.fori_proposal.compact import MODEL, POLICY
from benchmarks.corpus.fori_proposal.protocol import BUDGET_MS, HORIZONS, OUT, REPEATS, freeze
from benchmarks.corpus.fori_proposal.workloads import settings
from benchmarks.corpus.fori_regret.acquire import assert_same_graph, host
from benchmarks.corpus.fori_regret.calibrate import load_model
from benchmarks.corpus.fori_regret.live import break_even_calls, measure_calls
from benchmarks.corpus.fori_targeted.auto_fori import through_auto_fori
from benchmarks.corpus.fori_targeted.specialization import check_output
from benchmarks.phase4 import TimedEngine
from benchmarks.sweep import _warm_backend
from jax_unroll.baselines import llvm_style_heuristic
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.model import LogCosts
from jax_unroll.schema import FACTORS, FEATURE_VERSION
from jax_unroll.selector import PayoffPolicy

ARMS = ("native", "incumbent", "compact", "payoff", "size_heuristic", "oracle")
OPPORTUNITY_HORIZON = 20_000
OPPORTUNITY_ARMS = ("native", "compact", "oracle")


class SizeModel:
    """CostModel interface for the Jaxpr equation-count baseline."""

    backend = "cpu"
    feature_version = FEATURE_VERSION
    jax_versions = jax.__version__, jaxlib.__version__
    version = "size-128"

    def supports(self, row):
        return row["loop_kind"] == "fori_loop"

    def predict_candidates(self, row):
        best = llvm_style_heuristic(row)
        return {factor: LogCosts(0, int(factor != best)) for factor in FACTORS}


def native_choices():
    with (OUT / "native_decisions.csv").open(newline="") as handle:
        return {(r["workload"], r["variant"], int(r["trip_count"])): r for r in csv.DictReader(handle)}


def experiment(opportunities, choices):
    configs = [c for c in settings() if c.live]
    if opportunities:
        configs = [c for c in configs if c.cohort == "fresh"
                   and int(choices[c.spec.name, c.variant, c.trip]["oracle_factor"]) != 1]
        return configs, (OPPORTUNITY_HORIZON,), OPPORTUNITY_ARMS
    return configs, HORIZONS, ARMS


def load_arm(arm):
    if arm in ("native", "oracle"):
        return None, None, 0.
    if arm == "size_heuristic":
        return SizeModel(), None, 0.
    start = perf_counter_ns()
    model = load_model("incumbent") if arm == "incumbent" else CompactForiPredictor.load(MODEL)
    payoff = None
    if arm == "payoff":
        entry = json.loads(POLICY.read_text())
        if model.version != entry["version"]:
            raise ValueError("compact model and payoff calibration are out of sync")
        payoff = PayoffPolicy(**entry["policy"])
    return model, payoff, (perf_counter_ns() - start) / 1e6


def worker(arm, repeat, opportunities=False):
    jax.config.update("jax_enable_compilation_cache", False)
    _warm_backend()
    choices = native_choices()
    if arm in ("incumbent", "compact", "payoff") and "sklearn" in sys.modules:
        raise ValueError("training dependencies loaded before the startup timer")
    model, payoff, load_ms = load_arm(arm)
    planned, horizons, _ = experiment(opportunities, choices)
    configs = list(product(planned, horizons))
    random.Random(583_722 + repeat).shuffle(configs)
    rows, checks = [], []
    for cfg, calls in configs:
        spec, size, trip = cfg.spec, cfg.variant, cfg.trip
        k = spec.name, size, trip
        pick = choices[k]
        baseline = spec.build(trip, 1, size)
        args = jax.block_until_ready(jax.device_put(baseline.args))
        factor = 1
        fn = baseline.fn
        engine = None
        if arm == "oracle":
            factor = int(pick["oracle_factor"])
            fn = spec.build(trip, factor, size).fn
        elif model is not None:
            engine = TimedEngine(model, payoff=payoff)
            fn = through_auto_fori(fn, engine, throughput_only=payoff is None, R=calls,
                                  B=BUDGET_MS if payoff is not None else None)
        output, timing = measure_calls(fn, args, calls)
        if engine is not None:
            decision = engine.decision
            allowed = "insufficient_calls" if payoff is not None and calls == 1 else None
            label = f"payoff_{calls}" if payoff is not None else "size_heuristic" if arm == "size_heuristic" else "incumbent"
            if decision.fallback_reason != allowed or decision.cache_hit or decision.factor != int(pick[f"{label}_factor"]):
                raise ValueError(f"inconsistent live decision for {arm}, {k}: {decision}")
            factor = decision.factor
        rows.append({
            "workload": spec.name, "family": spec.family, "variant": size, "trip_count": trip,
            "cohort": cfg.cohort, "opportunity": int(pick["oracle_factor"]) != 1,
            "calls": calls, "factor": factor, "decision_ms": engine.decision_ms if engine else 0.,
            **timing,
        })
        checks.append((spec, baseline.fn, fn, args, output, factor))
    for spec, baseline, fn, args, output, factor in checks:
        assert_same_graph(fn, args, str(jax.make_jaxpr(baseline)(*args)), factor)
        expected = output if arm == "native" else jax.block_until_ready(jax.jit(baseline)(*args))
        check_output(output, expected, spec)
    return {"load_ms": load_ms, "rows": rows}


def summarize(runs, planned, horizons, arms):
    indexed = {}
    expected = {(c.spec.name, c.variant, c.trip, calls)
                for c, calls in product(planned, horizons)}
    for run in runs:
        if set(run) != set(arms):
            raise ValueError("live run is missing an arm")
        for arm, data in run.items():
            entries = {(r["workload"], r["variant"], r["trip_count"], r["calls"]): r for r in data["rows"]}
            if len(entries) != len(data["rows"]) or set(entries) != expected:
                raise ValueError(f"incomplete runtime settings for {arm}")
            for k, row in entries.items():
                indexed.setdefault((arm, k), []).append((row, data["load_ms"]))

    direct, break_even = {}, {}
    for calls in horizons:
        keys = sorted(k for k in expected if k[-1] == calls)
        direct[calls] = {}
        for cohort in ("all", "fresh", "fresh_with_opportunity", "fresh_other",
                       "proposal_control", "historical_control", "regression_control"):
            selected = []
            for k in keys:
                row = indexed["native", k][0][0]
                label = row["cohort"]
                if (cohort in ("all", label)
                        or (cohort == "fresh_with_opportunity" and label == "fresh" and row["opportunity"])
                        or (cohort == "fresh_other" and label == "fresh" and not row["opportunity"])):
                    selected.append(k)
            if not selected:
                continue
            by_arm = {}
            native_total = sum(r["direct_ms"] for k in selected for r, _ in indexed["native", k])
            for arm in arms:
                total, invocation_ms = 0., 0.
                first_delta, warm_gain = 0., 0.
                ratios, comparisons = [], []
                for k in selected:
                    paired = list(zip(indexed[arm, k], indexed["native", k], strict=True))
                    elapsed = [r["direct_ms"] + load / len(keys) for (r, load), _ in paired]
                    base = [r["direct_ms"] for _, (r, _) in paired]
                    ratio = median(a / n for a, n in zip(elapsed, base, strict=True))
                    ratios.append(ratio)
                    total += sum(elapsed)
                    invocation_ms += sum(r["direct_ms"] for (r, _), _ in paired)
                    if calls == horizons[-1]:
                        deltas = [r["first_ms"] + load / len(keys) - n["first_ms"] for (r, load), (n, _) in paired]
                        gains = [(n["warm_ms"] - r["warm_ms"]) / (calls - 1) if r["factor"] != 1 else 0.
                                 for (r, _), (n, _) in paired]
                        first_delta += sum(deltas)
                        warm_gain += sum(gains)
                        comparisons.append({
                            "workload": k[0], "variant": k[1], "trip_count": k[2], "time_ratio": ratio,
                            "factor": paired[0][0][0]["factor"],
                            "estimated_break_even": break_even_calls(median(deltas), median(gains)),
                        })
                if calls == horizons[-1]:
                    break_even.setdefault(cohort, {})[arm] = {
                        "pooled_estimated_break_even": break_even_calls(first_delta, warm_gain),
                        "configurations": comparisons,
                    }
                by_arm[arm] = {
                    "pooled_ms": total / len(runs),
                    "pooled_ms_excluding_load": invocation_ms / len(runs),
                    "pooled_speedup": native_total / total,
                    "geometric_speedup": 1 / geometric_mean(ratios),
                    "wins_over_5pct": sum(r <= 1 / 1.05 for r in ratios),
                    "slowdowns_over_3pct": sum(r > 1.03 for r in ratios),
                    "worst_time_ratio": max(ratios),
                    "median_decision_ms": median(r["decision_ms"] for k in selected for r, _ in indexed[arm, k]),
                }
            direct[calls][cohort] = {"configurations": len(selected), "arms": by_arm}
    return {
        "direct": direct, "estimated_break_even_from_long_horizon": break_even,
        "median_load_ms": {arm: median(run[arm]["load_ms"] for run in runs) for arm in arms},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--repeat-opportunities", action="store_true",
                        help="directly time 20,000 calls on all live fresh cases with a verified warm opportunity")
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.arm, args.worker, args.repeat_opportunities)))
        return
    freeze()
    planned, horizons, arms = experiment(args.repeat_opportunities, native_choices())
    stem = "opportunities" if args.repeat_opportunities else "live"
    runs = []
    for repeat in range(REPEATS):
        order = list(arms)
        random.Random(583_723 + repeat).shuffle(order)
        run = {}
        for arm in order:
            print(f"{stem} process {repeat + 1}/{REPEATS} {arm}", file=sys.stderr, flush=True)
            cmd = [sys.executable, "-m", __spec__.name, "--worker", str(repeat), "--arm", arm]
            if args.repeat_opportunities:
                cmd.append("--repeat-opportunities")
            run[arm] = json.loads(subprocess.check_output(
                cmd, text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"},
            ))
        runs.append(run)
        (OUT / f"{stem}_runs.json").write_text(
            json.dumps({"host": host(), "runs": runs}, separators=(",", ":")) + "\n",
        )
    (OUT / f"{stem}_summary.json").write_text(json.dumps(summarize(runs, planned, horizons, arms), indent=2) + "\n")


if __name__ == "__main__":
    main()
