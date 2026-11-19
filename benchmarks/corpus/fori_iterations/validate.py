"""Audit the frozen native sweep and measure the actual wrapper on new families."""

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
import numpy as np

from benchmarks.corpus.fori_iterations.confirmation import HORIZONS, REPEATS, freeze, matrix
from benchmarks.corpus.fori_iterations.storage import OUT, UNCOMPRESSED
from benchmarks.corpus.fori_proposal.compact import MODEL, POLICY as ORIGINAL_POLICY
from benchmarks.corpus.fori_regret.acquire import assert_same_graph, host
from benchmarks.corpus.fori_regret.live import break_even_calls, measure_calls
from benchmarks.corpus.fori_targeted.auto_fori import through_auto_fori
from benchmarks.corpus.fori_targeted.specialization import check_output
from benchmarks.phase4 import TimedEngine
from benchmarks.sweep import _warm_backend
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.schema import FACTORS
from jax_unroll.selector import PayoffPolicy, SavingsBand, SavingsPolicy, select_factor

REFIT = OUT / "refit.npz"
SAVINGS = OUT / "04_savings_policy.json"
ARMS = ("native", "incumbent", "refit", "old_payoff", "savings", "oracle")
BUDGET = 125


def row_key(row):
    return row["workload"], row["variant"], row["trip_count"]


def model_path(arm, rules):
    if arm == "savings":
        return UNCOMPRESSED if rules["model"] == "incumbent" else REFIT
    return REFIT if arm == "refit" else MODEL


def read_rule(arm):
    if arm == "savings":
        artifact = json.loads(SAVINGS.read_text())
        policy = artifact["policy"]
        bands = tuple(SavingsBand(**row) for row in policy["bands"])
        return artifact, SavingsPolicy(bands, policy["decision_ms"], policy["model_load_ms"])
    if arm == "old_payoff":
        artifact = json.loads(ORIGINAL_POLICY.read_text())
        return artifact, PayoffPolicy(**artifact["policy"])
    return {}, None


def confirmations():
    data = json.loads((OUT / "confirmation_native.json").read_text())["runs"]
    keys = {(spec.name, size, trip) for spec, size, trip in matrix()}
    indexed = [{row_key(row): row for row in run} for run in data]
    if len(indexed) != REPEATS or any(set(index) != keys or len(run) != len(keys)
                                     for index, run in zip(indexed, data, strict=True)):
        raise ValueError("incomplete native confirmation")
    return indexed


def static_score():
    data = confirmations()
    with (OUT / "training_keys.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    families = {r["family"] for r in history}
    hashes = {r[key] for r in history for key in ("body_hash", "current_body_hash")}
    models = {arm: CompactForiPredictor.load(model_path(arm, {})) for arm in ("incumbent", "refit")}
    alpha = []
    for spec, size, trip in matrix():
        key = spec.name, size, trip
        rows = [index[key] for index in data]
        features = rows[0]["features"]
        if spec.family in families or features["body_hash"] in hashes:
            raise ValueError(f"confirmation loop overlaps development: {key}")
        if any(row["features"]["body_hash"] != features["body_hash"] for row in rows):
            raise ValueError(f"inconsistent trace for {key}")
        warm = np.asarray([[median(s) for s in row["warm_samples_us"]] for row in rows])
        cold = [row["cold_ms"] for row in rows]
        if not np.all(np.isfinite(warm) & (warm > 0)) or not np.all(np.isfinite(cold) & (np.asarray(cold) > 0)):
            raise ValueError(f"invalid timing for {key}")
        medians = np.median(warm, axis=0)
        ratios = warm[:, :1] / warm
        eligible = (np.median(ratios, axis=0) >= 1.05) & (ratios > 1).all(axis=0)
        oracle = min((i for i, good in enumerate(eligible) if good), key=lambda i: medians[i], default=0)
        row = {"workload": key[0], "variant": key[1], "trip_count": key[2], "family": spec.family,
               "oracle_factor": FACTORS[oracle], "warm_us": medians.tolist()}
        for arm, model in models.items():
            if not model.supports(features):
                raise ValueError(f"model {arm} does not support {key}")
            prices = model.predict_candidates(features)
            effect = np.array([prices[f].log_warm_us - prices[1].log_warm_us for f in FACTORS[1:]])
            row[arm] = {
                "factor": select_factor(prices, throughput_only=True),
                "effect_mae": float(np.mean(abs(effect - np.log(medians[1:] / medians[0])))),
            }
        alpha.append(row)
    (OUT / "confirmation_static.json").write_text(json.dumps(alpha, indent=2) + "\n")
    return alpha


def worker(arm, repeat):
    jax.config.update("jax_enable_compilation_cache", False)
    _warm_backend()
    oracle = {row_key(row): row["oracle_factor"]
              for row in json.loads((OUT / "confirmation_static.json").read_text())}
    rules, payoff = read_rule(arm)
    load_ms = dict.fromkeys(HORIZONS, 0.)
    eager_model = None

    def load(call_count):
        start = perf_counter_ns()
        model = CompactForiPredictor.load(model_path(arm, rules))
        if rules and model.version != rules["version"]:
            raise ValueError("model does not match payoff rule")
        elapsed = (perf_counter_ns() - start) / 1e6
        if call_count is None:
            load_ms.update(dict.fromkeys(HORIZONS, elapsed))
        else:
            load_ms[call_count] = elapsed
        return model

    if arm == "savings":
        engines = {h: TimedEngine(model_loader=lambda h=h: load(h), payoff=payoff) for h in HORIZONS}
    elif arm not in ("native", "oracle"):
        eager_model = load(None)

    configs = list(product(matrix(), HORIZONS))
    random.Random(583_833 + repeat).shuffle(configs)
    rows, checks = [], []
    for (spec, size, trip), horizon in configs:
        native = spec.build(trip, 1, size)
        args = jax.block_until_ready(jax.device_put(native.args))
        factor, fn = 1, native.fn
        engine = None
        prior_load = load_ms[horizon]
        if arm == "oracle":
            factor = oracle[spec.name, size, trip]
            fn = spec.build(trip, factor, size).fn
        elif arm == "savings" or eager_model is not None:
            engine = engines[horizon] if arm == "savings" else TimedEngine(eager_model, payoff=payoff)
            fn = through_auto_fori(native.fn, engine, throughput_only=payoff is None, R=horizon,
                                  B=BUDGET if payoff is not None else None)
        out, timing = measure_calls(fn, args, horizon)
        if engine is not None:
            expect = "insufficient_calls" if payoff is not None and horizon == 1 else None
            if engine.decision.fallback_reason != expect or engine.decision.cache_hit:
                raise ValueError(f"unexpected live decision for {arm}, {spec.name}: {engine.decision}")
            factor = engine.decision.factor
        rows.append({"workload": spec.name, "variant": size, "trip_count": trip, "calls": horizon,
                     "factor": factor, "decision_ms": engine.decision_ms if engine else 0.,
                     "inline_load_ms": load_ms[horizon] - prior_load, **timing})
        checks.append((spec, native.fn, fn, args, out, factor))

    for spec, baseline, fn, args, output, factor in checks:
        graph = str(jax.make_jaxpr(baseline)(*args))
        assert_same_graph(fn, args, graph, factor)
        expect = output if arm == "native" else jax.block_until_ready(jax.jit(baseline)(*args))
        check_output(output, expect, spec)
    if arm == "savings" and (load_ms[1] or any(load_ms[h] <= 0 for h in HORIZONS[1:])):
        raise ValueError("unexpected deferred model load")
    return {"load_ms": load_ms, "rows": rows}


def summarize(runs, static):
    expected = {(*row_key(r), h) for r, h in product(static, HORIZONS)}
    indexed = {}
    for run in runs:
        if set(run) != set(ARMS):
            raise ValueError("live confirmation is missing arms")
        for arm, block in run.items():
            seen = {(*row_key(r), r["calls"]): r for r in block["rows"]}
            if set(seen) != expected or len(seen) != len(block["rows"]):
                raise ValueError(f"incomplete live timings for {arm}")
            for k, row in seen.items():
                loads = block["load_ms"]
                load = loads[str(k[-1])] if str(k[-1]) in loads else loads[k[-1]]
                indexed.setdefault((arm, k), []).append((row, load))
    if any(len({r["factor"] for r, _ in group}) != 1 for group in indexed.values()):
        raise ValueError("live factors changed across repeats")
    lookup = {row_key(r): r for r in static}
    direct = {}
    for h in HORIZONS:
        keys = sorted(k for k in expected if k[-1] == h)
        direct[h] = {}
        for cohort in ("all", "opportunity", "other"):
            member = [k for k in keys if cohort == "all"
                      or (lookup[k[:3]]["oracle_factor"] != 1) == (cohort == "opportunity")]
            if not member:
                continue
            result = {}
            for arm in ARMS:
                native_total, arm_total = 0., 0.
                first_addition, warm_gain = 0., 0.
                ratios, picks, cases = [], [], []
                for k in member:
                    pairs = zip(indexed[arm, k], indexed["native", k], strict=True)
                    selected = indexed[arm, k][0][0]["factor"]
                    deltas = []
                    for (row, load), (base, _) in pairs:
                        added_load = load / len(keys) - row["inline_load_ms"]
                        total = row["direct_ms"] + added_load
                        deltas.append(total / base["direct_ms"])
                        arm_total += total
                        native_total += base["direct_ms"]
                        first_addition += row["first_ms"] + added_load - base["first_ms"]
                        if h > 1 and selected != 1:
                            warm_gain += (base["warm_ms"] - row["warm_ms"]) / (h - 1)
                    ratios.append(median(deltas))
                    picks.append(selected)
                    cases.append({"workload": k[0], "variant": k[1], "trip_count": k[2],
                                  "factor": selected, "time_ratio": ratios[-1]})
                result[arm] = {
                    "ms_per_process": arm_total / len(runs), "pooled_speedup": native_total / arm_total,
                    "geometric_speedup": 1 / geometric_mean(ratios),
                    "factors": {f: picks.count(f) for f in FACTORS},
                    "wins_over_5pct": sum(r <= 1 / 1.05 for r in ratios),
                    "losses_over_3pct": sum(r > 1.03 for r in ratios),
                    "worst_time_ratio": max(ratios),
                    "median_load_ms": median(load for _, load in indexed[arm, member[0]]),
                    "estimated_crossover_calls": break_even_calls(first_addition, warm_gain) if h == HORIZONS[-1] else None,
                    "cases": cases,
                }
            direct[h][cohort] = result
    return {"direct": direct}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.arm, args.worker)))
        return
    freeze()
    static = static_score()
    runs = []
    for repeat in range(REPEATS):
        order = list(ARMS)
        random.Random(583_834 + repeat).shuffle(order)
        data = {}
        for arm in order:
            print(f"confirmation process {repeat + 1}/{REPEATS} {arm}", file=sys.stderr, flush=True)
            command = [sys.executable, "-m", __spec__.name, "--worker", str(repeat), "--arm", arm]
            data[arm] = json.loads(subprocess.check_output(
                command, text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"},
            ))
        runs.append(data)
        (OUT / "confirmation_live.json").write_text(
            json.dumps({"host": host(), "runs": runs}, separators=(",", ":")) + "\n",
        )
    (OUT / "confirmation_summary.json").write_text(json.dumps(summarize(runs, static), indent=2) + "\n")


if __name__ == "__main__":
    main()
