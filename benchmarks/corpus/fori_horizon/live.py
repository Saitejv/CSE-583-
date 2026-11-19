"""Confirm preselected policies with one fresh CPU process per arm, horizon and repeat."""

import argparse
import csv
from hashlib import sha256
from itertools import product
import json
import os
import random
import subprocess
import sys
from time import perf_counter_ns

import jax
import numpy as np

from benchmarks.corpus.fori_horizon import coverage
from benchmarks.corpus.fori_horizon.cohort import HORIZONS, OUT, matrix
from benchmarks.corpus.fori_horizon.transfer import freeze_all, settings
from benchmarks.corpus.fori_iterations.storage import UNCOMPRESSED
from benchmarks.corpus.fori_proposal.score import repeated_winner
from benchmarks.corpus.fori_regret.acquire import REPEATS, assert_same_graph, host
from benchmarks.corpus.fori_regret.live import measure_calls
from benchmarks.corpus.fori_targeted.auto_fori import through_auto_fori
from benchmarks.corpus.fori_targeted.specialization import check_output
from benchmarks.phase4 import TimedEngine
from benchmarks.sweep import _warm_backend
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.schema import FACTORS
from jax_unroll.selector import HorizonPolicy

ARMS = ("native", "incumbent", "crossover", "candidate", "oracle")


def key(row):
    return row["workload"], row["variant"], row["trip_count"]


def sweep_choices(configs, horizons, directory):
    from benchmarks.corpus.fori_horizon.replay import curves, total

    with open("results/fori_iterations/training_keys.csv", newline="") as stream:
        history = list(csv.DictReader(stream))
    for path in ("fori_regret/holdout_runs", "fori_proposal/native_runs", "fori_iterations/confirmation_native"):
        history.extend({**r, **r["features"]} for r in json.load(open(f"results/{path}.json"))["runs"][0])
    seen = {r.get(k) for r in history for k in ("family", "workload", "body_hash", "current_body_hash")}
    if directory == coverage.DIRECTORY:
        history_rows = json.loads((OUT / "sweep.json").read_text())["runs"][0]
        seen.update(x for r in history_rows for x in (r["workload"], r["family"], r["features"]["body_hash"]))
    frozen = {(s.name,v,t): s for s,v,t in configs}
    probes = {s.name for s,_,_ in settings()[:-1]}
    cases = []
    for row, cold, warm in curves(directory / "sweep.json"):
        name, size, trip = key(row)
        spec = frozen[name,size,trip]
        cohort = "historical" if name == "dense_gram_accum" else "transfer" if name in probes else "mixed"
        if directory == coverage.DIRECTORY:
            cohort = "coverage"
        if cohort != "historical" and seen.intersection((name, spec.family, row["features"]["body_hash"])):
            raise ValueError(f"development overlap for {name}")
        first, steady = np.median(cold, axis=0), np.median(warm, axis=0)
        times = {h: total(first, steady, h) for h in horizons}
        cases.append(dict(workload=name, variant=size, trip_count=trip, family=spec.family, cohort=cohort,
                          opportunity=bool(repeated_winner(warm)), first_ms=first.tolist(), warm_us=steady.tolist(),
                          oracle={h: FACTORS[t.argmin()] for h,t in times.items()},
                          replay_ms={h: t.tolist() for h,t in times.items()}))
    if {key(r) for r in cases} != set(frozen) or len(cases) != len(frozen):
        raise ValueError("incomplete sweep")
    (directory / "oracle_replay.json").write_text(json.dumps(cases, indent=2) + "\n")


def worker(arm, h, repeat, configs, directory):
    jax.config.update("jax_enable_compilation_cache", False)
    _warm_backend()
    oracle = {key(r): r["oracle"][str(h)] for r in json.loads((directory / "oracle_replay.json").read_text())}
    load_ms = 0.
    engine = None

    def load():
        nonlocal load_ms
        start = perf_counter_ns()
        model = CompactForiPredictor.load(UNCOMPRESSED)
        load_ms += (perf_counter_ns() - start) / 1e6
        return model

    start = perf_counter_ns()
    if arm == "incumbent":
        engine = TimedEngine(load())
    elif arm in ("crossover", "candidate"):
        policy = json.loads((OUT / "policy.json").read_text())
        engine = TimedEngine(model_loader=load, payoff=HorizonPolicy(**policy[arm]))
    setup_ms = (perf_counter_ns() - start) / 1e6 if engine else 0.
    configs = list(configs)
    random.Random(583922 + 31 * repeat + h).shuffle(configs)
    rows, checks = [], []
    for spec, size, trip in configs:
        native = spec.build(trip, 1, size)
        args = jax.block_until_ready(jax.device_put(native.args))
        factor, fn = 1, native.fn
        if engine:
            fn = through_auto_fori(native.fn, engine, R=h, B=None if arm == "incumbent" else 125.,
                                   throughput_only=arm == "incumbent")
        elif arm == "oracle":
            factor = oracle[spec.name, size, trip]
            fn = spec.build(trip, factor, size).fn
        prior = load_ms
        out, timing = measure_calls(fn, args, h)
        if engine:
            if engine.decision.fallback_reason not in (None, "insufficient_calls") or engine.decision.cache_hit:
                raise ValueError(f"unexpected decision {engine.decision}")
            factor = engine.decision.factor
        rows.append(dict(workload=spec.name, variant=size, trip_count=trip, calls=h, factor=factor,
                         decision_ms=engine.decision_ms if engine else 0., inline_load_ms=load_ms-prior, **timing))
        checks.append((spec, native.fn, fn, args, out, factor))
    for spec, native, fn, args, out, factor in checks:
        assert_same_graph(fn, args, str(jax.make_jaxpr(native)(*args)), factor)
        check_output(out, jax.block_until_ready(jax.jit(native)(*args)), spec)
    if engine and engine.model is not None and arm != "incumbent" and engine.model.version != policy["model_version"]:
        raise ValueError("calibration/model mismatch")
    return dict(arm=arm, calls=h, repeat=repeat, setup_ms=setup_ms, load_ms=load_ms, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--calls", type=int, choices=sorted(set(HORIZONS + coverage.DIRECT)))
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    freeze_all()
    directory, configs, horizons = OUT, matrix() + settings(), HORIZONS
    flags = []
    if args.coverage:
        coverage.freeze()
        directory, configs, horizons = coverage.DIRECTORY, coverage.matrix(), coverage.DIRECT
        flags = ["--coverage"]
    if args.sweep:
        from benchmarks.corpus.fori_regret.acquire import measure, run_workers
        if args.worker is not None:
            print(json.dumps(measure(configs, args.worker)))
        else:
            run_workers(__spec__.name, flags + ["--sweep"], REPEATS, directory / "sweep.json")
        return
    if args.worker is not None:
        print(json.dumps(worker(args.arm, args.calls, args.worker, configs, directory)))
        return
    sweep_choices(configs, horizons, directory)
    sources = ([OUT / "policy.json", directory / "matrix.json", directory / "oracle_replay.json"]
               if args.coverage else [OUT / p for p in ("policy.json", "matrix.json", "transfer_matrix.json",
                                                         "oracle_replay.json")])
    stamp = {os.path.relpath(path, directory): sha256(path.read_bytes()).hexdigest() for path in sources}
    blocks = []
    for repeat in range(REPEATS):
        order = list(product(ARMS, horizons))
        random.Random(583923 + repeat).shuffle(order)
        for arm, h in order:
            print(repeat+1, arm, h, flush=True, file=sys.stderr)
            cmd = [sys.executable, "-m", __spec__.name, "--worker", str(repeat), "--arm", arm, "--calls", str(h), *flags]
            blocks.append(json.loads(subprocess.check_output(cmd, text=True,
                                                             env={**os.environ, "JAX_PLATFORMS":"cpu"})))
            (directory / "live.json").write_text(json.dumps(dict(host=host(), freeze=stamp, blocks=blocks),
                                                       separators=(",",":")) + "\n")


if __name__ == "__main__":
    main()
