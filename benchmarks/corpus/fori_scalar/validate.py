"""Direct process-isolated evaluation of the fitted scalar costs and the previous engine."""

import argparse
import json
import os
import random
import subprocess
import sys
from hashlib import sha256
from itertools import product
from time import perf_counter_ns

import jax
import numpy as np
import pandas as pd

from benchmarks.corpus.fori_application import scan_ports
from benchmarks.corpus.fori_application.live import Recorder
from benchmarks.corpus.fori_iterations.storage import UNCOMPRESSED
from benchmarks.corpus.fori_regret.acquire import REPEATS, assert_same_graph, host
from benchmarks.corpus.fori_regret.live import measure_calls
from benchmarks.corpus.fori_scalar import cases
from benchmarks.corpus.fori_scalar.dataset import OUT, load
from benchmarks.corpus.fori_scalar.fit import BUNDLED
from benchmarks.corpus.fori_targeted.auto_fori import through_auto_fori
from benchmarks.corpus.fori_targeted.specialization import check_output
from benchmarks.sweep import _warm_backend
from jax_unroll.compact_fori import application_engine
from jax_unroll.schema import FACTORS

ARMS = ("native", "incumbent", "candidate", "oracle")
CALLS = (300, 5000, 20000)


def configs():
    return [scan_ports.settings()[0], *cases.settings("test")]


def oracle():
    result = {}
    for split in ("train", "test"):
        rows, times = load(split)
        for r, cost in zip(rows.itertuples(), times, strict=True):
            if split == "test" or r.workload == "scalar":
                result[f"{r.workload}/{r.trip_count}"] = {
                    str(h): FACTORS[np.argmin(cost[:, 0] + (h - 1) * cost[:, 1] / 1000)] for h in CALLS
                }
    return result


def worker(arm, calls, repeat):
    jax.config.update("jax_enable_compilation_cache", False)
    _warm_backend()
    batch = configs()
    random.Random(583209 + repeat + calls).shuffle(batch)
    choices = json.loads((OUT / "oracle.json").read_text())
    start = perf_counter_ns()
    engine = None
    if arm in ("candidate", "incumbent"):
        engine = Recorder(application_engine(model_path=BUNDLED if arm == "candidate" else UNCOMPRESSED))
    setup = (perf_counter_ns() - start) / 1e6 if engine else 0.
    checks, rows = [], []
    for spec, variant, trip in batch:
        fn, args = spec.build(trip, 1, variant)
        args = jax.block_until_ready(jax.device_put(args))
        factor, run = 1, fn
        if arm == "oracle":
            factor = choices[f"{spec.name}/{trip}"][str(calls)]
            run = spec.build(trip, factor, variant).fn
        elif engine:
            run = through_auto_fori(fn, engine, R=calls, B=125., throughput_only=False)
        output, times = measure_calls(run, args, calls)
        reason = None
        if engine:
            factor, reason = engine.decision.factor, engine.decision.fallback_reason
            if engine.decision.cache_hit or reason not in (None, "insufficient_calls", "unsupported_features"):
                raise ValueError(engine.decision)
        rows.append(dict(workload=spec.name, trip=trip, factor=factor, reason=reason, **times))
        checks.append((spec, args, fn, run, output, factor))
    for spec, args, fn, run, output, factor in checks:
        assert_same_graph(run, args, str(jax.make_jaxpr(fn)(*args)), factor)
        check_output(output, jax.jit(fn)(*args), spec)
    return dict(arm=arm, calls=calls, repeat=repeat, setup_ms=setup, rows=rows)


def report(data):
    records = []
    expected = {(s.name, t) for s, _, t in configs()}
    blocks = set(product(ARMS, CALLS, range(REPEATS)))
    for b in data["blocks"]:
        blocks.remove((b["arm"], b["calls"], b["repeat"]))
        if {(r["workload"], r["trip"]) for r in b["rows"]} != expected or len(b["rows"]) != len(expected):
            raise ValueError("incomplete process")
        for r in b["rows"]:
            seconds = (r["direct_ms"] + b["setup_ms"] / len(expected)) / 1000
            records.append(dict(arm=b["arm"], calls=b["calls"], repeat=b["repeat"], workload=r["workload"],
                                trip=r["trip"], factor=r["factor"], seconds=seconds))
    if blocks:
        raise ValueError("missing process")
    frame = pd.DataFrame(records)
    index = ["workload", "trip", "calls"]
    paired = frame.pivot(index=[*index, "repeat"], columns="arm", values="seconds").reset_index()
    paired["partition"] = np.where(paired.workload == "scalar", "source_train", "held_out")
    pooled = paired.groupby(["partition", "calls", "repeat"])[list(ARMS)].sum()
    pooled["wins_vs_native"] = pooled.candidate < pooled.native
    totals = pooled.groupby(["partition", "calls"]).mean()
    totals["wins_vs_native"] = pooled.wins_vs_native.groupby(["partition", "calls"]).sum()
    totals["saved_s"] = totals.native - totals.candidate
    totals.to_csv(OUT / "totals.csv")
    summary = paired.groupby(index)[list(ARMS)].mean()
    factors = frame[frame.arm == "candidate"].groupby(index).factor
    if factors.nunique().max() != 1:
        raise ValueError("inconsistent decisions")
    summary["factor"] = factors.first()
    summary["saved_s"] = summary.native - summary.candidate
    summary.to_csv(OUT / "cases.csv")
    print(totals.round(4))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", type=int)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--calls", choices=CALLS, type=int)
    args = p.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.arm, args.calls, args.worker)))
        return
    oracle_path = OUT / "oracle.json"
    oracle_path.write_text(json.dumps(oracle(), indent=2) + "\n")
    hashes = {p.name: sha256(p.read_bytes()).hexdigest()
              for p in (BUNDLED, UNCOMPRESSED, OUT / "split.json", OUT / "fit.json", oracle_path)}
    if hashes[BUNDLED.name] != json.loads((OUT / "fit.json").read_text())["model"]:
        raise ValueError("fit does not match deployment artifact")
    path = OUT / "live.json"
    data = json.loads(path.read_text()) if path.exists() else dict(host=host(), hashes=hashes, blocks=[])
    if data["hashes"] != hashes or data["host"] != host():
        raise ValueError("incompatible saved timings")
    done = {(b["arm"], b["calls"], b["repeat"]) for b in data["blocks"]}
    for repeat in range(REPEATS):
        order = list(product(ARMS, CALLS))
        random.Random(583211 + repeat).shuffle(order)
        for arm, calls in order:
            if (arm, calls, repeat) in done:
                continue
            cmd = [sys.executable, "-m", __spec__.name, "--worker", str(repeat),
                   "--arm", arm, "--calls", str(calls)]
            print(repeat + 1, arm, calls, file=sys.stderr, flush=True)
            block = subprocess.check_output(cmd, text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
            data["blocks"].append(json.loads(block))
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, separators=(",", ":")) + "\n")
            tmp.replace(path)
    report(data)


if __name__ == "__main__":
    main()
