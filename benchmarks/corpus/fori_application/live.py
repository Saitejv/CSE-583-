"""Fresh-process, whole-program validation on the predeclared applications and holdout."""

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict
from hashlib import sha256
from itertools import product
from time import perf_counter_ns

import jax
import numpy as np

from benchmarks.corpus.fori_agent_cases.dataset import OUT as AGENT_OUT
from benchmarks.corpus.fori_agent_cases.dataset import settings as agent_settings
from benchmarks.corpus.fori_application.applications import settings
from benchmarks.corpus.fori_application.protocol import ARMS, HORIZONS, OUT, check
from benchmarks.corpus.fori_horizon.cohort import OUT as PRIOR
from benchmarks.corpus.fori_horizon.replay import curves, total
from benchmarks.corpus.fori_iterations.storage import UNCOMPRESSED
from benchmarks.corpus.fori_regret.acquire import REPEATS, assert_same_graph, host
from benchmarks.corpus.fori_regret.live import measure_calls
from benchmarks.corpus.fori_targeted.auto_fori import through_auto_fori
from benchmarks.corpus.fori_targeted.specialization import check_output
from benchmarks.phase4 import TimedEngine
from benchmarks.sweep import _warm_backend
from jax_unroll.compact_fori import CompactForiPredictor, application_engine
from jax_unroll.schema import FACTORS
from jax_unroll.selector import HorizonPolicy


def specs(cohort):
    if cohort == "transfer":
        from benchmarks.corpus.fori_application.transfer import (
            settings as transfer_settings,
        )

        return transfer_settings()
    return settings() if cohort == "applications" else agent_settings("eval")


def key(row):
    return row["workload"], row["variant"], row["trip_count"]


LONG_CALLS = 20_000


def oracle_name(transfer, long_run):
    prefix = "transfer_" if transfer else ""
    suffix = "_long" if long_run else ""
    return f"{prefix}oracle{suffix}.json"


def prepare(transfer=False, long_run=False):
    horizons = (LONG_CALLS,) if long_run else HORIZONS
    result = []
    sources = (
        (("transfer", OUT / "transfer_runs.json"),)
        if transfer
        else (("applications", OUT / "application_runs.json"), ("holdout", AGENT_OUT / "eval_runs.json"))
    )
    for cohort, path in sources:
        if long_run and cohort == "holdout":
            continue
        expected = {(s.name, v, t) for s, v, t in specs(cohort)}
        for r, c, w in curves(path):
            expected.remove(key(r))
            ratios = w[:, :1] / w
            cold, warm = np.median(c, axis=0), np.median(w, axis=0)
            result.append(
                dict(
                    cohort=cohort,
                    workload=r["workload"],
                    variant=r["variant"],
                    trip_count=r["trip_count"],
                    opportunity=bool(np.any((np.median(ratios, axis=0) >= 1.05) & (ratios > 1).all(axis=0))),
                    oracle={h: FACTORS[total(cold, warm, h).argmin()] for h in horizons},
                )
            )
        if expected:
            raise ValueError("native sweep missing cases")
    (OUT / oracle_name(transfer, long_run)).write_text(
        json.dumps(result, indent=2) + "\n"
    )


class Recorder:
    def __init__(self, engine):
        self.engine = engine
        self.payoff = engine.payoff

    def decide(self, *args, **kwargs):
        self.decision = self.engine.decide(*args, **kwargs)
        return self.decision


def worker(cohort, arm, h, repeat):
    jax.config.update("jax_enable_compilation_cache", False)
    _warm_backend()
    oracle = OUT / oracle_name(cohort == "transfer", h == LONG_CALLS)
    choices = {key(r): r["oracle"][str(h)] for r in json.loads(oracle.read_text())}
    engine = None
    start = perf_counter_ns()
    if arm == "candidate":
        engine = Recorder(application_engine())
    elif arm == "warm":
        engine = TimedEngine(CompactForiPredictor.load(UNCOMPRESSED))
    elif arm == "previous":
        policy = json.loads((PRIOR / "policy.json").read_text())["candidate"]
        engine = TimedEngine(
            model_loader=lambda: CompactForiPredictor.load(UNCOMPRESSED), payoff=HorizonPolicy(**policy)
        )
    setup = (perf_counter_ns() - start) / 1e6 if engine else 0.0
    configs = specs(cohort)
    random.Random(5831901 + h + repeat).shuffle(configs)
    records, checks = [], []
    for spec, size, trip in configs:
        native = spec.build(trip, 1, size)
        args = jax.block_until_ready(jax.device_put(native.args))
        factor, fn = 1, native.fn
        if engine:
            fn = through_auto_fori(
                fn, engine, R=h, B=None if arm == "warm" else 125.0, throughput_only=arm == "warm"
            )
        if arm == "oracle":
            factor = choices[spec.name, size, trip]
            fn = spec.build(trip, factor, size).fn
        out, timing = measure_calls(fn, args, h)
        if engine:
            if engine.decision.cache_hit or engine.decision.fallback_reason not in (
                None,
                "insufficient_calls",
            ):
                raise ValueError(engine.decision)
            factor = engine.decision.factor
        records.append(dict(workload=spec.name, variant=size, trip_count=trip, factor=factor, **timing))
        checks.append((spec, native.fn, fn, args, out, factor))
    for spec, native, fn, args, out, factor in checks:
        assert_same_graph(fn, args, str(jax.make_jaxpr(native)(*args)), factor)
        check_output(out, jax.block_until_ready(jax.jit(native)(*args)), spec)
        if spec.name == "class_pixel_peaks":
            np.testing.assert_array_equal(out, np.maximum.reduceat(*map(np.asarray, args), axis=0))
    return dict(cohort=cohort, arm=arm, calls=h, repeat=repeat, setup_ms=setup, rows=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--calls", type=int, choices=(*HORIZONS, LONG_CALLS))
    parser.add_argument("--long", action="store_true", help="Measure the real-data 20,000-call extension")
    parser.add_argument("--cohort", choices=("applications", "holdout", "transfer"))
    opts = parser.parse_args()
    if opts.worker is not None:
        print(json.dumps(worker(opts.cohort, opts.arm, opts.calls, opts.worker)))
        return
    check()
    fit = json.loads((OUT / "fit.json").read_text())
    if asdict(application_engine().payoff) != fit["policy"]:
        raise ValueError("deployment policy differs from development choice")
    transfer = opts.cohort == "transfer"
    if transfer:
        from benchmarks.corpus.fori_application.transfer import check as check_transfer

        check_transfer()
    if opts.long and opts.cohort not in ("applications", "transfer"):
        parser.error("--long requires an explicit real-data cohort")
    prepare(transfer, opts.long)
    horizons = (LONG_CALLS,) if opts.long else HORIZONS
    for cohort in [opts.cohort] if opts.cohort else ("applications", "holdout"):
        path = OUT / (cohort + ("_long" if opts.long else "") + "_live.json")
        inputs = (
            ("transfer_manifest.json", "fit.json", oracle_name(True, opts.long))
            if transfer
            else ("manifest.json", "fit.json", oracle_name(False, opts.long))
        )
        stamp = {p: sha256((OUT / p).read_bytes()).hexdigest() for p in inputs}
        old = json.loads(path.read_text()) if path.exists() else dict(host=host(), freeze=stamp, blocks=[])
        if old["freeze"] != stamp or old["host"] != host():
            raise ValueError("incompatible saved run")
        done = {(b["arm"], b["calls"], b["repeat"]) for b in old["blocks"]}
        for repeat in range(REPEATS):
            order = list(product(ARMS, horizons))
            random.Random(5831902 + repeat).shuffle(order)
            for arm, h in order:
                if (arm, h, repeat) in done:
                    continue
                print(cohort, repeat + 1, arm, h, flush=True, file=sys.stderr)
                cmd = [
                    sys.executable,
                    "-m",
                    __spec__.name,
                    "--worker",
                    str(repeat),
                    "--arm",
                    arm,
                    "--calls",
                    str(h),
                    "--cohort",
                    cohort,
                ]
                old["blocks"].append(
                    json.loads(
                        subprocess.check_output(cmd, text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
                    )
                )
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(old, separators=(",", ":")) + "\n")
                tmp.replace(path)


if __name__ == "__main__":
    main()
