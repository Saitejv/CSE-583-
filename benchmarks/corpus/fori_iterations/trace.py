"""Cycle 2. Compare a reused scan trace with the original wrapper path."""

import argparse
from functools import partial
from itertools import product
import json
import random
from statistics import median
from unittest.mock import patch

import jax
from jax.extend.core import jaxpr_as_fun

from benchmarks.corpus.fori_iterations.storage import OUT, UNCOMPRESSED
from benchmarks.corpus.fori_proposal.workloads import FRESH
from benchmarks.corpus.fori_regret.acquire import assert_same_graph, run_workers
from benchmarks.corpus.fori_regret.live import measure_calls
from benchmarks.corpus.fori_targeted.auto_fori import through_auto_fori
from benchmarks.corpus.fori_targeted.specialization import check_output
from benchmarks.phase4 import TimedEngine
from benchmarks.sweep import _warm_backend
from jax_unroll import selector
from jax_unroll.auto_loop import auto_fori_loop
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.features import extract_features


def reuse_scan(lower, upper, body, initial, *, engine, throughput_only, R, B):
    def rolled(carry):
        return jax.lax.fori_loop(lower, upper, body, carry)

    closed, shape = jax.make_jaxpr(rolled, return_shape=True)(initial)
    eqns = closed.jaxpr.eqns
    if closed.effects or len(eqns) != 1 or eqns[0].primitive.name != "scan" or eqns[0].params["unroll"] != 1:
        raise ValueError("experiment requires exactly one pure rolled scan")
    replay = partial(extract_features, closed_jaxpr=closed)
    with patch.object(selector, "extract_features", replay):
        factor = engine.decide(
            trip_count=upper - lower, loop_kind="fori_loop", B=B, R=R, throughput_only=throughput_only,
        ).factor
    eqn = eqns[0].replace(params={**eqns[0].params, "unroll": factor})
    chosen = closed.replace(jaxpr=closed.jaxpr.replace(eqns=[eqn]))
    return jax.tree.unflatten(jax.tree.structure(shape), jaxpr_as_fun(chosen)(*jax.tree.leaves(initial)))


def worker(repeat):
    jax.config.update("jax_enable_compilation_cache", False)
    _warm_backend()
    model = CompactForiPredictor.load(UNCOMPRESSED)
    settings = list(product(FRESH, (("small", 1024), ("large", 255))))
    random.Random(583_802 + repeat).shuffle(settings)
    rows = []
    for index, (spec, (size, trip)) in enumerate(settings):
        work = spec.build(trip, 1, size)
        args = jax.block_until_ready(jax.device_put(work.args))
        measured, outputs, wrappers = {}, {}, {}
        for reuse in (bool((repeat + index) % 2), not bool((repeat + index) % 2)):
            engine = TimedEngine(model)
            fn = through_auto_fori(work.fn, engine, wrapper=reuse_scan if reuse else auto_fori_loop)
            output, timing = measure_calls(fn, args, 1)
            if engine.decision.fallback_reason:
                raise ValueError(engine.decision.fallback_reason)
            arm = "reused" if reuse else "original"
            measured[arm] = {**timing, "factor": engine.decision.factor}
            outputs[arm], wrappers[arm] = output, fn
        graph = str(jax.make_jaxpr(work.fn)(*args))
        for arm, fn in wrappers.items():
            assert_same_graph(fn, args, graph, measured[arm]["factor"])
        check_output(outputs["reused"], outputs["original"], spec)
        if measured["original"]["factor"] != measured["reused"]["factor"]:
            raise ValueError("trace reuse changed the selection")
        rows.append({"workload": spec.name, "size": size, "trip_count": trip, **measured})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.worker)))
        return
    runs = run_workers(__spec__.name, [], 3, OUT / "02_trace_runs.json")
    rows = [r for run in runs for r in run]
    summary = {}
    for field in ("trace_lower_ms", "first_ms"):
        summary[field] = {
            arm: median(r[arm][field] for r in rows) for arm in ("original", "reused")
        } | {
            "median_paired_saving_ms": median(r["original"][field] - r["reused"][field] for r in rows),
            "pooled_speedup": sum(r["original"][field] for r in rows) / sum(r["reused"][field] for r in rows),
        }
    text = json.dumps(summary, indent=2) + "\n"
    (OUT / "02_trace_summary.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
