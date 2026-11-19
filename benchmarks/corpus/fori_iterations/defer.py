"""Cycle 5. Measure eager and deferred loading for a known single-call loop."""

import argparse
import json
import random
from statistics import median
from time import perf_counter_ns

import jax

from benchmarks.corpus.fori_iterations.storage import OUT, UNCOMPRESSED
from benchmarks.corpus.fori_proposal.workloads import FRESH
from benchmarks.corpus.fori_regret.acquire import run_workers
from benchmarks.corpus.fori_regret.live import measure_calls
from benchmarks.corpus.fori_targeted.auto_fori import through_auto_fori
from benchmarks.phase4 import TimedEngine
from benchmarks.sweep import _warm_backend
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.selector import SavingsBand, SavingsPolicy


def worker(repeat):
    jax.config.update("jax_enable_compilation_cache", False)
    _warm_backend()
    workload = FRESH[0].build(255, 1, "small")
    args = jax.block_until_ready(jax.device_put(workload.args))
    policy = SavingsPolicy((SavingsBand(None, 0, 0, 0),), 1)
    rng = random.Random(583_805 + repeat)
    samples = {mode: [] for mode in ("eager", "lazy")}
    for _ in range(9):
        order = list(samples)
        rng.shuffle(order)
        for mode in order:
            loads = []

            def load():
                start = perf_counter_ns()
                model = CompactForiPredictor.load(UNCOMPRESSED)
                loads.append((perf_counter_ns() - start) / 1e6)
                return model

            if mode == "eager":
                engine = TimedEngine(load(), payoff=policy)
            else:
                engine = TimedEngine(model_loader=load, payoff=policy)
            _, timing = measure_calls(through_auto_fori(
                workload.fn, engine, throughput_only=False, R=1,
            ), args, 1)
            assert engine.decision.factor == 1 and engine.decision.fallback_reason == "insufficient_calls"
            assert len(loads) == int(mode == "eager")
            samples[mode].append({"load_ms": sum(loads), "direct_ms": timing["direct_ms"] + sum(loads)})
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.worker)))
        return
    runs = run_workers(__spec__.name, [], 3, OUT / "05_defer_runs.json")
    summary = {mode: {name: median(row[name] for run in runs for row in run[mode])
                      for name in ("load_ms", "direct_ms")} for mode in ("eager", "lazy")}
    value = json.dumps(summary, indent=2) + "\n"
    (OUT / "05_defer_summary.json").write_text(value)
    print(value)


if __name__ == "__main__":
    main()
