"""Run the final demo or replay the recorded full-caller CPU evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import jax
import jaxlib
import numpy as np

from benchmarks import phase4
from jax_unroll.model import CostModel

ARCHIVE = phase4.ROOT / "results/phase4"


def demo():
    print(f"JAX {jax.__version__}, backend {jax.default_backend()}", flush=True)
    phase4.demo()

    # Exercise the experimental tracing-time path on an integer carry.
    case = "debounced_counter"
    build, samples = phase4.learned_case(case, np.random.default_rng(583))
    inputs = jax.block_until_ready(jax.device_put(samples))
    engine = phase4.TimedEngine(CostModel.load(phase4.MODEL))
    native = jax.jit(build())
    learned = jax.jit(build(engine=engine, B=100, R=phase4.HORIZON))
    phase4.check_outputs([native, learned], inputs, exact=True)
    print(
        f"Experimental {case}: factor {engine.decision.factor}, "
        f"B=100 ms, R={phase4.HORIZON:,}, exact parity, "
        f"trace-time decision {engine.decision_ms:.2f} ms"
    )


def replay(results: Path):
    freeze = json.loads((results / "freeze.json").read_text())
    rows = json.loads((results / "runs.json").read_text())
    recorded = json.loads((results / "analysis.json").read_text())
    analysis = phase4.analyze(rows)
    # JSON represents the invocation counts used as dictionary keys as strings.
    if json.loads(json.dumps(analysis)) != recorded:
        raise ValueError("Replayed run analysis differs from the recorded analysis")

    print(f"{freeze['hardware']} ({freeze['backend']}), "
          f"JAX/jaxlib {freeze['jax']}/{freeze['jaxlib']}. "
          f"Replayed {len(rows)} process measurements.")
    print()
    print("| Caller | Factor | First native / deployed (ms) | Warm native / deployed (µs) "
          "| Paired warm speedup [min, max] | Warm wins | Break-even calls [min, max] |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for case in phase4.REDUCTIONS:
        pairs = [row for row in analysis["reductions"] if row["case"] == case]
        speedups = [row["warm_speedup"] for row in pairs]
        wins = sum(speedup > 1 for speedup in speedups)
        crossovers = [row["break_even_calls"] for row in pairs if row["break_even_calls"] is not None]
        missing = len(pairs) - len(crossovers)
        crossover = (f"{median(crossovers):,.0f} [{min(crossovers):,}, {max(crossovers):,}]"
                     if crossovers else "—")
        if missing:
            crossover += f" ({missing} without crossover)"
        firsts = [median(row[key] for row in pairs) for key in ("native_first_ms", "deployed_first_ms")]
        warm = [median(row[key] for row in pairs) for key in ("native_us", "deployed_us")]
        print(f"| {case} | {pairs[0]['factor']} | {firsts[0]:.2f} / {firsts[1]:.2f} "
              f"| {warm[0]:.2f} / {warm[1]:.2f} "
              f"| {median(speedups):.3f}× [{min(speedups):.3f}, {max(speedups):.3f}] "
              f"| {wins}/{len(pairs)} | {crossover} |")

    print()
    print(f"Experimental selection at {phase4.HORIZON:,} calls")
    print()
    print("| Budget | Predicted / measured violations | Median vs native | Median vs LLVM "
          "| Oracle agreement | Median regret |")
    print("|---|---:|---:|---:|---:|---:|")
    for budget in phase4.CONTEXTS.values():
        pairs = [row for row in analysis["learned"] if row["budget_ms"] == budget]
        predicted, measured = (sum(row[key] for row in pairs)
                               for key in ("predicted_violation", "measured_violation"))
        violations = f"{predicted}/{len(pairs)} / {measured}/{len(pairs)}" if budget is not None else "N/A"
        feasible = [row for row in pairs if row["oracle_factor"] is not None]
        agreement = f"{sum(row['oracle_agreement'] for row in feasible)}/{len(feasible)}" if feasible else "N/A"
        regret = f"{median(row['oracle_regret'] for row in feasible):.2%}" if feasible else "N/A"
        label = f"{budget} ms" if budget is not None else "Unlimited"
        native = median(row["speedup_vs_native"] for row in pairs)
        llvm = median(row["speedup_vs_llvm"] for row in pairs)
        print(f"| {label} | {violations} | {native:.3f}× | {llvm:.3f}× | {agreement} | {regret} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("demo", help="run live CPU calls on fresh inputs")
    replay_parser = modes.add_parser("replay", help="rebuild the report tables from measurement JSON")
    replay_parser.add_argument("--results", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    if args.mode == "demo":
        if jax.default_backend() != "cpu" or (jax.__version__, jaxlib.__version__) != ("0.12.0.1", "0.12.0.1"):
            parser.error("use JAX_PLATFORMS=cpu with JAX/jaxlib 0.12.0.1")
        demo()
    else:
        replay(args.results)


if __name__ == "__main__":
    main()
