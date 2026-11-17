"""Measure static decision-path latency on representative CPU development loops."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jaxlib
import numpy as np

from benchmarks.workloads import build_workload
from jax_unroll.features import extract_features_from_workload
from jax_unroll.model import CostModel
from jax_unroll.schema import FEATURE_VERSION
from jax_unroll.selector import DecisionEngine, select_factor
from jax_unroll.splits import assert_holdout_excluded

LOOPS = (
    ("scalar_recurrence", "scalar", 1024),
    ("elementwise_sine", "n1024", 100),
    ("reduction_vector", "n256", 256),
    ("dense_matvec", "n64", 256),
)
MEASUREMENTS = ("extraction_ms", "prediction_policy_ms", "miss_ms", "trace_then_hit_ms", "known_signature_hit_ms")


def measure(model: CostModel, repeats: int) -> dict:
    if jax.default_backend() != "cpu":
        raise ValueError("overhead measurements require the CPU backend")
    if (jax.__version__, jaxlib.__version__) != model.jax_versions:
        raise ValueError("JAX/jaxlib version mismatch with the model")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    def describe(samples: list[float]) -> dict[str, float]:
        return {"median": statistics.median(samples), "p95": float(np.quantile(samples, 0.95))}

    per_loop = {}
    pooled = {stage: [] for stage in MEASUREMENTS}
    for name, variant, trip in LOOPS:
        samples = {stage: [] for stage in MEASUREMENTS}
        for sample in range(repeats + 1):
            workload = build_workload(name, trip, 1, variant)
            extraction_loop = build_workload(name, trip, 1, variant)
            assert_holdout_excluded((workload.family,))
            engine = DecisionEngine(model)

            def record(stage, call):
                start = time.perf_counter_ns()
                result = call()
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                if sample:  # discard one full warmup per representative loop
                    samples[stage].append(elapsed_ms)
                return result

            features = record("extraction_ms", lambda: extract_features_from_workload(extraction_loop))
            record("prediction_policy_ms", lambda: select_factor(model.predict_candidates(features), R=10_000))
            options = {"trip_count": trip, "loop_kind": workload.loop_kind, "R": 10_000}
            decision = record("miss_ms", lambda: engine.decide(workload.fn, workload.args, **options))
            retraced = record("trace_then_hit_ms", lambda: engine.decide(workload.fn, workload.args, **options))
            cached = record("known_signature_hit_ms", lambda: engine.decide(signature=decision.signature, **options))
            if decision.cache_hit or decision.fallback_reason or not retraced.cache_hit or not cached.cache_hit:
                raise RuntimeError("overhead benchmark did not exercise the intended cache paths")
        for stage, timings in samples.items():
            pooled[stage].extend(timings)
        per_loop[f"{name}/{variant}/T={trip}"] = {stage: describe(values) for stage, values in samples.items()}
    return {
        "backend": "cpu", "architecture": platform.machine(), "python_version": platform.python_version(),
        "jax_version": jax.__version__, "jaxlib_version": jaxlib.__version__,
        "model_version": model.version, "feature_version": FEATURE_VERSION, "repeats_per_loop": repeats,
        "per_loop": per_loop, "pooled": {stage: describe(values) for stage, values in pooled.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/cost_cpu.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args(argv)
    result = measure(CostModel.load(args.model), args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for stage, timings in result["pooled"].items():
        print(f"{stage}: {timings['median']:.4f} ms median, {timings['p95']:.4f} ms p95")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
