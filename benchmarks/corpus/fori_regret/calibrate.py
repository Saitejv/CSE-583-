"""Calibrate the payoff policy from development forecasts and wrapper overhead."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import jax
import numpy as np

from benchmarks.corpus.fori_regret.acquire import OUT, REPEATS, run_workers
from benchmarks.corpus.fori_regret.workloads import cases
from benchmarks.corpus.fori_targeted.auto_fori import through_auto_fori
from benchmarks.phase4 import TimedEngine
from benchmarks.sweep import _warm_backend
from jax_unroll.selector import PayoffPolicy

MODELS = {
    "incumbent": Path("results/fori_predictor/model.joblib"),
    "refit": OUT / "model.joblib",
}
CALIBRATION = OUT / "payoff.json"
QUANTILE = .80
SETTINGS = (("small", 256), ("large", 1023))
SHARED_SPECIALIZATIONS = 6


def load_model(name):
    # Import inside the load interval so each fresh process pays its own startup.
    if name == "incumbent":
        from jax_unroll.fori_model import ForiCostPredictor
        return ForiCostPredictor.load(MODELS[name])

    from jax_unroll.fori_v2 import ForiPrediction
    return ForiPrediction.load(MODELS[name])


def worker(name, *, loader=load_model):
    jax.config.update("jax_enable_compilation_cache", False)
    _warm_backend()
    start = perf_counter_ns()
    model = loader(name)
    engine = TimedEngine(model)
    load_ms = (perf_counter_ns() - start) / 1e6
    rows = []
    for spec in cases("train").values():
        for size, trip in SETTINGS:
            workload = spec.build(trip, 1, size)
            args = jax.block_until_ready(jax.device_put(workload.args))
            jax.clear_caches()
            fn = through_auto_fori(workload.fn, engine)
            jax.make_jaxpr(fn)(*args)
            if engine.decision.fallback_reason or engine.decision.cache_hit:
                raise ValueError(f"expected a fresh decision on {spec.name}: {engine.decision}")
            rows.append({"workload": spec.name, "size": size, "trip_count": trip,
                         "decision_ms": engine.decision_ms})
    return {"version": model.version, "load_ms": load_ms, "rows": rows}


def margins(logs, costs):
    """One-sided residual quantiles when the out-of-fold forecast offers a warm win."""
    measured = np.log(costs)
    predicted_effect = logs[:, 1:, 1] - logs[:, :1, 1]
    true_effect = measured[:, 1:, 1] - measured[:, :1, 1]
    forecast_win = predicted_effect < 0
    if not forecast_win.any():
        raise ValueError("no predicted warm wins to calibrate")
    cold_error = costs[:, :, 0] - np.exp(logs[:, :, 0])

    def upper(values):
        return max(0., float(np.quantile(values, QUANTILE, method="higher")))

    return {
        "warm_effect_margin": upper((true_effect - predicted_effect)[forecast_win]),
        "baseline_warm_margin": upper((logs[:, 0, 1] - measured[:, 0, 1])[forecast_win.any(axis=1)]),
        "extra_cold_ms": upper((cold_error[:, 1:] - cold_error[:, :1])[forecast_win]),
        "candidate_cold_ms": upper(cold_error[:, 1:][forecast_win]),
    }, int(np.count_nonzero(forecast_win))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, help=argparse.SUPPRESS)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.model)))
        return

    with np.load(OUT / "oof.npz") as oof:
        comparable = np.isin(oof["source"], ("prediction", "regret"))
        result = {"quantile": QUANTILE, "configurations": int(comparable.sum()),
                  "model_load_shared_across": SHARED_SPECIALIZATIONS, "models": {}}
        for name in MODELS:
            runs = run_workers(__spec__.name, ["--model", name], REPEATS, OUT / f"{name}_overhead.json")
            values, n = margins(oof[name][comparable], oof["costs"][comparable])
            load_ms = float(np.median([run["load_ms"] for run in runs]))
            decision_ms = float(np.quantile(
                [row["decision_ms"] for run in runs for row in run["rows"]], QUANTILE, method="higher",
            ))
            values["decision_ms"] = decision_ms
            values["model_load_ms"] = load_ms / SHARED_SPECIALIZATIONS
            PayoffPolicy(**values)
            result["models"][name] = {
                "version": runs[0]["version"], "eligible_forecasts": n, "policy": values,
                "median_load_ms": load_ms, "decision_ms_at_quantile": decision_ms,
            }
    data = json.dumps(result, indent=2) + "\n"
    CALIBRATION.write_text(data)
    print(data, flush=True)


if __name__ == "__main__":
    main()
