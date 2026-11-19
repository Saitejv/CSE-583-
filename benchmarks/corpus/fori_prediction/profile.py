"""Evaluate shorter-trip residual profiles for automatic factor selection."""

from itertools import product
import json

import jax
import numpy as np

from benchmarks.corpus.fori_prediction.acquire import OUT, cases
from benchmarks.corpus.fori_prediction.train import (
    REFERENCE, decisions, metrics, paired_intervals, read_generated, reference_logs,
)
from benchmarks.corpus.fori_targeted.auto_fori import through_auto_fori
from jax_unroll.fori_model import ForiCostPredictor
from jax_unroll.profile_model import WarmResidualProfile
from jax_unroll.schema import FACTORS
from jax_unroll.selector import DecisionEngine, select_factor


def pairs(split, reference):
    short, short_cost = read_generated("holdout" if split == "confirm" else split)
    frame, timing = read_generated(split)
    mapping = {767: 511, 768: 512} if split == "confirm" else {511: 127, 512: 256}
    keep = frame.trip_count.isin(mapping)
    frame, timing = frame[keep].reset_index(drop=True), timing[keep]
    lookup = {(r.workload, r.variant, r.trip_count): i for i, r in short.iterrows()}
    indices = [lookup[r.workload, r.variant, mapping[r.trip_count]] for r in frame.itertuples()]
    profile_rows = short.iloc[indices].to_dict("records")
    rows = frame.to_dict("records")
    profiles = short_cost[indices]
    return frame, timing, rows, profile_rows, profiles, reference_logs(reference, rows)


def validation(reference):
    frame, timing, rows, profiles, costs, base = pairs("train", reference)
    static = metrics(frame, timing, base, decisions(base, 0))
    short = reference_logs(reference, profiles)
    target_trip = np.array([r["trip_count"] for r in rows])
    short_trip = np.array([r["trip_count"] for r in profiles])
    ratio = (target_trip / short_trip)[:, None]
    sampled = costs[:, :, 1]
    dispatch = np.minimum(5, sampled.min(axis=1, keepdims=True) * 0.9)
    methods = {
        "proportional": np.log(sampled * ratio),
        "dispatch_floor": np.log(dispatch + (sampled - dispatch) * ratio),
        "residual": np.log(sampled) + base[:, :, 1] - short[:, :, 1],
        "common_model_scaling": np.log(sampled) + base[:, :1, 1] - short[:, :1, 1],
    }
    trials = []
    for method, weight, margin in product(methods, (0.3, 0.5, 0.7, 0.85, 1.0), (0, 0.015, 0.03)):
        pred = base.copy()
        pred[:, :, 1] = (1 - weight) * base[:, :, 1] + weight * methods[method]
        stats = metrics(frame, timing, pred, decisions(pred, margin))
        gain = stats["weighted_utility"] - static["weighted_utility"]
        accuracy = 1 - stats["effect_log_mae"] / static["effect_log_mae"]
        trials.append({
            "method": method, "weight": weight, "margin": margin,
            "joint_improvement": gain > 0 and accuracy > 0, "objective": gain + 0.03 * accuracy, **stats,
        })
    trials.sort(key=lambda row: (row["joint_improvement"], row["objective"]), reverse=True)
    return trials


def score(split, reference, winner):
    frame, timing, rows, profile_rows, costs, base = pairs(split, reference)
    models = [
        WarmResidualProfile(reference, row, times[:, 1], weight=winner["weight"], margin=winner["margin"])
        for row, times in zip(profile_rows, costs, strict=True)
    ]
    predictions, chosen = [], []
    for model, row in zip(models, rows, strict=True):
        if not model.matches(row):
            raise ValueError(f"profile does not match {row['workload']}")
        estimates = model.forecast(row)
        predictions.append([[estimates[f].log_cold_ms, estimates[f].log_warm_us] for f in FACTORS])
        chosen.append(select_factor(model.predict_candidates(row), throughput_only=True))
    forecast = np.asarray(predictions)
    choose = decisions(forecast, winner["margin"])
    np.testing.assert_array_equal(chosen, np.take(FACTORS, choose))

    for index, row in enumerate(rows):
        if row["variant"] != "large" or row["trip_count"] != frame.trip_count.max():
            continue
        spec = cases(split)[row["workload"]]
        workload = spec.build(row["trip_count"], 1, row["variant"])
        engine = DecisionEngine(models[index])
        native = str(jax.make_jaxpr(workload.fn)(*workload.args))
        auto = str(jax.make_jaxpr(through_auto_fori(workload.fn, engine))(*workload.args))
        if auto != native.replace("unroll=1\n", f"unroll={chosen[index]}\n"):
            raise ValueError(f"unexpected live profile decision for {spec.name}")

    static_choice = decisions(base, 0.0)
    return {
        "configurations": len(frame),
        "workloads": int(frame.workload.nunique()),
        "static": metrics(frame, timing, base, static_choice),
        "profiled": metrics(frame, timing, forecast, choose),
        "paired": paired_intervals(frame, timing, base, forecast, static_choice, choose),
    }


def main():
    reference = ForiCostPredictor.load(REFERENCE)
    trials = validation(reference)
    winner = trials[0]
    if winner["method"] != "residual":
        raise ValueError("the frozen residual policy differs from the development winner")
    results = {split: score(split, reference, winner) for split in ("holdout", "confirm")}
    payload = {"selection": winner, "validation": trials, **results}
    (OUT / "profile_evaluation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
