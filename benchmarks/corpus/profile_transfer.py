"""Evaluate a fixed 1023-iteration profile on the fresh smoke-body sweep."""
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from benchmarks.corpus.nonlinear_cost import score
from benchmarks.corpus.registry import build_workload, tier_specs
from benchmarks.dataset import aggregate_labels, build_feature_table
from benchmarks.oracle import load_raw
from benchmarks.recovery_data import JOIN_KEY, MODEL
from benchmarks.recovery_metrics import paired_rows
from jax_unroll.model import CostModel
from jax_unroll.profile_model import ProfileCostModel
from jax_unroll.schema import FACTORS
from jax_unroll.selector import select_factor
from jax_unroll.validation import forecast

OUT = Path("results/profile_transfer")


def main():
    profile = ProfileCostModel(pd.read_csv("results/static_features/pilot_features.csv").to_dict("records"))
    raw = load_raw(sorted(OUT.glob("*_raw.csv")))
    expected = {(s.name, s.primary_variant, t, u, r) for s in tier_specs("smoke")
                for t in (775, 1015) for u in FACTORS for r in range(3)}
    observed = raw[["workload", "variant", "trip_count", "factor", "process_repeat"]]
    if (set(observed.itertuples(index=False, name=None)) != expected or len(raw) != len(expected)
            or not raw.correct.astype(str).str.lower().eq("true").all() or raw.error.notna().any()):
        raise ValueError("incomplete or incorrect validation sweep")
    labels = aggregate_labels(raw)
    features = build_feature_table(labels, workload_builder=build_workload)
    versions = raw[["jax_version", "jaxlib_version"]].drop_duplicates().to_dict("records")
    if len(versions) != 1:
        raise ValueError("validation requires a single compiler environment")
    rows = features.merge(labels, on=JOIN_KEY, validate="one_to_one").assign(**versions[0]).to_dict("records")
    frozen = forecast(CostModel.load(MODEL), rows)
    routed = []
    overhead = {}
    hits = 0
    for case in frozen:
        prediction = profile.predict_candidates(case.features)
        hits += prediction is not None
        if prediction is None:
            raise ValueError(f"missing validation profile for {case.features['workload']}")
        routed.append(replace(case, predictions=prediction, uncertain=False))
        row = case.features
        key = row["workload"], row["variant"], int(row["trip_count"])
        loop = build_workload(key[0], key[2], 1, key[1])
        start = perf_counter()
        actual = profile.choose(loop.fn, loop.args, workload=key[0], variant=key[1],
                                trip_count=key[2], loop_kind=row["loop_kind"], R=100_000)
        overhead[key] = (perf_counter() - start) * 1000
        if actual != select_factor(prediction, R=100_000):
            raise ValueError("live decision disagrees with replay")
    pd.DataFrame([dict(workload=name, variant=variant, trip_count=trip, decision_ms=ms)
                  for (name, variant, trip), ms in overhead.items()]).to_csv(OUT / "overhead.csv", index=False)
    print(f"Profile coverage {hits}/{len(frozen)}")
    summary = []
    for method, cases in (("frozen", frozen), ("profile", routed)):
        for trip in (775, 1015):
            batch = [c for c in cases if c.features["trip_count"] == trip]
            grid = pd.DataFrame(paired_rows(batch))
            long = grid[grid.feasible_long]
            grid.assign(method=method).to_csv(OUT / f"{method}_{trip}_decisions.csv", index=False)
            timed = pd.DataFrame(paired_rows(batch, overhead)) if method == "profile" else None
            timed_long = timed[timed.feasible_long] if timed is not None else None
            relative_errors = [
                np.abs(np.asarray(c.predictions[u].linear()) / c.measured[u] - 1)
                for c in batch for u in FACTORS
            ]
            cold_median_error, warm_median_error = np.median(relative_errors, axis=0)
            summary.append(dict(method=method, trip=trip, **score(batch),
                                cold_median_relative_error=cold_median_error,
                                warm_median_relative_error=warm_median_error,
                                timed_long_regret=(
                                    (timed_long.selector_total_ms / timed_long.measured_oracle_total_ms - 1).clip(lower=0).mean()
                                    if timed_long is not None else None),
                                median_speedup_vs_1=(long.factor_1_total_ms / long.selector_total_ms).median(),
                                within_10pct_oracle=(long.selector_total_ms <= 1.10 * long.measured_oracle_total_ms).mean(),
                                factor1_regret=(long.factor_1_total_ms / long.measured_oracle_total_ms - 1).clip(lower=0).mean()))
    table = pd.DataFrame(summary)
    print(table.to_string(index=False))
    table.to_csv(OUT / "summary.csv", index=False)


if __name__ == "__main__":
    main()
