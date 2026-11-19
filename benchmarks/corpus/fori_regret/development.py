"""Inspect the incumbent's most expensive decisions on development data."""

from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.corpus import train_fori, train_targeted
from benchmarks.corpus.fori_prediction.train import REFERENCE, decisions, read_generated, reference_logs
from jax_unroll.fori_model import ForiCostPredictor
from jax_unroll.schema import FACTORS

OUT = Path("results/fori_regret")


def archive():
    """Read only partitions previously assigned to model development."""
    expansion = train_fori.read(
        train_fori.DATA / "train_features.csv", {"train"}, pd.read_csv(train_fori.SPLIT),
    )
    targeted = train_targeted.audited_table("train")
    prediction = read_generated("train")
    sources = ("expansion", "targeted", "prediction")
    frame = pd.concat([
        table.assign(source=source)
        for source, (table, _) in zip(sources, (expansion, targeted, prediction), strict=True)
    ], ignore_index=True)
    return frame, np.concatenate([cost for _, cost in (expansion, targeted, prediction)])


def rank(frame, costs, choices):
    warm = costs[:, :, 1]
    oracle = warm.argmin(axis=1)
    selected = warm[np.arange(len(warm)), choices]
    best = warm[np.arange(len(warm)), oracle]
    regret = selected - best
    mode = np.select(
        [(choices != 0) & (selected > warm[:, 0]), (choices == 0) & (regret > 0)],
        ["harmful_unroll", "missed_speedup"],
        default="other",
    )
    cols = ["source", "family", "workload", "variant", "trip_count", "body_hash"]
    return frame[cols].assign(
        selected_factor=np.take(FACTORS, choices), best_factor=np.take(FACTORS, oracle),
        selected_us=selected, baseline_us=warm[:, 0], best_us=best,
        regret_us=regret, selected_over_best=selected / best, mode=mode,
    ).sort_values("regret_us", ascending=False, kind="stable")


def main():
    frame, timing = archive()
    model = ForiCostPredictor.load(REFERENCE)
    picked = decisions(reference_logs(model, frame.to_dict("records")), 0)
    ranked = rank(frame, timing, picked)
    OUT.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(OUT / "development_regret.csv", index=False)
    for mode in ("harmful_unroll", "missed_speedup"):
        rows = ranked[ranked["mode"] == mode].drop_duplicates(["source", "workload"]).head(8)
        print(rows.drop(columns="body_hash").to_string(index=False))


if __name__ == "__main__":
    main()
