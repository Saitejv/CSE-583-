"""Cycle 3. Refit compact trees with development curves from inexpensive loops."""

import json
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from benchmarks.corpus.fori_iterations.storage import OUT
from benchmarks.corpus.fori_proposal.score import read_native
from benchmarks.corpus.fori_regret.train import scored, training_data
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.corpus_model import CorpusPredictor
from jax_unroll.fori_model import ForiCostPredictor

REFIT = OUT / "refit.npz"
CV = OUT / "03_forecasts.npz"
TRAINING = OUT / "training_keys.csv"


def data():
    frame, timing, groups = training_data()
    extra, costs = [], []
    for cfg, row, cold, warm in read_native():
        if cfg.cohort == "fresh":
            extra.append({**row, "source": "proposal_dev", "workload": cfg.spec.name,
                          "family": cfg.spec.family, "variant": cfg.variant, "current_body_hash": row["body_hash"]})
            costs.append(np.column_stack((np.median(cold, axis=0), np.median(warm, axis=0))))
    recent = pd.DataFrame(extra)
    # Keep all settings of each previously benchmarked proposal family together.
    extra_groups = pd.factorize(recent.family)[0] + int(groups.max()) + 1
    return pd.concat([frame, recent], ignore_index=True), np.concatenate([timing, costs]), np.r_[groups, extra_groups]


def fit(frame, costs, leaf, weight_mode):
    model = ForiCostPredictor()
    model.fit_model.set_params(min_samples_leaf=leaf)
    weight = None
    if weight_mode != "uniform":
        cases = frame.groupby(["source", "workload"]).workload.transform("size")
        bodies = frame.groupby("source").workload.transform("nunique")
        weight = 1 / (cases * bodies).to_numpy()
        if weight_mode == "opportunity":
            warm = costs[:, :, 1]
            log_gain = np.maximum(0, np.log(warm[:, 0] / warm.min(axis=1)))
            weight *= 1 + np.minimum(2., log_gain / .15)
        weight /= weight.mean()
    return CorpusPredictor.fit(model, frame, costs, sample_weight=weight)


def main():
    frame, timing, groups = data()
    splits = list(GroupKFold(5).split(frame, groups=groups))
    baseline = np.empty_like(timing)
    folds = np.empty(len(frame), dtype=int)
    for i, (train, valid) in enumerate(splits):
        expansion = train[frame.source.iloc[train].eq("expansion")]
        model = ForiCostPredictor().fit(frame.iloc[expansion], timing[expansion])
        baseline[valid] = np.log(model.predict(frame.iloc[valid]))
        folds[valid] = i
    old = scored(frame, timing, baseline)

    candidates = []
    for leaf, weight in product((2, 4), ("uniform", "balanced", "opportunity")):
        forecast = np.empty_like(timing)
        for train, valid in splits:
            model = fit(frame.iloc[train], timing[train], leaf, weight)
            forecast[valid] = np.log(model.predict(frame.iloc[valid]))
        points = scored(frame, timing, forecast)
        objective = (points["weighted_utility"] + .25 * np.log(points["pooled_warm_speedup"])
                     + .03 * (1 - points["weighted_effect_log_mae"] / old["weighted_effect_log_mae"]))
        candidates.append((dict(leaf=leaf, weight=weight, objective=float(objective), **points), forecast))
    candidates.sort(key=lambda candidate: candidate[0]["objective"], reverse=True)
    winner, forecast = candidates[0]
    chosen = fit(frame, timing, winner["leaf"], winner["weight"])
    source = OUT / "refit.joblib"
    joblib.dump(chosen, source, compress=3)
    CompactForiPredictor.export(source, REFIT, compressed=False)
    np.savez_compressed(CV, incumbent=baseline, refit=forecast, fold=folds, timing=timing,
                       source=frame.source.to_numpy(dtype=str), workload=frame.workload.to_numpy(dtype=str))
    frame[["workload", "family", "source", "variant", "trip_count", "body_hash", "current_body_hash"]].assign(
        validation_group=groups, fold=folds,
    ).to_csv(TRAINING, index=False)
    output = json.dumps({
        "configurations": len(frame), "bodies": int(frame.workload.nunique()),
        "family_groups": len(set(groups)), "incumbent": old, "selected": winner,
        "trials": [row for row, _ in candidates],
    }, indent=2) + "\n"
    (OUT / "03_refit.json").write_text(output)
    print(output)


if __name__ == "__main__":
    main()
