"""Compare regret-weighted static training with the frozen static incumbent."""

from __future__ import annotations

import argparse
from itertools import product
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from benchmarks.corpus import train_fori, train_targeted
from benchmarks.corpus.fori_expansion import registry as expansion
from benchmarks.corpus.fori_prediction.train import (
    KEY, REFERENCE, decisions, extras, metrics, paired_intervals, reference_logs,
)
from benchmarks.corpus.fori_regret.acquire import read
from benchmarks.corpus.fori_regret.development import OUT, archive, rank
from benchmarks.corpus.fori_targeted import acquire as targeted_data, registry as targeted
from jax_unroll.fori_model import ForiCostPredictor
from jax_unroll.fori_v2 import ForiPrediction
from jax_unroll.schema import FACTORS

MODEL = OUT / "model.joblib"
ABLATION = OUT / "archive_only.joblib"
METHODS = ("extra", "shared_huber")
ANCHORS = (0., .5, .75)
REGRET_STRENGTHS = (0., 1., 3.)


def weights(frame, timing, picks, strength):
    """Balance collections and bodies, then upweight expensive mistakes."""
    counts = frame.groupby(["source", "workload"]).workload.transform("size")
    bodies = frame.groupby("source").workload.transform("nunique")
    base = 1 / (counts * bodies).to_numpy()
    ranked = rank(frame, timing, picks).reindex(frame.index)
    costly = ranked["mode"].isin(["harmful_unroll", "missed_speedup"])
    # Saturate at 30 us per call so a few large absolute errors cannot dominate.
    severity = np.sqrt(np.clip(ranked.regret_us / 30, 0, 1))
    weighted = base * (1 + strength * severity.to_numpy() * costly.to_numpy())
    return weighted / weighted.mean()


def scored(frame, timing, logs):
    choice = decisions(logs, 0.)
    result = metrics(frame, timing, logs, choice)
    selected = timing[np.arange(len(choice)), choice, 1]
    ratios = selected / timing[:, 0, 1]
    result["slowdowns"] = int((ratios > 1).sum())
    result["median_slowdown_over_3pct"] = (
        float(np.median(ratios[ratios > 1.03])) if (ratios > 1.03).any() else None
    )
    result["mean_regret_us"] = float(np.mean(selected - timing[:, :, 1].min(axis=1)))
    result["selected_factors"] = {str(f): int((choice == i).sum()) for i, f in enumerate(FACTORS)}
    return result


def training_data():
    history, old_costs = archive()
    parts = []
    for source, builder in (("expansion", expansion.build_workload), ("targeted", targeted.build_workload)):
        archived = history[history.source == source].dropna(axis="columns", how="all")
        parts.append(extras(archived, builder))
    parts.append(history[history.source == "prediction"])
    recent, recent_costs = read("train")
    recent = recent.assign(current_body_hash=recent.body_hash)
    frame = pd.concat((*parts, recent), ignore_index=True)
    timing = np.concatenate((old_costs, recent_costs))

    generated = frame[frame.source.isin(["prediction", "regret"])]
    manifest = generated[["workload", "family"]].drop_duplicates().assign(
        related_group=lambda x: x.workload, partition="train",
    )
    groups = train_targeted.grouped_families(
        frame, pd.read_csv(train_fori.SPLIT),
        pd.concat((pd.read_csv(targeted_data.SPLIT_CSV), manifest)),
    )
    if groups.isna().any():
        raise ValueError("a training configuration is missing its family group")
    return frame, timing, groups


def fit():
    frame, timing, groups = training_data()
    rows = frame.to_dict("records")
    splits = list(GroupKFold(5).split(frame, groups=groups))
    baseline = np.empty_like(timing)
    fold_picks = []
    for train, valid in splits:
        base_train = train[frame.source.iloc[train].eq("expansion")]
        model = ForiCostPredictor().fit(frame.iloc[base_train], timing[base_train])
        model._prepare_trees()
        baseline[valid] = reference_logs(model, [rows[i] for i in valid])
        fold_picks.append(decisions(reference_logs(model, [rows[i] for i in train]), 0))
    base = scored(frame, timing, baseline)

    trials = []
    for method, strength in product(METHODS, REGRET_STRENGTHS):
        oof = np.empty_like(timing)
        for (train, valid), picked in zip(splits, fold_picks, strict=True):
            sample_weight = weights(frame.iloc[train], timing[train], picked, strength)
            model = ForiPrediction(method, extended=False).fit(
                [rows[i] for i in train], timing[train], sample_weight,
            )
            oof[valid] = model.predict_new([rows[i] for i in valid])
        for anchor in ANCHORS:
            logs = (1 - anchor) * oof + anchor * baseline
            result = scored(frame, timing, logs)
            accuracy = 1 - result["weighted_effect_log_mae"] / base["weighted_effect_log_mae"]
            utility = result["weighted_utility"] - base["weighted_utility"]
            trials.append(({
                "method": method, "strength": strength, "anchor": anchor,
                "joint_improvement": bool(utility > 0 and accuracy > 0),
                "objective": float(utility + .03 * accuracy), **result,
            }, logs))
    trials.sort(key=lambda t: (t[0]["joint_improvement"], t[0]["objective"]), reverse=True)
    weighted, oof = next(t for t in trials if t[0]["strength"] > 0)
    unweighted, _ = next(t for t in trials if t[0]["strength"] == 0)
    pd.DataFrame([trial for trial, _ in trials]).to_csv(OUT / "validation.csv", index=False)

    reference = ForiCostPredictor.load(REFERENCE)
    incumbent = decisions(reference_logs(reference, rows), 0)
    for path, keep in ((MODEL, np.ones(len(frame), dtype=bool)), (ABLATION, frame.source.ne("regret"))):
        subset = frame[keep]
        candidate = ForiPrediction(weighted["method"], extended=False).fit(
            subset.to_dict("records"), timing[keep],
            weights(subset, timing[keep], incumbent[keep], weighted["strength"]),
        )
        candidate.anchor_weight = weighted["anchor"]
        candidate.reference = reference if candidate.anchor_weight else None
        joblib.dump(candidate, path, compress=3)

    frame[[*KEY, "source", "family", "body_hash", "current_body_hash"]].assign(
        validation_group=groups,
    ).to_csv(OUT / "training_keys.csv", index=False)
    rank(frame, timing, incumbent).to_csv(OUT / "training_regret.csv", index=False)
    np.savez_compressed(OUT / "oof.npz", incumbent=baseline, refit=oof, costs=timing,
                        source=frame.source.to_numpy(dtype=str))
    summary = {
        "configurations": len(frame), "workloads": frame.workload.nunique(), "groups": groups.nunique(),
        "incumbent_grouped_cv": base, "selected_regret_weighted": weighted, "best_unweighted_cv": unweighted,
    }
    save_summary("fit.json", summary)


def save_summary(path, summary):
    value = json.dumps(summary, indent=2) + "\n"
    (OUT / path).write_text(value)
    print(value, flush=True)


def score():
    frame, timing = read("holdout")
    used = pd.read_csv(OUT / "training_keys.csv")
    if any(set(used[field]) & set(frame[field]) for field in ("workload", "family", "body_hash")) or (
        set(used.current_body_hash) & set(frame.body_hash)
    ):
        raise ValueError("holdout shares a training workload, family or body hash")
    rows = frame.to_dict("records")
    models = {
        "incumbent": ForiCostPredictor.load(REFERENCE),
        "refit": ForiPrediction.load(MODEL),
        "archive_only": ForiPrediction.load(ABLATION),
    }
    comparisons, forecasts = {}, {}
    for name, model in models.items():
        if not all(model.supports(row) for row in rows):
            raise ValueError(f"{name} does not support all held-out workloads")
        logs = reference_logs(model, rows)
        forecasts[name] = logs
        comparisons[name] = scored(frame, timing, logs)
        rank(frame, timing, decisions(logs, 0.)).to_csv(OUT / f"{name}_holdout.csv", index=False)
    old, new = forecasts["incumbent"], forecasts["refit"]
    oracle = metrics(frame, timing, None, timing[:, :, 1].argmin(axis=1))
    save_summary("holdout.json", {
        "configurations": len(frame), "workloads": frame.workload.nunique(),
        "families": frame.family.nunique(), **comparisons, "oracle": oracle,
        "paired_refit_vs_incumbent": paired_intervals(
            frame, timing, old, new, decisions(old, 0.), decisions(new, 0.),
        ),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "score"))
    {"fit": fit, "score": score}[parser.parse_args().stage]()
