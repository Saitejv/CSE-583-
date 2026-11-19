"""Fit using development families, then score the frozen predictor on held-out bodies."""

from __future__ import annotations

import argparse
from itertools import product
import json

import jax
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from benchmarks.corpus import train_fori, train_targeted
from benchmarks.corpus.fori_expansion import registry as expansion
from benchmarks.corpus.fori_prediction.acquire import CONFIRM_TRIPS, OUT, SIZES, TRIPS, cases
from benchmarks.corpus.fori_targeted import acquire as targeted_data, registry as targeted
from jax_unroll.features import extract_features
from jax_unroll.fori_model import ForiCostPredictor
from jax_unroll.fori_v2 import ForiPrediction, loop_features
from jax_unroll.schema import FACTORS
from jax_unroll.selector import DecisionEngine, select_factor

MODEL = OUT / "model.joblib"
ARCHIVE_MODEL = OUT / "archive_only.joblib"
REFERENCE = train_fori.OUT / "model.joblib"
METHODS = ("extra", "forest", "shared_huber", "shared_absolute")
ANCHORS = (0.0, 0.25, 0.5, 0.75, 0.9)
MARGINS = (0.0, 0.01, 0.025, 0.05)
KEY = ["workload", "variant", "trip_count"]


def read_generated(split):
    runs = json.loads((OUT / f"{split}_runs.json").read_text())
    lookup = {(r["workload"], r["variant"], r["trip_count"], r["repeat"]): r for r in runs}
    expected = set(product(cases(split), SIZES, CONFIRM_TRIPS if split == "confirm" else TRIPS, range(3)))
    if len(lookup) != len(runs) or set(lookup) != expected:
        raise ValueError(f"invalid or incomplete generated {split} matrix")
    frame, costs = [], []
    for name, variant, trip in sorted({key[:3] for key in lookup}):
        group = [lookup[name, variant, trip, r] for r in range(3)]
        row = group[0]
        if len({r["features"]["body_hash"] for r in group}) != 1:
            raise ValueError(f"inconsistent loop trace: {name} {variant} {trip}")
        frame.append({
            **row["features"], "workload": name, "family": row["family"], "variant": variant,
            "partition": split, "source": "generated", "current_body_hash": row["features"]["body_hash"],
        })
        costs.append(np.median(np.asarray([
            np.column_stack((r["cold_ms"], r["warm_us"])) for r in group
        ]), axis=0))
    return pd.DataFrame(frame), np.asarray(costs)


def extras(frame, builder):
    extension_rows = []
    for i, row in enumerate(frame.itertuples(index=False)):
        if i % 64 == 0:
            jax.clear_caches()
        loop = builder(row.workload, int(row.trip_count), 1, row.variant)
        traced = extract_features(
            loop.fn, loop.args, loop_kind="fori_loop", trip_count=int(row.trip_count),
            factor=1, feature_block=loop_features,
        )
        # Some older archives use a different Jaxpr string fingerprint.
        # Require agreement on all recorded structural v1 features and keep
        # both hashes so the overlap audit checks both representations.
        expected = row._asdict()
        for name in traced.keys() & expected.keys() - {"body_hash", "extraction_ms", "B", "R"}:
            value = traced[name]
            same = np.isclose(value, expected[name], rtol=1e-10) if isinstance(value, float) else value == expected[name]
            if not same:
                raise ValueError(f"archived feature {name} changed: {row.workload} {row.variant} {row.trip_count}")
        extension_rows.append({
            **{name: value for name, value in traced.items() if name not in frame},
            "current_body_hash": traced["body_hash"],
        })
    return frame.join(pd.DataFrame(extension_rows, index=frame.index))


def reference_logs(model, rows):
    return np.asarray([
        [[cost.log_cold_ms, cost.log_warm_us] for cost in model.predict_candidates(row).values()]
        for row in rows
    ])


def decisions(logs, margin):
    warm = logs[:, :, 1].copy()
    warm[:, 1:] += margin
    return np.argmin(warm, axis=1)


def effects(logs):
    return logs[:, 1:, 1] - logs[:, :1, 1]


def body_means(frame, values):
    return pd.Series(values).groupby([frame.source, frame.workload]).mean()


def metrics(frame, timing, forecasts, choice):
    picked = timing[np.arange(len(choice)), choice, 1]
    baseline = timing[:, 0, 1]
    oracle = timing[:, :, 1].min(axis=1)
    log_gain = np.log(baseline / picked)
    log_actual = np.log(timing)
    result = {
        "workload_warm_speedup": float(np.exp(body_means(frame, log_gain).mean())),
        "weighted_utility": float(body_means(frame, log_gain + np.minimum(log_gain, 0))
                                  .groupby(level=0).mean().mean()),
        "pooled_warm_speedup": float(baseline.sum() / picked.sum()),
        "within_5pct_oracle": float(np.mean(picked <= 1.05 * oracle)),
        "within_10pct_oracle": float(np.mean(picked <= 1.10 * oracle)),
        "fastest_factor_rate": float(np.mean(picked == oracle)),
        "worst_warm_slowdown": float(np.max(picked / baseline)),
        "wins_over_3pct": int(np.count_nonzero(picked < 0.97 * baseline)),
        "losses_over_3pct": int(np.count_nonzero(picked > 1.03 * baseline)),
    }
    if forecasts is not None:
        error = np.abs(effects(forecasts) - effects(log_actual)).mean(axis=1)
        result["effect_log_mae"] = float(body_means(frame, error).mean())
        result["weighted_effect_log_mae"] = float(body_means(frame, error).groupby(level=0).mean().mean())
        result["median_warm_relative_error"] = float(np.median(np.abs(np.expm1(
            forecasts[:, :, 1] - log_actual[:, :, 1]
        ))))
        result["median_cold_relative_error"] = float(np.median(np.abs(np.expm1(
            forecasts[:, :, 0] - log_actual[:, :, 0]
        ))))
    return result


def fit():
    manifest = pd.read_csv(train_fori.SPLIT)
    old, old_cost = train_fori.read(train_fori.DATA / "train_features.csv", {"train"}, manifest)
    target, target_cost = train_targeted.audited_table("train")
    new, new_cost = read_generated("train")
    print(f"Training configurations: {len(old)} expansion, {len(target)} targeted, {len(new)} generated", flush=True)
    old = extras(old, expansion.build_workload).assign(source="expansion")
    target = extras(target, targeted.build_workload).assign(source="targeted")
    frame = pd.concat((old, target, new), ignore_index=True)
    costs = np.concatenate((old_cost, target_cost, new_cost))
    raw_sources = frame.source.to_numpy()
    rows = frame.to_dict("records")
    generated_manifest = new[["workload", "family"]].drop_duplicates().assign(
        related_group=lambda x: x.workload, partition="train",
    )
    target_manifest = pd.concat((pd.read_csv(targeted_data.SPLIT_CSV), generated_manifest))
    groups = train_targeted.grouped_families(frame, manifest, target_manifest)
    splits = list(GroupKFold(5).split(frame, groups=groups))
    baseline_logs = np.empty_like(costs)
    for a, b in splits:
        base_train = a[raw_sources[a] == "expansion"]
        predictor = ForiCostPredictor().fit(frame.iloc[base_train], costs[base_train])
        predictor._prepare_trees()
        baseline_logs[b] = reference_logs(predictor, [rows[i] for i in b])
    baseline = metrics(frame, costs, baseline_logs, decisions(baseline_logs, 0.0))

    def weights(subset):
        data = frame.iloc[subset]
        sizes = data.groupby(["source", "workload"]).workload.transform("size")
        bodies = data.groupby("source").workload.transform("nunique")
        weight = 1 / (sizes * bodies)
        return weight.to_numpy() / weight.mean()

    trials = []
    for method, extended, balanced in product(METHODS, (False, True), (False, True)):
        print(f"CV {method} extended={extended} balanced={balanced}", flush=True)
        oof = np.empty_like(costs)
        for a, b in splits:
            model = ForiPrediction(method, extended).fit(
                [rows[i] for i in a], costs[a], weights(a) if balanced else None,
            )
            oof[b] = model.predict_new([rows[i] for i in b])
        for anchor, margin in product(ANCHORS, MARGINS):
            forecasts = (1 - anchor) * oof + anchor * baseline_logs
            results = metrics(frame, costs, forecasts, decisions(forecasts, margin))
            utility = results["weighted_utility"] - baseline["weighted_utility"]
            accuracy = 1 - results["weighted_effect_log_mae"] / baseline["weighted_effect_log_mae"]
            trials.append({
                "method": method, "extended": extended, "balanced": balanced,
                "anchor": anchor, "margin": margin, "joint_improvement": utility > 0 and accuracy > 0,
                "objective": utility + 0.03 * accuracy, **results,
            })
    validation = pd.DataFrame(trials).sort_values(["joint_improvement", "objective"], ascending=False)
    validation.to_csv(OUT / "validation.csv", index=False)
    winner = validation.iloc[0]
    model = ForiPrediction(winner.method, bool(winner.extended)).fit(
        rows, costs, weights(np.arange(len(frame))) if winner.balanced else None,
    )
    model.reference = ForiCostPredictor.load(REFERENCE) if winner.anchor else None
    model.anchor_weight, model.margin = float(winner.anchor), float(winner.margin)
    joblib.dump(model, MODEL, compress=3)
    frame[[*KEY, "source", "family", "body_hash", "current_body_hash"]].to_csv(OUT / "training_keys.csv", index=False)
    (OUT / "selection.json").write_text(json.dumps({
        "baseline": baseline, "winner": winner.to_dict(), "configs": len(frame),
        "bodies": int(frame.workload.nunique()), "group_count": int(groups.nunique()),
    }, indent=2) + "\n")
    print(json.dumps(winner.to_dict(), indent=2))


def fit_archive_only():
    """Refit the frozen design on historical development data only."""
    winner = json.loads((OUT / "selection.json").read_text())["winner"]
    old, old_cost = train_fori.read(
        train_fori.DATA / "train_features.csv", {"train"}, pd.read_csv(train_fori.SPLIT),
    )
    target, target_cost = train_targeted.audited_table("train")
    old, target = old.assign(source="expansion"), target.assign(source="targeted")
    if winner["extended"]:
        old = extras(old, expansion.build_workload)
        target = extras(target, targeted.build_workload)
    frame = pd.concat((old, target), ignore_index=True)
    weights = None
    if winner["balanced"]:
        sizes = frame.groupby(["source", "workload"]).workload.transform("size")
        bodies = frame.groupby("source").workload.transform("nunique")
        weights = 1 / (sizes * bodies).to_numpy()
        weights /= weights.mean()
    model = ForiPrediction(winner["method"], winner["extended"]).fit(
        frame.to_dict("records"), np.concatenate((old_cost, target_cost)), weights,
    )
    model.anchor_weight, model.margin = winner["anchor"], winner["margin"]
    model.reference = ForiCostPredictor.load(REFERENCE) if model.anchor_weight else None
    joblib.dump(model, ARCHIVE_MODEL, compress=3)


def paired_intervals(frame, timing, old, new, old_choice, new_choice):
    warm = timing[:, :, 1]
    old_result = warm[np.arange(len(warm)), old_choice]
    new_result = warm[np.arange(len(warm)), new_choice]
    accuracy = (
        np.abs(effects(old) - effects(np.log(timing))) - np.abs(effects(new) - effects(np.log(timing)))
    ).mean(axis=1)
    pairs = np.column_stack((
        body_means(frame, np.log(old_result / new_result)).to_numpy(),
        body_means(frame, accuracy).to_numpy(),
    ))
    sample = np.random.default_rng(583).integers(0, len(pairs), (10_000, len(pairs)))
    draws = pairs[sample].mean(axis=1)
    return {
        "candidate_warm_speedup_vs_incumbent": float(np.exp(pairs[:, 0].mean())),
        "speedup_ci95": np.exp(np.quantile(draws[:, 0], [0.025, 0.975])).tolist(),
        "effect_log_mae_reduction": float(pairs[:, 1].mean()),
        "effect_log_mae_reduction_ci95": np.quantile(draws[:, 1], [0.025, 0.975]).tolist(),
    }


def score():
    model = ForiPrediction.load(MODEL)
    reference = ForiCostPredictor.load(REFERENCE)
    archive_only = ForiPrediction.load(ARCHIVE_MODEL) if ARCHIVE_MODEL.exists() else None
    training = pd.read_csv(OUT / "training_keys.csv")
    result, detail = {}, []
    new, new_cost = read_generated("holdout")
    targeted_eval, targeted_cost = train_targeted.audited_table("sealed_eval")
    archive = extras(targeted_eval, targeted.build_workload).assign(source="targeted")
    for name, frame, timing in (("generated", new, new_cost), ("archived_targeted", archive, targeted_cost)):
        test_hashes = set(frame.body_hash) | set(frame.current_body_hash)
        train_hashes = set(training.body_hash) | set(training.current_body_hash)
        if test_hashes & train_hashes or set(frame.workload) & set(training.workload):
            raise ValueError(f"training and evaluation overlap: {name}")
        rows = frame.to_dict("records")
        if not all(model.supports(row) for row in rows):
            raise ValueError(f"unsupported evaluation features: {name}")
        old, new_logs = reference_logs(reference, rows), model.predict_log(rows)
        old_choice, new_choice = decisions(old, 0.0), decisions(new_logs, model.margin)
        live_choices = [select_factor(model.predict_candidates(row), throughput_only=True) for row in rows]
        np.testing.assert_array_equal(live_choices, np.take(FACTORS, new_choice))
        if name == "generated":
            engine = DecisionEngine(model)
            for i, row in frame.iterrows():
                if row.variant != "large" or row.trip_count != max(TRIPS):
                    continue
                fn = cases("holdout")[row.workload].build(int(row.trip_count), 1, row.variant)
                decision = engine.decide(
                    fn.fn, fn.args, trip_count=int(row.trip_count), loop_kind="fori_loop", throughput_only=True,
                )
                if decision.fallback_reason or decision.factor != live_choices[i]:
                    raise ValueError(f"runtime decision mismatch: {row.workload}")
        result[name] = {"configurations": len(frame), "workloads": frame.workload.nunique()}
        comparisons = [
            ("incumbent", old, old_choice), ("candidate", new_logs, new_choice),
            ("factor1", None, np.zeros(len(frame), dtype=int)),
            ("oracle", None, timing[:, :, 1].argmin(axis=1)),
        ]
        if archive_only is not None:
            archive_logs = archive_only.predict_log(rows)
            comparisons.append(("archive_only", archive_logs, decisions(archive_logs, archive_only.margin)))
        for label, logs, choice in comparisons:
            result[name][label] = metrics(frame, timing, logs, choice)
            selected = frame[[*KEY, "family"]].copy()
            selected["factor"] = np.take(FACTORS, choice)
            selected["warm_us"] = timing[np.arange(len(frame)), choice, 1]
            detail.append(selected.assign(cohort=name, policy=label))
        result[name]["paired"] = paired_intervals(frame, timing, old, new_logs, old_choice, new_choice)
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.concat(detail, ignore_index=True).to_csv(OUT / "decisions.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "ablate", "score"))
    args = parser.parse_args()
    {"fit": fit, "ablate": fit_archive_only, "score": score}[args.stage]()
