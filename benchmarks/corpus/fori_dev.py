"""Development-only grouped replay for the static-bound public fori_loop pilot."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from benchmarks.corpus import pilot
from benchmarks.corpus.manifest import DEFAULT_MANIFEST
from benchmarks.corpus.registry import SPECS, build_workload, related_groups
from benchmarks.dataset import aggregate_labels, build_feature_table
from benchmarks.oracle import load_raw
from benchmarks.recovery_data import JOIN_KEY, MODEL, VERSIONS, config_key, dump, require_cpu, sha, write_csv
from benchmarks.recovery_metrics import STRATA, metrics, paired_rows, select_stratum
from jax_unroll.model import CostModel, LogCosts, _design, candidate_features, group_configs
from jax_unroll.schema import FACTORS
from jax_unroll.validation import forecast, prediction_errors

OUT = Path("results/fori_dev")
RAW = pilot.PILOT / "primary_1023_raw.csv"
PROTOCOL = Path("docs/fori_dev_protocol.md")
BASELINES = ("zero", "factor_median", "frozen", "warm_refit")
PRIMARY = "half_frozen"


def eligible_folds():
    """Close family links across the full manifest before filtering by public API."""
    eligible = {spec.name for spec in SPECS if spec.loop_kind == "fori_loop"}
    components = pilot.family_components(SPECS, related_groups())
    return {fold: sorted(eligible.intersection(members)) for fold, members in components.items()
            if eligible.intersection(members)}


def prepare(folds):
    names = {body for members in folds.values() for body in members}
    raw = load_raw(RAW)
    scoped = raw.loc[raw.workload.isin(names)]
    valid, dropped, repeats = pilot.validate_acquisition(scoped)
    excluded = {name: reason for name, reason in dropped.items() if name in names}
    labels = aggregate_labels(valid)
    # One static trace per configuration. Only arithmetic depends on candidate u.
    bases = build_feature_table(labels.loc[labels.factor.eq(1)], workload_builder=build_workload)
    if not (bases.loop_kind.eq("fori_loop") & bases.loop_primitive.eq("scan")).all():
        raise ValueError("expected a public fori_loop with static bounds")
    features = pd.DataFrame([
        candidate_features(base, u) for base in bases.drop(columns="extraction_ms").to_dict("records")
        for u in FACTORS
    ])
    joined = features.merge(labels, on=JOIN_KEY, validate="one_to_one")
    if len(joined) != len(labels) or not joined[["B", "R"]].isna().all().all():
        raise ValueError("incomplete or policy-conditioned join")
    return [dict(row, **VERSIONS) for row in joined.to_dict("records")], repeats, excluded


def grouped_splits(groups, folds):
    bodies = [body for members in folds.values() for body in members]
    if len(set(bodies)) != len(bodies) or set(bodies) != {key[0] for key in groups}:
        raise ValueError("each observed body must belong to exactly one fold")
    for fold, members in folds.items():
        test = sorted(key for key in groups if key[0] in members)
        train = sorted(set(groups) - set(test))
        if not test or not train:
            raise ValueError("empty train or test partition")
        yield fold, train, test


def effect_design(row):
    return _design(row, "memory_work")[3]


def measured_effect(group, u):
    return group[u]["log_warm_us"] - group[1]["log_warm_us"]


def predict(rows, folds, model, *, baselines_only=False):
    groups = {config_key(g[1]): g for g in group_configs(rows)}
    frozen = {config_key(case.features): case for case in forecast(model, rows)}
    forecasts = {}
    splits = {}
    for fold, train, test in grouped_splits(groups, folds):
        splits[fold] = {"train": train, "test": test}
        x = [effect_design(groups[key][u]) for key in train for u in FACTORS]
        y = [measured_effect(groups[key], u) for key in train for u in FACTORS]
        refit = Ridge(alpha=0.025, fit_intercept=False).fit(x, y)
        ridge_fits = {"warm_refit": refit}
        if not baselines_only:
            ridge_fits["ridge_1"] = Ridge(alpha=1.0, fit_intercept=False).fit(x, y)
        medians = {u: np.median([measured_effect(groups[key], u) for key in train]) for u in FACTORS}
        for key in test:
            case = frozen[key]
            base = case.predictions[1].log_warm_us
            predictions = {}
            for u in FACTORS:
                fixed = case.predictions[u]
                design = effect_design(candidate_features(case.features, u))
                effects = {m: float(np.dot(fit.coef_, design)) for m, fit in ridge_fits.items()}
                effects.update(zero=0., factor_median=medians[u], frozen=fixed.log_warm_us - base)
                if not baselines_only:
                    effects[PRIMARY] = 0.5 * effects["frozen"]
                for method, effect in effects.items():
                    predictions.setdefault(method, {})[u] = LogCosts(fixed.log_cold_ms, base + effect)
            for method, costs in predictions.items():
                forecasts.setdefault(method, []).append(replace(case, predictions=costs))
    return forecasts, splits


def warm_rows(forecasts, folds, repeats):
    membership = {body: fold for fold, members in folds.items() for body in members}
    configs = {m: {config_key(c.features): c for c in cases} for m, cases in forecasts.items()}
    rows = []
    for key, case in sorted(configs["frozen"].items()):
        for index, u in enumerate(FACTORS[1:]):
            repeated = repeats[key[0]][:, index]
            row = {"body": key[0], "variant": key[1], "trip_count": key[2], "factor": u,
                   "fold": membership[key[0]], "measured_log_ratio": np.log(case.measured[u][1] / case.measured[1][1]),
                   "repeat_sign_consistent": bool(np.all(np.sign(repeated) == np.sign(repeated[0]))),
                   "repeat_log_range": np.ptp(repeated)}
            for method, by_key in configs.items():
                costs = by_key[key].predictions
                row[method] = costs[u].log_warm_us - costs[1].log_warm_us
            rows.append(row)
    return pd.DataFrame(rows)


def warm_summary(ratios, methods):
    errors = ratios[list(methods)].sub(ratios.measured_log_ratio, axis=0).abs()
    return {"mae": errors.mean().to_dict(),
            "fold_mae": errors.groupby(ratios.fold).mean().T.to_dict()}


def paired_intervals(ratios, methods, baseline):
    errors = ratios[list(methods)].sub(ratios.measured_log_ratio, axis=0).abs()
    body_delta = errors.sub(errors[baseline], axis=0).groupby(ratios.body, sort=True).mean()
    values = body_delta.to_numpy()
    draws = np.random.default_rng(583).integers(len(values), size=(2000, len(values)))
    lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975], axis=0)
    return {m: {"mean": float(mean), "ci95": [float(l), float(h)]}
            for m, mean, l, h in zip(body_delta.columns, values.mean(axis=0), lo, hi, strict=True)}


def save_evaluation(out, ratios, forecasts, splits, excluded, summary):
    pairs, predictions, replay, calibration = [], [], {}, {}
    fold_by_body = ratios.set_index("body").fold.to_dict()
    for method, cases in forecasts.items():
        matched = paired_rows(cases)
        pairs.extend(dict(row, method=method, fold=fold_by_body[row["workload"]]) for row in matched)
        replay[method] = {s: metrics(select_stratum(matched, s)) for s in STRATA}
        calibration[method] = prediction_errors(cases)
        for case in cases:
            body, variant, trip = config_key(case.features)
            for u in FACTORS:
                cold, warm = case.predictions[u].linear()
                predictions.append({"body": body, "variant": variant, "trip_count": trip, "factor": u,
                                    "fold": fold_by_body[body], "method": method, "predicted_cold_ms": cold,
                                    "predicted_warm_us": warm, "measured_cold_ms": case.measured[u][0],
                                    "measured_warm_us": case.measured[u][1]})

    intervals = {b: paired_intervals(ratios, forecasts, b) for b in ("zero", "factor_median", "frozen")}
    mae = summary["mae"]
    wins = sum(fold[PRIMARY] < fold["zero"] for fold in summary["fold_mae"].values())
    useful = replay[PRIMARY]["oracle_improves_long"]
    primary_all = replay[PRIMARY]["all"]["policies"]["selector"]
    practical = None
    if useful["settings"]:
        regret = useful["policies"]["selector"]["mean_positive_regret"]
        practical = (
            regret < min(useful["policies"]["factor_1"]["mean_positive_regret"],
                         replay["frozen"]["oracle_improves_long"]["policies"]["selector"]["mean_positive_regret"])
            and primary_all["budget_violations_when_feasible"] <= min(
                replay["frozen"]["all"]["policies"]["selector"]["budget_violations_when_feasible"],
                replay[PRIMARY]["all"]["policies"]["factor_1"]["budget_violations_when_feasible"])
        )
    report = {
        **summary, "folds": splits, "excluded_bodies": excluded, "baseline_methods": BASELINES,
        "input_sha256": {str(p): sha(p) for p in (RAW, MODEL, PROTOCOL, DEFAULT_MANIFEST.relative_to(Path.cwd()))},
        "environment": {"backend": "cpu", **VERSIONS}, "paired_mae_differences": intervals,
        "repeat_stability": {"stable": int(ratios.repeat_sign_consistent.sum()), "effects": len(ratios),
                             "median_log_range": float(ratios.repeat_log_range.median()),
                             "median_abs_effect": float(ratios.measured_log_ratio.abs().median())},
        "absolute_calibration": calibration, "diagnostic_replay_without_decision_overhead": replay,
        "primary": PRIMARY, "fold_wins_vs_zero": wins,
        "predictive_criterion_met": mae[PRIMARY] < min(mae["zero"], mae["factor_median"])
        and wins > len(splits) / 2 and intervals["zero"][PRIMARY]["ci95"][1] < 0,
        "practical_criterion_met": practical,
    }
    out.mkdir(parents=True, exist_ok=True)
    ratios.to_csv(out / "warm_ratios.csv", index=False)
    write_csv(out / "predictions.csv", predictions, predictions[0])
    write_csv(out / "policy_pairs.csv", pairs, pairs[0])
    dump(out / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()
    require_cpu()
    folds = eligible_folds()
    rows, repeats, excluded = prepare(folds)
    folds = {fold: [b for b in members if b not in excluded] for fold, members in folds.items()}
    forecasts, splits = predict(rows, folds, CostModel.load(MODEL), baselines_only=args.baselines_only)
    ratios = warm_rows(forecasts, folds, repeats)
    summary = warm_summary(ratios, forecasts)
    print(summary)
    if args.baselines_only:
        dump(args.out / "baselines.json", dict(summary, folds=splits, excluded_bodies=excluded))
    else:
        save_evaluation(args.out, ratios, forecasts, splits, excluded, summary)


if __name__ == "__main__":
    main()
