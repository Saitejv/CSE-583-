"""Analyze the frozen, development-only, long-trip corpus pilot."""

import argparse
import subprocess
from collections import defaultdict
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from benchmarks.corpus.manifest import DEFAULT_MANIFEST
from benchmarks.corpus.registry import SPECS, build_workload, related_groups
from benchmarks.dataset import aggregate_labels, build_feature_table
from benchmarks.oracle import load_raw
from benchmarks.recovery_data import JOIN_KEY, MODEL, RESULTS, VERSIONS, dump, require_cpu, sha, write_csv
from benchmarks.recovery_metrics import STRATA, metrics, paired_rows, select_stratum
from jax_unroll.model import CostModel, LogCosts, _design, group_configs
from jax_unroll.schema import FACTORS
from jax_unroll.validation import forecast, prediction_errors

PILOT = RESULTS.parent / "corpus_pilot"
PROTOCOL = Path("docs/corpus_pilot_protocol.md")
TRIP = 1023
REPEATS = 3
COMPARISONS = ("zero", "frozen", "factor_median", "warm_refit")


def family_components(specs, related):
    family_by_body = {spec.name: spec.family for spec in specs}
    components = [{family} for family in sorted(set(family_by_body.values()))]
    for names in related.values():
        families = {family_by_body[name] for name in names}
        merged = [component for component in components if component & families]
        components = [component for component in components if not component & families]
        components.append(families.union(*merged))
    return {
        "+".join(sorted(component)): sorted(name for name, family in family_by_body.items() if family in component)
        for component in sorted(components, key=lambda group: sorted(group))
    }


def validate_acquisition(raw):
    specs = {spec.name: spec for spec in SPECS}
    repeats = set(product(FACTORS, range(REPEATS)))
    groups = dict(tuple(raw.groupby("workload")))
    if set(groups) - set(specs):
        raise ValueError("unexpected workload in the pilot input")
    complete, dropped, repeats_by_body = [], {}, {}
    for name, spec in specs.items():
        body = groups.get(name)
        if body is None:
            dropped[name] = "missing body"
            continue
        static = {"family": spec.family, "variant": spec.primary_variant, "trip_count": TRIP, "loop_kind": spec.loop_kind}
        if any(not body[field].eq(expected).all() for field, expected in static.items()):
            raise ValueError(f"{name}: outside the predeclared measurement matrix")
        observed = set(zip(body["factor"], body["process_repeat"], strict=True))
        runtime = {"backend": "cpu", "device": "cpu:0", **VERSIONS}
        error = body["error"].astype("string").fillna("").ne("").any()
        correct = body["correct"].astype(str).str.lower().eq("true").all()
        costs = body[["cold_total_ms", "warm_median_us"]].to_numpy(dtype=float)
        if len(body) != len(repeats) or observed != repeats:
            dropped[name] = "missing, duplicate, or extra factor repeat"
        elif error or not correct or not (np.isfinite(costs).all() and (costs > 0).all()):
            dropped[name] = "sweep error, correctness failure, or nonpositive/nonfinite label"
        elif any(body[field].nunique() != 1 or body[field].isna().any() for field in ("shape", "dtype")):
            dropped[name] = "inconsistent shape or dtype across factor repeats"
        elif any(not body[column].eq(expected).all() for column, expected in runtime.items()):
            raise ValueError(f"{name}: not measured in pinned CPU environment")
        else:
            complete.append(body)
            warm = body.pivot(index="process_repeat", columns="factor", values="warm_median_us")
            repeats_by_body[name] = np.log(warm[list(FACTORS[1:])].to_numpy() / warm[[1]].to_numpy())
    if not complete:
        raise ValueError("no complete pilot bodies")
    return pd.concat(complete, ignore_index=True), dropped, repeats_by_body


def as_cases(rows, model):
    return {case.features["workload"]: case for case in forecast(model, rows)}


def analysis(rows, repeats_by_body, model):
    cases = as_cases(rows, model)
    by_body = {group[1]["workload"]: group for group in group_configs(rows)}
    components = family_components(SPECS, related_groups())
    detailed, refitted, report_by_fold = [], [], {}
    for fold, members in components.items():
        test = set(members) & cases.keys()
        train = sorted(cases.keys() - test)
        if not test or not train:
            raise ValueError(f"{fold}: lacks complete test or training bodies")

        def ratio(name, factor):
            return float(by_body[name][factor]["log_warm_us"]) - float(by_body[name][1]["log_warm_us"])

        design = [_design(by_body[name][factor], "memory_work")[3] for name in train for factor in FACTORS]
        targets = [ratio(name, factor) for name in train for factor in FACTORS]
        fit = Ridge(alpha=0.025, fit_intercept=False).fit(design, targets)
        medians = {factor: float(np.median([ratio(name, factor) for name in train])) for factor in FACTORS[1:]}
        fold_rows = []
        for name in sorted(test):
            case = cases[name]
            base = case.predictions[1].log_warm_us
            predictions = {1: case.predictions[1]}
            for index, factor in enumerate(FACTORS[1:]):
                fixed = case.predictions[factor]
                effect = float(fit.predict([_design(by_body[name][factor], "memory_work")[3]])[0])
                predictions[factor] = LogCosts(fixed.log_cold_ms, base + effect)
                actual = ratio(name, factor)
                repeats = repeats_by_body[name][:, index]
                fold_rows.append({
                    "body": name, "family": case.features["family"], "fold": fold, "factor": factor,
                    "measured_log_ratio": actual, "repeat_sign_consistent": bool(np.all(np.sign(repeats) == np.sign(repeats[0]))),
                    "repeat_log_range": float(np.ptp(repeats)), "zero": 0.0, "frozen": fixed.log_warm_us - base,
                    "factor_median": medians[factor], "warm_refit": effect,
                })
            refitted.append(replace(case, predictions=predictions))
        report_by_fold[fold] = {
            "train_bodies": train, "test_bodies": sorted(test), "metrics": warm_metrics(fold_rows),
            "paired_mae_delta_vs_zero": paired_mae_differences(fold_rows),
        }
        detailed.extend(fold_rows)
    return detailed, list(cases.values()), refitted, report_by_fold


def warm_metrics(rows):
    actual = np.array([row["measured_log_ratio"] for row in rows])
    return {
        name: {
            "mae": float(np.mean(abs(estimated - actual))),
            "bias": float(np.mean(estimated - actual)),
            "direction_agreement": float(np.mean(np.sign(estimated) == np.sign(actual))),
        }
        for name in COMPARISONS
        for estimated in [np.array([row[name] for row in rows])]
    }


def paired_mae_differences(rows):
    errors = defaultdict(list)
    for row in rows:
        actual = row["measured_log_ratio"]
        errors[row["body"]].append([abs(row[name] - actual) - abs(actual) for name in COMPARISONS])
    values = np.array([np.mean(errors[body], axis=0) for body in sorted(errors)])
    draws = np.random.default_rng(583).integers(len(values), size=(2_000, len(values)))
    lower, upper = np.quantile(np.mean(values[draws], axis=1), (0.025, 0.975), axis=0)
    return {
        name: {"mean": float(mean), "ci95": [float(low), float(high)]}
        for name, mean, low, high in zip(COMPARISONS, np.mean(values, axis=0), lower, upper, strict=True)
    }


def policy_metrics(cases):
    rows = paired_rows(cases)
    return {
        "calibration": prediction_errors(cases),
        "not_supported": sum(case.uncertain for case in cases),
        "strata": {stratum: metrics(select_stratum(rows, stratum)) for stratum in STRATA},
    }


def run(raw_path, out):
    require_cpu()
    receipt = (PILOT / "acquisition.sha256").read_text(encoding="utf-8").splitlines()
    if receipt != [f"{sha(path)}  {path}" for path in (PROTOCOL, DEFAULT_MANIFEST.relative_to(Path.cwd()), MODEL)]:
        raise ValueError("protocol, corpus manifest, or frozen model does not match the pre-sweep hashes")
    commit = (PILOT / "corpus_commit.txt").read_text(encoding="utf-8").strip()
    subprocess.run(["git", "diff", "--exit-code", commit, "--", "benchmarks/corpus/registry.py", "benchmarks/corpus/indirect.py",
                    "benchmarks/corpus/structured.py", "benchmarks/corpus/common.py", "benchmarks/corpus/spec.py",
                    "benchmarks/sweep.py", "src/jax_unroll"], check=True)
    raw = load_raw(raw_path)
    valid, dropped, repeat_effects = validate_acquisition(raw)
    labels = aggregate_labels(valid)
    features = build_feature_table(labels, workload_builder=build_workload).drop(columns="extraction_ms")
    joined = features.merge(labels, on=JOIN_KEY, validate="one_to_one")
    if len(joined) != len(repeat_effects) * len(FACTORS) or not joined[["B", "R"]].isna().all().all():
        raise ValueError("unexpected rows or policy context in the feature/label join")
    rows = [dict(row, **VERSIONS) for row in joined.to_dict("records")]
    model = CostModel.load(MODEL)
    detailed, frozen, refit, folds = analysis(rows, repeat_effects, model)
    pooled = warm_metrics(detailed)
    diff = paired_mae_differences(detailed)
    fold_wins = sum(fold["metrics"]["warm_refit"]["mae"] < fold["metrics"]["zero"]["mae"] for fold in folds.values())
    scale = (
        pooled["warm_refit"]["mae"] < min(pooled["zero"]["mae"], pooled["factor_median"]["mae"])
        and fold_wins > len(folds) / 2 and diff["warm_refit"]["ci95"][1] < 0
    )
    result = {
        "source_commit": commit, "raw_sha256": sha(raw_path), "protocol_sha256": sha(PROTOCOL),
        "backend": "cpu", **VERSIONS,
        "frozen_model": model.version, "raw_rows": len(raw), "valid_bodies": len(frozen), "dropped_bodies": dropped,
        "warm_metrics": pooled, "paired_mae_delta_vs_zero": diff, "folds": folds,
        "folds_refit_beats_zero": fold_wins, "refit_justifies_scaling": scale,
        "repeat_consistency": {
            "stable_factor_effects": sum(row["repeat_sign_consistent"] for row in detailed),
            "factor_effects": len(detailed),
            "median_repeat_log_range": float(np.median([row["repeat_log_range"] for row in detailed])),
            "median_abs_measured_log_effect": float(np.median([abs(row["measured_log_ratio"]) for row in detailed])),
        },
        "policies_without_decision_overhead": {"frozen": policy_metrics(frozen), "warm_refit": policy_metrics(refit)},
    }
    write_csv(out / "warm_ratios.csv", detailed, detailed[0])
    dump(out / "report.json", result)
    print(f"{len(frozen)} bodies in {len(folds)} folds; pooled MAE: " + ", ".join(
        f"{name}={stats['mae']:.4f}" for name, stats in pooled.items()
    ))
    print(f"Refit wins {fold_wins}/{len(folds)} folds. Scale timing for training: {scale}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=PILOT / "primary_1023_raw.csv")
    parser.add_argument("--out", type=Path, default=PILOT)
    args = parser.parse_args()
    run(args.raw, args.out)


if __name__ == "__main__":
    main()
