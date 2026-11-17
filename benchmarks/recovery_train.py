"""Select the small recovery model from development labels only."""

import argparse
import platform
from pathlib import Path

import numpy as np
import sklearn

from benchmarks.recovery_data import (
    DATA, LEGACY_MODEL, MODEL, PROTOCOL, RESULTS, dump, joined_rows, prepare, sha, write_csv,
)
from benchmarks.recovery_metrics import describe, paired_rows
from jax_unroll.model import CostModel
from jax_unroll.recovery_split import DEV, NEW_DEV, check_rows
from jax_unroll.validation import B_GRID, R_GRID, forecast

TRIALS = (("memory_work", 0.025), ("indexed", 0.025), ("indexed", 0.1),
          ("indexed_traffic", 0.025), ("indexed_traffic", 0.1))
SELECTION = "Plateau at or below refit warm relative MAE on new dev; minimum worst-new-family mean positive regret on oracle-improving feasible long settings; tie-break by warm relative MAE, smaller design, stronger shrinkage."


def folds(rows):
    check_rows(rows, "dev")
    families = sorted({row["family"] for row in rows})
    if set(families) != set(DEV.values()):
        raise ValueError("recovery LOFO requires rows from all seven development families")
    return {
        family: ([row for row in rows if row["family"] != family], [row for row in rows if row["family"] == family])
        for family in families
    }


def report(forecasts):
    cases = [case for group in forecasts.values() for case in group]
    new = [case for case in cases if case.features["workload"] in NEW_DEV]
    legacy = [case for case in cases if case.features["workload"] not in NEW_DEV]
    all_groups = {**forecasts, "new_dev": new, "legacy_dev": legacy, "pooled": cases}
    return {group: describe(group_cases, paired_rows(group_cases)) for group, group_cases in all_groups.items()}


def choose(trials):
    limit = trials[0]["warm_relative_log_mae_new_dev"]
    short = [trial for trial in trials if trial["warm_relative_log_mae_new_dev"] <= limit]
    return min(short, key=lambda trial: (
        trial["worst_new_family_regret"] if trial["worst_new_family_regret"] is not None else 0.0,
        trial["warm_relative_log_mae_new_dev"], trial["warm_weights"], -trial["warm_alpha"],
    ))


def train(joined, artifact_path, out):
    rows, metadata = joined_rows(joined, "dev")
    dev_folds = folds(rows)
    legacy = CostModel.load(LEGACY_MODEL)
    frozen_cases = {family: forecast(legacy, test) for family, (_, test) in dev_folds.items()}
    trials = []
    cases_by_trial = {}
    for terms, alpha in TRIALS:
        fold_models = {
            family: CostModel(warm_terms=terms, warm_alpha=alpha).fit_recovery(training)
            for family, (training, _) in dev_folds.items()
        }
        cases = {family: forecast(fold_models[family], test) for family, (_, test) in dev_folds.items()}
        cases_by_trial[(terms, alpha)] = cases
        diagnostics = report(cases)
        useful = [diagnostics[family]["strata"]["oracle_improves_long"] for family in sorted(set(NEW_DEV.values()))]
        regressions = [group["policies"]["selector"]["mean_positive_regret"] for group in useful if group["settings"]]
        trials.append({
            "warm_terms": terms, "warm_alpha": alpha,
            "warm_weights": len(next(iter(fold_models.values())).to_dict()["regressions"][3]["weights"]),
            "warm_relative_log_mae_new_dev": diagnostics["new_dev"]["calibration"]["warm_relative"]["mae_log"],
            "worst_new_family_regret": max(regressions, default=None), "diagnostics": diagnostics,
        })
    selected = choose(trials)
    terms, alpha = selected["warm_terms"], selected["warm_alpha"]
    model = CostModel(warm_terms=terms, warm_alpha=alpha).fit_recovery(rows)
    comparisons = {
        "frozen_week3": frozen_cases, "week3_refit": cases_by_trial[TRIALS[0]], "selected": cases_by_trial[(terms, alpha)],
    }
    detail = [
        {"comparison": name, **row} for name, split in comparisons.items()
        for row in paired_rows([case for cases in split.values() for case in cases])
    ]
    csv_path = out / "dev_pairs.csv"
    report_path = out / "dev_report.json"
    write_csv(csv_path, detail, detail[0])
    dump(report_path, {
        "grid": {"R": R_GRID, "B_ms": B_GRID}, "selected": [terms, alpha], "selection": SELECTION,
        "trial_reports": trials, "frozen_week3": report(frozen_cases),
        "frozen_week3_note": "Historical legacy dev is in-sample; all new dev is unseen by the frozen model.",
        "fold_workloads": {
            family: {"train": sorted({row["workload"] for row in training}), "test": sorted({row["workload"] for row in test})}
            for family, (training, test) in dev_folds.items()
        },
    })
    dump(artifact_path, {
        "model": model.to_dict(), "protocol_sha256": sha(PROTOCOL), "dev_provenance": metadata,
        "comparisons": {"frozen_week3_sha256": sha(LEGACY_MODEL), "dev_report_sha256": sha(report_path), "dev_pairs_sha256": sha(csv_path)},
        "training_versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "selection": SELECTION,
    })
    for trial in trials:
        print(f"{trial['warm_terms']} alpha={trial['warm_alpha']}: new-dev warm log-ratio MAE {trial['warm_relative_log_mae_new_dev']:.4f}; worst-family useful regret {trial['worst_new_family_regret']}")
    print(f"Selected {model.version}: {terms}, alpha={alpha}; SHA-256 {sha(artifact_path)}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DATA / "dev_raw.csv")
    parser.add_argument("--joined", type=Path, default=DATA / "dev.csv")
    parser.add_argument("--model-out", type=Path, default=MODEL)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    prepare(args.raw, "dev", args.joined)
    train(args.joined, args.model_out, args.results)


if __name__ == "__main__":
    main()
