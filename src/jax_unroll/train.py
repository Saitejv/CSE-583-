"""Choose and freeze a cost model using only development-family measurements."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

from jax_unroll.model import CostModel, WARM_TERMS, read_rows
from jax_unroll.validation import B_GRID, LONG_R, R_GRID, cv_metrics, lofo_forecasts

ARTIFACT_NAME = "cost_cpu.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="joined feature/label CSV")
    parser.add_argument("--out", type=Path, required=True, help="directory for the frozen model")
    args = parser.parse_args(argv)

    import numpy as np
    import sklearn

    dev = read_rows(args.data, split="dev")
    trials = []
    for terms in WARM_TERMS:
        for alpha in (0.025, 0.1, 1.0):
            folds = lofo_forecasts(dev, warm_terms=terms, warm_alpha=alpha)
            metrics = cv_metrics(folds)
            worst_long_regret = max(fold["mean_positive_regret_long"] for name, fold in metrics.items() if name != "pooled")
            trials.append({
                "warm_terms": terms, "warm_alpha": alpha, "cv": metrics,
                "worst_long_regret": worst_long_regret,
            })
    best = min(
        trials, key=lambda trial: (
            trial["worst_long_regret"],
            trial["cv"]["pooled"]["mean_positive_regret"],
            trial["cv"]["pooled"]["prediction_errors"]["log_warm_us"]["log_mae"],
        ),
    )
    model = CostModel(warm_terms=best["warm_terms"], warm_alpha=best["warm_alpha"]).fit(dev)
    artifact = {
        "model": model.to_dict(),
        "train_date": datetime.now().astimezone().date().isoformat(),
        "training_versions": {
            "python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        },
        "cv_grid": {"R": R_GRID, "B_ms": B_GRID},
        "selection": {
            "primary": "minimum worst-family mean positive regret at long horizons",
            "long_R_min": LONG_R,
            "tie_break": ["pooled mean positive regret", "pooled warm log MAE"],
        },
        "cv": best["cv"],
        "dev_trials": [
            {
                "warm_terms": trial["warm_terms"], "warm_alpha": trial["warm_alpha"],
                "mean_positive_regret": trial["cv"]["pooled"]["mean_positive_regret"],
                "worst_long_regret": trial["worst_long_regret"],
                "warm_log_mae": trial["cv"]["pooled"]["prediction_errors"]["log_warm_us"]["log_mae"],
            }
            for trial in trials
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ARTIFACT_NAME
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{model.version} | {len(dev)} dev rows | warm terms={model.warm_terms}, alpha={model.warm_alpha}")
    for fold, row in best["cv"].items():
        errors = row["prediction_errors"]
        print(
            f"{fold}: C/T log MAE {errors['log_cold_ms']['log_mae']:.3f}/{errors['log_warm_us']['log_mae']:.3f}, "
            f"agreement {row['factor_agreement']:.1%}, mean positive regret {row['mean_positive_regret']:.2%}"
        )
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
