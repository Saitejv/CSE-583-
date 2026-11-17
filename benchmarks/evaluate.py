"""Compare frozen CPU unroll policies against offline sweep measurements."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from datetime import date
from pathlib import Path

import numpy as np

from jax_unroll.model import CostModel, dev_rows_digest, read_rows
from jax_unroll.schema import DATA_DIR, FACTORS, PROCESSED_RESULTS_DIR, ensure_parent, utc_timestamp
from jax_unroll.selector import total_ms
from jax_unroll.splits import HELD_OUT_FAMILIES
from jax_unroll.validation import (
    B_GRID, POLICIES, R_GRID, Forecast, choices, cv_metrics, forecast, lofo_forecasts, prediction_errors,
)

GATE_LOG = PROCESSED_RESULTS_DIR / "selector_cpu_holdout_gates.jsonl"
DEFAULT_BOOTSTRAPS = 2_000
BOOTSTRAP_SEED = 583


def _budget(B: float | None) -> str:
    return "unlimited" if B is None else f"{B:g}"


def _bootstrap_summary(values: np.ndarray, draws: np.ndarray) -> dict[str, float]:
    medians = ("total_ms", "regret", "speedup_vs_factor_1", "speedup_vs_llvm")
    averages = ("budget_violation_rate", "factor_agreement")
    sampled = values[draws]
    result = {}
    for index, field in enumerate((*medians, *averages)):
        reducer = np.median if field in medians else np.mean
        low, high = np.quantile(reducer(sampled[..., index], axis=1), (0.025, 0.975))
        result[field] = round(float(reducer(values[:, index])), 8)
        result[f"{field}_ci_low"] = round(float(low), 8)
        result[f"{field}_ci_high"] = round(float(high), 8)
    return result


def evaluate(
    groups: dict[str, list[Forecast]], *, miss_ms: float, bootstraps: int,
) -> tuple[list[dict], list[dict]]:
    """Aggregate per-setting results and collect paired holdout gate wins."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    table, gate_wins = [], []
    for group, cases in groups.items():
        draws = rng.integers(len(cases), size=(bootstraps, len(cases)))
        for B in B_GRID:
            for R in R_GRID:
                records = {policy: [] for policy in POLICIES}
                counts = {policy: dict.fromkeys(FACTORS, 0) for policy in POLICIES}
                no_feasible = 0
                for case in cases:
                    selected, feasible = choices(case, R, B)
                    no_feasible += not feasible
                    totals = {
                        policy: total_ms(*case.measured[u], R) + (miss_ms if policy == "selector" else 0)
                        for policy, u in selected.items()
                    }
                    oracle = selected["measured_oracle"]
                    for policy, u in selected.items():
                        elapsed = totals[policy]
                        violates = B is not None and case.measured[u][0] > B
                        counts[policy][u] += 1
                        records[policy].append((
                            elapsed, elapsed / totals["measured_oracle"] - 1,
                            totals["factor_1"] / elapsed, totals["llvm_size"] / elapsed,
                            violates, u == oracle,
                        ))
                    if group == "holdout" and _gate_win(case, totals, selected, B):
                        gate_wins.append({
                            "case": case, "R": R, "B": B, "factors": selected, "totals": totals,
                        })
                for policy in POLICIES:
                    table.append({
                        "group": group, "R": R, "B_ms": _budget(B), "policy": policy,
                        "configurations": len(cases), "no_measured_feasible": no_feasible,
                        **{f"choices_u{u}": count for u, count in counts[policy].items()},
                        **_bootstrap_summary(np.asarray(records[policy]), draws),
                    })
    return table, gate_wins


def _gate_win(case: Forecast, totals: dict[str, float], factors: dict[str, int], B: float | None) -> bool:
    compared = ("selector", "factor_1", "llvm_size")
    return (
        totals["selector"] < min(totals["factor_1"], totals["llvm_size"])
        and (B is None or all(case.measured[factors[policy]][0] <= B for policy in compared))
    )


def _record_gate(handle, record: dict) -> None:
    handle.seek(0, os.SEEK_END)
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _setting(row: dict) -> str:
    factors = ", ".join(f"{u}×{row[f'choices_u{u}']}" for u in FACTORS if row[f"choices_u{u}"])
    return f"| {row['R']:,} | {row['B_ms']} | {row['policy']} | {row['total_ms']:.3f} | {row['regret']:.2%} [{row['regret_ci_low']:.2%}, {row['regret_ci_high']:.2%}] | {row['budget_violation_rate']:.1%} | {row['factor_agreement']:.1%} | {factors} |"


def _markdown(
    model: CostModel, table: list[dict], cv: dict, *, miss_ms: float, bootstraps: int,
    holdout: list[Forecast], gate_wins: list[dict], gate_count: int,
) -> str:
    jax_version, jaxlib_version = model.jax_versions
    lines = [
        "# CPU unroll selector evaluation", "",
        f"Frozen model `{model.version}`. CPU sweep labels use JAX {jax_version} and jaxlib {jaxlib_version}. "
        f"Selector total includes a {miss_ms:.4f} ms decision charge per configuration.",
        "",
        "Totals use measured cold ms + R × measured warm µs / 1,000. The oracle chooses the fastest measured feasible factor "
        "and uses factor 1 when every measured factor is infeasible. Violation uses measured cold time. "
        "A budget-violating choice can have negative signed regret. The CSV includes paired speedups and 95% percentile "
        f"intervals from {bootstraps:,} bootstrap draws of entire configurations within each group (seed {BOOTSTRAP_SEED}).",
        "", "## Dev leave-one-family-out", "",
        "Factor agreement, budget violations, and mean positive regret pool the R/B grid before overhead. "
        "Relative MAE uses measured cost as denominator.",
        "",
        "| Fold | Configs | C log MAE | T log MAE | C rel. MAE | T rel. MAE | Agreement | Violations | Mean positive regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold, metrics in cv.items():
        c = metrics["prediction_errors"]["log_cold_ms"]
        t = metrics["prediction_errors"]["log_warm_us"]
        lines.append(
            f"| {fold.removeprefix('lofo_')} | {metrics['configurations']} | {c['log_mae']:.3f} | {t['log_mae']:.3f} | "
            f"{c['mean_absolute_relative_error']:.1%} | {t['mean_absolute_relative_error']:.1%} | "
            f"{metrics['factor_agreement']:.1%} | {metrics['budget_violation_rate']:.1%} | {metrics['mean_positive_regret']:.2%} |"
        )
    view = "holdout" if holdout else "dev_pooled"
    lines.extend([
        "", f"## {'Held-out' if holdout else 'Dev pooled'} policy comparison", "",
        "Unlimited budget rows show all seven policies. The CSV includes every budget and family.",
        "",
        "| R | B (ms) | Policy | Median total (ms) | Median regret [95% CI] | Violations | Agreement | Factor × count |",
        "|---:|---:|---|---:|---:|---:|---:|---|",
    ])
    lines.extend(_setting(row) for row in table if row["group"] == view and row["B_ms"] == "unlimited")
    if holdout:
        lines.extend(_holdout_markdown(holdout, gate_wins, gate_count))
    return "\n".join(lines) + "\n"


def _holdout_markdown(cases: list[Forecast], wins: list[dict], count: int) -> list[str]:
    lines = [
        "", "## Frozen Week 3 gate", "",
        f"**{'PASS' if wins else 'FAIL'}** on evaluation #{count}. "
        f"{len(cases)} configurations scored once. A win requires strictly lower measured total including selector overhead "
        "against both factor 1 and LLVM size on the same configuration, with all three choices within any finite budget.",
        "",
        "| R | B (ms) | Indexed memory wins | Iterative solver wins |",
        "|---:|---:|---:|---:|",
    ]
    for R in R_GRID:
        for B in B_GRID:
            counts = [
                sum(win["R"] == R and win["B"] == B and win["case"].features["family"] == family for win in wins)
                for family in HELD_OUT_FAMILIES
            ]
            lines.append(f"| {R:,} | {_budget(B)} | {counts[0]} | {counts[1]} |")
    if wins:
        lines.extend([
            "", "| Winning configuration | R | B (ms) | Chosen u | Selector ms | Factor 1 ms | LLVM size ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for win in wins:
            row = win["case"].features
            totals = win["totals"]
            lines.append(
                f"| {row['workload']}/{row['variant']}/T={row['trip_count']} | {win['R']:,} | {_budget(win['B'])} | "
                f"{win['factors']['selector']} | {totals['selector']:.3f} | {totals['factor_1']:.3f} | {totals['llvm_size']:.3f} |"
            )
    lines.extend([
        "", "## Held-out prediction diagnostics", "",
        "| Family | C log MAE | T log MAE | C median prediction / measurement | T median prediction / measurement |",
        "|---|---:|---:|---:|---:|",
    ])
    for family in HELD_OUT_FAMILIES:
        metrics = prediction_errors([case for case in cases if case.features["family"] == family])
        c, t = metrics["log_cold_ms"], metrics["log_warm_us"]
        lines.append(
            f"| {family} | {c['log_mae']:.3f} | {t['log_mae']:.3f} | "
            f"{c['median_predicted_over_measured']:.3f} | {t['median_predicted_over_measured']:.3f} |"
        )
    return lines


def _write_results(
    out: Path, model: CostModel, groups: dict[str, list[Forecast]], cv: dict,
    *, miss_ms: float, bootstraps: int, gate_count: int = 0,
) -> bool:
    table, wins = evaluate(groups, miss_ms=miss_ms, bootstraps=bootstraps)
    with ensure_parent(out.with_suffix(".csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    out.with_suffix(".md").write_text(
        _markdown(model, table, cv, miss_ms=miss_ms, bootstraps=bootstraps,
                  holdout=groups.get("holdout", []), gate_wins=wins, gate_count=gate_count),
        encoding="utf-8",
    )
    return bool(wins)


def main(argv: list[str] | None = None) -> int:
    today = date.today().isoformat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "features_cpu_2026-11-17.csv")
    parser.add_argument("--model", type=Path, default=Path("models/cost_cpu.json"))
    parser.add_argument("--overhead", type=Path, required=True, help="decision_overhead JSON for this model")
    parser.add_argument("--out", type=Path, help="output prefix for the CSV and Markdown")
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--holdout-gate", action="store_true", help="run and record the one gate for this frozen model")
    args = parser.parse_args(argv)
    if args.bootstraps < 1:
        parser.error("--bootstraps must be positive")
    out = args.out or PROCESSED_RESULTS_DIR / f"selector_cpu_{today}_dev"

    artifact_bytes = args.model.read_bytes()
    artifact = json.loads(artifact_bytes)
    model = CostModel.from_dict(artifact)
    timings = json.loads(args.overhead.read_text(encoding="utf-8"))
    if timings["model_version"] != model.version or timings["backend"] != model.backend:
        parser.error("--overhead must come from the evaluated CPU model")
    miss_ms = float(timings["pooled"]["miss_ms"]["median"])
    if miss_ms < 0 or not math.isfinite(miss_ms):
        parser.error("decision overhead must be finite and non-negative")
    dev = read_rows(args.data, split="dev")
    if dev_rows_digest(dev) != artifact["model"]["dev_rows_sha256"]:
        parser.error("dev data does not match the frozen artifact")
    folds = lofo_forecasts(dev, warm_terms=model.warm_terms, warm_alpha=model.warm_alpha)
    cv = cv_metrics(folds)
    dev_pooled = [case for cases in folds.values() for case in cases]
    groups = {**folds, "dev_pooled": dev_pooled}
    options = {"miss_ms": miss_ms, "bootstraps": args.bootstraps}
    if args.holdout_gate:
        with ensure_parent(GATE_LOG).open("a+", encoding="utf-8") as log:
            fcntl.flock(log, fcntl.LOCK_EX)
            log.seek(0)
            starts = [entry for entry in map(json.loads, log) if entry["status"] == "started"]
            if any(entry["model_version"] == model.version for entry in starts):
                parser.error(f"the gate for {model.version} has already been recorded in {GATE_LOG}")
            run_number = len(starts) + 1
            suffix = "" if run_number == 1 else f"_gate{run_number}"
            out = args.out or PROCESSED_RESULTS_DIR / f"selector_cpu_{today}{suffix}"
            if any(entry["output"] == out.name for entry in starts):
                parser.error(f"output {out.name} is already recorded for a prior gate")
            run = {
                "run": run_number, "model_version": model.version,
                "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "output": out.name, "started_utc": utc_timestamp(), "decision_overhead_ms": miss_ms,
                "R": R_GRID, "B_ms": B_GRID,
            }
            _record_gate(log, {**run, "status": "started"})
            holdout = forecast(model, read_rows(args.data, split="holdout"))
            groups["holdout"] = holdout
            for family in HELD_OUT_FAMILIES:
                groups[f"holdout_{family}"] = [case for case in holdout if case.features["family"] == family]
            passed = _write_results(out, model, groups, cv, gate_count=run["run"], **options)
            _record_gate(log, {
                **run, "status": "passed" if passed else "failed", "configurations": len(holdout),
                "finished_utc": utc_timestamp(),
            })
        print(f"Week 3 holdout gate #{run['run']}: {'PASS' if passed else 'FAIL'}")
    else:
        _write_results(out, model, groups, cv, **options)
    print(f"Wrote {out.with_suffix('.csv')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
