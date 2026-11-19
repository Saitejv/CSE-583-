"""Cycle 4. Calibrate joint savings errors and score budgeted choices by call horizon."""

from dataclasses import asdict
from itertools import product
import json
from statistics import median

import numpy as np
import pandas as pd

from benchmarks.corpus.fori_iterations.confirmation import HORIZONS
from benchmarks.corpus.fori_iterations.refit import CV, REFIT
from benchmarks.corpus.fori_iterations.storage import OUT, UNCOMPRESSED
from benchmarks.corpus.fori_proposal.compact import POLICY as ORIGINAL_POLICY
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.model import LogCosts
from jax_unroll.schema import FACTORS
from jax_unroll.selector import PayoffPolicy, SavingsBand, SavingsPolicy, select_factor

POLICY = OUT / "04_savings_policy.json"
BUDGET = 125


def bounds(predictions, timing, boundaries, quantile, decision_ms, load_ms):
    estimate = np.exp(predictions)
    warm1 = estimate[:, 0, 1]
    predicted_gain = warm1[:, None] - estimate[:, 1:, 1]
    measured_gain = timing[:, :1, 1] - timing[:, 1:, 1]
    offer = predicted_gain > 0
    warm_error = (predicted_gain - measured_gain) / warm1[:, None]
    cold_error = timing[:, :, 0] - estimate[:, :, 0]
    additional_error = cold_error[:, 1:] - cold_error[:, :1]

    def upper(values, mask):
        return max(0., float(np.quantile(values[mask], quantile, method="higher")))

    bands, lower = [], 0.
    for edge in (*boundaries, None):
        scope = (warm1 >= lower) & (warm1 < (np.inf if edge is None else edge))
        local = offer & scope[:, None]
        if local.sum() < 20:
            local = offer
        bands.append(SavingsBand(
            edge, upper(warm_error, local), upper(additional_error, local),
            upper(cold_error[:, 1:], local),
        ))
        if edge is not None:
            lower = edge
    return SavingsPolicy(tuple(bands), decision_ms, load_ms)


def choices(logs, rule, horizon):
    as_costs = [{f: LogCosts(*log) for f, log in zip(FACTORS, row, strict=True)} for row in logs]
    if rule is None:
        picked = [select_factor(row, throughput_only=True) for row in as_costs]
    else:
        picked = [rule.choose(row, R=horizon, B=BUDGET) for row in as_costs]
    return np.searchsorted(FACTORS, picked)


def replay(timing, workload, picks, horizon, overhead):
    native = timing[:, 0, 0] + (horizon - 1) * timing[:, 0, 1] / 1_000
    selected = timing[np.arange(len(timing)), picks]
    actual = selected[:, 0] + (horizon - 1) * selected[:, 1] / 1_000 + overhead
    gain = np.log(native / actual)
    utility = pd.Series(gain + np.minimum(gain, 0)).groupby(workload).mean().mean()
    return {
        "pooled_speedup": float(native.sum() / actual.sum()),
        "body_speedup": float(np.exp(pd.Series(gain).groupby(workload).mean().mean())),
        "utility": float(utility),
        "unrolled": int(np.count_nonzero(picks)),
        "regressions_over_3pct": int((actual > 1.03 * native).sum()),
    }


def main():
    old_values = json.loads(ORIGINAL_POLICY.read_text())["policy"]
    load = json.loads((OUT / "01_storage.json").read_text())
    trace = json.loads((OUT / "02_trace_runs.json").read_text())["runs"]
    # Original trace + lower is an upper bound on decision time for this path.
    decision_ms = median(row["original"]["trace_lower_ms"] for run in trace for row in run)
    load_ms = load["uncompressed"]["median_ms"] / 6
    older = PayoffPolicy(**{**old_values, "decision_ms": decision_ms, "model_load_ms": load_ms})
    candidates = []
    with np.load(CV) as data:
        timing, folds, names = data["timing"], data["fold"], data["workload"]
        horizon_list = HORIZONS[1:]
        comparison = {
            label: {str(h): replay(timing, names, choices(data["incumbent"], rule, h), h, decision_ms + load_ms)
                    for h in horizon_list}
            for label, rule in (("incumbent_warm", None), ("original_payoff", older))
        }
        comparison["refit_warm"] = {
            str(h): replay(timing, names, choices(data["refit"], None, h), h, decision_ms + load_ms)
            for h in horizon_list
        }
        for name, boundaries, quantile in product(("incumbent", "refit"), ((), (15., 60.)), (.55, .70, .85)):
            prediction = data[name]
            selections = {h: np.zeros(len(timing), dtype=int) for h in horizon_list}
            for fold in np.unique(folds):
                train, test = folds != fold, folds == fold
                policy = bounds(prediction[train], timing[train], boundaries, quantile, decision_ms, load_ms)
                for h in horizon_list:
                    selections[h][test] = choices(prediction[test], policy, h)
            report = {str(h): replay(timing, names, picks, h, decision_ms + load_ms)
                      for h, picks in selections.items()}
            score = float(np.mean([r["utility"] + .35 * np.log(r["pooled_speedup"]) for r in report.values()]))
            candidates.append({
                "model": name, "boundaries": boundaries, "quantile": quantile, "score": score, "replay": report,
            })
        candidates.sort(key=lambda trial: trial["score"], reverse=True)
        winner = candidates[0]
        frozen = bounds(data[winner["model"]], timing, winner["boundaries"], winner["quantile"], decision_ms, load_ms)
    model = CompactForiPredictor.load(UNCOMPRESSED if winner["model"] == "incumbent" else REFIT)
    POLICY.write_text(json.dumps({"model": winner["model"], "version": model.version,
                                  "policy": asdict(frozen), "cold_budget_ms": BUDGET}, indent=2) + "\n")
    output = json.dumps({"baseline": comparison, "selected": winner, "trials": candidates}, indent=2) + "\n"
    (OUT / "04_development_replay.json").write_text(output)
    print(json.dumps(winner, indent=2))


if __name__ == "__main__":
    main()
