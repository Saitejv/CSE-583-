"""Audit the native sweep and score static choices against its paired measurements."""

import csv
import json
from statistics import geometric_mean, median

import numpy as np

from benchmarks.corpus.fori_proposal.compact import MODEL, POLICY
from benchmarks.corpus.fori_proposal.protocol import BUDGET_MS, HORIZONS, OUT, REPEATS, freeze
from benchmarks.corpus.fori_proposal.workloads import settings
from benchmarks.corpus.fori_regret.calibrate import MODELS
from jax_unroll.baselines import llvm_style_heuristic
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.fori_model import ForiCostPredictor
from jax_unroll.schema import FACTORS
from jax_unroll.selector import PayoffPolicy, select_factor

DETAIL = OUT / "native_decisions.csv"


def key(row):
    return row["workload"], row["variant"], int(row["trip_count"])


def repeated_winner(samples):
    """Require 5% median speedup and a warm win in every process."""
    ratios = samples[:, :1] / samples
    qualifies = (np.median(ratios, axis=0) >= 1.05) & (ratios > 1).all(axis=0)
    warm = np.median(samples, axis=0)
    return min((i for i, good in enumerate(qualifies) if good), key=lambda i: warm[i], default=0)


def read_native():
    runs = json.loads((OUT / "native_runs.json").read_text())["runs"]
    configurations = {(c.spec.name, c.variant, c.trip): c for c in settings()}
    indexed = [{key(row): row for row in run} for run in runs]
    if len(runs) != REPEATS or any(len(run) != len(configurations) or set(index) != set(configurations)
                                  for run, index in zip(runs, indexed, strict=True)):
        raise ValueError("incomplete or duplicate native timing matrix")
    for k, cfg in configurations.items():
        group = [index[k] for index in indexed]
        features = group[0]["features"]
        if any(r["features"]["body_hash"] != features["body_hash"] or r["family"] != cfg.spec.family
               for r in group):
            raise ValueError(f"inconsistent features for {k}")
        warm = np.asarray([[median(samples) for samples in r["warm_samples_us"]] for r in group])
        cold = np.asarray([r["cold_ms"] for r in group])
        if any(a.shape != (REPEATS, len(FACTORS)) or not (np.isfinite(a) & (a > 0)).all() for a in (warm, cold)):
            raise ValueError(f"invalid timing rows for {k}")
        yield cfg, features, cold, warm


def main():
    freeze()
    reference = ForiCostPredictor.load(MODELS["incumbent"])
    compact = CompactForiPredictor.load(MODEL)
    rule = PayoffPolicy(**json.loads(POLICY.read_text())["policy"])
    with open("results/fori_regret/training_keys.csv", newline="") as handle:
        training = list(csv.DictReader(handle))
    seen_names = {r["workload"] for r in training}
    seen_families = {r["family"] for r in training}
    seen_hashes = {r[k] for r in training for k in ("body_hash", "current_body_hash")}

    detail = []
    for cfg, features, cold, warm in read_native():
        if cfg.cohort == "fresh" and (cfg.spec.name in seen_names or cfg.spec.family in seen_families
                                      or features["body_hash"] in seen_hashes):
            raise ValueError(f"fresh configuration overlaps training: {cfg.spec.name}")
        if not (reference.supports(features) and compact.supports(features)):
            raise ValueError(f"unsupported features for {cfg.spec.name}")
        predictions = compact.predict_candidates(features)
        if compact.source_version != reference.version or predictions != reference.predict_candidates(features):
            raise ValueError("compact and original incumbent forecasts differ")
        times = np.median(warm, axis=0)
        first = np.median(cold, axis=0)
        picks = {
            "oracle": FACTORS[repeated_winner(warm)],
            "minimum_median": FACTORS[int(times.argmin())],
            "incumbent": select_factor(predictions, throughput_only=True),
            "size_heuristic": llvm_style_heuristic(features),
        }
        picks.update({f"payoff_{calls}": rule.choose(predictions, R=calls, B=BUDGET_MS)
                      for calls in HORIZONS})
        row = {
            "workload": cfg.spec.name, "family": cfg.spec.family, "variant": cfg.variant,
            "trip_count": cfg.trip, "cohort": cfg.cohort, "live": cfg.live,
            "body_hash": features["body_hash"],
            **{f"warm_{f}_us": t for f, t in zip(FACTORS, times, strict=True)},
            **{f"first_{f}_ms": t for f, t in zip(FACTORS, first, strict=True)},
            **{f"{label}_factor": u for label, u in picks.items()},
        }
        detail.append(row)
    temporary = DETAIL.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    temporary.replace(DETAIL)

    summary = {}
    for cohort in sorted({r["cohort"] for r in detail}):
        rows = [r for r in detail if r["cohort"] == cohort]
        summary[cohort] = {
            "configurations": len(rows), "bodies": len({r["workload"] for r in rows}),
            "repeat_supported_opportunities": sum(r["oracle_factor"] != 1 for r in rows),
            "oracle_factor_counts": {u: sum(r["oracle_factor"] == u for r in rows) for u in FACTORS},
            "policies": {},
        }
        for name in picks:
            chosen = [r[f'warm_{r[f"{name}_factor"]}_us'] for r in rows]
            base = [r["warm_1_us"] for r in rows]
            ratios = [float(b / a) for a, b in zip(chosen, base, strict=True)]
            summary[cohort]["policies"][name] = {
                "pooled_speedup": float(sum(base) / sum(chosen)), "geometric_speedup": geometric_mean(ratios),
                "wins_over_5pct": sum(r >= 1.05 for r in ratios),
                "slowdowns_over_3pct": sum(r < 1 / 1.03 for r in ratios),
                "worst_time_ratio": 1 / min(ratios),
            }
    path = OUT / "native_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
