"""Train and evaluate across the complete 66-workload CPU corpus."""
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from benchmarks.corpus.registry import build_workload
from benchmarks.corpus.static_experiment import components
from benchmarks.dataset import aggregate_labels, build_feature_table
from benchmarks.oracle import load_raw
from benchmarks.recovery_workloads import RECOVERY_WORKLOADS
from benchmarks.workloads import WORKLOADS
from benchmarks.recovery_data import JOIN_KEY, LEGACY_DATA, MODEL
from jax_unroll.corpus_model import CorpusPredictor
from jax_unroll.model import CostModel, group_configs
from jax_unroll.recovery_split import CONFIRMATION
from jax_unroll.schema import FACTORS
from jax_unroll.selector import DecisionEngine, select_factor
from jax_unroll.validation import B_GRID, R_GRID, LONG_R

OUT = Path("results/all_workloads")
TEST_FAMILIES = {"indexed_memory", "iterative_solver", "stencil", "segmented_scan", *CONFIRMATION.values()}
def dataset():
    tables = [pd.read_csv(p) for p in (LEGACY_DATA, "data/recovery/dev.csv",
              "data/recovery/confirmation.csv", "results/static_features/augmented_features.csv")]
    paths = [Path("results/raw/corpus_smoke_cpu_2026-11-17.csv"),
             *sorted(Path("results/profile_transfer").glob("*_raw.csv"))]
    labels = aggregate_labels(load_raw(paths))
    tables.append(build_feature_table(labels, workload_builder=build_workload).merge(labels, on=JOIN_KEY))
    rows = pd.concat(tables, ignore_index=True)
    costs = ["cold_median_ms", "warm_median_us"]
    if rows.groupby(JOIN_KEY)[costs].nunique().gt(1).any().any():
        raise ValueError("overlapping inputs disagree")
    groups = group_configs(rows.drop_duplicates(JOIN_KEY).to_dict("records"))
    base = pd.DataFrame([g[1] for g in groups])
    measured = np.array([[[g[u][k] for k in costs] for u in FACTORS] for g in groups])
    if not np.isfinite(measured).all() or measured.min() <= 0 or base.workload.nunique() != 66:
        raise ValueError("invalid measurement matrix")
    return base, measured


def decisions(measured, estimates, margin=0, fixed=False):
    estimates = estimates.copy()
    estimates[:, 1:, 1] *= np.exp(margin)
    output = []
    for budget in B_GRID:
        limit = np.inf if budget is None else budget
        feasible = measured[:, :, 0] <= limit
        for calls in R_GRID:
            actual = measured[:, :, 0] + calls * measured[:, :, 1] / 1000
            expected = estimates[:, :, 0] + calls * estimates[:, :, 1] / 1000
            oracle = np.where(feasible, actual, np.inf).argmin(axis=1)
            selected = np.where(estimates[:, :, 0] <= limit, expected, np.inf).argmin(axis=1)
            if fixed:
                selected.fill(0)
            i = np.arange(len(measured))
            output.append(pd.DataFrame(dict(i=i, budget=limit, calls=calls,
                long=(calls >= LONG_R) & (feasible.sum(axis=1) >= 2),
                feasible=feasible.any(axis=1), violation=~feasible[i, selected],
                oracle=np.take(FACTORS, oracle), chosen=np.take(FACTORS, selected),
                excess=np.maximum(0, actual[i, selected] / actual[i, oracle] - 1))))
    return pd.concat(output, ignore_index=True)


def metrics(grid):
    long = grid[grid.long]
    return dict(settings=len(long), exact=(long.chosen == long.oracle).mean(),
                within=((long.excess <= .1) & ~long.violation).mean(), mean_excess=long.excess.mean(),
                violations=grid.loc[grid.feasible, "violation"].mean())


def verify_live(model, held, measured, grid):
    timings = []
    for i, row in enumerate(held.to_dict("records")):
        name, trip, variant = row["workload"], row["trip_count"], row["variant"]
        spec = RECOVERY_WORKLOADS.get(name) or WORKLOADS.get(name)
        loop = spec.build(trip, 1, variant) if spec else build_workload(name, trip, 1, variant)
        engine = DecisionEngine(model)
        start = perf_counter()
        choice = engine.decide(loop.fn, loop.args, trip_count=trip, loop_kind=loop.loop_kind, R=100_000)
        elapsed = (perf_counter() - start) * 1000
        costs = model.predict_candidates(row)
        if choice.fallback_reason or choice.factor != select_factor(costs, R=100_000):
            raise ValueError("runtime selection differs from offline prediction")
        for setting in grid[grid.i == i].itertuples():
            if select_factor(costs, B=setting.budget, R=setting.calls) != setting.chosen:
                raise ValueError("batch evaluation differs from selector")
        timings.append(elapsed)
    long = grid[grid.long]
    indices = long.i.to_numpy()
    factors = np.searchsorted(FACTORS, long.oracle)
    cold, warm = measured[indices, factors].T
    oracle_ms = cold + warm * long.calls / 1000
    return (long.excess + np.asarray(timings)[indices] / oracle_ms).mean()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base, measured = dataset()
    test = base.family.isin(TEST_FAMILIES)
    split = base[["workload", "family", "variant", "trip_count"]].assign(split=np.where(test, "test", "train"))
    split.to_csv(OUT / "split.csv", index=False)
    if split.groupby("workload").split.nunique().max() != 1 or set(base[~test].body_hash) & set(base[test].body_hash):
        raise ValueError("workload or body hash crosses split")
    train = base[~test].reset_index(drop=True)
    y = measured[~test]
    membership = {w: group for group, names in components().items() for w in names}
    indexed = membership["gather_scalar_table"]
    groups = [membership.get(w, indexed if w.startswith("matched_") else f) for w, f in zip(train.workload, train.family)]
    cv = []
    for method in ("linear", "kernel", "forest", "trees"):
        predictions = np.zeros_like(y)
        for fit, val in GroupKFold(5).split(train, groups=groups):
            model = CorpusPredictor(method, 0).fit(train.iloc[fit], y[fit])
            predictions[val] = model.predict(train.iloc[val])
        for margin in (0, .05, .15, .3):
            grid = decisions(y, predictions, margin)
            cv.append(dict(method=method, margin=margin, **metrics(grid)))
    cv = pd.DataFrame(cv)
    cv["loss"] = cv.mean_excess + (1-cv.within) + cv.violations
    cv.sort_values("loss").to_csv(OUT / "validation.csv", index=False)
    best = cv.loc[cv.loss.idxmin()]
    model = CorpusPredictor(best.method, best.margin).fit(train, y)
    # Persist fitted sklearn state with a stable import path.
    joblib.dump((best.method, best.margin, model.scale, model.center, model.fit_model), OUT / "model.joblib", compress=3)
    held = base[test].reset_index(drop=True)
    frozen = CostModel.load(MODEL)
    prior = np.array([[frozen.predict_candidates(r)[u].linear() for u in FACTORS] for r in held.to_dict("records")])
    loaded = CorpusPredictor.load(OUT / "model.joblib")
    if not all(loaded.supports(row) and frozen.supports(row) for row in held.to_dict("records")):
        raise ValueError("evaluation outside supported model domain")
    summary = []
    for name, pred, margin in [(best.method, loaded.predict(held), best.margin), ("recovery", prior, 0), ("factor1", prior, 0)]:
        grid = decisions(measured[test], pred, margin, fixed=name=="factor1")
        grid = grid.join(held[["workload","variant","trip_count"]], on="i")
        grid.to_csv(OUT / f"{name}_test.csv", index=False)
        result = dict(method=name, **metrics(grid))
        if name != "factor1":
            cold, warm = np.median(abs(pred / measured[test] - 1), axis=(0, 1))
            result.update(cold_median_relative_error=cold, warm_median_relative_error=warm)
        if name == best.method:
            result["live_mean_excess"] = verify_live(loaded, held, measured[test], grid)
        summary.append(result)
    print(split.groupby("split").workload.agg(["nunique","size"]),best, pd.DataFrame(summary), sep="\n")
    pd.DataFrame(summary).to_csv(OUT / "test_summary.csv", index=False)


if __name__ == "__main__":
    main()
