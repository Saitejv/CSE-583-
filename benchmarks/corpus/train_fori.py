"""Fit on expansion development data, replay fixed warm-only policies by cohort."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from jax_unroll.corpus_model import CorpusPredictor
from jax_unroll.fori_model import ForiCostPredictor
from jax_unroll.model import group_configs
from jax_unroll.schema import FACTORS
from jax_unroll.selector import select_factor

DATA = Path("results/fori_expansion")
OUT = Path("results/fori_predictor")
SPLIT = Path("benchmarks/corpus/fori_expansion/split.csv")
METHODS = ("trees", "effect_trees", "effect_forest", "effect_linear")


def read(path, partitions, manifest):
    df = pd.read_csv(path)
    expect = manifest[manifest.partition.isin(partitions)]
    if (set(df.workload) != set(expect.workload) or
            not df.partition.eq(df.workload.map(expect.set_index("workload").partition)).all() or
            not df.loop_kind.eq("fori_loop").all() or not df.correct_all.eq(True).all()):
        raise ValueError("dataset disagrees with split or workload contract")
    groups = group_configs(df.to_dict("records"))
    frame = pd.DataFrame([g[1] for g in groups])
    costs = np.array([[[g[u]["cold_median_ms"], g[u]["warm_median_us"]] for u in FACTORS] for g in groups])
    if not (np.isfinite(costs).all() and costs.min() > 0):
        raise ValueError("invalid cost")
    return frame, costs


def choices(predictions, margin):
    warm = predictions[:, :, 1].copy()
    warm[:, 1:] *= np.exp(margin)
    return warm.argmin(axis=1)


def evaluate(frame, warm, selected):
    chosen = warm[np.arange(len(warm)), selected]
    best = warm.min(axis=1)
    detail = frame[["workload", "variant", "trip_count", "partition"]].copy()
    detail["factor"] = np.take(FACTORS, selected)
    detail["chosen_us"], detail["default_us"], detail["oracle_us"] = chosen, warm[:, 0], best
    by_body = np.log(warm[:, 0] / chosen)
    body_logs = pd.Series(by_body, index=frame.workload).groupby(level=0).mean().to_numpy()
    draws = np.random.default_rng(583).choice(body_logs, size=(5000, len(body_logs)))
    lo, hi = np.exp(np.quantile(draws.mean(axis=1), [.025, .975]))
    summary = dict(configs=len(frame), bodies=frame.workload.nunique(), geo_speedup=np.exp(by_body.mean()),
                   body_speedup=np.exp(body_logs.mean()), body_ci_lo=lo, body_ci_hi=hi,
                   pooled_speedup=warm[:, 0].sum()/chosen.sum(), exact=np.mean(chosen == best),
                   within10=np.mean(chosen <= best*1.1), worst_slowdown=max(chosen / warm[:, 0]),
                   wins=int(sum(chosen < warm[:, 0])), losses=int(sum(chosen > warm[:, 0])))
    return detail, summary


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    manifest = pd.read_csv(SPLIT)
    if manifest.workload.duplicated().any() or manifest.groupby("related_group").partition.nunique().max() != 1:
        raise ValueError("invalid lineage split")
    train, y = read(DATA/"train_features.csv", {"train"}, manifest)
    cv = []
    # Keep whole families together; also merge any families linked by related-group IDs.
    family_sets = [{f} for f in train.family.unique()]
    for _, rows in manifest[manifest.partition == "train"].groupby("related_group"):
        joined = [s for s in family_sets if s.intersection(rows.family)]
        family_sets = [s for s in family_sets if s not in joined] + [set.union(*joined)]
    family_groups = train.family.map({f: i for i,s in enumerate(family_sets) for f in s})
    for method in METHODS:
        oof = np.empty_like(y)
        for a,b in GroupKFold(5).split(train, groups=family_groups):
            model = ForiCostPredictor(method).fit(train.iloc[a], y[a])
            oof[b] = model.predict(train.iloc[b])
        for margin in (0, .015, .04, .08):
            _, result = evaluate(train, y[:,:,1], choices(oof, margin))
            cv.append(dict(method=method, margin=margin, **result))
    cv = pd.DataFrame(cv).sort_values("body_speedup", ascending=False, kind="stable")
    cv.to_csv(OUT/"validation.csv", index=False)
    winner = cv.iloc[0]
    model = ForiCostPredictor(winner.method, winner.margin).fit(train, y)
    joblib.dump(model, OUT/"model.joblib", compress=3)
    model = ForiCostPredictor.load(OUT/"model.joblib")
    previous = CorpusPredictor.load("results/all_workloads/model.joblib")
    test, costs = read(DATA/"eval_features.csv", {"historical_eval","fresh_eval"}, manifest)
    if set(train.body_hash) & set(test.body_hash):
        raise ValueError("body hash leakage")
    summaries, details = [], []
    for cohort, indices in test.groupby("partition").groups.items():
        frame, actual = test.loc[indices], costs[indices]
        pred = model.predict(frame)
        old = previous.predict(frame)
        warm = actual[:,:,1]
        policy = choices(pred, model.margin)
        runtime = [select_factor(model.predict_candidates(r), throughput_only=True)
                   for r in frame.to_dict("records")]
        if not np.array_equal(np.take(FACTORS, policy), runtime):
            raise ValueError("runtime policy mismatch")
        for name, pick in (("expanded", policy), ("previous", choices(old, previous.margin)),
                           ("factor1", np.zeros(len(frame),dtype=int)), ("oracle", warm.argmin(axis=1))):
            detail, result = evaluate(frame, warm, pick)
            if name in ("expanded","previous"):
                estimate = pred if name=="expanded" else old
                result["warm_median_abs_relative_error"] = np.median(abs(estimate[:,:,1]/warm - 1))
            summaries.append(dict(cohort=cohort, policy=name, **result))
            details.append(detail.assign(policy=name))
    pd.concat(details).to_csv(OUT/"decisions.csv", index=False)
    result = pd.DataFrame(summaries)
    result.to_csv(OUT/"summary.csv", index=False)
    print(cv.head(4).to_string(index=False), result.to_string(index=False), sep="\n")


if __name__ == "__main__":
    main()
