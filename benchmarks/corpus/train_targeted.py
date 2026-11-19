"""Train on completed development matrices, freeze the winner, then score sealed loops.

Run with python -m benchmarks.corpus.train_targeted [fit|score].
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from benchmarks.corpus import train_fori
from benchmarks.corpus.fori_targeted import acquire
from benchmarks.corpus.fori_targeted.registry import TRAIN_SPECS
from jax_unroll.fori_model import ForiCostPredictor
from jax_unroll.schema import FACTORS
from jax_unroll.selector import DecisionEngine, select_factor
from jax_unroll.targeted_model import TargetedCostPredictor

DIR = Path("results/targeted_predictor")
MODEL = DIR / "model.joblib"
RAW_KEY = ["workload", "variant", "trip_count", "factor", "process_repeat"]
METHODS = ("trees", "joint_forest", "cost_ratio", "blend", "kernel", "shared_effect", "boosted", "upper_quartile")
MARGINS = (0, .015, .04, .08)


def audited_table(partition):
    """Verify planned keys and recompute all factor medians for the relevant partition."""
    manifest = pd.read_csv(acquire.SPLIT_CSV)
    names = manifest.loc[manifest.partition == partition, "workload"]
    raw = pd.read_csv(acquire.RAW_PATH)
    raw = raw[raw.workload.isin(names)]
    planned = pd.DataFrame(acquire.plan_acquisition_matrix())
    expected = planned[planned.workload.isin(names)]
    raw_keys = pd.MultiIndex.from_frame(raw[RAW_KEY])
    expected_keys = pd.MultiIndex.from_frame(expected[RAW_KEY])
    if not raw_keys.is_unique or not raw_keys.sort_values().equals(expected_keys.sort_values()):
        raise ValueError(f"incomplete or duplicate {partition} acquisition keys")
    if (not raw.correct.eq(True).all() or raw.error.notna().any() or
            not raw.backend.eq("cpu").all() or not raw.loop_kind.eq("fori_loop").all()):
        raise ValueError(f"invalid measurements in {partition}")
    path = acquire.TRAIN_PATH if partition == "train" else acquire.EVAL_PATH
    table = pd.read_csv(path).set_index(RAW_KEY[:-1]).sort_index()
    medians = raw.groupby(RAW_KEY[:-1])[["cold_total_ms", "warm_median_us"]].median()
    if not table.index.equals(medians.index):
        raise ValueError(f"feature table keys differ from raw {partition} medians")
    np.testing.assert_allclose(table[["cold_median_ms", "warm_median_us"]], medians, rtol=1e-9)
    return train_fori.read(path, {partition}, manifest)


def grouped_families(frame, old_manifest, new_manifest):
    """Keep each family, explicit lineage, declared adaptation and identical IR together."""
    rows = pd.concat([old_manifest, new_manifest])
    rows = rows[rows.workload.isin(frame.workload)]
    components = [{family} for family in rows.family.unique()]

    def connect(families):
        linked = [s for s in components if s.intersection(families)]
        for s in linked:
            components.remove(s)
        components.append(set.union(*linked))

    for _, block in rows.groupby("related_group"):
        connect(set(block.family))
    for _, block in frame.groupby("body_hash"):
        connect(set(block.family))
    for spec in TRAIN_SPECS:
        source = spec.provenance.source
        if source.startswith("Adapted from "):
            ancestor = source.split()[2]
            matches = old_manifest[(old_manifest.workload == ancestor) | (old_manifest.mechanism == ancestor)]
            if len(matches) != 1 or matches.iloc[0].partition != "train":
                raise ValueError(f"unresolved training lineage {spec.name}: {source}")
            connect({spec.family, matches.iloc[0].family})
    ids = {family: n for n, group in enumerate(components) for family in group}
    return frame.family.map(ids)


def validation_utility(detail, sources):
    """Equal weight per collection, then per workload. Charge twice for slowdowns."""
    log_gain = np.log(detail.default_us / detail.chosen_us)
    utility = log_gain + log_gain.clip(upper=0)
    return utility.groupby([sources, detail.workload]).mean().groupby(level=0).mean().mean()


def fit():
    old_manifest = pd.read_csv(train_fori.SPLIT)
    old, timing = train_fori.read(train_fori.DATA/"train_features.csv", {"train"}, old_manifest)
    new, added = audited_table("train")
    frame = pd.concat([old.assign(source="expansion"), new.assign(source="targeted")], ignore_index=True)
    y = np.concatenate((timing, added))
    groups = grouped_families(frame, old_manifest, pd.read_csv(acquire.SPLIT_CSV))
    splits = list(GroupKFold(5).split(frame, groups=groups))
    cv = []
    for method in METHODS:
        pred = np.empty_like(y)
        for a,b in splits:
            model = TargetedCostPredictor(method).fit(frame.iloc[a], y[a])
            pred[b] = model.predict(frame.iloc[b])
        for margin in MARGINS:
            choices = train_fori.choices(pred, margin)
            detail, result = train_fori.evaluate(frame, y[:,:,1], choices)
            cv.append(dict(method=method, margin=margin, utility=validation_utility(detail, frame.source),
                           **result))
    validation = pd.DataFrame(cv).sort_values("utility", ascending=False, kind="stable")
    validation.to_csv(DIR / "validation.csv", index=False)
    winner = validation.iloc[0]
    model = TargetedCostPredictor(winner.method, winner.margin).fit(frame, y)
    joblib.dump(model, MODEL, compress=3)
    frame[["workload","family","variant","trip_count","body_hash","source"]].assign(
        validation_group=groups).to_csv(DIR / "training_split.csv", index=False)
    print(validation.head().to_string(index=False))


def score():
    frozen = TargetedCostPredictor.load(MODEL)
    reference = ForiCostPredictor.load(train_fori.OUT/"model.joblib")
    test, costs = audited_table("sealed_eval")
    used = pd.read_csv(DIR/"training_split.csv")
    if set(used.workload) & set(test.workload) or set(used.body_hash) & set(test.body_hash):
        raise ValueError("sealed configurations overlap fitted data")
    selections = {}
    summaries, rows = [], []
    for name, model in (("targeted", frozen), ("incumbent", reference)):
        if not all(model.supports(row) for row in test.to_dict("records")):
            raise ValueError("unsupported holdout features")
        pred = model.predict(test)
        selected = train_fori.choices(pred, model.margin)
        live = [select_factor(model.predict_candidates(r), throughput_only=True) for r in test.to_dict("records")]
        np.testing.assert_array_equal(live, np.take(FACTORS, selected))
        selections[name] = selected
    # Check the public entry point with a real factor-1 function for every sealed body.
    engine = DecisionEngine(frozen)
    probes = test[(test.variant == "large") & (test.trip_count == 1024)]
    if len(probes) != test.workload.nunique():
        raise ValueError("missing runtime smoke configuration")
    for i, row in probes.iterrows():
        loop = acquire.build_workload(row.workload, int(row.trip_count), 1, row.variant)
        decision = engine.decide(loop.fn, loop.args, trip_count=int(row.trip_count),
                                 loop_kind="fori_loop", throughput_only=True)
        if decision.fallback_reason or decision.factor != FACTORS[selections["targeted"][i]]:
            raise ValueError(f"live trace mismatch on {row.workload}")
    warm = costs[:,:,1]
    for i, factor in enumerate(FACTORS):
        selections[f"factor{factor}"] = np.full(len(test), i, dtype=int)
    selections["oracle"] = warm.argmin(axis=1)
    for name, idx in selections.items():
        detail, summary = train_fori.evaluate(test, warm, idx)
        rows.append(detail.assign(policy=name))
        summaries.append(dict(policy=name, **summary))
    comparison = pd.concat(rows).merge(test[["workload", "family"]].drop_duplicates(),
                                       on="workload", validate="many_to_one")
    comparison.to_csv(DIR/"sealed_decisions.csv", index=False)
    repeat_rows = []
    raw = pd.read_csv(acquire.RAW_PATH)
    raw = raw[raw.workload.isin(test.workload)]
    key = ["workload", "variant", "trip_count"]
    index = pd.MultiIndex.from_frame(test[key])
    for repeat, block in raw.groupby("process_repeat"):
        times = block.pivot(index=key, columns="factor", values="warm_median_us").loc[index, list(FACTORS)].to_numpy()
        for name in ("targeted", "incumbent", "factor1"):
            _, metrics = train_fori.evaluate(test, times, selections[name])
            repeat_rows.append(dict(repeat=repeat, policy=name, **metrics))
    pd.DataFrame(repeat_rows).to_csv(DIR/"repeat_summary.csv", index=False)
    results = pd.DataFrame(summaries)
    results.to_csv(DIR/"sealed_summary.csv", index=False)
    wide = comparison.pivot(index=["workload","variant","trip_count"], columns="policy", values="chosen_us")
    paired = np.log(wide.incumbent / wide.targeted).groupby(level="workload").mean().to_numpy()
    draws = np.random.default_rng(583).choice(paired, size=(5000,len(paired)))
    ci = np.exp(np.quantile(draws.mean(axis=1), [.025,.975]))
    (DIR/"paired.json").write_text(json.dumps(dict(speedup_vs_incumbent=np.exp(paired.mean()),
                                                    ci95=ci.tolist()), indent=2)+"\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit","score"))
    args = parser.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "fit":
        fit()
    else:
        score()
