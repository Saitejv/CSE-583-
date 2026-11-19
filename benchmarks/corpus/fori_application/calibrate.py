"""Select a matrix action using family-held-out end-to-end development replay."""

import json
from dataclasses import asdict, replace
from itertools import product

import numpy as np
import pandas as pd

from benchmarks.corpus.fori_agent_cases.dataset import load
from benchmarks.corpus.fori_application.protocol import OUT
from benchmarks.corpus.fori_horizon.calibrate import features
from benchmarks.corpus.fori_horizon.cohort import OUT as PRIOR
from benchmarks.corpus.fori_horizon.replay import curves
from benchmarks.corpus.fori_iterations.refit import CV, TRAINING
from benchmarks.corpus.fori_iterations.storage import UNCOMPRESSED
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.schema import FACTORS
from jax_unroll.selector import HorizonPolicy

CALLS = (300, 1000, 5000, 20000)
BUDGET = 125.0


def development():
    keys = pd.read_csv(TRAINING)
    frame = pd.DataFrame(features(keys))
    frame["family"] = keys.family
    frame["group"] = keys.validation_group.astype(str)
    frame["workload"] = keys.workload
    more, times = [], []
    for path in (PRIOR / "sweep.json", PRIOR / "coverage/sweep.json"):
        for r, cold, warm in curves(path):
            if r["workload"] == "dense_gram_accum":  # historical control already in training
                continue
            more.append(dict(**r["features"], family=r["family"], workload=r["workload"]))
            times.append(np.column_stack((np.median(cold, axis=0), np.median(warm, axis=0))))
    agent, agent_times = load("train")
    extra = pd.concat([pd.DataFrame(more), agent], ignore_index=True)
    extra["group"] = extra.family
    model = CompactForiPredictor.load(UNCOMPRESSED)
    extra_pred = [
        [p[f].linear() for f in FACTORS]
        for p in (model.predict_candidates(r) for r in extra.to_dict("records"))
    ]
    with np.load(CV) as db:
        np.testing.assert_array_equal(db["workload"], frame.workload)
        pred = np.concatenate([np.exp(db["incumbent"]), extra_pred])
        observed = np.concatenate([db["timing"], times, agent_times])
    return pd.concat([frame, extra], ignore_index=True), pred, observed


def main():
    previous = HorizonPolicy(**json.loads((PRIOR / "policy.json").read_text())["candidate"])
    frame, forecasts, measured = development()
    matrix = frame.carry_shapes.map(lambda s: any(len(shape) == 2 for shape in json.loads(s))).to_numpy()
    feasible = forecasts[:, :, 0] <= BUDGET - previous.decision_ms
    warm = np.where(feasible, forecasts[:, :, 1], np.inf).argmin(axis=1)
    rows = np.arange(len(warm))
    times = np.array([measured[:, :, 0] + (h - 1) * measured[:, :, 1] / 1000 for h in CALLS])
    weight = 1 / frame.groupby("workload").workload.transform("size").to_numpy()
    elementwise = (frame.eq_dot.eq(0) & frame.eq_reduce.eq(0)).to_numpy()
    indexes = frame.eq_gather_scatter.to_numpy()

    def scores(rule):
        spatial = matrix & (frame.carry_bytes >= rule.matrix_bytes).to_numpy()
        mask = spatial & elementwise & (indexes <= rule.max_index_ops)
        index = FACTORS.index(rule.matrix_factor)
        costs = []
        for k, h in enumerate(CALLS):
            pick = warm.copy() if h >= rule.min_calls else np.zeros(len(warm), int)
            if h >= rule.matrix_min_calls:
                pick[mask & feasible[:, index]] = index
            active = (h >= rule.min_calls) | (spatial & (h >= rule.matrix_min_calls))
            costs.append(times[k, rows, pick] + active * (rule.decision_ms + 0.3))
        return np.array(costs)

    base = scores(previous)
    # Minimize regret where the decision can change. Charge relative slowdowns
    # twice and balance workloads while retaining a term for absolute ms saved.
    trials = [previous]
    for size, ops, factor, calls in product((512, 1024, 2048), (2, 3, 4), (2, 4), (300, 1000)):
        trials.append(
            replace(
                previous, matrix_bytes=size, max_index_ops=ops, matrix_factor=factor, matrix_min_calls=calls
            )
        )
    scope = matrix & elementwise & (frame.carry_bytes >= 512) & (indexes <= 4)
    costs = np.array([scores(p) for p in trials])

    def best(mask):
        mask = np.asarray(mask & scope)
        relative = 1 - costs[:, :, mask] / base[:, mask]
        benefit = np.average(relative + np.minimum(relative, 0), axis=2, weights=weight[mask])
        pooled = 1 - costs[:, :, mask].sum(axis=2) / base[:, mask].sum(axis=1)
        return int((benefit + pooled).mean(axis=1).argmax())

    grouped = base.copy()
    folds = []
    for group in frame.group.unique():
        valid = frame.group.eq(group).to_numpy()
        winner = best(~valid)
        grouped[:, valid] = costs[winner][:, valid]
        folds.append(dict(group=group, policy=asdict(trials[winner])))
    winner = best(np.ones(len(frame), bool))
    result = dict(
        policy=asdict(trials[winner]),
        model_version=CompactForiPredictor.load(UNCOMPRESSED).version,
        configurations=len(frame),
        families=frame.family.nunique(),
        groups=len(folds),
        calls=CALLS,
        grouped_saved_vs_previous_s=((base - grouped).sum(axis=1) / 1000).tolist(),
        folds=folds,
    )
    (OUT / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "folds"})


if __name__ == "__main__":
    main()
