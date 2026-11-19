"""Choose actions by end-to-end regret on family-grouped development folds."""

from dataclasses import asdict, replace
from itertools import product
import json

import numpy as np
import pandas as pd

from benchmarks.corpus.fori_horizon.cohort import OUT
from benchmarks.corpus.fori_horizon.replay import history, total
from benchmarks.corpus.fori_iterations.refit import CV, TRAINING
from benchmarks.corpus.fori_iterations.storage import UNCOMPRESSED
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.schema import FACTORS
from jax_unroll.selector import HorizonPolicy

GRID = (300, 1000, 2000, 5000, 20000)
BUDGET = 125.


def features(keys):
    lookup = {}
    for cohort, r, f, _, _ in history():
        source = {"proposal": "proposal_dev", "regret_train": "regret"}.get(cohort, cohort)
        key = source, r["workload"], r["variant"], int(r["trip_count"])
        lookup[key] = f
    return [lookup[k] for k in keys[["source","workload","variant","trip_count"]].itertuples(index=False,name=None)]


def choices(policy, frame, pred, h):
    feasible = pred[:, :, 0] <= BUDGET - policy.decision_ms
    predicted = pred[:, :, 1]
    picked = np.where(feasible, predicted, np.inf).argmin(axis=1)
    active = np.full(len(frame), h >= policy.min_calls)
    picked[~active] = 0
    if policy.matrix_bytes is not None and h >= policy.matrix_min_calls:
        index = FACTORS.index(policy.matrix_factor)
        guard = ((frame.carry_bytes >= policy.matrix_bytes) & frame.matrix & (frame.eq_dot == 0)
                 & (frame.eq_reduce == 0) & (frame.eq_gather_scatter <= policy.max_index_ops))
        active |= (frame.carry_bytes >= policy.matrix_bytes) & frame.matrix
        picked[guard & feasible[:, index]] = index
    return picked, active


def utility(seconds, native, mask, weights):
    # Match deployment time while charging twice for relative workload regressions.
    selected, base = seconds[:, mask], native[:, mask]
    gain = 1 - selected / base
    relative = np.average(gain + np.minimum(gain, 0), weights=weights[mask], axis=1)
    pooled = 1 - selected.sum(axis=1) / base.sum(axis=1)
    return float(np.mean(relative + pooled))


def main():
    keys = pd.read_csv(TRAINING)
    frame = pd.DataFrame(features(keys))
    frame["matrix"] = frame.carry_shapes.map(lambda shapes: any(len(s) == 2 for s in json.loads(shapes)))
    with np.load(CV) as db:
        pred, timing, folds = np.exp(db["incumbent"]), db["timing"], db["fold"]
        np.testing.assert_array_equal(db["workload"], keys.workload)
    assert keys.groupby("family").fold.nunique().max() == 1 and np.array_equal(keys.fold, folds)
    seconds = np.array([total(timing[:,:,0], timing[:,:,1], h) / 1000 for h in GRID])
    native = seconds[:, :, 0]
    weights = 1 / keys.groupby(["source", "workload"]).workload.transform("size").to_numpy()
    # Historical decision time includes feature extraction and inference.
    from benchmarks.corpus.fori_iterations.storage import OUT as prior
    raw = json.loads((prior / "confirmation_live.json").read_text())
    decision = float(np.median([r["decision_ms"] for run in raw["runs"] for r in run["incumbent"]["rows"]]))
    load = json.loads((prior / "01_storage.json").read_text())["uncompressed"]["median_ms"]
    trials = []
    # Previous mixed direct measurements place the warm crossover around 1800.
    # Grouped development did not support changing the long-horizon warm ranking.
    base = HorizonPolicy(1800, decision)
    candidates = [base] + [replace(base, matrix_bytes=n, max_index_ops=ops,
                                   matrix_min_calls=cut, matrix_factor=f)
                           for n, ops, cut, f in product((512, 1024, 2048, 4096), (1, 2),
                                                         (300, 1000, 1800), (2, 4))]
    for policy in candidates:
        rows = []
        for i, h in enumerate(GRID):
            picks, active = choices(policy, frame, pred, h)
            times = seconds[i, np.arange(len(frame)), picks] + active * (decision + load / 12) / 1000
            rows.append(times)
        trials.append((policy, np.array(rows)))
    all_rows = np.ones(len(keys), bool)
    output = {"grid_calls": GRID, "configurations": len(frame), "folds": [], "budget_ms": BUDGET,
              "model_version": CompactForiPredictor.load(UNCOMPRESSED).version}
    for name in ("crossover", "candidate"):
        eligible = [(p,t) for p,t in trials if name == "candidate" or p.matrix_bytes is None]
        held = np.zeros_like(native)
        for fold in np.unique(folds):
            train, test = folds != fold, folds == fold
            p, t = max(eligible, key=lambda entry: utility(entry[1], native, train, weights))
            held[:,test] = t[:,test]
            output["folds"].append(dict(arm=name, fold=int(fold), policy=asdict(p),
                                        families=sorted(keys.family[test].unique())))
        p, _ = max(eligible, key=lambda entry: utility(entry[1], native, all_rows, weights))
        output[name] = asdict(p)
        output[name + "_grouped_replay_s"] = held.sum(axis=1).tolist()
    output["native_replay_s"] = native.sum(axis=1).tolist()
    (OUT / "policy.json").write_text(json.dumps(output, indent=2) + "\n")
    print({k:v for k,v in output.items() if k != "folds"})


if __name__ == "__main__":
    main()
