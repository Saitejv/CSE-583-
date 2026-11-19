"""Calibrate on direct-factor training sweeps and choose coverage by leave-family-out replay."""

import json
from hashlib import sha256

import numpy as np

from benchmarks.corpus.fori_scalar.fit import BUNDLED
from benchmarks.corpus.fori_workflows.cases import settings
from benchmarks.corpus.fori_workflows.dataset import OUT, load
from jax_unroll.compact_fori import application_engine
from jax_unroll.features import extract_features
from jax_unroll.schema import FACTORS
from jax_unroll.workflow_fori import workflow_features


def calibrate(rows, costs, cap):
    eligible = rows.streams.between(1, cap)
    if rows[eligible].family.nunique() < 2:
        return None
    tables = []
    for t in sorted(rows.trip_count.unique()):
        cs = costs[eligible & rows.trip_count.eq(t)]
        if not len(cs):
            return None
        warm = cs[:, :, 1]
        tables.append(
            np.column_stack(
                (cs[:, :, 0].max(axis=0), np.median(warm / warm[:, :1], axis=0) * np.median(warm[:, 0]))
            )
        )
    return np.stack(tables)


def main():
    rows, costs = load("train")
    lookup = {(s.name, t): (s, v) for s, v, t in settings("train")}
    streams = []
    for r in rows.itertuples():
        s, v = lookup[r.workload, r.trip_count]
        fn, args = s.build(r.trip_count, 1, v)
        f = extract_features(
            fn,
            args,
            trip_count=r.trip_count,
            factor=1,
            loop_kind="fori_loop",
            feature_block=workflow_features,
        )
        streams.append(f.get("workflow_streams", 0))
    rows["streams"] = streams
    trips = sorted(map(int, rows.trip_count.unique()))
    headroom = 125 - application_engine().payoff.decision_ms
    scores, folds = {}, []
    for cap in (0, 1, 4):
        gain = []
        for family in rows.family.unique():
            test = rows.family.eq(family)
            table = calibrate(rows[~test], costs[~test], cap)
            actual, baseline = 0.0, 0.0
            for r in rows[test].itertuples():
                for h in (5000, 20000):
                    elapsed = costs[r.Index, :, 0] + (h - 1) * costs[r.Index, :, 1] / 1000
                    idx = 0
                    if table is not None and 0 < r.streams <= cap:
                        estimates = table[trips.index(r.trip_count)]
                        predicted = np.where(
                            estimates[:, 0] <= headroom,
                            estimates[:, 0] + (h - 1) * estimates[:, 1] / 1000,
                            np.inf,
                        )
                        idx = int(predicted.argmin())
                    actual += elapsed[idx]
                    baseline += elapsed[0]
            relative = 1 - actual / baseline
            gain.append(relative + min(relative, 0))
            folds.append(dict(max_streams=cap, family=family, replay_saved_ms=baseline - actual))
        scores[cap] = float(np.mean(gain))
    cap = max(scores, key=scores.get)
    table = calibrate(rows, costs, cap)
    # With no positive grouped improvement this profile delegates every decision.
    if table is None:
        table = np.ones((2, len(FACTORS), 2))
    profile = dict(
        base=sha256(BUNDLED.read_bytes()).hexdigest(), max_streams=cap, trips=trips, costs=table.tolist()
    )
    (OUT / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    (OUT / "fit.json").write_text(json.dumps(dict(scores=scores, folds=folds), indent=2) + "\n")
    labels = rows[["workload", "family", "trip_count", "body_hash", "streams"]].copy()
    for i, factor in enumerate(FACTORS):
        labels[f"first_ms_{factor}"], labels[f"warm_us_{factor}"] = costs[:, i].T
    labels.to_csv(OUT / "training_labels.csv", index=False)
    print("mean grouped utility", scores, "selected", cap)


if __name__ == "__main__":
    main()
