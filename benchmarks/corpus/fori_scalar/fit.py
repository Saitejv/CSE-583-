"""Update the compact artifact using the four supplied labels and two calibration families."""

import json
from pathlib import Path

import numpy as np

from benchmarks.corpus.fori_application import scan_ports
from benchmarks.corpus.fori_iterations.storage import UNCOMPRESSED
from benchmarks.corpus.fori_scalar import cases
from benchmarks.corpus.fori_scalar.dataset import OUT, load
from jax_unroll.compact_fori import CompactForiPredictor, is_scalar_stream, scalar_stream_features
from jax_unroll.features import extract_features
from jax_unroll.schema import FACTORS

BUNDLED = Path("src/jax_unroll/data/fori_cpu.npz")


def main():
    rows, costs = load("train")
    configs = {(s.name, v, t): s for s, v, t in scan_ports.settings() + cases.settings("train")}
    eligible = []
    for r in rows.itertuples():
        case = configs[r.workload, r.variant, r.trip_count].build(r.trip_count, 1, r.variant)
        features = extract_features(case.fn, case.args, trip_count=r.trip_count, factor=1,
                                    loop_kind="fori_loop", feature_block=scalar_stream_features)
        eligible.append(is_scalar_stream(features))
    keys = rows[["workload", "family", "variant", "trip_count", "body_hash", "split"]].copy()
    keys["scalar_profile"] = eligible
    for i, f in enumerate(FACTORS):
        keys[f"first_ms_{f}"], keys[f"warm_us_{f}"] = costs[:, i].T
    for h in (300, 5000, 20000):
        keys[f"oracle_{h}"] = np.take(FACTORS, (costs[:, :, 0] + (h - 1) * costs[:, :, 1] / 1000).argmin(axis=1))
    keys.to_csv(OUT / "training_labels.csv", index=False)

    trips = np.unique(rows.trip_count[eligible])
    tables = []
    for trip in trips:
        group = costs[np.asarray(eligible) & (rows.trip_count == trip)]
        # Bound cold cost by the most expensive family's median. Equal weight per
        # training case at this length prevents a single large warm time dominating.
        cold = group[:, :, 0].max(axis=0)
        warm = group[:, :, 1]
        warm = np.median(warm / warm[:, :1], axis=0) * np.median(warm[:, 0])
        tables.append(np.column_stack((cold, warm)))
    with np.load(UNCOMPRESSED, allow_pickle=False) as base:
        arrays = dict(base)
    tmp = BUNDLED.with_suffix(".tmp")
    with tmp.open("wb") as out:
        np.savez(out, **arrays, scalar_trips=trips, scalar_costs=tables)
    candidate = CompactForiPredictor.load(tmp)
    tmp.replace(BUNDLED)
    (OUT / "fit.json").write_text(json.dumps(dict(
        model=candidate.version, base=CompactForiPredictor.load(UNCOMPRESSED).version,
        training_cases=len(rows), scalar_cases=sum(map(int, eligible)), trips=trips.tolist(),
        scalar_costs=[t.tolist() for t in tables]), indent=2) + "\n")


if __name__ == "__main__":
    main()
