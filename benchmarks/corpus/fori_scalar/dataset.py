"""Labeled incremental training data and a separate family-level confirmation split."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.corpus.fori_application import scan_ports
from benchmarks.corpus.fori_horizon.replay import curves
from benchmarks.corpus.fori_regret.acquire import REPEATS, measure, run_workers
from benchmarks.corpus.fori_scalar import cases
from jax_unroll.features import extract_features
from jax_unroll.schema import FACTORS

OUT = Path("results/fori_scalar")
SOURCE = scan_ports.OUT / "scan_ports_runs.json"


def freeze(create=False):
    rows = []
    for split in ("train", "test"):
        configs = cases.settings(split)
        if split == "train":
            configs = scan_ports.settings() + configs
        for spec, size, trip in configs:
            fn, args = spec.build(trip, 1, size)
            row = extract_features(fn, args, loop_kind="fori_loop", trip_count=trip, factor=1)
            rows.append(dict(split=split, workload=spec.name, family=spec.family, variant=size,
                             trip_count=trip, body_hash=row["body_hash"]))
    manifest = dict(rows=rows, factors=FACTORS, repeats=REPEATS, calls=[300, 5000, 20000],
                    source_sha256=sha256(SOURCE.read_bytes()).hexdigest(),
                    cases_sha256=sha256(Path(cases.__file__).read_bytes()).hexdigest())
    path = OUT / "split.json"
    encoded = json.dumps(manifest, indent=2) + "\n"
    if create and not path.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        with path.open("x") as f:
            f.write(encoded)
    if path.read_text() != encoded:
        raise ValueError("scalar split or source measurements changed")
    return rows


def load(split):
    """Return features and [case, factor, (first_ms, warm_us)] median measurements."""
    expected = {(r["workload"], r["variant"], r["trip_count"]): r
                for r in freeze() if r["split"] == split}
    paths = [OUT / f"{split}_runs.json"]
    if split == "train":
        paths.insert(0, SOURCE)
    features, targets = [], []
    for path in paths:
        for r, cold, warm in curves(path):
            key = r["workload"], r["variant"], r["trip_count"]
            rec = expected.pop(key)
            if r["features"]["body_hash"] != rec["body_hash"] or r["family"] != rec["family"]:
                raise ValueError(f"incorrect measurement for {key}")
            features.append({**r["features"], **rec})
            targets.append(np.column_stack((np.median(cold, axis=0), np.median(warm, axis=0))))
    if expected:
        raise ValueError(f"missing measurements for {list(expected)}")
    return pd.DataFrame(features), np.stack(targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "train", "test"))
    parser.add_argument("--worker", type=int)
    opts = parser.parse_args()
    freeze(create=opts.stage == "freeze")
    if opts.worker is not None:
        print(json.dumps(measure(cases.settings(opts.stage), opts.worker)))
    elif opts.stage != "freeze":
        run_workers(__spec__.name, [opts.stage], REPEATS, OUT / f"{opts.stage}_runs.json")
