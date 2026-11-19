"""Freeze researched workflows and collect paired native factor labels by partition."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.corpus.fori_horizon.replay import curves
from benchmarks.corpus.fori_regret.acquire import REPEATS, measure, run_workers
from benchmarks.corpus.fori_workflows.cases import settings
from jax_unroll.features import extract_features
from jax_unroll.schema import FACTORS

OUT = Path("results/fori_workflows")
CALLS = (300, 5000, 20000)


def freeze(create=False):
    rows = []
    for split in ("train", "test"):
        for spec, size, trip in settings(split):
            fn, args = spec.build(trip, 1, size)
            features = extract_features(fn, args, loop_kind="fori_loop", trip_count=trip, factor=1)
            rows.append(
                dict(
                    workload=spec.name,
                    family=spec.family,
                    split=split,
                    trip_count=trip,
                    variant=size,
                    body_hash=features["body_hash"],
                    inputs=[
                        [
                            list(np.shape(a)),
                            str(np.asarray(a).dtype),
                            hashlib.sha256(np.asarray(a).tobytes()).hexdigest(),
                        ]
                        for a in args
                    ],
                )
            )
    for key in ("family", "body_hash"):
        if {r[key] for r in rows if r["split"] == "train"} & {r[key] for r in rows if r["split"] == "test"}:
            raise ValueError(f"{key} crosses partitions")
    files = [*Path(__file__).parent.glob("*_cases.py"), Path(__file__).with_name("cases.py")]
    data = dict(
        rows=rows,
        source={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
        factors=FACTORS,
        calls=CALLS,
        repeats=REPEATS,
        promotion="Faster pooled totals against incumbent and native in all three 20k repeats; report every case.",
    )
    path = OUT / "split.json"
    encoded = json.dumps(data, indent=2) + "\n"
    if create and not path.exists():
        OUT.mkdir(exist_ok=True, parents=True)
        with path.open("x") as f:
            f.write(encoded)
    if path.read_text() != encoded:
        raise ValueError("changed workflow manifest")
    return rows


def load(split):
    if split not in ("train", "test"):
        raise ValueError(split)
    expected = {(r["workload"], r["variant"], r["trip_count"]): r for r in freeze() if r["split"] == split}
    rows, labels = [], []
    for r, cold, warm in curves(OUT / f"{split}_runs.json"):
        rec = expected.pop((r["workload"], r["variant"], r["trip_count"]))
        if r["features"]["body_hash"] != rec["body_hash"] or r["family"] != rec["family"]:
            raise ValueError("mismatched workflow")
        rows.append({**r["features"], **rec})
        labels.append(np.column_stack((np.median(cold, axis=0), np.median(warm, axis=0))))
    if expected:
        raise ValueError("incomplete measurements")
    return pd.DataFrame(rows), np.stack(labels)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze", "train", "test"))
    p.add_argument("--worker", type=int)
    args = p.parse_args()
    freeze(create=args.stage == "freeze")
    if args.worker is not None:
        print(json.dumps(measure(settings(args.stage), args.worker)))
    elif args.stage != "freeze":
        run_workers(__spec__.name, [args.stage], REPEATS, OUT / f"{args.stage}_runs.json")
