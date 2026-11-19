"""Freeze whole workload families, collect native labels and load a single split."""

import argparse
import csv
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import jax
import numpy as np

from benchmarks.corpus.fori_agent_cases.workloads import CASES
from benchmarks.corpus.fori_regret.acquire import REPEATS, measure, run_workers
from jax_unroll.features import extract_features
from jax_unroll.schema import FACTORS

OUT = Path("results/fori_agent_cases")
MANIFEST = OUT / "split.json"
SIZES = ("small", "large")
TRIPS = (255, 1024)
SPLITS = ("train", "eval")


def settings(split):
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return list(product((case for case in CASES if case.partition == split), SIZES, TRIPS))


def historical_keys():
    """Read identifying fields without using prior evaluation labels for selection."""
    hashes, names = set(), set()
    for pattern in ("fori_*/training_keys.csv", "fori_expansion/*features.csv", "fori_targeted/*features.csv"):
        for path in Path("results").glob(pattern):
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    names.add(row["workload"])
                    hashes.update(row.get(k) for k in ("body_hash", "current_body_hash"))
    for pattern in ("fori_prediction/*_runs.json", "fori_regret/train_runs.json", "fori_regret/holdout_runs.json",
                    "fori_proposal/native_runs.json", "fori_iterations/confirmation_native.json",
                    "fori_horizon/**/sweep.json"):
        for path in Path("results").glob(pattern):
            parts = json.loads(path.read_text())
            rows = parts["runs"][0] if isinstance(parts, dict) else parts
            for row in rows:
                names.add(row["workload"])
                hashes.add(row["features"]["body_hash"])
    return names, hashes


def sources():
    folder = Path(__file__).parent
    paths = {folder / "workloads.py", *(folder / (case.factory.__module__.split(".")[-1] + ".py") for case in CASES)}
    return {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def freeze():
    names, previous = historical_keys()
    if names.intersection(c.name for c in CASES) or len({c.name for c in CASES}) != len(CASES):
        raise ValueError("new workload names must be unique")
    seen_families, seen_bodies, rows = {}, {}, []
    for split in SPLITS:
        for case, size, trip in settings(split):
            if seen_families.setdefault(case.family, split) != split:
                raise ValueError(f"family crosses partitions: {case.family}")
            build = case.build(trip, 1, size)
            features = extract_features(build.fn, jax.device_put(build.args), trip_count=trip, factor=1,
                                        loop_kind="fori_loop")
            body = features["body_hash"]
            if body in previous or seen_bodies.setdefault(body, case.name) != case.name:
                raise ValueError(f"duplicate loop IR: {case.name}, {size}, {trip}")
            rows.append(dict(workload=case.name, family=case.family, partition=split, variant=size,
                             trip_count=trip, body_hash=body, author=case.factory.__module__.split(".")[-1]))
    data = dict(configurations=rows, sha256=sources(), factors=list(FACTORS),
                sizes=list(SIZES), trips=list(TRIPS), repeats=REPEATS)
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text()) != data:
            raise ValueError("frozen manifest changed")
    else:
        OUT.mkdir(exist_ok=True, parents=True)
        with MANIFEST.open("x") as stream:
            stream.write(json.dumps(data, indent=2) + "\n")


def check_manifest():
    data = json.loads(MANIFEST.read_text())
    if (data["sha256"] != sources() or data["factors"] != list(FACTORS) or data["sizes"] != list(SIZES)
            or data["trips"] != list(TRIPS) or data["repeats"] != REPEATS):
        raise ValueError("sources or acquisition settings changed after freeze")
    return data


def load(split):
    """Return factor-1 features and [config, factor, (first_ms, warm_us)] labels."""
    import pandas as pd
    from benchmarks.corpus.fori_horizon.replay import curves

    settings(split)
    expected = {(r["workload"], r["variant"], r["trip_count"]): r
                for r in check_manifest()["configurations"] if r["partition"] == split}
    features, costs = [], []
    for row, cold, warm in curves(OUT / f"{split}_runs.json"):
        k = (row["workload"], row["variant"], row["trip_count"])
        record = expected.pop(k)
        if row["features"]["body_hash"] != record["body_hash"] or row["family"] != record["family"]:
            raise ValueError(f"timed a different case: {k}")
        features.append({**row["features"], **record, "source": "agent_cases"})
        costs.append(np.column_stack((np.median(cold, axis=0), np.median(warm, axis=0))))
    if expected:
        raise ValueError(f"incomplete {split} acquisition")
    return pd.DataFrame(features), np.stack(costs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", *SPLITS, "verify"))
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "verify":
        for split in SPLITS:
            frame, labels = load(split)
            print(f"{split}: {len(frame)} configurations, {frame.family.nunique()} families, labels {labels.shape}")
    else:
        check_manifest()
        if args.worker is not None:
            print(json.dumps(measure(settings(args.stage), args.worker)))
        else:
            run_workers(__spec__.name, [args.stage], REPEATS, OUT / f"{args.stage}_runs.json")
            load(args.stage)


if __name__ == "__main__":
    main()
