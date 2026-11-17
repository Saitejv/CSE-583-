"""Validate complete recovery measurements and join the original static features."""

import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import jax
import jaxlib
import numpy as np
import pandas as pd

from benchmarks.dataset import aggregate_labels, build_feature_table
from benchmarks.oracle import load_raw
from jax_unroll.model import group_configs, read_rows
from jax_unroll.recovery_split import PARTITIONS, PINNED_JAX, REPEATS, TRIPS, VARIANTS, check_rows, manifest
from jax_unroll.schema import DATASET_COLUMNS, DATASET_KEY_COLUMNS, DATASET_VERSION, FACTORS, FEATURE_VERSION

DATA = Path("data/recovery")
RESULTS = Path("results/recovery")
MODEL = Path("models/cost_cpu_recovery_v1.json")
LEGACY_DATA = Path("data/features_cpu_2026-11-17.csv")
LEGACY_MODEL = Path("models/cost_cpu.json")
PROTOCOL = Path("docs/recovery_protocol.md")
VERSIONS = {"jax_version": PINNED_JAX, "jaxlib_version": PINNED_JAX}
CONFIG = ("workload", "variant", "trip_count")
JOIN_KEY = [*DATASET_KEY_COLUMNS, "shape"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(fields), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_cpu():
    if jax.default_backend() != "cpu" or (jax.__version__, jaxlib.__version__) != (PINNED_JAX, PINNED_JAX):
        raise ValueError(f"requires CPU with JAX and jaxlib {PINNED_JAX}")


def config_key(row):
    return (row["workload"], row["variant"], int(row["trip_count"]))


def complete_configs(raw, partition):
    members = PARTITIONS[partition]
    expected = set(product(members, VARIANTS, TRIPS))
    expected_pairs = set(product(FACTORS, range(REPEATS)))
    groups = dict(tuple(raw.groupby(list(CONFIG))))
    if set(groups) - expected:
        raise ValueError("raw sweep contains configurations outside its frozen partition")
    good, dropped = [], []
    for key in sorted(expected):
        rows = groups.get(key)
        if rows is None:
            dropped.append({"configuration": key, "reason": "missing configuration"})
            continue
        pairs = set(zip(rows["factor"], rows["process_repeat"], strict=True))
        costs = rows[["cold_total_ms", "warm_median_us"]].to_numpy(dtype=float)
        if len(rows) != len(expected_pairs) or pairs != expected_pairs:
            reason = "missing, extra or duplicate factor repeat"
        elif not rows["correct"].astype(str).str.lower().eq("true").all() or rows["error"].fillna("").ne("").any():
            reason = "sweep error or correctness failure"
        elif not (np.isfinite(costs).all() and (costs > 0).all()):
            reason = "invalid timing label"
        else:
            good.append(rows)
            continue
        dropped.append({"configuration": key, "reason": reason})
    return good, dropped


def prepare(raw_path, partition, out):
    require_cpu()
    raw = load_raw(raw_path)
    members = PARTITIONS[partition]
    fields = {"backend": "cpu", "device": "cpu:0", "dtype": "float32", **VERSIONS}
    measured = raw[raw["error"].fillna("").eq("")]
    if any(not measured[field].eq(expected).all() for field, expected in fields.items()):
        raise ValueError("mixed, unsupported, or unpinned sweep runtime")
    if not raw["workload"].map(members).eq(raw["family"]).all():
        raise ValueError("raw sweep contains a workload outside its frozen partition")
    good, dropped = complete_configs(raw, partition)
    if good:
        labels = aggregate_labels(pd.concat(good, ignore_index=True))
        table = build_feature_table(labels).merge(labels, on=JOIN_KEY, validate="one_to_one")
        table["split"] = partition
        table["extraction_ms"] = None
        table["B"] = table["R"] = None
        rows = table.astype(object).where(pd.notna(table), "").to_dict("records")
    else:
        rows = []
    if partition == "dev":
        rows.extend({**row, "extraction_ms": ""} for row in read_rows(LEGACY_DATA, split="dev"))
    rows.sort(key=lambda row: (*config_key(row), int(row["factor"])))
    write_csv(out, rows, DATASET_COLUMNS)
    meta = {
        "dataset_version": DATASET_VERSION, "feature_version": FEATURE_VERSION, **VERSIONS,
        "split_manifest": manifest(), "protocol_sha256": sha(PROTOCOL), "partition": partition,
        "backend": "cpu", "device": "cpu:0", "configurations": len(rows) // len(FACTORS),
        "raw_rows": len(raw), "raw_sha256": sha(raw_path), "joined_sha256": sha(out), "dropped": dropped,
    }
    if partition == "dev":
        meta["legacy_csv_sha256"] = sha(LEGACY_DATA)
        meta["legacy_meta_sha256"] = sha(LEGACY_DATA.with_suffix(".meta.json"))
    dump(out.with_suffix(".meta.json"), meta)
    return meta


def joined_rows(path, partition):
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if (
        meta["partition"] != partition or meta["backend"] != "cpu" or meta["joined_sha256"] != sha(path)
        or meta["protocol_sha256"] != sha(PROTOCOL) or meta["feature_version"] != FEATURE_VERSION
        or meta["dataset_version"] != DATASET_VERSION or meta["split_manifest"] != json.loads(json.dumps(manifest()))
        or any(meta[field] != version for field, version in VERSIONS.items())
    ):
        raise ValueError("joined recovery data does not match its provenance")
    rows = [{**row, **VERSIONS} for row in read_csv(path)]
    check_rows(rows, partition)
    groups = group_configs(rows)
    if len({config_key(group[1]) for group in groups}) != len(groups) or len(groups) != meta["configurations"]:
        raise ValueError("joined data contains duplicate or missing configurations")
    if any(row["B"] or row["R"] or row["correct_all"] != "True" for row in rows):
        raise ValueError("joined data must be correctness-checked and omit policy context")
    return rows, meta
