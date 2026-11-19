"""Serial acquisition, dataset aggregation, and export for fori_loop corpus expansion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from benchmarks.corpus.all_workloads import dataset as load_historical_base
from benchmarks.corpus.fori_expansion.registry import (
    ALL_FORI_SPECS,
    CONVERTED_WORKLOAD_SPECS,
    HISTORICAL_FORI_NAMES,
    HISTORICAL_FORI_SPECS,
    NEW_INDEPENDENT_SPECS,
    build_workload,
    get_spec,
)
from benchmarks.dataset import (
    DATASET_KEY_COLUMNS,
    DATASET_LABEL_COLUMNS,
    GROUP_KEY,
    aggregate_labels,
    build_feature_table,
)
from benchmarks.oracle import load_raw
from benchmarks.sweep import SWEEP_COLUMNS
from jax_unroll.schema import FACTORS

RAW_PATH = Path("results/fori_expansion/raw_sweep.csv")
TRAIN_PATH = Path("results/fori_expansion/train_features.csv")
EVAL_PATH = Path("results/fori_expansion/eval_features.csv")
SPLIT_CSV = Path("benchmarks/corpus/fori_expansion/split.csv")
MANIFEST_JSON = Path("benchmarks/corpus/fori_expansion/manifest.json")

# Representative bodies across mechanisms and partitions for depth coverage
DEPTH_WORKLOADS: tuple[str, ...] = (
    "sig_cascaded_biquad",
    "opt_coordinate_descent",
    "sp_csr_matvec_row",
    "graph_edge_message_passing",
    "dp_edit_distance_row",
    "sim_finite_volume_burgers",
    "tiled_online_softmax",
    "idx_double_indirect_gather",
    "stream_count_min_sketch",
    "sched_priority_queue_step",
    "conv_dense_matvec",
    "conv_elementwise_sine",
    "conv_indexed_update",
    "conv_rolling_mean_shift",
)


def load_existing_raw_keys(path: Path) -> set[tuple[str, str, int, int, int]]:
    """Return set of (workload, variant, trip_count, factor, process_repeat) already measured."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        keys = set()
        for _, row in df.iterrows():
            keys.add((
                str(row["workload"]),
                str(row["variant"]),
                int(row["trip_count"]),
                int(row["factor"]),
                int(row["process_repeat"]),
            ))
        return keys
    except Exception:
        return set()


def plan_acquisition_matrix(repeats: int = 3) -> list[dict[str, Any]]:
    """Generate configuration matrix for breadth (trips 64, 255) and depth coverage."""
    configs: list[dict[str, Any]] = []

    # 1. Breadth matrix: All 123 bodies (70 new + 53 converted) at primary shape, trips (64, 255), 4 factors
    breadth_specs = [s for s in ALL_FORI_SPECS if s.name not in HISTORICAL_FORI_NAMES]
    for spec in breadth_specs:
        var = spec.primary_variant
        for trip in (64, 255):
            for factor in FACTORS:
                for rep in range(repeats):
                    configs.append({
                        "workload": spec.name,
                        "variant": var,
                        "trip_count": trip,
                        "factor": factor,
                        "process_repeat": rep,
                    })

    # 2. Depth matrix: representative bodies with secondary variants or longer trips
    for name in DEPTH_WORKLOADS:
        spec = get_spec(name)
        sec_variants = [v for v in sorted(spec.shape_variants) if v != spec.primary_variant]
        if sec_variants:
            sec_var = sec_variants[0]
            for trip in (64, 255):
                for factor in FACTORS:
                    for rep in range(repeats):
                        configs.append({
                            "workload": spec.name,
                            "variant": sec_var,
                            "trip_count": trip,
                            "factor": factor,
                            "process_repeat": rep,
                        })
        # Longer trips for representative bodies
        for long_trip in (512, 1023):
            for factor in FACTORS:
                for rep in range(repeats):
                    configs.append({
                        "workload": spec.name,
                        "variant": spec.primary_variant,
                        "trip_count": long_trip,
                        "factor": factor,
                        "process_repeat": rep,
                    })

    # Deduplicate configs
    unique_configs = {}
    for c in configs:
        k = (c["workload"], c["variant"], c["trip_count"], c["factor"], c["process_repeat"])
        unique_configs[k] = c
    return list(unique_configs.values())


def run_serial_acquisition(
    configs: list[dict[str, Any]],
    out_path: Path = RAW_PATH,
    max_count: int | None = None,
) -> int:
    """Run serial CPU timing on all planned configurations, resuming automatically."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = load_existing_raw_keys(out_path)

    pending = [
        c for c in configs
        if (c["workload"], c["variant"], c["trip_count"], c["factor"], c["process_repeat"]) not in existing_keys
    ]

    print(
        f"Acquisition: total={len(configs)}, already_measured={len(existing_keys)}, "
        f"pending={len(pending)}, out={out_path}"
    )

    if max_count is not None:
        pending = pending[:max_count]

    if not pending:
        print("All planned configurations already measured!")
        return 0

    mode = "a" if out_path.exists() and out_path.stat().st_size > 0 else "w"
    with out_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_COLUMNS, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

        total = len(pending)
        started = time.perf_counter()
        for idx, config in enumerate(pending, start=1):
            cmd = [
                sys.executable,
                "-m",
                "benchmarks.sweep",
                "--single-json",
                json.dumps(config),
                "--warmup",
                "3",
                "--sample-budget-ms",
                "50",
                "--min-samples",
                "10",
                "--max-samples",
                "1000",
                "--r-calls",
                "100",
                "--workload-module",
                "benchmarks.corpus.fori_expansion.registry",
            ]
            env = os.environ.copy()
            env["JAX_PLATFORMS"] = "cpu"

            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            elapsed = time.perf_counter() - t0

            if proc.returncode != 0:
                print(f"[{idx}/{total}] ERROR {config['workload']}: {proc.stderr[:120]}")
                continue

            # Extract row JSON from stdout
            row = None
            for line in reversed(proc.stdout.splitlines()):
                if line.startswith("ROW_JSON:"):
                    row = json.loads(line[len("ROW_JSON:"):])
                    break

            if row:
                writer.writerow(row)
                handle.flush()
                warm = row.get("warm_median_us", "n/a")
                if idx % 10 == 0 or idx == total or idx <= 5:
                    print(
                        f"[{idx}/{total}] r{config['process_repeat']} {config['workload']}/{config['variant']} "
                        f"T={config['trip_count']} u={config['factor']} warm={warm} us ({elapsed:.2f}s)"
                    )

        wall_min = (time.perf_counter() - started) / 60.0
        print(f"Acquisition finished {total} measurements in {wall_min:.1f} min.")
    return 0


def _load_historical_fori_all_factors() -> pd.DataFrame:
    """Load all 140 historical fori_loop rows across all 4 factors (1, 2, 4, 8)."""
    from benchmarks.recovery_data import LEGACY_DATA, JOIN_KEY
    from benchmarks.corpus.registry import build_workload as hist_workload_builder
    tables = [pd.read_csv(p) for p in (
        LEGACY_DATA,
        "data/recovery/dev.csv",
        "data/recovery/confirmation.csv",
        "results/static_features/augmented_features.csv",
    )]
    paths = [
        Path("results/raw/corpus_smoke_cpu_2026-11-17.csv"),
        *sorted(Path("results/profile_transfer").glob("*_raw.csv")),
    ]
    labels = aggregate_labels(load_raw(paths))
    tables.append(build_feature_table(labels, workload_builder=hist_workload_builder).merge(labels, on=JOIN_KEY))
    all_historical = pd.concat(tables, ignore_index=True)
    hist_fori = all_historical[all_historical["workload"].isin(HISTORICAL_FORI_NAMES)].copy()
    hist_fori = hist_fori.drop_duplicates(subset=[c for c in DATASET_KEY_COLUMNS if c in hist_fori.columns])
    return hist_fori


def build_and_export_datasets(
    raw_path: Path = RAW_PATH,
    split_csv: Path = SPLIT_CSV,
    train_out: Path = TRAIN_PATH,
    eval_out: Path = EVAL_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate raw timing rows, incorporate historical fori rows, join features, and export."""
    print("Loading and aggregating raw timings...")
    raw_df = pd.read_csv(raw_path)

    # 1. Filter valid rows
    error_mask = raw_df["error"].fillna("").astype(str).str.strip().eq("")
    correct_mask = raw_df["correct"].astype(str).str.lower().isin(("true", "1"))
    valid_raw = raw_df[error_mask & correct_mask].copy()
    print(f"Loaded {len(valid_raw)} valid raw rows from {raw_path}")

    # 2. Aggregate raw labels
    fresh_labels = aggregate_labels(valid_raw)
    print(f"Aggregated fresh labels: {len(fresh_labels)} rows across {fresh_labels['workload'].nunique()} workloads")

    # 3. Extract factor-1 static features for all fresh configurations
    print("Extracting factor-1 static features from loop body Jaxprs...")
    fresh_features = build_feature_table(fresh_labels, workload_builder=build_workload)

    # 4. Merge features and labels
    merge_keys = [*DATASET_KEY_COLUMNS, "shape"]
    merged_fresh = fresh_features.merge(fresh_labels, on=merge_keys, how="inner")
    print(f"Merged fresh feature table: {merged_fresh.shape}")

    # 5. Load compatible historical fori_loop rows across all 4 factors
    print("Loading compatible historical fori_loop rows...")
    hist_fori = _load_historical_fori_all_factors()
    print(f"Historical fori rows available: {len(hist_fori)} across {hist_fori['workload'].nunique()} workloads")

    # Align columns
    common_cols = [c for c in merged_fresh.columns if c in hist_fori.columns]
    combined_rows = pd.concat([merged_fresh[common_cols], hist_fori[common_cols]], ignore_index=True)
    combined_rows = combined_rows.drop_duplicates(subset=[c for c in DATASET_KEY_COLUMNS if c in combined_rows.columns])

    # 6. Verify loop_kind is strictly fori_loop on 100% of rows
    assert (combined_rows["loop_kind"] == "fori_loop").all(), "Found non-fori_loop rows in dataset!"

    # 7. Map partitions from official split
    split_df = pd.read_csv(split_csv)
    partition_map = dict(zip(split_df["workload"], split_df["partition"]))
    combined_rows["partition"] = combined_rows["workload"].map(partition_map)

    # Split into train and eval tables
    train_table = combined_rows[combined_rows["partition"] == "train"].copy().reset_index(drop=True)
    eval_table = combined_rows[combined_rows["partition"].isin(["historical_eval", "fresh_eval"])].copy().reset_index(drop=True)

    # Export
    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_table.to_csv(train_out, index=False)
    eval_table.to_csv(eval_out, index=False)

    print(f"Exported training table: {train_out} ({len(train_table)} rows, {train_table['workload'].nunique()} workloads)")
    print(f"Exported evaluation table: {eval_out} ({len(eval_table)} rows, {eval_table['workload'].nunique()} workloads)")
    print(f"  historical_eval: {len(eval_table[eval_table['partition'] == 'historical_eval'])} rows")
    print(f"  fresh_eval: {len(eval_table[eval_table['partition'] == 'fresh_eval'])} rows")

    return train_table, eval_table


if __name__ == "__main__":
    matrix = plan_acquisition_matrix(repeats=3)
    print(f"Planned matrix size: {len(matrix)} raw configs")
    run_serial_acquisition(matrix)
    build_and_export_datasets()
