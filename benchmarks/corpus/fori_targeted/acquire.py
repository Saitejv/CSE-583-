"""Serial acquisition, dataset aggregation, and export for targeted fori_loop collection.

Implements the protocol from docs/gemini_fori_targeted_collection.md:
- 48 bodies x 2 sizes x 4 trip counts (255, 256, 1023, 1024) x 4 factors x 3 repeats
- Randomized execution order with 3 passes
- Resume support without duplicate rows
- Development and sealed evaluation table export
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from benchmarks.corpus.fori_targeted.registry import (
    ALL_SPECS,
    TRAIN_SPECS,
    EVAL_SPECS,
    build_workload,
    get_spec,
)
from benchmarks.corpus.fori_targeted.split import (
    compute_split_hash,
    generate_split_csv,
    validate_split,
)
from benchmarks.dataset import (
    DATASET_KEY_COLUMNS,
    GROUP_KEY,
    aggregate_labels,
    build_feature_table,
)
from benchmarks.sweep import SWEEP_COLUMNS
from jax_unroll.schema import FACTORS

# File paths
RAW_PATH = Path("results/fori_targeted/raw_sweep.csv")
TRAIN_PATH = Path("results/fori_targeted/train_features.csv")
EVAL_PATH = Path("results/fori_targeted/sealed_eval_features.csv")
SPLIT_CSV = Path("benchmarks/corpus/fori_targeted/split.csv")
MANIFEST_JSON = Path("benchmarks/corpus/fori_targeted/manifest.json")
MATRIX_JSON = Path("results/fori_targeted/planned_matrix.json")

TRIP_COUNTS = (255, 256, 1023, 1024)
SIZES = ("small", "large")
REPEATS = 3
SEED = 583_42


def plan_acquisition_matrix() -> list[dict[str, Any]]:
    """Generate the full acquisition matrix: 48 bodies x 2 sizes x 4 trips x 4 factors x 3 repeats."""
    configs: list[dict[str, Any]] = []
    for spec in ALL_SPECS:
        for variant in SIZES:
            if variant not in spec.shape_variants:
                continue
            for trip in TRIP_COUNTS:
                for factor in FACTORS:
                    for rep in range(REPEATS):
                        configs.append({
                            "workload": spec.name,
                            "variant": variant,
                            "trip_count": trip,
                            "factor": factor,
                            "process_repeat": rep,
                        })
    return configs


def load_existing_raw_keys(path: Path) -> set[tuple[str, str, int, int, int]]:
    """Return set of already-measured keys."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        return {
            (str(row["workload"]), str(row["variant"]), int(row["trip_count"]),
             int(row["factor"]), int(row["process_repeat"]))
            for _, row in df.iterrows()
        }
    except Exception:
        return set()


def randomize_configs(
    configs: list[dict[str, Any]],
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Deterministic randomized order with 3 passes separating repeats.

    Within each pass, the four factors for each (body, size, trip) block are
    randomized together, and blocks are shuffled. Repeats of the same body
    are spread across the three passes.
    """
    rng = random.Random(seed)

    # Group configs by repeat
    by_repeat: dict[int, list[dict]] = {r: [] for r in range(REPEATS)}
    for c in configs:
        by_repeat[c["process_repeat"]].append(c)

    ordered = []
    for rep in range(REPEATS):
        rep_configs = by_repeat[rep]
        # Group into blocks of 4 (one per factor) for each (body, size, trip)
        blocks: dict[tuple, list] = {}
        for c in rep_configs:
            key = (c["workload"], c["variant"], c["trip_count"])
            blocks.setdefault(key, []).append(c)

        # Shuffle factor order within each block
        block_list = list(blocks.values())
        for block in block_list:
            rng.shuffle(block)
        # Shuffle block order
        rng.shuffle(block_list)
        for block in block_list:
            ordered.extend(block)

    return ordered


ACQUISITION_LOG = Path("results/fori_targeted/acquisition_log.csv")
HARDWARE_JSON = Path("results/fori_targeted/hardware_env.json")


def save_hardware_env():
    """Record Apple M2 CPU hardware and environment details."""
    import platform
    jax_ver = "0.12.0.1"
    devices = "cpu:0"
    try:
        import jax
        jax_ver = jax.__version__
        devices = str(jax.devices())
    except Exception:
        pass

    info = {
        "hardware": "Apple M2",
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "backend": "cpu",
        "jax_version": jax_ver,
        "devices": devices,
        "recorded_at": datetime.now().isoformat(),
    }
    HARDWARE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with HARDWARE_JSON.open("w") as f:
        json.dump(info, f, indent=2)


def run_serial_acquisition(
    configs: list[dict[str, Any]],
    out_path: Path = RAW_PATH,
    max_count: int | None = None,
    smoke: bool = False,
) -> int:
    """Run serial CPU timing on all planned configurations, resuming automatically."""
    save_hardware_env()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = load_existing_raw_keys(out_path)

    pending = [
        c for c in configs
        if (c["workload"], c["variant"], c["trip_count"], c["factor"], c["process_repeat"])
        not in existing_keys
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
    log_mode = "a" if ACQUISITION_LOG.exists() and ACQUISITION_LOG.stat().st_size > 0 else "w"
    
    log_file = ACQUISITION_LOG.open(log_mode, newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if log_mode == "w":
        log_writer.writerow(["order_idx", "pass_idx", "timestamp", "seed", "workload", "variant",
                             "trip_count", "factor", "process_repeat", "elapsed_s", "warm_median_us", "correct"])

    with out_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_COLUMNS, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

        total = len(pending)
        started = time.perf_counter()
        errors = 0
        for idx, config in enumerate(pending, start=1):
            warmup = "1" if smoke else "2"
            budget = "15" if smoke else "25"
            min_s = "5" if smoke else "10"
            max_s = "50" if smoke else "200"
            r_calls = "10" if smoke else "50"
            cmd = [
                sys.executable,
                "-m",
                "benchmarks.sweep",
                "--single-json",
                json.dumps(config),
                "--warmup", warmup,
                "--sample-budget-ms", budget,
                "--min-samples", min_s,
                "--max-samples", max_s,
                "--r-calls", r_calls,
                "--workload-module",
                "benchmarks.corpus.fori_targeted.registry",
            ]
            env = os.environ.copy()
            env["JAX_PLATFORMS"] = "cpu"

            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            elapsed = time.perf_counter() - t0

            if proc.returncode != 0:
                print(f"[{idx}/{total}] ERROR {config['workload']}: {proc.stderr[:200]}")
                errors += 1
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
                correct = row.get("correct", False)
                ts = datetime.now().isoformat()
                pass_idx = config["process_repeat"]
                log_writer.writerow([idx, pass_idx, ts, SEED, config["workload"], config["variant"],
                                     config["trip_count"], config["factor"], config["process_repeat"],
                                     f"{elapsed:.3f}", warm, correct])
                log_file.flush()

                if idx % 50 == 0 or idx == total or idx <= 5:
                    pct = idx / total * 100
                    eta = (time.perf_counter() - started) / idx * (total - idx) / 60.0
                    print(
                        f"[{idx}/{total} {pct:.0f}%] r{config['process_repeat']} "
                        f"{config['workload']}/{config['variant']} "
                        f"T={config['trip_count']} u={config['factor']} warm={warm} us "
                        f"({elapsed:.2f}s, ETA {eta:.1f}min)"
                    )
            else:
                errors += 1
                print(f"[{idx}/{total}] NO ROW {config['workload']}: stdout empty")

        log_file.close()
        wall_min = (time.perf_counter() - started) / 60.0
        print(f"Acquisition finished {total} measurements ({errors} errors) in {wall_min:.1f} min.")
    return errors


def build_and_export_datasets(
    raw_path: Path = RAW_PATH,
    split_csv: Path = SPLIT_CSV,
    train_out: Path = TRAIN_PATH,
    eval_out: Path = EVAL_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate raw timings, extract features, partition, and export."""
    print("Loading and aggregating raw timings...")
    raw_df = pd.read_csv(raw_path)

    # Filter valid rows
    error_mask = raw_df["error"].fillna("").astype(str).str.strip().eq("")
    correct_mask = raw_df["correct"].astype(str).str.lower().isin(("true", "1"))
    valid_raw = raw_df[error_mask & correct_mask].copy()
    print(f"Valid raw rows: {len(valid_raw)} / {len(raw_df)}")

    # Aggregate labels (median across repeats)
    labels = aggregate_labels(valid_raw)
    print(f"Aggregated labels: {len(labels)} rows, {labels['workload'].nunique()} workloads")

    # Extract factor-1 static features
    print("Extracting factor-1 static features from loop body Jaxprs...")
    features = build_feature_table(labels, workload_builder=build_workload)

    # Merge features and labels
    merge_keys = [c for c in DATASET_KEY_COLUMNS if c in features.columns and c in labels.columns]
    if "shape" in features.columns and "shape" in labels.columns:
        merge_keys.append("shape")
    merged = features.merge(labels, on=merge_keys, how="inner")
    print(f"Merged feature table: {merged.shape}")

    # Verify all fori_loop
    assert (merged["loop_kind"] == "fori_loop").all(), "Found non-fori_loop rows!"

    # Map partitions from split
    split_df = pd.read_csv(split_csv)
    partition_map = dict(zip(split_df["workload"], split_df["partition"]))
    merged["partition"] = merged["workload"].map(partition_map)

    # Split into train and eval tables
    train_table = merged[merged["partition"] == "train"].copy().reset_index(drop=True)
    eval_table = merged[merged["partition"] == "sealed_eval"].copy().reset_index(drop=True)

    # Verify four-factor completeness
    for name, df in [("train", train_table), ("eval", eval_table)]:
        config_groups = df.groupby(["workload", "variant", "trip_count"])["factor"].apply(
            lambda s: tuple(sorted(s.unique()))
        )
        incomplete = config_groups[config_groups != tuple(FACTORS)]
        if len(incomplete) > 0:
            print(f"WARNING: {name} has {len(incomplete)} incomplete factor groups")

    # Export
    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_table.to_csv(train_out, index=False)
    eval_table.to_csv(eval_out, index=False)

    print(f"Exported train: {train_out} ({len(train_table)} rows, {train_table['workload'].nunique()} workloads)")
    print(f"Exported eval: {eval_out} ({len(eval_table)} rows, {eval_table['workload'].nunique()} workloads)")

    return train_table, eval_table


def freeze_matrix_and_split():
    """Freeze the split CSV and planned matrix JSON before timing."""
    print("=" * 60)
    print("FREEZING SPLIT AND MATRIX")
    print("=" * 60)

    # Generate split CSV
    records = generate_split_csv(ALL_SPECS, SPLIT_CSV)
    result = validate_split(records)
    print(f"Split validation: {result}")

    # Generate planned matrix
    matrix = plan_acquisition_matrix()
    randomized = randomize_configs(matrix)

    MATRIX_JSON.parent.mkdir(parents=True, exist_ok=True)
    with MATRIX_JSON.open("w") as f:
        json.dump({
            "planned_configs": len(matrix),
            "planned_raw_rows": len(matrix),
            "bodies": len(ALL_SPECS),
            "train_bodies": len(TRAIN_SPECS),
            "eval_bodies": len(EVAL_SPECS),
            "sizes": list(SIZES),
            "trip_counts": list(TRIP_COUNTS),
            "factors": list(FACTORS),
            "repeats": REPEATS,
            "seed": SEED,
            "split_hash": result["split_hash"],
            "frozen_at": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"Split CSV: {SPLIT_CSV}")
    print(f"Planned matrix: {MATRIX_JSON}")
    print(f"Split hash: {result['split_hash']}")
    print(f"Total planned measurements: {len(matrix)}")
    print(f"  = {len(ALL_SPECS)} bodies x {len(SIZES)} sizes x {len(TRIP_COUNTS)} trips x {len(FACTORS)} factors x {REPEATS} repeats")

    return randomized


def main():
    parser = argparse.ArgumentParser(description="Targeted fori_loop acquisition")
    parser.add_argument("--freeze-only", action="store_true", help="Only freeze split and matrix")
    parser.add_argument("--export-only", action="store_true", help="Only build and export datasets")
    parser.add_argument("--smoke", action="store_true", help="Smoke test with reduced timing")
    parser.add_argument("--max-count", type=int, default=None, help="Max measurements")
    args = parser.parse_args()

    if args.freeze_only:
        freeze_matrix_and_split()
        return

    if args.export_only:
        build_and_export_datasets()
        return

    # Full pipeline: freeze -> acquire -> export
    randomized = freeze_matrix_and_split()
    print()
    print("=" * 60)
    print("RUNNING SERIAL ACQUISITION")
    print("=" * 60)
    errors = run_serial_acquisition(randomized, max_count=args.max_count, smoke=args.smoke)
    print()
    print("=" * 60)
    print("BUILDING AND EXPORTING DATASETS")
    print("=" * 60)
    build_and_export_datasets()


if __name__ == "__main__":
    main()
