"""Join static loop features with measured sweep labels into a dataset table.

The builder reuses the oracle aggregation in ``benchmarks/oracle.py`` so the
dataset and oracle share one definition of a valid measurement. Feature
extraction only traces JAX loops; it never compiles or benchmarks a candidate
factor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from benchmarks.oracle import GROUP_KEY, aggregate_raw, load_raw
from benchmarks.workloads import build_workload
from jax_unroll.features import extract_features_from_workload
from jax_unroll.schema import (
    DATASET_COLUMNS,
    DATASET_KEY_COLUMNS,
    DATASET_LABEL_COLUMNS,
    DATASET_VERSION,
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    ensure_parent,
    git_commit,
    utc_timestamp,
)
from jax_unroll.splits import (
    DEV_FAMILIES,
    HELD_OUT_FAMILIES,
    SPLIT_VERSION,
    manifest as split_manifest,
    split_for_family,
)

__all__ = [
    "aggregate_labels",
    "build_dataset",
    "build_feature_table",
    "build_parser",
    "main",
    "run",
    "write_dataset",
]

_MERGE_KEY_COLUMNS: tuple[str, ...] = (*DATASET_KEY_COLUMNS, "shape")
_CONFIG_COLUMNS: tuple[str, ...] = tuple(
    column for column in _MERGE_KEY_COLUMNS if column != "factor"
)
_FEATURE_OUTPUT_COLUMNS: tuple[str, ...] = tuple(
    column for column in FEATURE_COLUMNS if column not in DATASET_KEY_COLUMNS
)

WorkloadBuilder = Callable[[str, int, int, str | None], Any]


def _valid_mask(frame: pd.DataFrame) -> pd.Series:
    error_ok = frame["error"].astype("string").fillna("").str.strip().eq("")
    correct_ok = (
        frame["correct"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(("true", "1", "1.0"))
    )
    return error_ok & correct_ok


def _valid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[_valid_mask(frame)]


def _log_positive(value: Any, column: str, key: Mapping[str, Any]) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{column} is not numeric for configuration {dict(key)!r}: {value!r}"
        ) from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(
            f"{column} must be finite and positive before taking a log for "
            f"configuration {dict(key)!r}; got {number!r}"
        )
    return math.log(number)


def _whole_number_series(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    if values.notna().all():
        as_array = values.to_numpy()
        if np.all(np.mod(as_array, 1.0) == 0.0):
            return values.astype(int)
    return values


def aggregate_labels(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per aggregated configuration/factor with dataset labels."""

    if "r_calls" not in raw.columns:
        raise ValueError("raw rows are missing required column 'r_calls'")
    valid = _valid_rows(raw).copy()
    aggregated = aggregate_raw(valid)

    key_columns = list(GROUP_KEY)
    r_calls = valid.loc[:, [*key_columns, "r_calls"]].copy()
    r_calls["r_calls"] = pd.to_numeric(r_calls["r_calls"], errors="coerce")
    r_calls_medians = (
        r_calls.groupby(key_columns, dropna=False, sort=False)["r_calls"]
        .median()
        .reset_index()
    )

    merged = aggregated.merge(
        r_calls_medians,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )
    merged["r_calls"] = _whole_number_series(merged["r_calls"])
    merged["correct_all"] = True

    log_cold: list[float] = []
    log_warm: list[float] = []
    for row in merged.itertuples(index=False):
        key = {column: getattr(row, column) for column in _MERGE_KEY_COLUMNS}
        log_cold.append(_log_positive(row.cold_median_ms, "cold_median_ms", key))
        log_warm.append(_log_positive(row.warm_median_us, "warm_median_us", key))
    merged["log_cold_ms"] = log_cold
    merged["log_warm_us"] = log_warm

    columns = [*DATASET_KEY_COLUMNS, "shape", *DATASET_LABEL_COLUMNS]
    return merged.reindex(columns=columns).reset_index(drop=True)


def _assert_unique(
    frame: pd.DataFrame,
    columns: Sequence[str],
    label: str,
) -> None:
    duplicated = frame.duplicated(subset=list(columns), keep=False)
    if duplicated.any():
        sample = (
            frame.loc[duplicated, list(columns)]
            .drop_duplicates()
            .head(3)
            .to_dict("records")
        )
        raise ValueError(f"{label} has duplicate keys after dedupe: {sample}")


def build_feature_table(
    aggregated: pd.DataFrame,
    *,
    workload_builder: WorkloadBuilder = build_workload,
) -> pd.DataFrame:
    """Extract static features for each aggregated configuration row."""

    missing = set(_MERGE_KEY_COLUMNS).difference(aggregated.columns)
    if missing:
        raise ValueError(
            f"aggregated rows are missing key columns: {sorted(missing)}"
        )

    working = aggregated.drop_duplicates(
        subset=list(_MERGE_KEY_COLUMNS), keep="first"
    )
    _assert_unique(working, _MERGE_KEY_COLUMNS, "aggregated input")

    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, int, int, str | None], dict[str, Any]] = {}
    for row in working.itertuples(index=False):
        workload_name = str(row.workload)
        variant = None if pd.isna(row.variant) else str(row.variant)
        trip_count = int(row.trip_count)
        factor = int(row.factor)
        cache_key = (workload_name, trip_count, factor, variant)

        features = cache.get(cache_key)
        if features is None:
            workload = workload_builder(
                workload_name, trip_count, factor, variant
            )
            features = extract_features_from_workload(workload)
            cache[cache_key] = features

        key = {column: getattr(row, column) for column in _MERGE_KEY_COLUMNS}
        if features["loop_kind"] != str(row.loop_kind):
            raise ValueError(
                f"extracted loop_kind {features['loop_kind']!r} does not match "
                f"raw row {row.loop_kind!r} for configuration {key!r}"
            )
        if int(features["trip_count"]) != trip_count:
            raise ValueError(
                f"extracted trip_count {features['trip_count']!r} does not "
                f"match raw row {trip_count!r} for configuration {key!r}"
            )
        if int(features["factor"]) != factor:
            raise ValueError(
                f"extracted factor {features['factor']!r} does not match raw "
                f"row {factor!r} for configuration {key!r}"
            )

        output = {column: getattr(row, column) for column in _MERGE_KEY_COLUMNS}
        output.update(
            {column: features[column] for column in _FEATURE_OUTPUT_COLUMNS}
        )
        rows.append(output)

    return pd.DataFrame(
        rows,
        columns=[*_MERGE_KEY_COLUMNS, *_FEATURE_OUTPUT_COLUMNS],
    ).reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_backend(raw: pd.DataFrame, backend: str | None) -> str:
    values = sorted(
        {str(value) for value in raw["backend"].dropna().unique()}
    )
    if backend is not None:
        if backend not in values:
            raise ValueError(
                f"backend {backend!r} is not present in the raw rows; "
                f"available backends: {values}"
            )
        return backend
    if len(values) != 1:
        raise ValueError(
            f"raw rows span multiple backends {values}; pass backend=... or "
            "--backend to select one"
        )
    return values[0]


def _consistent_value(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    values = sorted(
        {
            str(value).strip()
            for value in frame[column].dropna()
            if str(value).strip()
        }
    )
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(
            f"raw rows span multiple {column} values {values}; mixed versions "
            "must not be combined in one dataset"
        )
    return values[0]


def _device_value(frame: pd.DataFrame) -> str | list[str] | None:
    if "device" not in frame.columns:
        return None
    values = sorted(
        {
            str(value).strip()
            for value in frame["device"].dropna()
            if str(value).strip()
        }
    )
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _metadata(
    table: pd.DataFrame,
    *,
    raw: pd.DataFrame,
    raw_files: list[dict[str, Any]],
    valid_rows: pd.DataFrame,
    backend: str,
    device: str | list[str] | None,
    jax_version: str | None,
    jaxlib_version: str | None,
) -> dict[str, Any]:
    extraction_ms = table["extraction_ms"].astype(float).to_numpy()
    extraction_stats = {
        "mean": float(np.mean(extraction_ms)),
        "median": float(np.median(extraction_ms)),
        "p95": float(np.percentile(extraction_ms, 95)),
    }

    r_calls_values = sorted(
        {_jsonable(value) for value in table["r_calls"].dropna().unique()}
    )
    r_calls = (
        r_calls_values[0] if len(r_calls_values) == 1 else r_calls_values
    )

    n_raw_rows = int(len(raw))
    n_valid_rows = int(len(valid_rows))
    families = sorted(str(value) for value in table["family"].unique())
    n_unique_configs = int(
        table[list(_CONFIG_COLUMNS)].drop_duplicates().shape[0]
    )
    trip_counts = sorted(int(value) for value in table["trip_count"].unique())
    factors = sorted(int(value) for value in table["factor"].unique())
    variants = sorted(str(value) for value in table["variant"].unique())
    workloads = sorted(str(value) for value in table["workload"].unique())

    meta: dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "split_version": SPLIT_VERSION,
        "split_manifest": split_manifest(),
        "created_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "backend": backend,
        "device": device,
        "jax_version": jax_version,
        "jaxlib_version": jaxlib_version,
        "raw_files": raw_files,
        "n_raw_rows": n_raw_rows,
        "n_valid_rows": n_valid_rows,
        "n_dropped_rows": n_raw_rows - n_valid_rows,
        "n_rows": int(len(table)),
        "n_unique_configs": n_unique_configs,
        "families": families,
        "held_out_families": list(HELD_OUT_FAMILIES),
        "dev_families": list(DEV_FAMILIES),
        "feature_columns": list(FEATURE_COLUMNS),
        "label_columns": list(DATASET_LABEL_COLUMNS),
        "r_calls": r_calls,
        "extraction_ms": extraction_stats,
        "extraction_ms_mean": extraction_stats["mean"],
        "extraction_ms_median": extraction_stats["median"],
        "extraction_ms_p95": extraction_stats["p95"],
        "config": {
            "builder": "benchmarks.dataset",
            "backend": backend,
            "raw_files": [entry["path"] for entry in raw_files],
            "workloads": workloads,
            "families": families,
            "variants": variants,
            "trip_counts": trip_counts,
            "factors": factors,
            "r_calls": r_calls,
        },
    }
    return _jsonable(meta)


def build_dataset(
    paths: Sequence[Path],
    *,
    backend: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the feature/label dataset and its metadata from raw sweep files."""

    path_list = [Path(path) for path in paths]
    if not path_list:
        raise ValueError("at least one raw sweep file is required")

    raw = load_raw(path_list)
    raw_files = []
    for path in path_list:
        frame = pd.read_csv(path)
        raw_files.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "row_count": int(len(frame)),
            }
        )

    resolved_backend = _resolve_backend(raw, backend)
    selected = raw.loc[
        raw["backend"].astype(str) == resolved_backend
    ].reset_index(drop=True)
    if selected.empty:
        raise ValueError(
            f"no raw rows remain for backend {resolved_backend!r}"
        )

    jax_version = _consistent_value(selected, "jax_version")
    jaxlib_version = _consistent_value(selected, "jaxlib_version")
    device = _device_value(selected)

    valid = _valid_rows(selected).reset_index(drop=True)
    if valid.empty:
        raise ValueError(
            "no valid, correct raw rows to aggregate for backend "
            f"{resolved_backend!r}"
        )

    aggregated = aggregate_labels(valid)
    features = build_feature_table(aggregated)
    _assert_unique(features, _MERGE_KEY_COLUMNS, "feature table")

    labels = aggregated[
        [*_MERGE_KEY_COLUMNS, *DATASET_LABEL_COLUMNS]
    ]
    table = features.merge(
        labels,
        on=list(_MERGE_KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    if len(table) != len(aggregated) or len(table) != len(features):
        raise ValueError(
            "feature/label join lost rows: "
            f"{len(features)} features, {len(aggregated)} labels, "
            f"{len(table)} joined"
        )

    table["split"] = table["family"].map(split_for_family)
    table = table.reindex(columns=list(DATASET_COLUMNS)).reset_index(drop=True)
    _assert_unique(table, _MERGE_KEY_COLUMNS, "dataset")

    meta = _metadata(
        table,
        raw=raw,
        raw_files=raw_files,
        valid_rows=valid,
        backend=resolved_backend,
        device=device,
        jax_version=jax_version,
        jaxlib_version=jaxlib_version,
    )
    return table, meta


def write_dataset(
    table: pd.DataFrame,
    out_path: Path,
    meta: Mapping[str, Any],
    meta_path: Path,
) -> tuple[Path, Path]:
    """Write the dataset CSV and JSON metadata, creating parent directories."""

    out_path = Path(out_path)
    meta_path = Path(meta_path)
    ensure_parent(out_path)
    ensure_parent(meta_path)
    table.to_csv(out_path, index=False)
    meta_path.write_text(
        json.dumps(_jsonable(meta), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path, meta_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        nargs="+",
        default=None,
        help="one or more raw sweep CSV paths",
    )
    parser.add_argument("--out", default=None, help="dataset CSV output path")
    parser.add_argument(
        "--meta-out",
        default=None,
        help="metadata JSON output path (default: <out stem>.meta.json)",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="backend to select when raw rows span several backends",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    meta_path = (
        Path(args.meta_out)
        if args.meta_out
        else out_path.with_suffix(".meta.json")
    )
    table, meta = build_dataset(
        [Path(path) for path in args.raw],
        backend=args.backend,
    )
    write_dataset(table, out_path, meta, meta_path)
    print(f"dataset: wrote {len(table)} rows to {out_path}")
    print(f"dataset: metadata at {meta_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.raw or not args.out:
        parser.error("--raw and --out are required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
