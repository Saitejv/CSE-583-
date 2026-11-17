from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.dataset import (
    aggregate_labels,
    build_dataset,
    build_feature_table,
    build_parser,
    write_dataset,
)
from benchmarks.workloads import build_workload
from jax_unroll.schema import (
    DATASET_COLUMNS,
    DATASET_KEY_COLUMNS,
    DATASET_LABEL_COLUMNS,
    DATASET_VERSION,
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    SWEEP_COLUMNS,
)
from jax_unroll.splits import (
    DEV_FAMILIES,
    HELD_OUT_FAMILIES,
    SPLIT_VERSION,
)

CONFIGS = {
    "scalar_recurrence": {
        "family": "recurrence",
        "variant": "scalar",
        "loop_kind": "scan",
        "shape": '{"carry":[],"xs":[16]}',
    },
    "indexed_update": {
        "family": "indexed_memory",
        "variant": "n256",
        "loop_kind": "scan",
        "shape": '{"carry":[256],"indices":[16],"xs":[16,16]}',
    },
}

FEATURE_OUTPUT_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in DATASET_KEY_COLUMNS
]


def _row(
    name: str,
    *,
    trip_count: int,
    factor: int,
    process_repeat: int,
    warm_median_us: float,
    cold_total_ms: float,
    correct: bool = True,
    error: str = "",
    backend: str = "cpu",
    device: str = "cpu:0",
    r_calls: int = 100,
    jax_version: str = "test-jax",
    jaxlib_version: str = "test-jaxlib",
) -> dict[str, object]:
    config = CONFIGS[name]
    row: dict[str, object] = {column: "" for column in SWEEP_COLUMNS}
    row.update(
        {
            "workload": name,
            "family": config["family"],
            "variant": config["variant"],
            "loop_kind": config["loop_kind"],
            "backend": backend,
            "device": device,
            "jax_version": jax_version,
            "jaxlib_version": jaxlib_version,
            "trip_count": trip_count,
            "shape": config["shape"],
            "dtype": "float32",
            "factor": factor,
            "divides_trip": trip_count % factor == 0,
            "process_repeat": process_repeat,
            "trace_ms": cold_total_ms * 0.1,
            "lower_ms": cold_total_ms * 0.2,
            "compile_ms": cold_total_ms * 0.7,
            "cold_total_ms": cold_total_ms,
            "warm_median_us": warm_median_us,
            "warm_p95_us": warm_median_us * 1.1,
            "warm_min_us": warm_median_us * 0.9,
            "warm_std_us": warm_median_us * 0.02,
            "warm_samples": 10,
            "end_to_end_ms_R": cold_total_ms + r_calls * warm_median_us / 1e3,
            "r_calls": r_calls,
            "stablehlo_bytes": 1000 + factor,
            "hlo_bytes": 2000 + factor,
            "correct": correct,
            "max_abs_diff": 0.0,
            "error": error,
            "notes": "",
        }
    )
    return row


def _make_raw() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scalar_factor1 = [(1000.0, 11.0), (900.0, 12.0), (1100.0, 13.0)]
    scalar_factor2 = [(700.0, 21.0), (750.0, 22.0), (800.0, 23.0)]
    indexed_factor1 = [(2000.0, 30.0), (2100.0, 31.0), (2200.0, 32.0)]

    for repeat, (warm, cold) in enumerate(scalar_factor1):
        rows.append(
            _row(
                "scalar_recurrence",
                trip_count=16,
                factor=1,
                process_repeat=repeat,
                warm_median_us=warm,
                cold_total_ms=cold,
            )
        )
    for repeat, (warm, cold) in enumerate(scalar_factor2):
        rows.append(
            _row(
                "scalar_recurrence",
                trip_count=16,
                factor=2,
                process_repeat=repeat,
                warm_median_us=warm,
                cold_total_ms=cold,
            )
        )
    for repeat, (warm, cold) in enumerate(indexed_factor1):
        rows.append(
            _row(
                "indexed_update",
                trip_count=16,
                factor=1,
                process_repeat=repeat,
                warm_median_us=warm,
                cold_total_ms=cold,
            )
        )

    rows.append(
        _row(
            "scalar_recurrence",
            trip_count=16,
            factor=2,
            process_repeat=99,
            warm_median_us=1.0,
            cold_total_ms=1.0,
            correct=False,
        )
    )
    rows.append(
        _row(
            "scalar_recurrence",
            trip_count=16,
            factor=1,
            process_repeat=99,
            warm_median_us=1.0,
            cold_total_ms=1.0,
            error="synthetic failure",
        )
    )
    return pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))


def _write_raw(tmp_path: Path, frame: pd.DataFrame, name: str) -> Path:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _pick(frame: pd.DataFrame, workload: str, trip: int, factor: int) -> pd.Series:
    subset = frame[
        (frame["workload"] == workload)
        & (frame["trip_count"] == trip)
        & (frame["factor"] == factor)
    ]
    assert len(subset) == 1
    return subset.iloc[0]


@pytest.fixture(scope="module")
def raw_frame() -> pd.DataFrame:
    return _make_raw()


@pytest.fixture(scope="module")
def aggregated(raw_frame: pd.DataFrame) -> pd.DataFrame:
    return aggregate_labels(raw_frame)


def test_aggregate_labels_medians_drops_r_calls_and_logs(
    raw_frame: pd.DataFrame,
) -> None:
    agg = aggregate_labels(raw_frame)

    assert list(agg.columns) == [
        *DATASET_KEY_COLUMNS,
        "shape",
        *DATASET_LABEL_COLUMNS,
    ]
    assert len(agg) == 3

    factor1 = _pick(agg, "scalar_recurrence", 16, 1)
    assert factor1.repeats == 3
    assert factor1.cold_median_ms == pytest.approx(12.0)
    assert factor1.warm_median_us == pytest.approx(1000.0)
    assert factor1.warm_p95_us == pytest.approx(1100.0)
    assert factor1.warm_min_us == pytest.approx(900.0)
    assert bool(factor1.correct_all) is True
    assert factor1.r_calls == 100
    assert factor1.log_cold_ms == pytest.approx(math.log(12.0))
    assert factor1.log_warm_us == pytest.approx(math.log(1000.0))
    assert math.isfinite(factor1.log_cold_ms)
    assert math.isfinite(factor1.log_warm_us)

    factor2 = _pick(agg, "scalar_recurrence", 16, 2)
    assert factor2.repeats == 3
    assert factor2.cold_median_ms == pytest.approx(22.0)
    assert factor2.warm_median_us == pytest.approx(750.0)

    indexed = _pick(agg, "indexed_update", 16, 1)
    assert indexed.repeats == 3
    assert indexed.cold_median_ms == pytest.approx(31.0)
    assert indexed.warm_median_us == pytest.approx(2100.0)
    assert indexed.log_warm_us == pytest.approx(math.log(2100.0))


def test_build_feature_table_extracts_static_features(
    aggregated: pd.DataFrame,
) -> None:
    table = build_feature_table(aggregated)

    assert list(table.columns) == [
        *DATASET_KEY_COLUMNS,
        "shape",
        *FEATURE_OUTPUT_COLUMNS,
    ]
    assert len(table) == 3

    scalar = _pick(table, "scalar_recurrence", 16, 1)
    assert scalar.num_equations > 0
    assert scalar.carry_bytes > 0
    assert scalar.loop_primitive == "scan"
    assert scalar.extraction_ms >= 0.0

    indexed = _pick(table, "indexed_update", 16, 1)
    assert indexed.num_equations > 0
    assert indexed.carry_bytes > 0
    assert indexed.eq_gather_scatter >= 1


def test_build_feature_table_uses_injected_builder(
    aggregated: pd.DataFrame,
) -> None:
    calls: list[tuple[str, int, int, str | None]] = []

    def builder(
        name: str, trip_count: int, factor: int, variant: str | None
    ) -> object:
        calls.append((name, trip_count, factor, variant))
        return build_workload(name, trip_count, factor, variant)

    table = build_feature_table(aggregated, workload_builder=builder)

    assert len(table) == 3
    assert len(calls) == 3
    assert ("scalar_recurrence", 16, 1, "scalar") in calls


def test_build_dataset_schema_split_metadata_and_write(
    tmp_path: Path,
    raw_frame: pd.DataFrame,
) -> None:
    path = _write_raw(tmp_path, raw_frame, "raw.csv")
    table, meta = build_dataset([path])

    assert list(table.columns) == list(DATASET_COLUMNS)
    assert len(table.columns) == len(set(table.columns))
    assert len(table) == 3
    assert table["correct_all"].all()
    assert table["log_cold_ms"].tolist() == pytest.approx(
        [math.log(value) for value in table["cold_median_ms"]]
    )
    assert table["log_warm_us"].tolist() == pytest.approx(
        [math.log(value) for value in table["warm_median_us"]]
    )

    scalar = _pick(table, "scalar_recurrence", 16, 1)
    assert scalar.split == "dev"
    assert scalar.num_equations > 0
    assert scalar.carry_bytes > 0
    assert scalar.factor == 1

    indexed = _pick(table, "indexed_update", 16, 1)
    assert indexed.split == "holdout"

    assert meta["dataset_version"] == DATASET_VERSION
    assert meta["feature_version"] == FEATURE_VERSION
    assert meta["split_version"] == SPLIT_VERSION
    assert meta["backend"] == "cpu"
    assert meta["device"] == "cpu:0"
    assert meta["jax_version"] == "test-jax"
    assert meta["jaxlib_version"] == "test-jaxlib"
    assert meta["n_raw_rows"] == len(raw_frame)
    assert meta["n_valid_rows"] == 9
    assert meta["n_dropped_rows"] == 2
    assert meta["n_rows"] == 3
    assert meta["n_unique_configs"] == 2
    assert meta["families"] == ["indexed_memory", "recurrence"]
    assert meta["dev_families"] == list(DEV_FAMILIES)
    assert meta["held_out_families"] == list(HELD_OUT_FAMILIES)
    assert meta["split_manifest"]["split_version"] == SPLIT_VERSION
    assert meta["split_manifest"]["held_out_families"] == list(HELD_OUT_FAMILIES)
    assert meta["config"]["backend"] == "cpu"
    assert meta["config"]["trip_counts"] == [16]
    assert meta["config"]["factors"] == [1, 2]
    assert meta["config"]["workloads"] == ["indexed_update", "scalar_recurrence"]
    assert meta["feature_columns"] == list(FEATURE_COLUMNS)
    assert meta["label_columns"] == list(DATASET_LABEL_COLUMNS)
    assert meta["r_calls"] == 100
    assert set(meta["extraction_ms"]) == {"mean", "median", "p95"}
    assert meta["extraction_ms"]["mean"] > 0.0
    assert meta["extraction_ms_mean"] == meta["extraction_ms"]["mean"]

    raw_entry = meta["raw_files"][0]
    assert raw_entry["path"] == str(path)
    assert raw_entry["row_count"] == len(raw_frame)
    assert raw_entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    out_path, meta_path = write_dataset(
        table,
        tmp_path / "features.csv",
        meta,
        tmp_path / "features.meta.json",
    )
    assert out_path.exists()
    assert meta_path.exists()
    written = pd.read_csv(out_path)
    assert list(written.columns) == list(DATASET_COLUMNS)
    assert len(written) == 3
    parsed = json.loads(meta_path.read_text(encoding="utf-8"))
    assert parsed["n_rows"] == 3
    assert parsed["r_calls"] == 100
    assert parsed["raw_files"][0]["sha256"] == raw_entry["sha256"]


def test_duplicate_raw_rows_across_files_aggregate(
    tmp_path: Path,
    raw_frame: pd.DataFrame,
) -> None:
    first = _write_raw(tmp_path, raw_frame, "raw_a.csv")
    second = _write_raw(tmp_path, raw_frame, "raw_b.csv")

    table, meta = build_dataset([first, second])

    assert len(table) == 3
    assert not table.duplicated(
        subset=[*DATASET_KEY_COLUMNS, "shape"]
    ).any()
    assert meta["n_raw_rows"] == 2 * len(raw_frame)
    assert meta["n_valid_rows"] == 18
    assert meta["n_dropped_rows"] == 4

    scalar = _pick(table, "scalar_recurrence", 16, 1)
    assert scalar.repeats == 6
    assert scalar.cold_median_ms == pytest.approx(12.0)
    assert scalar.warm_median_us == pytest.approx(1000.0)
    assert scalar.log_cold_ms == pytest.approx(math.log(12.0))


def test_backend_filter_and_multiple_backend_error(
    tmp_path: Path,
    raw_frame: pd.DataFrame,
) -> None:
    gpu = raw_frame.copy()
    gpu["backend"] = "gpu"
    gpu["device"] = "gpu:0"
    combined = pd.concat([raw_frame, gpu], ignore_index=True)
    path = _write_raw(tmp_path, combined, "mixed_backend.csv")

    with pytest.raises(ValueError, match="multiple backends"):
        build_dataset([path])

    table, meta = build_dataset([path], backend="cpu")
    assert len(table) == 3
    assert meta["backend"] == "cpu"
    assert meta["device"] == "cpu:0"
    assert meta["n_raw_rows"] == len(combined)
    assert meta["n_valid_rows"] == 9
    assert meta["n_dropped_rows"] == len(combined) - 9

    with pytest.raises(ValueError, match="not present"):
        build_dataset([path], backend="tpu")


def test_non_positive_medians_are_rejected(raw_frame: pd.DataFrame) -> None:
    warm_bad = raw_frame.copy()
    warm_mask = (
        (warm_bad["workload"] == "scalar_recurrence")
        & (warm_bad["factor"] == 2)
        & warm_bad["correct"]
        & (warm_bad["error"] == "")
    )
    warm_bad.loc[warm_mask, "warm_median_us"] = 0.0
    with pytest.raises(ValueError, match="warm_median_us"):
        aggregate_labels(warm_bad)

    cold_bad = raw_frame.copy()
    cold_mask = (
        (cold_bad["workload"] == "scalar_recurrence")
        & (cold_bad["factor"] == 1)
        & cold_bad["correct"]
        & (cold_bad["error"] == "")
    )
    cold_bad.loc[cold_mask, "cold_total_ms"] = -1.0
    with pytest.raises(ValueError, match="cold_median_ms"):
        aggregate_labels(cold_bad)


def test_cli_parser_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["--raw", "a.csv", "b.csv", "--out", "data/features.csv"]
    )
    assert args.raw == ["a.csv", "b.csv"]
    assert args.out == "data/features.csv"
    assert args.backend is None
    assert args.meta_out is None

    explicit = parser.parse_args(
        [
            "--raw",
            "a.csv",
            "--out",
            "out.csv",
            "--backend",
            "gpu",
            "--meta-out",
            "out.json",
        ]
    )
    assert explicit.backend == "gpu"
    assert explicit.meta_out == "out.json"
