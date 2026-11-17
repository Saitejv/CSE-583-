from __future__ import annotations

import csv
import io
from collections import Counter

import jax
import pytest

from benchmarks.sweep import _empty_row, enumerate_configs, measure_config
from benchmarks.workloads import WORKLOADS
from jax_unroll.schema import FACTORS, SWEEP_COLUMNS, TRIP_COUNTS

PER_REPEAT = len(WORKLOADS) * len(TRIP_COUNTS) * len(FACTORS)


def _keys(configs):
    return [
        (c["workload"], c["variant"], c["trip_count"], c["factor"]) for c in configs
    ]


def test_sweep_columns_match_dictwriter_header():
    row = _empty_row({"workload": "scalar_recurrence", "factor": 2})
    assert set(row) == set(SWEEP_COLUMNS)
    assert len(row) == len(SWEEP_COLUMNS)

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=SWEEP_COLUMNS, extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerow(row)

    parsed = list(csv.reader(io.StringIO(buffer.getvalue())))
    assert parsed[0] == list(SWEEP_COLUMNS)
    assert len(parsed[1]) == len(SWEEP_COLUMNS)
    assert parsed[1][0] == "scalar_recurrence"


def test_enumerate_configs_repeat_count_and_total():
    configs = enumerate_configs()
    assert PER_REPEAT == 6 * 6 * 4
    assert len(configs) == PER_REPEAT * 3
    assert len(configs) == 432
    counts = Counter(config["process_repeat"] for config in configs)
    assert counts == {0: PER_REPEAT, 1: PER_REPEAT, 2: PER_REPEAT}
    assert all(
        set(config) == {
            "workload",
            "family",
            "variant",
            "loop_kind",
            "trip_count",
            "factor",
            "process_repeat",
        }
        for config in configs
    )


def test_enumerate_configs_order_is_seeded_and_shuffled():
    first = enumerate_configs(repeats=2, seed=583)
    second = enumerate_configs(repeats=2, seed=583)
    other = enumerate_configs(repeats=2, seed=584)

    assert _keys(first) == _keys(second)
    assert _keys(first) != _keys(other)

    repeat_zero = _keys(first)[:PER_REPEAT]
    repeat_one = _keys(first)[PER_REPEAT:]
    assert set(repeat_zero) == set(repeat_one)
    assert len(set(repeat_zero)) == PER_REPEAT
    assert repeat_zero != sorted(repeat_zero)


def test_enumerate_configs_shape_modes():
    primary = enumerate_configs(shapes="primary", repeats=1)
    secondary = enumerate_configs(shapes="secondary", repeats=1)
    all_shapes = enumerate_configs(shapes="all", repeats=1)

    assert len(primary) == PER_REPEAT
    expected_primary = {
        (name, spec.primary_variant) for name, spec in WORKLOADS.items()
    }
    assert {(c["workload"], c["variant"]) for c in primary} == expected_primary

    expected_secondary = {
        (name, variant)
        for name, spec in WORKLOADS.items()
        for variant in spec.shape_variants
        if variant != spec.primary_variant
    }
    expected_all = {
        (name, variant)
        for name, spec in WORKLOADS.items()
        for variant in spec.shape_variants
    }
    assert {(c["workload"], c["variant"]) for c in secondary} == expected_secondary
    assert {(c["workload"], c["variant"]) for c in all_shapes} == expected_all

    per_variant = len(TRIP_COUNTS) * len(FACTORS)
    assert len(secondary) == len(expected_secondary) * per_variant
    assert len(all_shapes) == len(expected_all) * per_variant
    assert len(secondary) + len(primary) == len(all_shapes)


def test_enumerate_configs_limit_is_applied_after_ordering():
    full = enumerate_configs(repeats=2)
    limited = enumerate_configs(repeats=2, limit=150)

    assert len(limited) == 150
    assert limited == full[:150]
    assert Counter(c["process_repeat"] for c in limited) == {
        0: PER_REPEAT,
        1: 150 - PER_REPEAT,
    }

    assert enumerate_configs(repeats=1, limit=0) == []
    assert enumerate_configs(repeats=1, limit=-5) == []
    assert len(enumerate_configs(repeats=1, limit=10_000)) == PER_REPEAT


def test_enumerate_configs_rejects_bad_arguments():
    with pytest.raises(ValueError, match="repeats"):
        enumerate_configs(repeats=0)
    with pytest.raises(ValueError, match="shapes"):
        enumerate_configs(shapes="bogus")
    with pytest.raises(ValueError, match="unknown families"):
        enumerate_configs(families=["bogus"])


def test_measure_config_single_tiny_run():
    config = {
        "workload": "scalar_recurrence",
        "variant": "scalar",
        "trip_count": 16,
        "factor": 2,
    }
    row = measure_config(
        config,
        warmup=1,
        sample_budget_s=0.01,
        min_samples=3,
        max_samples=5,
        r_calls=2,
    )

    assert set(row) == set(SWEEP_COLUMNS)
    assert row["backend"] == jax.default_backend()
    assert row["device"] == str(jax.devices()[0])
    assert row["family"] == "recurrence"
    assert row["loop_kind"] == "scan"
    assert row["variant"] == "scalar"
    assert row["trip_count"] == 16
    assert row["factor"] == 2
    assert row["divides_trip"] is True
    assert row["dtype"] == "float32"
    assert row["shape"]

    for column in (
        "trace_ms",
        "compile_ms",
        "cold_total_ms",
        "warm_median_us",
        "warm_p95_us",
        "warm_min_us",
        "end_to_end_ms_R",
        "stablehlo_bytes",
        "hlo_bytes",
    ):
        assert row[column] > 0, column
    assert row["lower_ms"] >= 0
    assert row["warm_std_us"] >= 0
    assert row["warm_min_us"] <= row["warm_p95_us"]

    assert 3 <= row["warm_samples"] <= 5
    assert row["r_calls"] == 2
    assert row["correct"] is True
    assert row["max_abs_diff"] == 0.0
    assert row["error"] == ""
    assert "adaptive_samples=" in row["notes"]

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=SWEEP_COLUMNS, extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerow(row)
    parsed = list(csv.reader(io.StringIO(buffer.getvalue())))
    assert parsed[0] == list(SWEEP_COLUMNS)
    assert len(parsed) == 2
    assert len(parsed[1]) == len(SWEEP_COLUMNS)
