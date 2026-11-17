from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.oracle import (
    AGGREGATED_COLUMNS,
    CONFIG_KEY,
    GROUP_KEY,
    aggregate_raw,
    build_oracle,
    gate_evidence,
    gate_markdown,
    load_raw,
    oracle_markdown,
)
from jax_unroll.schema import FACTORS, ORACLE_COLUMNS, SWEEP_COLUMNS

ALPHA_WARM = {1: 1000.0, 2: 700.0, 4: 500.0, 8: 400.0}
BETA_WARM = {1: 400.0, 2: 420.0, 4: 500.0, 8: 900.0}
COLD = {1: 11.0, 2: 12.0, 4: 13.0, 8: 14.0}

SPECS = {
    "alpha": {
        "family": "recurrence",
        "variant": "scalar",
        "loop_kind": "scan",
        "warm": ALPHA_WARM,
    },
    "beta": {
        "family": "iterative_solver",
        "variant": "n32",
        "loop_kind": "fori_loop",
        "warm": BETA_WARM,
    },
}

KEYS = ["workload", "variant", "trip_count", "factor", "process_repeat"]


def _make_raw(*, specs=None, trips=(16, 100), repeats=3):
    specs = SPECS if specs is None else specs
    rows = []
    for name, spec in specs.items():
        spec_trips = spec.get("trips", trips)
        spec_repeats = spec.get("repeats", repeats)
        spec_cold = spec.get("cold", COLD)
        warm_by_trip = spec.get("warm_by_trip", {})
        for trip in spec_trips:
            warm_map = warm_by_trip.get(trip, spec["warm"])
            for factor in FACTORS:
                warm = warm_map[factor]
                cold = spec_cold[factor]
                for repeat in range(spec_repeats):
                    row = {column: "" for column in SWEEP_COLUMNS}
                    row.update(
                        {
                            "workload": name,
                            "family": spec["family"],
                            "variant": spec["variant"],
                            "loop_kind": spec["loop_kind"],
                            "backend": "cpu",
                            "device": "cpu:0",
                            "jax_version": "test",
                            "jaxlib_version": "test",
                            "trip_count": trip,
                            "shape": '{"carry":[],"xs":[%d]}' % trip,
                            "dtype": "float32",
                            "factor": factor,
                            "divides_trip": trip % factor == 0,
                            "process_repeat": repeat,
                            "trace_ms": 1.0,
                            "lower_ms": 0.5,
                            "compile_ms": 5.0,
                            "cold_total_ms": cold,
                            "warm_median_us": warm,
                            "warm_p95_us": warm * 1.1,
                            "warm_min_us": warm * 0.9,
                            "warm_std_us": warm * 0.01,
                            "warm_samples": 10,
                            "end_to_end_ms_R": cold + 100.0 * warm / 1e3,
                            "r_calls": 100,
                            "stablehlo_bytes": 1000 + factor,
                            "hlo_bytes": 2000 + factor,
                            "correct": True,
                            "max_abs_diff": 0.0,
                            "error": "",
                            "notes": "",
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))


def _pick(frame, workload, trip, factor):
    subset = frame[
        (frame["workload"] == workload)
        & (frame["trip_count"] == trip)
        & (frame["factor"] == factor)
    ]
    assert len(subset) == 1
    return subset.iloc[0]


def _pick_oracle(oracle, workload, trip):
    subset = oracle[
        (oracle["workload"] == workload) & (oracle["trip_count"] == trip)
    ]
    assert len(subset) == 1
    return subset.iloc[0]


@pytest.fixture(scope="module")
def raw_frame():
    return _make_raw()


@pytest.fixture(scope="module")
def aggregated(raw_frame):
    return aggregate_raw(raw_frame)


@pytest.fixture(scope="module")
def oracle(aggregated):
    return build_oracle(aggregated, r_calls=100)


def test_synthetic_raw_matches_schema(raw_frame):
    assert list(raw_frame.columns) == list(SWEEP_COLUMNS)
    assert len(raw_frame) == 2 * 2 * 4 * 3
    assert set(GROUP_KEY).issubset(raw_frame.columns)
    assert set(CONFIG_KEY).issubset(raw_frame.columns)
    assert raw_frame["error"].eq("").all()
    assert raw_frame["correct"].all()


def test_aggregate_raw_medians_and_repeats(aggregated):
    assert list(aggregated.columns) == list(AGGREGATED_COLUMNS)
    assert len(aggregated) == 2 * 2 * 4

    alpha = _pick(aggregated, "alpha", 16, 4)
    assert alpha["repeats"] == 3
    assert alpha["warm_median_us"] == pytest.approx(500.0)
    assert alpha["cold_median_ms"] == pytest.approx(13.0)
    assert alpha["warm_p95_us"] == pytest.approx(550.0)
    assert alpha["warm_min_us"] == pytest.approx(450.0)
    assert alpha["stablehlo_bytes_median"] == pytest.approx(1004.0)

    beta = _pick(aggregated, "beta", 100, 1)
    assert beta["repeats"] == 3
    assert beta["warm_median_us"] == pytest.approx(400.0)
    assert beta["cold_median_ms"] == pytest.approx(11.0)


def test_aggregate_raw_drops_errored_and_incorrect_rows(raw_frame):
    mutated = raw_frame.copy()
    mask = (
        (mutated["workload"] == "alpha")
        & (mutated["trip_count"] == 16)
        & (mutated["factor"] == 2)
    )
    indices = mutated.index[mask]
    mutated.loc[indices[0], "error"] = "RuntimeError: boom"
    mutated.loc[indices[1], "correct"] = False

    partial = aggregate_raw(mutated)
    row = _pick(partial, "alpha", 16, 2)
    assert row["repeats"] == 1
    assert row["warm_median_us"] == pytest.approx(ALPHA_WARM[2])
    assert len(partial) == 16

    mutated_all = raw_frame.copy()
    mask_all = (
        (mutated_all["workload"] == "beta")
        & (mutated_all["trip_count"] == 100)
        & (mutated_all["factor"] == 8)
    )
    mutated_all.loc[mask_all, "error"] = "CompileError: bad"
    dropped = aggregate_raw(mutated_all)
    assert (
        (dropped["workload"] == "beta")
        & (dropped["trip_count"] == 100)
        & (dropped["factor"] == 8)
    ).sum() == 0
    assert len(dropped) == 15


def test_aggregate_raw_rejects_frames_with_no_valid_rows(raw_frame):
    broken = raw_frame.copy()
    broken["error"] = "boom"
    with pytest.raises(ValueError, match="no valid"):
        aggregate_raw(broken)


def test_build_oracle_schema(oracle):
    assert list(oracle.columns) == list(ORACLE_COLUMNS)
    assert len(oracle) == 4
    assert oracle[["workload", "variant", "trip_count"]].duplicated().sum() == 0


def test_build_oracle_best_factors_and_gains(oracle):
    alpha = _pick_oracle(oracle, "alpha", 16)
    assert alpha["factor_u1_warm_us"] == pytest.approx(1000.0)
    assert alpha["oracle_warm_factor"] == 8
    assert alpha["oracle_warm_us"] == pytest.approx(400.0)
    assert alpha["warm_gain_vs_u1"] == pytest.approx(2.5)
    assert alpha["oracle_total_factor_R"] == 8
    assert alpha["u1_total_ms_R"] == pytest.approx(111.0)
    assert alpha["oracle_total_ms_R"] == pytest.approx(54.0)
    assert alpha["total_gain_vs_u1"] == pytest.approx(111.0 / 54.0, rel=1e-6)
    assert alpha["factor_agreement_best_warm"] == 1
    assert alpha["repeats"] == 3

    beta = _pick_oracle(oracle, "beta", 100)
    assert beta["oracle_warm_factor"] == 1
    assert beta["oracle_warm_us"] == pytest.approx(400.0)
    assert beta["warm_gain_vs_u1"] == pytest.approx(1.0)
    assert beta["oracle_total_factor_R"] == 1
    assert beta["total_gain_vs_u1"] == pytest.approx(1.0)
    assert beta["factor_agreement_best_warm"] == 1


def test_oracle_and_gate_consider_non_dividing_factors(aggregated, oracle):
    """u=8 at T=100 must compete with u=1 like every other candidate factor."""
    alpha = _pick_oracle(oracle, "alpha", 100)
    assert alpha["oracle_warm_factor"] == 8
    assert alpha["oracle_warm_us"] == pytest.approx(400.0)
    assert alpha["oracle_total_factor_R"] == 8
    assert alpha["oracle_total_ms_R"] == pytest.approx(54.0)

    _, evidence = gate_evidence(aggregated, oracle, r_calls=100)
    by_workload = {row["workload"]: row for row in evidence}
    assert by_workload["alpha"]["dominance_warm"] == pytest.approx(1.0)
    assert by_workload["beta"]["median_warm_gain_vs_u1"] == pytest.approx(2.25)


def test_build_oracle_tie_breaks_to_smaller_factor():
    specs = {
        "tie": {
            "family": "recurrence",
            "variant": "scalar",
            "loop_kind": "scan",
            "warm": {1: 1000.0, 2: 500.0, 4: 500.0, 8: 900.0},
            "cold": {1: 10.0, 2: 20.0, 4: 20.0, 8: 30.0},
            "trips": (16,),
        }
    }
    aggregated_tie = aggregate_raw(_make_raw(specs=specs))
    oracle_tie = build_oracle(aggregated_tie, r_calls=100)
    row = oracle_tie.iloc[0]

    assert row["oracle_warm_factor"] == 2
    assert row["oracle_warm_us"] == pytest.approx(500.0)
    assert row["oracle_total_factor_R"] == 2
    assert row["oracle_total_ms_R"] == pytest.approx(70.0)
    assert row["factor_agreement_best_warm"] == 1


def test_gate_evidence_passes_with_two_distinct_dominant_factors(
    aggregated, oracle
):
    passed, evidence = gate_evidence(aggregated, oracle, r_calls=100)
    assert passed is True
    assert len(evidence) == 2

    by_workload = {row["workload"]: row for row in evidence}
    alpha = by_workload["alpha"]
    assert alpha["dominant_warm_factor"] == 8
    assert alpha["dominance_warm"] >= 0.5
    assert alpha["median_warm_gain_vs_u1"] == pytest.approx(2.5)
    assert alpha["useful_choice"] is True

    beta = by_workload["beta"]
    assert beta["dominant_warm_factor"] == 1
    assert beta["dominance_warm"] == pytest.approx(1.0)
    assert beta["median_warm_gain_vs_u1"] >= 1.05
    assert beta["dominant_total_factor_R"] == 1
    assert beta["useful_choice"] is True


def test_gate_evidence_fails_when_one_workload_is_not_dominant():
    specifications = {
        "alpha": SPECS["alpha"],
        "beta": {
            **SPECS["beta"],
            "trips": (16, 64, 256, 1000),
            "warm_by_trip": {
                16: {1: 400.0, 2: 900.0, 4: 500.0, 8: 420.0},
                64: {1: 500.0, 2: 400.0, 4: 450.0, 8: 900.0},
                256: {1: 500.0, 2: 600.0, 4: 400.0, 8: 700.0},
                1000: {1: 500.0, 2: 600.0, 4: 700.0, 8: 400.0},
            },
        },
    }
    raw = _make_raw(specs=specifications)
    aggregated_split = aggregate_raw(raw)
    oracle_split = build_oracle(aggregated_split, r_calls=100)

    passed, evidence = gate_evidence(aggregated_split, oracle_split, r_calls=100)
    assert passed is False
    by_workload = {row["workload"]: row for row in evidence}
    assert by_workload["alpha"]["useful_choice"] is True
    assert by_workload["beta"]["dominance_warm"] < 0.5
    assert by_workload["beta"]["useful_choice"] is False


def test_oracle_markdown_is_well_formed(oracle):
    markdown = oracle_markdown(
        oracle, raw_path="results/raw/test.csv", backend="cpu", r_calls=100
    )
    assert markdown.startswith("# Measured oracle table (cpu)")
    assert "Tie-breaking prefers the smaller factor." in markdown

    table_lines = [line for line in markdown.splitlines() if line.startswith("|")]
    assert len(table_lines) == 2 + len(oracle)
    widths = {len(line.strip().strip("|").split("|")) for line in table_lines}
    assert widths == {10}
    for line in table_lines:
        assert line.rstrip().endswith("|")
        assert "||" not in line
    assert "| alpha |" in markdown
    assert "| beta |" in markdown


def test_gate_markdown_decision_line_and_table(aggregated, oracle):
    passed, evidence = gate_evidence(aggregated, oracle, r_calls=100)
    markdown = gate_markdown(
        passed,
        evidence,
        raw_path="results/raw/test.csv",
        backend="cpu",
        r_calls=100,
        threshold=0.05,
    )
    assert "- Decision: **PASS**" in markdown
    table_lines = [line for line in markdown.splitlines() if line.startswith("|")]
    assert len(table_lines) == 2 + len(evidence)
    widths = {len(line.strip().strip("|").split("|")) for line in table_lines}
    assert widths == {10}

    failed_markdown = gate_markdown(
        False,
        evidence,
        raw_path="results/raw/test.csv",
        backend="cpu",
        r_calls=100,
        threshold=0.05,
    )
    assert "- Decision: **FAIL**" in failed_markdown
    assert "**PASS**" not in failed_markdown


def test_load_raw_concatenates_multiple_paths(tmp_path, raw_frame):
    first = raw_frame.iloc[:20]
    second = raw_frame.iloc[20:]
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)

    loaded = load_raw([first_path, second_path])
    assert len(loaded) == len(raw_frame)
    assert list(loaded.columns) == list(SWEEP_COLUMNS)
    assert loaded["workload"].value_counts().to_dict() == (
        raw_frame["workload"].value_counts().to_dict()
    )

    columns = KEYS + ["warm_median_us", "cold_total_ms", "correct"]
    pd.testing.assert_frame_equal(
        loaded.sort_values(KEYS).reset_index(drop=True)[columns],
        raw_frame.sort_values(KEYS).reset_index(drop=True)[columns],
        check_dtype=False,
    )
    assert len(load_raw(first_path)) == 20


def test_load_raw_rejects_missing_input(tmp_path, raw_frame):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="at least one"):
        load_raw([])

    incomplete = tmp_path / "incomplete.csv"
    raw_frame.drop(columns=["warm_median_us"]).to_csv(incomplete, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_raw(incomplete)
