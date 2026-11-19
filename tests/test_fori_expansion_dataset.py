"""Verification tests for the expanded fori_loop dataset tables."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from jax_unroll.schema import FACTORS, FEATURE_COLUMNS

TRAIN_PATH = Path("results/fori_expansion/train_features.csv")
EVAL_PATH = Path("results/fori_expansion/eval_features.csv")
RAW_PATH = Path("results/fori_expansion/raw_sweep.csv")
SPLIT_CSV = Path("benchmarks/corpus/fori_expansion/split.csv")
MANIFEST_JSON = Path("benchmarks/corpus/fori_expansion/manifest.json")


@pytest.fixture(scope="module")
def train_df():
    assert TRAIN_PATH.exists(), f"Missing {TRAIN_PATH}"
    return pd.read_csv(TRAIN_PATH)


@pytest.fixture(scope="module")
def eval_df():
    assert EVAL_PATH.exists(), f"Missing {EVAL_PATH}"
    return pd.read_csv(EVAL_PATH)


@pytest.fixture(scope="module")
def raw_df():
    assert RAW_PATH.exists(), f"Missing {RAW_PATH}"
    return pd.read_csv(RAW_PATH)


def test_public_loop_kind_is_fori_loop(train_df, eval_df):
    """Assert that every row in both model-ready tables has public loop_kind equal to fori_loop."""
    assert (train_df["loop_kind"] == "fori_loop").all(), "Non-fori_loop row found in train table!"
    assert (eval_df["loop_kind"] == "fori_loop").all(), "Non-fori_loop row found in eval table!"


def test_train_eval_disjointness(train_df, eval_df):
    """Assert zero leakage across train and evaluation partitions."""
    train_workloads = set(train_df["workload"].unique())
    eval_workloads = set(eval_df["workload"].unique())
    overlap = train_workloads.intersection(eval_workloads)
    assert not overlap, f"Workloads found in both train and eval: {overlap}"

    # Verify related group isolation
    split_df = pd.read_csv(SPLIT_CSV)
    group_map = dict(zip(split_df["workload"], split_df["related_group"]))
    train_groups = {group_map[w] for w in train_workloads if w in group_map}
    eval_groups = {group_map[w] for w in eval_workloads if w in group_map}
    group_overlap = train_groups.intersection(eval_groups)
    assert not group_overlap, f"Related groups found in both train and eval: {group_overlap}"


def test_workload_counts(train_df, eval_df):
    """Verify workload counts match expected taxonomy."""
    total_workloads = train_df["workload"].nunique() + eval_df["workload"].nunique()
    assert total_workloads == 136, f"Expected 136 total workloads, got {total_workloads}"
    assert train_df["workload"].nunique() == 104, f"Expected 104 train workloads, got {train_df['workload'].nunique()}"
    assert eval_df["workload"].nunique() == 32, f"Expected 32 eval workloads, got {eval_df['workload'].nunique()}"

    # Partition check
    hist_eval = eval_df[eval_df["partition"] == "historical_eval"]
    fresh_eval = eval_df[eval_df["partition"] == "fresh_eval"]
    assert hist_eval["workload"].nunique() == 14, f"Expected 14 historical eval workloads, got {hist_eval['workload'].nunique()}"
    assert fresh_eval["workload"].nunique() == 18, f"Expected 18 fresh eval workloads, got {fresh_eval['workload'].nunique()}"


def test_four_factor_prediction_groups(train_df, eval_df):
    """Verify four-factor prediction group can be reconstructed for every configuration."""
    for name, df in [("train", train_df), ("eval", eval_df)]:
        config_cols = ["workload", "variant", "trip_count"]
        grouped = df.groupby(config_cols)["factor"].apply(lambda s: tuple(sorted(s.unique())))
        for cfg, factors in grouped.items():
            assert factors == tuple(FACTORS), f"{name} configuration {cfg} has incomplete factors: {factors}"


def test_predictor_features_clean(train_df):
    """Verify that predictor features do not leak timing values, names, or partition tags."""
    # Standard static feature columns
    excluded_from_features = {"warm_median_us", "cold_median_ms", "trace_median_ms", "lower_median_ms",
                              "compile_median_ms", "workload", "family", "partition", "related_group"}
    # Check that FEATURE_COLUMNS do not contain forbidden column names
    for col in FEATURE_COLUMNS:
        assert col in train_df.columns, f"Missing feature column: {col}"
        assert col not in excluded_from_features, f"Leaked column in features: {col}"


def test_raw_rows_validity(raw_df):
    """Verify raw rows integrity."""
    assert len(raw_df) >= 3600, f"Expected at least 3600 raw rows, got {len(raw_df)}"
    assert (raw_df["backend"] == "cpu").all()
    has_error = raw_df["error"].notna() & (raw_df["error"].astype(str).str.strip() != "")
    assert not has_error.any(), "Found errors in raw sweep rows!"
    assert raw_df["correct"].astype(str).str.lower().isin(("true", "1")).all(), "Found incorrect rows in raw sweep!"
