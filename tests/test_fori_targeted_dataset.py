"""Verification tests for the targeted fori_loop dataset tables."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from jax_unroll.schema import FACTORS, FEATURE_COLUMNS

TRAIN_PATH = Path("results/fori_targeted/train_features.csv")
EVAL_PATH = Path("results/fori_targeted/sealed_eval_features.csv")
RAW_PATH = Path("results/fori_targeted/raw_sweep.csv")
SPLIT_CSV = Path("benchmarks/corpus/fori_targeted/split.csv")


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
    """Assert zero leakage across train and sealed evaluation partitions."""
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


def test_workload_and_config_counts(train_df, eval_df):
    """Verify workload counts match expected targeted matrix."""
    assert train_df["workload"].nunique() == 24, f"Expected 24 train workloads, got {train_df['workload'].nunique()}"
    assert eval_df["workload"].nunique() == 24, f"Expected 24 eval workloads, got {eval_df['workload'].nunique()}"
    
    # Each configuration should have 4 factors
    # 24 bodies * 2 sizes * 4 trips = 192 configurations -> 192 * 4 = 768 rows
    assert len(train_df) == 768, f"Expected 768 train rows, got {len(train_df)}"
    assert len(eval_df) == 768, f"Expected 768 eval rows, got {len(eval_df)}"


def test_four_factor_prediction_groups(train_df, eval_df):
    """Verify four-factor prediction group can be reconstructed for every configuration."""
    for name, df in [("train", train_df), ("eval", eval_df)]:
        config_cols = ["workload", "variant", "trip_count"]
        grouped = df.groupby(config_cols)["factor"].apply(lambda s: tuple(sorted(s.unique())))
        for cfg, factors in grouped.items():
            assert factors == tuple(FACTORS), f"{name} configuration {cfg} has incomplete factors: {factors}"


def test_predictor_features_clean(train_df):
    """Verify that predictor features do not leak timing values, names, or partition tags."""
    excluded_from_features = {"warm_median_us", "cold_median_ms", "trace_median_ms", "lower_median_ms",
                              "compile_median_ms", "workload", "family", "partition", "related_group"}
    for col in FEATURE_COLUMNS:
        assert col in train_df.columns, f"Missing feature column: {col}"
        assert col not in excluded_from_features, f"Leaked column in features: {col}"
