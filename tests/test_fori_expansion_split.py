"""Unit tests for the fori_expansion split validation and hash computation."""

import pandas as pd
import pytest

from benchmarks.corpus.fori_expansion.split import (
    compute_split_hash,
    validate_split_records,
)


def test_split_hash_deterministic():
    records_a = [
        {"workload": "w1", "family": "f1", "related_group": "g1", "partition": "train"},
        {"workload": "w2", "family": "f2", "related_group": "g2", "partition": "fresh_eval"},
    ]
    records_b = [
        {"workload": "w2", "family": "f2", "related_group": "g2", "partition": "fresh_eval"},
        {"workload": "w1", "family": "f1", "related_group": "g1", "partition": "train"},
    ]
    assert compute_split_hash(records_a) == compute_split_hash(records_b)


def test_leakage_detection():
    # Two workloads with same related_group in different partitions must raise ValueError
    leaking_records = [
        {
            "workload": "w1",
            "family": "f1",
            "related_group": "group_shared",
            "partition": "train",
            "mechanism": "dummy mechanism one",
            "tier": "full",
            "origin": "new_independent",
            "reason": "dummy reason 12345",
        },
        {
            "workload": "w2",
            "family": "f1",
            "related_group": "group_shared",
            "partition": "fresh_eval",
            "mechanism": "dummy mechanism two",
            "tier": "full",
            "origin": "new_independent",
            "reason": "dummy reason 12345",
        },
    ]
    with pytest.raises(ValueError, match="Related-group partition leakage"):
        validate_split_records(leaking_records, enforce_quotas=False)


def test_historical_lineage_enforcement():
    historical_split = {"w_hist_train": "train", "w_hist_test": "test"}
    
    # Correct mapping
    valid_records = [
        {
            "workload": "w_hist_train",
            "family": "f1",
            "related_group": "g1",
            "partition": "train",
            "mechanism": "dummy mechanism 1",
            "tier": "full",
            "origin": "historical_fori",
            "reason": "historical train workload",
        },
        {
            "workload": "w_hist_test",
            "family": "f2",
            "related_group": "g2",
            "partition": "historical_eval",
            "mechanism": "dummy mechanism 2",
            "tier": "full",
            "origin": "historical_fori",
            "reason": "historical test workload",
        },
    ]
    res = validate_split_records(valid_records, historical_split=historical_split, enforce_quotas=False)
    assert res["status"] == "passed"

    # Inverted mapping
    invalid_records = [
        {
            "workload": "w_hist_train",
            "family": "f1",
            "related_group": "g1",
            "partition": "historical_eval",  # Should be train!
            "mechanism": "dummy mechanism 1",
            "tier": "full",
            "origin": "historical_fori",
            "reason": "violating historical train",
        },
    ]
    with pytest.raises(ValueError, match="Lineage violation"):
        validate_split_records(invalid_records, historical_split=historical_split, enforce_quotas=False)


def test_fresh_eval_isolation():
    # 1. Non-new_independent origin in fresh_eval
    invalid_origin_records = [
        {
            "workload": "w_new",
            "family": "f1",
            "related_group": "g1",
            "partition": "fresh_eval",
            "mechanism": "dummy mechanism 1",
            "tier": "full",
            "origin": "historical_fori",
            "reason": "invalid origin in fresh_eval",
        },
    ]
    with pytest.raises(ValueError, match="Isolation violation"):
        validate_split_records(invalid_origin_records, enforce_quotas=False)

    # 2. Historical workload assigned to fresh_eval triggers lineage violation
    historical_split = {"w_hist": "test"}
    contaminated_records = [
        {
            "workload": "w_hist",
            "family": "f1",
            "related_group": "g1",
            "partition": "fresh_eval",
            "mechanism": "dummy mechanism 1",
            "tier": "full",
            "origin": "converted_scan",
            "reason": "historical in fresh_eval",
        },
    ]
    with pytest.raises(ValueError, match="Lineage violation"):
        validate_split_records(contaminated_records, historical_split=historical_split, enforce_quotas=False)
