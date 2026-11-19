"""Tests for the targeted fori_loop split validation."""

import pytest
from benchmarks.corpus.fori_targeted.registry import ALL_SPECS
from benchmarks.corpus.fori_targeted.split import (
    compute_split_hash,
    validate_split,
)


def test_split_hash_deterministic():
    records = [
        {"workload": "w1", "family": "f1", "related_group": "g1", "partition": "train"},
        {"workload": "w2", "family": "f2", "related_group": "g2", "partition": "sealed_eval"},
    ]
    h1 = compute_split_hash(records)
    h2 = compute_split_hash(list(reversed(records)))
    assert h1 == h2


def test_split_validates():
    records = []
    for s in ALL_SPECS:
        partition = "sealed_eval" if s.name.startswith("eval_") else "train"
        records.append({
            "workload": s.name,
            "family": s.family,
            "related_group": s.related_group,
            "partition": partition,
        })
    result = validate_split(records)
    assert result["status"] == "passed"
    assert result["train"] == 24
    assert result["sealed_eval"] == 24


def test_related_group_isolation():
    records = []
    for s in ALL_SPECS:
        partition = "sealed_eval" if s.name.startswith("eval_") else "train"
        records.append({
            "workload": s.name,
            "family": s.family,
            "related_group": s.related_group,
            "partition": partition,
        })
    # Verify no related_group appears in both train and eval
    train_groups = {r["related_group"] for r in records if r["partition"] == "train"}
    eval_groups = {r["related_group"] for r in records if r["partition"] == "sealed_eval"}
    overlap = train_groups & eval_groups
    assert not overlap, f"Related groups in both partitions: {overlap}"
