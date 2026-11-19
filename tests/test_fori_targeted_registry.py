"""Tests for the targeted fori_loop workload registry."""

import pytest
from benchmarks.corpus.fori_targeted.registry import (
    ALL_SPECS, TRAIN_SPECS, EVAL_SPECS,
    build_workload, get_spec,
)


def test_total_spec_count():
    assert len(ALL_SPECS) == 48, f"Expected 48 specs, got {len(ALL_SPECS)}"


def test_train_eval_counts():
    assert len(TRAIN_SPECS) == 24, f"Expected 24 train, got {len(TRAIN_SPECS)}"
    assert len(EVAL_SPECS) == 24, f"Expected 24 eval, got {len(EVAL_SPECS)}"


def test_unique_names():
    names = [s.name for s in ALL_SPECS]
    assert len(names) == len(set(names)), "Duplicate workload names found"


def test_all_fori_loop():
    for s in ALL_SPECS:
        assert s.loop_kind == "fori_loop", f"{s.name} is {s.loop_kind}, expected fori_loop"


def test_build_workload():
    spec = ALL_SPECS[0]
    wl = build_workload(spec.name, 64, 1, "small")
    assert wl.fn is not None


def test_all_have_two_variants():
    for s in ALL_SPECS:
        assert "small" in s.shape_variants, f"{s.name} missing 'small' variant"
        assert "large" in s.shape_variants, f"{s.name} missing 'large' variant"


def test_eval_prefix_convention():
    for s in EVAL_SPECS:
        assert s.name.startswith("eval_"), f"Eval spec {s.name} should start with eval_"
    for s in TRAIN_SPECS:
        assert not s.name.startswith("eval_"), f"Train spec {s.name} should not start with eval_"
