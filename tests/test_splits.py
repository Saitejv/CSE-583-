from __future__ import annotations

import json

import pytest

from benchmarks.workloads import FAMILIES, WORKLOADS
from jax_unroll.splits import (
    ALL_FAMILIES,
    DEV_FAMILIES,
    HELD_OUT_FAMILIES,
    SPLIT_FROZEN_DATE,
    SPLIT_VERSION,
    assert_holdout_excluded,
    lofo_folds,
    manifest,
    split_for_family,
)

EXPECTED_DEV = (
    "recurrence",
    "elementwise",
    "reduction",
    "dense_linear_algebra",
)
EXPECTED_HELD_OUT = ("indexed_memory", "iterative_solver")
EXPECTED_ALL = EXPECTED_DEV + EXPECTED_HELD_OUT


def test_frozen_split_constants():
    assert SPLIT_VERSION == 1
    assert SPLIT_FROZEN_DATE == "2026-11-17"
    assert ALL_FAMILIES == EXPECTED_ALL
    assert DEV_FAMILIES == EXPECTED_DEV
    assert HELD_OUT_FAMILIES == EXPECTED_HELD_OUT


def test_families_are_partitioned():
    assert len(ALL_FAMILIES) == 6
    assert len(set(ALL_FAMILIES)) == len(ALL_FAMILIES)
    assert set(DEV_FAMILIES).isdisjoint(HELD_OUT_FAMILIES)
    assert set(DEV_FAMILIES) | set(HELD_OUT_FAMILIES) == set(ALL_FAMILIES)


def test_split_matches_workload_registry():
    assert set(ALL_FAMILIES) == set(FAMILIES)
    assert set(DEV_FAMILIES) | set(HELD_OUT_FAMILIES) == set(FAMILIES)
    assert {spec.family for spec in WORKLOADS.values()} == set(FAMILIES)

    for family in FAMILIES:
        assert split_for_family(family) in {"dev", "holdout"}
    for spec in WORKLOADS.values():
        assert split_for_family(spec.family) in {"dev", "holdout"}


@pytest.mark.parametrize("family", DEV_FAMILIES)
def test_split_for_dev_family(family):
    assert split_for_family(family) == "dev"


@pytest.mark.parametrize("family", HELD_OUT_FAMILIES)
def test_split_for_held_out_family(family):
    assert split_for_family(family) == "holdout"


@pytest.mark.parametrize(
    "family", ["", "Recurrence", "a_bogus_family", "indexed-memory"]
)
def test_split_for_unknown_family_raises_helpful_error(family):
    with pytest.raises(ValueError) as excinfo:
        split_for_family(family)

    message = str(excinfo.value)
    assert "unknown family" in message
    assert repr(family) in message
    assert "expected one of" in message


def test_lofo_folds_cover_each_dev_family():
    folds = lofo_folds()

    assert len(folds) == len(DEV_FAMILIES)
    assert set(folds) == {f"lofo_{family}" for family in DEV_FAMILIES}

    for family in DEV_FAMILIES:
        fold = folds[f"lofo_{family}"]

        assert fold["test"] == (family,)
        assert fold["train"] == tuple(
            other for other in DEV_FAMILIES if other != family
        )
        assert set(fold["train"]) == set(DEV_FAMILIES) - {family}
        assert len(fold["train"]) == len(DEV_FAMILIES) - 1
        assert fold["train"]
        assert set(fold["train"]).isdisjoint(HELD_OUT_FAMILIES)
        assert set(fold["test"]).isdisjoint(HELD_OUT_FAMILIES)

    observed_train = set().union(
        *(set(fold["train"]) for fold in folds.values())
    )
    observed_test = {fold["test"][0] for fold in folds.values()}
    assert observed_train == set(DEV_FAMILIES)
    assert observed_test == set(DEV_FAMILIES)
    assert (observed_train | observed_test).isdisjoint(HELD_OUT_FAMILIES)


def test_lofo_folds_are_deterministic():
    first = lofo_folds()
    second = lofo_folds()

    assert first == second

    first["lofo_recurrence"]["train"] = ("bogus",)
    assert lofo_folds() == second


def test_lofo_rejects_unknown_family():
    with pytest.raises(ValueError, match="unknown families"):
        lofo_folds(["a_bogus_family"])


@pytest.mark.parametrize("family", HELD_OUT_FAMILIES)
def test_lofo_rejects_held_out_family(family):
    with pytest.raises(ValueError, match="held-out"):
        lofo_folds([family])


@pytest.mark.parametrize("family", DEV_FAMILIES)
def test_lofo_rejects_single_dev_family(family):
    with pytest.raises(ValueError, match="at least two development families"):
        lofo_folds([family])


def test_lofo_rejects_empty_family_list():
    with pytest.raises(ValueError, match="at least one development family"):
        lofo_folds([])


@pytest.mark.parametrize(
    "families",
    [
        (),
        DEV_FAMILIES,
        DEV_FAMILIES[:2],
        ("recurrence", "elementwise"),
    ],
)
def test_assert_holdout_excluded_accepts_dev_subsets(families):
    assert_holdout_excluded(families) is None


def test_assert_holdout_excluded_accepts_iterables():
    assert_holdout_excluded(family for family in DEV_FAMILIES) is None


@pytest.mark.parametrize("family", HELD_OUT_FAMILIES)
def test_assert_holdout_excluded_rejects_each_held_out_family(family):
    with pytest.raises(ValueError) as excinfo:
        assert_holdout_excluded(["recurrence", family])

    message = str(excinfo.value)
    assert "held-out" in message
    assert family in message


def test_manifest_is_json_serializable_with_expected_fields():
    record = manifest()

    json.dumps(record)

    assert record["split_version"] == SPLIT_VERSION
    assert record["frozen_date"] == SPLIT_FROZEN_DATE
    assert record["all_families"] == list(ALL_FAMILIES)
    assert record["dev_families"] == list(DEV_FAMILIES)
    assert record["held_out_families"] == list(HELD_OUT_FAMILIES)
    assert record["final_dev"] == list(DEV_FAMILIES)
    assert record["final_holdout"] == list(HELD_OUT_FAMILIES)

    expected_folds = {
        name: {"train": list(fold["train"]), "test": list(fold["test"])}
        for name, fold in lofo_folds().items()
    }
    assert record["lofo_folds"] == expected_folds
    assert set(record["lofo_folds"]) == {
        f"lofo_{family}" for family in DEV_FAMILIES
    }


def test_manifest_json_round_trip():
    record = manifest()

    assert json.loads(json.dumps(record)) == record


def test_manifest_is_repeatable_and_does_not_mutate_split_state():
    first = manifest()
    second = manifest()

    assert first == second
    assert manifest() is not second

    first["dev_families"].append("bogus")
    first["lofo_folds"]["lofo_recurrence"]["train"].append("bogus")

    third = manifest()
    assert third == second
    assert third["dev_families"] == list(DEV_FAMILIES)
    for fold in third["lofo_folds"].values():
        assert set(fold["train"]).isdisjoint(HELD_OUT_FAMILIES)
        assert set(fold["test"]).isdisjoint(HELD_OUT_FAMILIES)
