from itertools import product

import numpy as np
import pytest

from benchmarks.corpus.fori_agent_cases import dataset


def test_frozen_families_and_sizes():
    configs = dataset.check_manifest()["configurations"]
    train = [r for r in configs if r["partition"] == "train"]
    held = [r for r in configs if r["partition"] == "eval"]
    assert train and held
    for key in ("workload", "family", "body_hash"):
        assert {r[key] for r in train}.isdisjoint(r[key] for r in held)
    for case in dataset.CASES:
        rows = [r for r in configs if r["workload"] == case.name]
        expected = set(product(dataset.SIZES, dataset.TRIPS))
        assert len(rows) == len(expected)
        assert {(r["variant"], r["trip_count"]) for r in rows} == expected
        assert all(r["family"] == case.family and r["partition"] == case.partition for r in rows)


@pytest.mark.parametrize("split", dataset.SPLITS)
def test_single_split_training_interface(split):
    frame, labels = dataset.load(split)
    assert set(frame.partition) == {split}
    assert len(frame) == len(dataset.settings(split))
    assert not frame.duplicated(["workload", "variant", "trip_count"]).any()
    assert labels.shape == (len(frame), 4, 2)
    assert np.all(np.isfinite(labels) & (labels > 0))
    assert set(frame.factor) == {1}


def test_cannot_accidentally_request_combined_splits():
    with pytest.raises(ValueError, match="unknown split"):
        dataset.load("all")
