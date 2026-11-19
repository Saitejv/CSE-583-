from types import SimpleNamespace

import pandas as pd
import pytest

from benchmarks.corpus.pilot import COMPARISONS, REPEATS, TRIP, family_components, paired_mae_differences, validate_acquisition
from benchmarks.corpus.registry import SPECS, related_groups
from benchmarks.recovery_data import VERSIONS
from jax_unroll.schema import FACTORS


def test_components_are_transitive_and_disjoint():
    specs = [SimpleNamespace(name=name, family=family) for name, family in (
        ("first", "alpha"), ("second", "beta"), ("third", "beta"),
        ("fourth", "gamma"), ("other", "delta"),
    )]
    assert family_components(specs, {"left": ["first", "second"], "right": ["third", "fourth"]}) == {
        "alpha+beta+gamma": ["first", "fourth", "second", "third"], "delta": ["other"],
    }
    actual = family_components(SPECS, related_groups())
    assert len(actual) == 8
    assert len({member for members in actual.values() for member in members}) == sum(map(len, actual.values())) == len(SPECS)


def test_pilot_rejects_entire_body_when_a_repeat_is_missing():
    raw = pd.DataFrame([
        {
            "workload": spec.name, "family": spec.family, "variant": spec.primary_variant, "trip_count": TRIP,
            "loop_kind": spec.loop_kind, "factor": factor, "process_repeat": repeat, "cold_total_ms": 40,
            "warm_median_us": 10, "backend": "cpu", "device": "cpu:0", "correct": True, "error": "",
            "shape": "{}", "dtype": "float32", **VERSIONS,
        }
        for spec in SPECS[:2] for factor in FACTORS for repeat in range(REPEATS)
    ])
    target = SPECS[0].name
    valid, dropped, repeat_effects = validate_acquisition(raw.drop(index=raw.index[0]))
    assert dropped[target] == "missing, duplicate, or extra factor repeat"
    assert target not in valid["workload"].values
    assert target not in repeat_effects


def test_bootstrap_averages_within_body_first():
    rows = [
        {"body": name, "measured_log_ratio": 1.0, **dict.fromkeys(COMPARISONS, predicted)}
        for name, predicted in (("large", 1.0), ("large", 1.0), ("small", 3.0))
    ]
    result = paired_mae_differences(rows)["warm_refit"]
    assert result["mean"] == pytest.approx(0.0)
    assert result["ci95"] == [-1.0, 1.0]
