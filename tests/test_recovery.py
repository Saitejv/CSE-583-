import math

import jax
import pytest

from benchmarks.oracle import load_raw
from benchmarks.recovery_data import DATA, MODEL, complete_configs, joined_rows
from benchmarks.recovery_metrics import confirmation_summary, paired_rows
from benchmarks.recovery_train import folds
from benchmarks.recovery_workloads import RECOVERY_WORKLOADS
from benchmarks.workloads import WORKLOADS, build_workload
from jax_unroll.features import extract_features_from_workload
from jax_unroll.model import CostModel, LogCosts
from jax_unroll.recovery_split import CONFIRMATION, DEV, NEW_DEV
from jax_unroll.schema import FACTORS
from jax_unroll.selector import DecisionEngine
from jax_unroll.validation import Forecast


def test_recovery_groups_are_disjoint_and_lofo_keeps_workloads_whole():
    rows, _ = joined_rows(DATA / "dev.csv", "dev")
    assert not set(DEV) & set(CONFIRMATION)
    assert not set(DEV.values()) & set(CONFIRMATION.values())
    assert not set(RECOVERY_WORKLOADS) & set(WORKLOADS)
    assert len(rows) == (35 + 36) * len(FACTORS)
    for family, (train, test) in folds(rows).items():
        assert {row["family"] for row in test} == {family}
        assert not {row["workload"] for row in train} & {row["workload"] for row in test}
        assert len(train) + len(test) == len(rows)
    for workload, family in (*CONFIRMATION.items(), ("indexed_update", "indexed_memory")):
        with pytest.raises(ValueError, match="recovery requires pinned CPU dev rows"):
            CostModel().fit_recovery([dict(rows[0], workload=workload, family=family)])


def test_recovery_body_hashes_are_distinct():
    hashes = {
        extract_features_from_workload(build_workload(name, 64, 1))["body_hash"]
        for name in RECOVERY_WORKLOADS
    }
    assert len(hashes) == len(NEW_DEV) + len(CONFIRMATION)


def test_partial_raw_configuration_drops_all_factors():
    raw = load_raw(DATA / "dev_raw.csv")
    raw = raw.assign(error=raw["error"].astype("string"))
    valid, dropped = complete_configs(raw.drop(index=raw.index[0]), "dev")
    assert len(valid) == 35
    assert dropped == [{
        "configuration": tuple(raw.iloc[0][["workload", "variant", "trip_count"]]),
        "reason": "missing, extra or duplicate factor repeat",
    }]


def test_gate_requires_overhead_and_joint_feasibility():
    case = Forecast(
        {"workload": "synthetic", "family": "synthetic", "variant": "n64", "trip_count": 64, "num_equations": 2},
        {1: (5, 10), 2: (8, 8), 4: (12, 12), 8: (120, 11)},
        {factor: LogCosts(math.log(cold), math.log(warm)) for factor, (cold, warm) in {
            1: (5, 10), 2: (8, 8), 4: (12, 12), 8: (120, 11),
        }.items()},
        False,
    )
    key = ("synthetic", "n64", 64)
    rows = paired_rows([case], {key: 1.0})
    by_setting = {(row["R"], row["B_ms"]): row for row in rows}
    assert by_setting[(10_000, "unlimited")]["paired_win"]
    assert by_setting[(10_000, 100)]["llvm_size_violates_budget"]
    assert not by_setting[(10_000, 100)]["paired_win"]
    assert by_setting[(10_000, "unlimited")]["selector_total_ms"] == pytest.approx(89)
    assert confirmation_summary(rows)["gate_passed"]
    assert not confirmation_summary(paired_rows([case], {key: 250.0}))["gate_passed"]


def test_recovery_artifact_uses_static_features_and_signature_cache(monkeypatch):
    model = CostModel.load(MODEL)
    workload = build_workload("window_replace", 256, 1)
    options = {"trip_count": 256, "loop_kind": "scan", "B": 100, "R": 100_000, "throughput_only": True}
    engine = DecisionEngine(model)

    def unexpected(*args, **kwargs):
        pytest.fail("decision path must not compile candidates or retrace a known signature")

    monkeypatch.setattr(jax, "jit", unexpected)
    result = engine.decide(workload.fn, workload.args, **options)
    assert model.version.startswith("cpu-recovery-v1-")
    assert result.factor in FACTORS and result.fallback_reason is None
    monkeypatch.setattr(jax, "make_jaxpr", unexpected)
    hit = engine.decide(signature=result.signature, **options)
    assert hit.cache_hit and hit.factor == result.factor
