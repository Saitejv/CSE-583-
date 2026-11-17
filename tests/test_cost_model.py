from __future__ import annotations

import csv
import json
import math
import subprocess
import sys

import pytest

from benchmarks import evaluate
from jax_unroll.model import (
    CostModel, FACTOR_FEATURES, MODEL_FEATURES, candidate_features, group_configs, lofo_predictions, read_rows,
)
from jax_unroll.schema import DATA_DIR, PROJECT_ROOT
from jax_unroll.splits import DEV_FAMILIES, HELD_OUT_FAMILIES, lofo_folds
from jax_unroll.train import ARTIFACT_NAME

DATA = DATA_DIR / "features_cpu_2026-11-17.csv"
MODEL = PROJECT_ROOT / "models" / ARTIFACT_NAME


@pytest.fixture(scope="module")
def dev():
    return read_rows(DATA, split="dev")


@pytest.mark.parametrize("family", HELD_OUT_FAMILIES)
@pytest.mark.parametrize("entry", [CostModel().fit, lambda rows: lofo_predictions(rows, warm_terms="trip", warm_alpha=0.1)])
def test_fit_and_lofo_refuse_holdout_even_with_dev_tag(entry, family):
    with pytest.raises(ValueError, match="held-out families"):
        entry([{"family": family, "split": "dev"}])


def test_fit_rejects_wrong_split_tag(dev):
    with pytest.raises(ValueError, match="development rows"):
        CostModel().fit([{**dev[0], "split": "holdout"}])


def test_cost_predictions_are_finite_log_linear_and_factor_conditioned(dev):
    model = CostModel().fit(dev)
    cases = group_configs(dev)
    for case in cases:
        estimates = model.predict_candidates(case[1])
        assert set(estimates) == set(case)
        assert len(set(estimates.values())) == 4
        for factor, estimate in estimates.items():
            cold, warm = estimate.linear()
            assert all(math.isfinite(x) and x > 0 for x in (cold, warm))
            assert math.log(cold) == pytest.approx(estimate.log_cold_ms)
            assert math.log(warm) == pytest.approx(estimate.log_warm_us)
            assert estimate == model.predict(case[factor])


def test_selector_context_is_not_a_feature(dev):
    assert {"B", "R", "max_out_degree"}.isdisjoint(MODEL_FEATURES)
    model = CostModel.load(MODEL)
    assert model.predict(dev[0]) == model.predict({**dev[0], "B": 1, "R": 10_000})


def test_factor_features_from_single_trace(dev):
    row = candidate_features({**dev[0], "trip_count": 100, "num_equations": 3}, 8)
    assert tuple(row[key] for key in FACTOR_FEATURES) == (8, 12, 4, 0.04, False, True, 0.08, 24)


def test_lofo_uses_each_dev_family_once(dev):
    folds = lofo_predictions(dev, warm_terms="work_size", warm_alpha=0.1)
    assert folds.keys() == lofo_folds().keys()
    assert {row["family"] for _, rows in folds.values() for row in rows} == set(DEV_FAMILIES)
    assert sum(len(test) for _, test in folds.values()) == len(dev) == 140
    for name, (_, test) in folds.items():
        assert {row["family"] for row in test} == set(lofo_folds()[name]["test"])


def test_model_artifact_detects_modified_weights():
    artifact = json.loads(MODEL.read_text())
    artifact["model"]["regressions"][0]["weights"][0] += 1
    with pytest.raises(ValueError, match="corrupt"):
        CostModel.from_dict(artifact)


def test_train_cli_fits_and_writes_current_frozen_artifact(tmp_path):
    run = subprocess.run(
        [sys.executable, "-m", "jax_unroll.train", "--data", str(DATA), "--out", str(tmp_path)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    path = tmp_path / ARTIFACT_NAME
    artifact = json.loads(path.read_text())
    assert set(artifact["cv"]) == {*lofo_folds(), "pooled"}
    assert artifact["model"]["jax_version"] == artifact["model"]["jaxlib_version"] == "0.12.0.1"
    assert "140 dev rows" in run.stdout
    # Reproduce the formula and selected hyperparameters without depending on
    # platform-specific last-bit BLAS differences in the content hash.
    reproduced, frozen = CostModel.load(path).to_dict(), CostModel.load(MODEL).to_dict()
    assert reproduced["hyperparameters"] == frozen["hyperparameters"]
    for left, right in zip(reproduced["regressions"], frozen["regressions"], strict=True):
        assert left["intercept"] == pytest.approx(right["intercept"])
        assert left["weights"] == pytest.approx(right["weights"])


def test_evaluate_cli_smoke_on_four_configs(tmp_path, dev, monkeypatch):
    one_per_family = {}
    for config in group_configs(dev):
        one_per_family.setdefault(config[1]["family"], config)
    rows = [row for config in one_per_family.values() for row in config.values()]
    versions = {"jax_version", "jaxlib_version"}
    data = tmp_path / "tiny.csv"
    with data.open("w", newline="") as handle:
        fields = [key for key in rows[0] if key not in versions]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    data.with_suffix(".meta.json").write_text(DATA.with_suffix(".meta.json").read_text())
    model = CostModel().fit(rows)
    model_path, overhead_path = tmp_path / "model.json", tmp_path / "overhead.json"
    model_path.write_text(json.dumps({"model": model.to_dict()}))
    overhead_path.write_text(json.dumps({
        "model_version": model.version, "backend": "cpu", "pooled": {"miss_ms": {"median": 0.5}},
    }))
    monkeypatch.setattr(evaluate, "GATE_LOG", tmp_path / "gate.jsonl")
    output = tmp_path / "selector_cpu_test"
    assert evaluate.main([
        "--data", str(data), "--model", str(model_path), "--overhead", str(overhead_path),
        "--out", str(output), "--bootstraps", "20",
    ]) == 0
    with output.with_suffix(".csv").open(newline="") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 5 * len(evaluate.R_GRID) * len(evaluate.B_GRID) * len(evaluate.POLICIES)
    rows = [row for row in table if row["group"] == "dev_pooled"]
    assert {row["configurations"] for row in rows} == {"4"}
    assert {row["policy"] for row in table} == set(evaluate.POLICIES)
    assert "Dev leave-one-family-out" in output.with_suffix(".md").read_text()
    assert not evaluate.GATE_LOG.exists()
