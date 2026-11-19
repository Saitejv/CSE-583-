import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks.corpus.fori_proposal.protocol import freeze
from benchmarks.corpus.fori_proposal.score import repeated_winner
from benchmarks.corpus.fori_proposal.workloads import settings
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.fori_model import ForiCostPredictor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def exported_model(tmp_path_factory):
    source = ROOT / "results/fori_predictor/model.joblib"
    export = tmp_path_factory.mktemp("compact_fori") / "incumbent.npz"
    CompactForiPredictor.export(source, export)
    return source, export


def test_export_preserves_forecasts_across_archived_fori_configurations(exported_model):
    source, export = exported_model
    original = ForiCostPredictor.load(source)
    compact = CompactForiPredictor.load(export)
    assert compact.source_version == original.version
    assert compact.jax_versions == original.jax_versions
    for path in ("train_features.csv", "eval_features.csv"):
        with (ROOT / "results/fori_expansion" / path).open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["factor"] == "1"]
        assert rows
        for row in rows:
            assert compact.supports(row) == original.supports(row)
            assert compact.predict_candidates(row) == original.predict_candidates(row)
        assert not compact.supports({**rows[0], "loop_kind": "scan"})


def test_compact_load_in_clean_process_avoids_training_dependencies(exported_model):
    _, export = exported_model
    code = """
import sys
from jax_unroll.compact_fori import CompactForiPredictor
CompactForiPredictor.load(sys.argv[1])
assert {'sklearn', 'pandas', 'joblib'}.isdisjoint(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code, str(export)], cwd=ROOT, check=True)


def test_proposal_fresh_families_and_controls_are_distinct():
    freeze()
    matrix = settings()
    fresh = [cfg for cfg in matrix if cfg.cohort == "fresh"]
    controls = [cfg for cfg in matrix if cfg.cohort != "fresh"]
    assert len(fresh) == 32 and sum(cfg.live for cfg in fresh) == 24
    assert {cfg.spec.family for cfg in fresh}.isdisjoint(cfg.spec.family for cfg in controls)
    assert {cfg.cohort for cfg in controls} == {"proposal_control", "historical_control", "regression_control"}


def test_warm_opportunity_requires_repeatable_material_gain():
    samples = np.array([[10, 9, 4, 9.9], [10, 9, 4, 9.9], [10, 9, 11, 9.9]])
    assert repeated_winner(samples) == 1  # Factor 2 wins, despite apparent median win for factor 4.
    samples[:, 1] = 9.9
    assert repeated_winner(samples) == 0  # A small effect and an inconsistent effect.
