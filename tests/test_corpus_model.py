from pathlib import Path

import numpy as np
import pandas as pd

from jax_unroll.corpus_model import CorpusPredictor
from jax_unroll.schema import FACTORS


def test_live_tree_traversal_reproduces_sklearn_costs():
    root = Path(__file__).resolve().parents[1]
    model = CorpusPredictor.load(root / "results/all_workloads/model.joblib")
    rows = pd.read_csv(root / "data/recovery/confirmation.csv").query("factor == 1")
    expected = model.predict(rows)
    for row, target in zip(rows.to_dict("records"), expected, strict=True):
        actual = model.predict_candidates(row)
        np.testing.assert_allclose([actual[u].linear() for u in FACTORS], target, rtol=1e-12)
        assert model.supports(row)
        assert not model.supports(dict(row, backend="gpu"))
