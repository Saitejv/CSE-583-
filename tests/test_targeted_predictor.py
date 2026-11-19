import numpy as np
import pandas as pd
import pytest

from jax_unroll.schema import FACTORS
from jax_unroll.targeted_model import TargetedCostPredictor


@pytest.mark.parametrize("method", ["trees", "joint_forest", "upper_quartile", "cost_ratio", "kernel", "blend", "shared_effect", "boosted"])
def test_runtime_matches_batch_after_model_load(method, tmp_path):
    import joblib

    rows = pd.read_csv("results/fori_expansion/train_features.csv").query("factor == 1").iloc[:12]
    rng = np.random.default_rng(583)
    measured = np.exp(rng.normal(size=(len(rows), len(FACTORS), 2)))
    model = TargetedCostPredictor(method, margin=.04).fit(rows, measured)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = TargetedCostPredictor.load(path)
    row = rows.iloc[0]
    estimates = loaded.predict_candidates(row)
    expected = loaded.predict(rows.iloc[:1])[0]
    expected[1:, 1] *= np.exp(loaded.margin)
    np.testing.assert_allclose([estimates[u].linear() for u in FACTORS], expected)
