from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.corpus.fori_expansion.registry import build_workload
from jax_unroll.fori_model import ForiCostPredictor
from jax_unroll.profile_model import WarmResidualProfile
from jax_unroll.schema import FACTORS
from jax_unroll.selector import DecisionEngine, select_factor


def test_saved_fori_model_on_live_loop():
    root = Path(__file__).resolve().parents[1]
    model = ForiCostPredictor.load(root / "results/fori_predictor/model.joblib")
    data = pd.read_csv(root / "results/fori_expansion/eval_features.csv")
    row = data.query("partition == 'fresh_eval' and factor == 1").iloc[0]
    loop = build_workload(row.workload, int(row.trip_count), 1, row.variant)
    costs = model.predict_candidates(row)
    target = model.predict(row.to_frame().T)[0]
    target[1:, 1] *= np.exp(model.margin)
    np.testing.assert_allclose([costs[u].linear() for u in FACTORS], target)
    result = DecisionEngine(model).decide(loop.fn, loop.args, trip_count=int(row.trip_count),
                                          loop_kind="fori_loop", throughput_only=True)
    assert result.fallback_reason is None
    assert result.factor == select_factor(costs, throughput_only=True)
    assert not model.supports({**row, "loop_kind": "scan"})


def test_warm_profile_matches_signature_and_unroll_remainder():
    root = Path(__file__).resolve().parents[1]
    reference = ForiCostPredictor.load(root / "results/fori_predictor/model.joblib")
    profile_data = pd.read_csv(root / "results/fori_expansion/eval_features.csv")
    row = profile_data.query("partition == 'fresh_eval' and factor == 1 and trip_count <= 512").iloc[0].to_dict()
    measured_warm = np.array([10, 8, 7, 6], dtype=float)
    profiled = WarmResidualProfile(reference, row, measured_warm)

    target = {**row, "trip_count": row["trip_count"] + 8}
    assert profiled.matches(target)
    expected = reference.predict_candidates(target)
    short = reference.predict_candidates(row)
    forecast = profiled.forecast(target)
    for factor, measured in zip(FACTORS, measured_warm, strict=True):
        value = expected[factor]
        assert forecast[factor].log_cold_ms == value.log_cold_ms
        offset = 0.5 * (np.log(measured) - short[factor].log_warm_us)
        np.testing.assert_allclose(forecast[factor].log_warm_us, value.log_warm_us + offset)
    for other in ({**target, "trip_count": target["trip_count"] + 1}, {**target, "body_hash": "different"}):
        assert not profiled.matches(other)
        assert profiled.predict_candidates(other) == reference.predict_candidates(other)
