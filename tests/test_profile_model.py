import math

import pytest

from jax_unroll.profile_model import ProfileCostModel
from jax_unroll.schema import FACTORS


def profile_rows():
    base = dict(workload="example", family="recurrence", variant="small", backend="cpu",
                device="cpu:0", jax_version="1", jaxlib_version="1", loop_kind="scan",
                body_hash="abc", carry_shapes="[]", const_shapes="[]", input_shapes="[]",
                output_shapes="[]", dtypes="float32", trip_count=1023, dtype="float32")
    return [dict(base, factor=u, cold_median_ms=10*u, warm_median_us=80/u) for u in FACTORS]


def test_transfer_and_scope():
    rows = profile_rows()
    model = ProfileCostModel(rows)
    query = dict(rows[0], trip_count=775)
    costs = model.predict_candidates(query)
    assert costs[4].linear() == pytest.approx((40, 20*775/1023))
    for mismatch in (dict(trip_count=1024), dict(trip_count=511), dict(trip_count=1031), dict(variant="different"),
                     dict(jaxlib_version="2"), dict(body_hash="different")):
        assert model.predict_candidates({**query, **mismatch}) is None


def test_invalid_profile():
    rows = profile_rows()
    rows[1]["warm_median_us"] = math.nan
    with pytest.raises(ValueError, match="invalid profile timing"):
        ProfileCostModel(rows)


def test_choose_uses_named_profile_on_live_trace():
    import csv
    from pathlib import Path

    from benchmarks.corpus.registry import build_workload

    path = Path(__file__).resolve().parents[1] / "results/static_features/pilot_features.csv"
    with path.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["workload"] == "rolling_mean_shift"]
    for row in rows:
        row["warm_median_us"] = 2 if int(row["factor"]) == 4 else 10
    model = ProfileCostModel(rows)
    loop = build_workload("rolling_mean_shift", 1015, 1, "w8")
    options = dict(workload="rolling_mean_shift", variant="w8", trip_count=1015,
                   loop_kind=loop.loop_kind, throughput_only=True)
    assert model.choose(loop.fn, loop.args, **options) == 4
    assert model.choose(loop.fn, loop.args, **{**options, "variant": "unknown"}) == 1
