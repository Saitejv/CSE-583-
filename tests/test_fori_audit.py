import numpy as np
import pandas as pd
import pytest

from benchmarks.corpus.fori_audit import audit_body
from jax_unroll.schema import FACTORS


def test_raw_labels_and_matched_repeat_effects_use_distinct_aggregations():
    # For factor 2, the ratio of medians is 2.5 while the median of ratios is 1.5.
    times = {1: [1, 2, 100], 2: [5, 3, 10], 4: [2, 4, 200], 8: [1, 2, 100]}
    raw = pd.DataFrame([
        dict(factor=u, process_repeat=r, warm_median_us=t, cold_total_ms=3.,
             trace_ms=.5, lower_ms=.5, compile_ms=2.)
        for u in FACTORS for r, t in enumerate(times[u])
    ]).sample(frac=1., random_state=583)
    retained = pd.DataFrame([
        dict(factor=u, log_warm_us=np.log(np.median(times[u])), log_cold_ms=np.log(3.),
             hlo_bytes_median=1000)
        for u in FACTORS
    ])
    ratios = pd.DataFrame([
        dict(factor=u, measured_log_ratio=np.log(np.median(times[u]) / 2)) for u in FACTORS[1:]
    ])
    factor_two = audit_body(raw, retained, ratios)["relative_warm"][2]
    assert factor_two["log_ratio_of_warm_medians"] == pytest.approx(np.log(2.5))
    assert factor_two["paired_repeat_log_ratios"] == pytest.approx(np.log([5., 1.5, .1]))
    assert not factor_two["consistent_sign"]

    with pytest.raises(ValueError, match="complete positive"):
        audit_body(raw.iloc[1:], retained, ratios)
