import numpy as np
import pandas as pd

from benchmarks.corpus.fori_regret.development import rank
from benchmarks.corpus.fori_regret.freeze import check
from benchmarks.corpus.fori_regret.live import break_even_calls
from benchmarks.corpus.fori_regret.workloads import cases


def test_rank_uses_absolute_regret_and_keeps_both_failure_modes():
    frame = pd.DataFrame({
        "source": ["test"] * 3,
        "family": ["test"] * 3,
        "workload": ["harmful", "missed", "good"],
        "variant": ["small"] * 3,
        "trip_count": [256] * 3,
        "body_hash": ["a", "b", "c"],
    })
    warm = np.array([[5, 10, 8, 4], [100, 80, 70, 90], [6, 3, 4, 5]])
    costs = np.stack((np.ones_like(warm), warm), axis=-1)
    result = rank(frame, costs, np.array([1, 0, 1]))
    assert result.workload.tolist() == ["missed", "harmful", "good"]
    assert result.regret_us.tolist() == [30, 6, 0]
    assert result["mode"].tolist() == ["missed_speedup", "harmful_unroll", "other"]


def test_frozen_split_withholds_whole_families():
    check()
    train = {case.family for case in cases("train").values()}
    holdout = {case.family for case in cases("holdout").values()}
    assert train and holdout and train.isdisjoint(holdout)


def test_break_even_requires_strict_savings_over_setup():
    assert break_even_calls(10., .5) == 22
    assert break_even_calls(1., 0.) is None
    assert break_even_calls(-1., .5) == 1
