"""Partition by workflow; group the two level/trend estimators for validation."""

from benchmarks.corpus.fori_regret.workloads import Case
from benchmarks.corpus.fori_workflows import deepseek_cases as ds, gemini_cases as gm

TRAIN = (
    (ds.dc_blocker, "dc_rejection"),
    (ds.heikin_ashi, "ohlc_smoothing"),
    (ds.battery_ecm, "battery_rc"),
    (ds.holt_trend, "level_trend"),
    (gm.alpha_beta_tracker, "level_trend"),
    (gm.acoustic_resonator_bank, "modal_synthesis"),
)
TEST = (
    (gm.muskingum, "river_routing"),
    (gm.cumulative_trapezoid_integrator, "quadrature"),
    (gm.hysteretic_event_detector, "alarm_persistence"),
    (ds.rainflow_turning_points, "fatigue_preprocessing"),
    (ds.welford_online_moments, "online_moments"),
)


def settings(split):
    specs, trips = {"train": (TRAIN, (2048, 4096)), "test": (TEST, (2304, 3840))}[split]
    return [
        (
            Case(fn.__name__, family, split, fn),
            "turns_range" if fn is ds.rainflow_turning_points else "default",
            t,
        )
        for fn, family in specs
        for t in trips
    ]
