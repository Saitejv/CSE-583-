"""Reviewed partition by mechanism. All sizes and trip counts stay together."""

from benchmarks.corpus.fori_agent_cases import agy_eval, agy_train, opencode_eval, opencode_train, opencode_unmixing
from benchmarks.corpus.fori_regret.workloads import Case


def case(factory, family, partition):
    return Case(factory.__name__, family, partition, factory)


CASES = (
    case(opencode_train.order_book_depth, "indexed_order_depth", "train"),
    case(opencode_train.halftone_modulator, "quantization_error_feedback", "train"),
    case(agy_train.tile_stipple, "quantization_error_feedback", "train"),
    case(opencode_train.fog_transmittance, "optical_compositing", "train"),
    case(opencode_train.tournament_rating, "paired_rating_updates", "train"),
    case(agy_train.soft_kmeans, "online_centroid_assignment", "train"),
    case(agy_train.traffic_signal_debt, "cyclic_pressure_scheduler", "train"),
    case(opencode_eval.queue_workload, "reflected_resource_balance", "eval"),
    case(agy_eval.inventory_lost_sales, "reflected_resource_balance", "eval"),
    case(opencode_eval.warranty_ladder, "age_cohort_survival", "eval"),
    case(opencode_eval.bitplane_motion, "rolling_bit_motion", "eval"),
    case(agy_eval.flood_duration, "threshold_excursion_duration", "eval"),
    case(opencode_unmixing.adaptive_unmixing, "nonnegative_spectral_unmixing", "eval"),
)
