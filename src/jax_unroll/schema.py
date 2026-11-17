"""Shared dataset contract for factor sweeps.

Every sweep row is one (workload, variant, trip count, factor, process repeat)
measurement. Raw rows live in ``results/raw/sweep_<backend>_<date>.csv`` and are
aggregated by ``benchmarks/oracle.py``. See ``docs/data_schema.md`` for units
and measurement definitions.
"""

from __future__ import annotations

import os
from pathlib import Path

FACTORS: tuple[int, ...] = (1, 2, 4, 8)
TRIP_COUNTS: tuple[int, ...] = (16, 64, 100, 256, 1000, 1024)
REMAINDER_TRIP_COUNTS: tuple[int, ...] = (100, 1000)

FEATURE_VERSION = 1
DATASET_VERSION = 2

SWEEP_COLUMNS: tuple[str, ...] = (
    "workload",
    "family",
    "variant",
    "loop_kind",
    "backend",
    "device",
    "jax_version",
    "jaxlib_version",
    "trip_count",
    "shape",
    "dtype",
    "factor",
    "divides_trip",
    "process_repeat",
    "trace_ms",
    "lower_ms",
    "compile_ms",
    "cold_total_ms",
    "warm_median_us",
    "warm_p95_us",
    "warm_min_us",
    "warm_std_us",
    "warm_samples",
    "end_to_end_ms_R",
    "r_calls",
    "stablehlo_bytes",
    "hlo_bytes",
    "correct",
    "max_abs_diff",
    "error",
    "notes",
)

ORACLE_COLUMNS: tuple[str, ...] = (
    "workload",
    "family",
    "variant",
    "loop_kind",
    "backend",
    "device",
    "trip_count",
    "shape",
    "dtype",
    "divides_trip",
    "repeats",
    "factor_u1_cold_ms",
    "factor_u1_warm_us",
    "oracle_warm_factor",
    "oracle_warm_us",
    "warm_gain_vs_u1",
    "oracle_total_factor_R",
    "oracle_total_ms_R",
    "u1_total_ms_R",
    "total_gain_vs_u1",
    "factor_agreement_best_warm",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "loop_kind",
    "loop_primitive",
    "trip_count",
    "factor",
    "full_chunks",
    "remainder",
    "remainder_ratio",
    "divides_evenly",
    "has_tail",
    "factor_over_trip",
    "backend",
    "device",
    "dtype_bits",
    "B",
    "R",
    "num_consts",
    "num_carry",
    "num_loop_carried",
    "num_xs",
    "num_invars",
    "num_outvars",
    "num_equations",
    "unrolled_equations",
    "eq_elementwise",
    "eq_dot",
    "eq_reduce",
    "eq_gather_scatter",
    "eq_convert",
    "eq_other",
    "carry_bytes",
    "xs_bytes",
    "const_bytes",
    "input_bytes",
    "output_bytes",
    "intermediate_bytes",
    "max_tensor_bytes",
    "bytes_moved_proxy",
    "input_elements",
    "output_elements",
    "carry_elements",
    "max_rank",
    "carry_shapes",
    "const_shapes",
    "input_shapes",
    "output_shapes",
    "dtypes",
    "longest_path",
    "max_out_degree",
    "num_feed_forward_values",
    "dependence_density",
    "flops_proxy",
    "arithmetic_intensity_proxy",
    "equations_per_iteration",
    "body_hash",
    "extraction_ms",
)

DATASET_KEY_COLUMNS: tuple[str, ...] = (
    "workload",
    "family",
    "variant",
    "loop_kind",
    "backend",
    "device",
    "trip_count",
    "dtype",
    "factor",
)

DATASET_LABEL_COLUMNS: tuple[str, ...] = (
    "repeats",
    "divides_trip",
    "correct_all",
    "trace_median_ms",
    "lower_median_ms",
    "compile_median_ms",
    "cold_median_ms",
    "warm_median_us",
    "warm_p95_us",
    "warm_min_us",
    "warm_std_us",
    "stablehlo_bytes_median",
    "hlo_bytes_median",
    "end_to_end_median_ms",
    "r_calls",
    "log_cold_ms",
    "log_warm_us",
)

DATASET_COLUMNS: tuple[str, ...] = (
    *DATASET_KEY_COLUMNS,
    "shape",
    "split",
    *(column for column in FEATURE_COLUMNS if column not in DATASET_KEY_COLUMNS),
    *DATASET_LABEL_COLUMNS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_RESULTS_DIR = PROJECT_ROOT / "results" / "raw"
PROCESSED_RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"


def default_sweep_path(backend: str, date: str) -> Path:
    return RAW_RESULTS_DIR / f"sweep_{backend}_{date}.csv"


def default_oracle_path(backend: str, date: str) -> Path:
    return PROCESSED_RESULTS_DIR / f"oracle_{backend}_{date}.csv"


def default_gate_path(backend: str, date: str) -> Path:
    return PROCESSED_RESULTS_DIR / f"risk_gate_{backend}_{date}.md"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def git_commit() -> str:
    try:
        import subprocess

        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def utc_timestamp() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
