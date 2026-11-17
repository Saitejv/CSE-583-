"""Aggregate raw sweep rows into an oracle table and Week 1 risk-gate decision.

Input: ``results/raw/sweep_<backend>_<date>.csv`` produced by
``benchmarks/sweep.py``. Output: a per-configuration oracle CSV/Markdown table
and a risk-gate Markdown report. The oracle is built only from measured data;
no factor is benchmarked at decision time.
"""

from __future__ import annotations

import argparse
import statistics
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from jax_unroll.schema import (
    ORACLE_COLUMNS,
    PROJECT_ROOT,
    default_gate_path,
    default_oracle_path,
    ensure_parent,
    utc_timestamp,
)

CONFIG_KEY = (
    "workload",
    "family",
    "variant",
    "loop_kind",
    "backend",
    "device",
    "trip_count",
    "shape",
    "dtype",
)
GROUP_KEY = (*CONFIG_KEY, "factor")

NUMERIC_COLUMNS = (
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
    "stablehlo_bytes",
    "hlo_bytes",
    "max_abs_diff",
    "r_calls",
)

AGGREGATED_COLUMNS = (
    *CONFIG_KEY,
    "divides_trip",
    "factor",
    "repeats",
    "cold_median_ms",
    "trace_median_ms",
    "lower_median_ms",
    "compile_median_ms",
    "warm_median_us",
    "warm_p95_us",
    "warm_min_us",
    "warm_std_us",
    "end_to_end_median_ms",
    "stablehlo_bytes_median",
    "hlo_bytes_median",
)


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))


def load_raw(paths: Path | Iterable[Path]) -> pd.DataFrame:
    path_list = [paths] if isinstance(paths, Path) else list(paths)
    if not path_list:
        raise ValueError("at least one raw sweep file is required")
    frames: list[pd.DataFrame] = []
    for path in path_list:
        if not path.exists():
            raise FileNotFoundError(f"raw sweep file not found: {path}")
        frame = pd.read_csv(path)
        required = set(GROUP_KEY) | {"correct", "error", "cold_total_ms", "warm_median_us"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def aggregate_raw(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["_ok"] = (
        working["error"].astype("string").fillna("").str.strip().eq("")
    )
    working["_ok"] &= _as_bool(working["correct"])
    valid = working[working["_ok"]].copy()
    if valid.empty:
        raise ValueError("no valid, correct raw rows to aggregate")
    for column in NUMERIC_COLUMNS:
        if column in valid.columns:
            valid[column] = pd.to_numeric(valid[column], errors="coerce")

    rows: list[dict[str, Any]] = []
    for key, group in valid.groupby(list(GROUP_KEY), dropna=False):
        values = dict(zip(GROUP_KEY, key, strict=True))
        row: dict[str, Any] = dict(values)
        row["divides_trip"] = int(values["trip_count"]) % int(values["factor"]) == 0
        row["repeats"] = int(len(group))
        row["cold_median_ms"] = float(group["cold_total_ms"].median())
        row["trace_median_ms"] = float(group["trace_ms"].median())
        row["lower_median_ms"] = float(group["lower_ms"].median())
        row["compile_median_ms"] = float(group["compile_ms"].median())
        row["warm_median_us"] = float(group["warm_median_us"].median())
        row["warm_p95_us"] = float(group["warm_p95_us"].median())
        row["warm_min_us"] = float(group["warm_min_us"].median())
        row["warm_std_us"] = float(group["warm_std_us"].median())
        row["end_to_end_median_ms"] = float(group["end_to_end_ms_R"].median())
        row["stablehlo_bytes_median"] = float(group["stablehlo_bytes"].median())
        hlo = group["hlo_bytes"].dropna()
        row["hlo_bytes_median"] = float(hlo.median()) if not hlo.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=list(AGGREGATED_COLUMNS))


def _total_ms(row: Any, r_calls: int) -> float:
    return float(row.cold_median_ms) + r_calls * float(row.warm_median_us) / 1e3


def build_oracle(aggregated: pd.DataFrame, r_calls: int = 100) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in aggregated.groupby(list(CONFIG_KEY), dropna=False):
        values = dict(zip(CONFIG_KEY, key, strict=True))
        factors = {int(row.factor): row for row in group.itertuples()}
        if 1 not in factors:
            continue
        u1 = factors[1]
        best_warm = min(
            group.itertuples(), key=lambda row: (float(row.warm_median_us), int(row.factor))
        )
        best_total = min(
            group.itertuples(), key=lambda row: (_total_ms(row, r_calls), int(row.factor))
        )
        u1_total = _total_ms(u1, r_calls)
        best_total_ms = _total_ms(best_total, r_calls)
        warm_ratio = (
            float(u1.warm_median_us) / float(best_warm.warm_median_us)
            if float(best_warm.warm_median_us) > 0
            else float("nan")
        )
        total_ratio = u1_total / best_total_ms if best_total_ms > 0 else float("nan")
        rows.append(
            {
                **values,
                "divides_trip": bool(
                    int(values["trip_count"]) % int(best_warm.factor) == 0
                ),
                "repeats": int(u1.repeats),
                "factor_u1_cold_ms": round(float(u1.cold_median_ms), 6),
                "factor_u1_warm_us": round(float(u1.warm_median_us), 3),
                "oracle_warm_factor": int(best_warm.factor),
                "oracle_warm_us": round(float(best_warm.warm_median_us), 3),
                "warm_gain_vs_u1": round(warm_ratio, 6),
                "oracle_total_factor_R": int(best_total.factor),
                "oracle_total_ms_R": round(best_total_ms, 6),
                "u1_total_ms_R": round(u1_total, 6),
                "total_gain_vs_u1": round(total_ratio, 6),
                "factor_agreement_best_warm": int(
                    int(best_total.factor) == int(best_warm.factor)
                ),
            }
        )
    oracle = pd.DataFrame(rows, columns=list(ORACLE_COLUMNS))
    return oracle.sort_values(
        ["workload", "variant", "trip_count"], kind="stable"
    ).reset_index(drop=True)


def _cost_row(row: Any, r_calls: int) -> float:
    return _total_ms(row, r_calls)


def _dominant_factor(series: pd.Series) -> tuple[int, float]:
    counts = series.astype(int).value_counts()
    factor, count = min(
        ((int(factor), int(count)) for factor, count in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return factor, float(count) / float(len(series))


def gate_evidence(
    aggregated: pd.DataFrame,
    oracle: pd.DataFrame,
    *,
    r_calls: int = 100,
    threshold: float = 0.05,
) -> tuple[bool, list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    for workload, sub in oracle.groupby("workload", sort=False):
        dominant_warm, dominance_warm = _dominant_factor(sub["oracle_warm_factor"])
        dominant_total, dominance_total = _dominant_factor(
            sub["oracle_total_factor_R"]
        )

        warm_gains: list[float] = []
        warm_competitor: list[float] = []
        total_gains: list[float] = []
        total_competitor: list[float] = []
        for config_row in sub.itertuples():
            key = {
                "workload": config_row.workload,
                "family": config_row.family,
                "variant": config_row.variant,
                "loop_kind": config_row.loop_kind,
                "backend": config_row.backend,
                "device": config_row.device,
                "trip_count": config_row.trip_count,
                "shape": config_row.shape,
                "dtype": config_row.dtype,
            }
            group = aggregated
            for name, value in key.items():
                group = group[group[name] == value]
            if group.empty:
                continue
            by_factor = {int(row.factor): row for row in group.itertuples()}
            if 1 not in by_factor:
                continue
            u1 = by_factor[1]

            if int(config_row.oracle_warm_factor) == dominant_warm:
                chosen = by_factor.get(dominant_warm)
                if chosen is not None:
                    if dominant_warm == 1:
                        others = [
                            float(row.warm_median_us)
                            for factor, row in by_factor.items()
                            if factor != 1
                        ]
                        if others:
                            warm_gains.append(max(others) / float(u1.warm_median_us))
                    elif float(chosen.warm_median_us) > 0:
                        warm_gains.append(
                            float(u1.warm_median_us) / float(chosen.warm_median_us)
                        )
                    others = [
                        float(row.warm_median_us)
                        for factor, row in by_factor.items()
                        if factor != dominant_warm
                    ]
                    if others and float(chosen.warm_median_us) > 0:
                        warm_competitor.append(
                            min(others) / float(chosen.warm_median_us)
                        )

            if int(config_row.oracle_total_factor_R) == dominant_total:
                chosen = by_factor.get(dominant_total)
                if chosen is not None:
                    chosen_cost = _cost_row(chosen, r_calls)
                    u1_cost = _cost_row(u1, r_calls)
                    if dominant_total == 1:
                        others = [
                            _cost_row(row, r_calls)
                            for factor, row in by_factor.items()
                            if factor != 1
                        ]
                        if others and u1_cost > 0:
                            total_gains.append(max(others) / u1_cost)
                    elif chosen_cost > 0:
                        total_gains.append(u1_cost / chosen_cost)
                    others = [
                        _cost_row(row, r_calls)
                        for factor, row in by_factor.items()
                        if factor != dominant_total
                    ]
                    if others and chosen_cost > 0:
                        total_competitor.append(min(others) / chosen_cost)

        warm_gain = statistics.median(warm_gains) if warm_gains else 1.0
        total_gain = statistics.median(total_gains) if total_gains else 1.0
        warm_comp = statistics.median(warm_competitor) if warm_competitor else 1.0
        total_comp = statistics.median(total_competitor) if total_competitor else 1.0
        useful = dominance_warm >= 0.5 and warm_gain >= 1.0 + threshold
        evidence.append(
            {
                "workload": workload,
                "family": str(sub["family"].iloc[0]),
                "loop_kind": str(sub["loop_kind"].iloc[0]),
                "configs": int(len(sub)),
                "dominant_warm_factor": dominant_warm,
                "dominance_warm": round(dominance_warm, 3),
                "median_warm_gain_vs_u1": round(warm_gain, 4),
                "median_warm_gain_vs_best_competitor": round(warm_comp, 4),
                "dominant_total_factor_R": dominant_total,
                "dominance_total": round(dominance_total, 3),
                "median_total_gain_vs_u1": round(total_gain, 4),
                "median_total_gain_vs_best_competitor": round(total_comp, 4),
                "useful_choice": bool(useful),
            }
        )

    useful_factors = {
        row["dominant_warm_factor"] for row in evidence if row["useful_choice"]
    }
    passed = (
        sum(1 for row in evidence if row["useful_choice"]) >= 2
        and len(useful_factors) >= 2
    )
    return passed, evidence


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def oracle_markdown(
    oracle: pd.DataFrame,
    *,
    raw_path: str,
    backend: str,
    r_calls: int,
) -> str:
    columns = [
        "workload",
        "variant",
        "trip_count",
        "divides_trip",
        "factor_u1_warm_us",
        "oracle_warm_factor",
        "oracle_warm_us",
        "warm_gain_vs_u1",
        "oracle_total_factor_R",
        "total_gain_vs_u1",
    ]
    lines = [
        f"# Measured oracle table ({backend})",
        "",
        f"- Generated: {utc_timestamp()}",
        f"- Raw source: {raw_path}",
        f"- Total-cost objective uses R = {r_calls} warm calls.",
        "- Warm gain is `median(u=1 latency) / median(best-factor latency)`; "
        "total gain is the same ratio on `cold_total_ms + R * warm_median_us`.",
        "- `divides_trip` refers to the oracle warm factor; it is not part of the "
        "configuration key because it depends on the candidate factor.",
        "- Tie-breaking prefers the smaller factor.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in oracle.itertuples():
        lines.append(
            "| "
            + " | ".join(_fmt(getattr(row, column)) for column in columns)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def gate_markdown(
    passed: bool,
    evidence: list[dict[str, Any]],
    *,
    raw_path: str,
    backend: str,
    r_calls: int,
    threshold: float,
) -> str:
    lines = [
        f"# Week 1 risk gate ({backend})",
        "",
        f"- Generated: {utc_timestamp()}",
        f"- Raw source: {raw_path}",
        f"- Decision: **{'PASS' if passed else 'FAIL'}**",
        f"- Criterion: at least two workloads with distinct dominant warm factors, "
        f"each dominant for >= 50% of configurations and with a median warm gain "
        f">= {1.0 + threshold:.2f}x over factor 1. For a dominant factor of 1, the "
        f"gain compares u=1 against the worst candidate factor (the regression "
        f"avoided by not over-unrolling).",
        f"- Secondary total-cost view: R = {r_calls}; reported for the same workload.",
        "",
        "| workload | family | configs | dominant u (warm) | dominance | "
        "median gain over default | median gain vs best competitor | "
        "dominant u (total, R) | median total gain over default | useful |",
        "|---|---|---|---|---|---|---|---|---|---|",
        "",
        "For a dominant factor greater than 1, the gain column is "
        "`u=1 / chosen` (speedup over the default). For a dominant factor of 1, "
        "it is `worst other candidate / u=1`, the regression avoided by not "
        "over-unrolling. The total columns use the same rule on "
        "`cold_median_ms + R * warm_median_us / 1000`.",
        "",
    ]
    for row in evidence:
        lines.append(
            "| {workload} | {family} | {configs} | {dw} | {dom} | {gw} | {gw_c} | "
            "{dt} | {gt} | {useful}".format(
                workload=row["workload"],
                family=row["family"],
                configs=row["configs"],
                dw=row["dominant_warm_factor"],
                dom=_fmt(row["dominance_warm"]),
                gw=_fmt(row["median_warm_gain_vs_u1"], 4),
                gw_c=_fmt(row["median_warm_gain_vs_best_competitor"], 4),
                dt=row["dominant_total_factor_R"],
                gt=_fmt(row["median_total_gain_vs_u1"], 4),
                useful=_fmt(row["useful_choice"]),
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes and limitations",
            "",
            "- This is a feasibility gate on one CPU device, not a held-out evaluation.",
            "- Rows are medians over repeated fresh-process measurements; see the raw CSV.",
            "- Optimized-HLO size is a proxy for generated code size and is not a",
            "  correctness or performance claim.",
            "- Phase 4 must audit optimized HLO before attributing gains to JAX-level",
            "  unrolling; XLA may unroll or fuse independently.",
            "",
        ]
    )
    return "\n".join(lines)


def _latest_raw() -> list[Path]:
    candidates = sorted(
        (PROJECT_ROOT / "results" / "raw").glob("sweep_*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    primary = [
        path
        for path in candidates
        if not any(marker in path.stem for marker in ("_quick", "_shapes"))
    ]
    if not candidates:
        raise FileNotFoundError(
            "no results/raw/sweep_*.csv found; run `make sweep` first"
        )
    return [(primary or candidates)[-1]]


def run(args: argparse.Namespace) -> int:
    raw_paths = [Path(path) for path in args.raw] if args.raw else _latest_raw()
    raw = load_raw(raw_paths)
    raw_display = ", ".join(f"`{path}`" for path in raw_paths)
    backend = str(raw["backend"].dropna().iloc[0]) if "backend" in raw else "cpu"
    run_date = args.date or date.today().isoformat()
    out_path = (
        Path(args.out)
        if args.out
        else default_oracle_path(backend, run_date)
    )
    md_path = (
        Path(args.md_out)
        if args.md_out
        else out_path.with_suffix(".md")
    )
    gate_path = (
        Path(args.gate_out)
        if args.gate_out
        else default_gate_path(backend, run_date)
    )

    aggregated = aggregate_raw(raw)
    oracle = build_oracle(aggregated, r_calls=args.r_calls)
    passed, evidence = gate_evidence(
        aggregated,
        oracle,
        r_calls=args.r_calls,
        threshold=args.threshold,
    )

    ensure_parent(out_path)
    oracle.to_csv(out_path, index=False)
    ensure_parent(md_path)
    md_path.write_text(
        oracle_markdown(
            oracle, raw_path=raw_display, backend=backend, r_calls=args.r_calls
        ),
        encoding="utf-8",
    )
    ensure_parent(gate_path)
    gate_path.write_text(
        gate_markdown(
            passed,
            evidence,
            raw_path=raw_display,
            backend=backend,
            r_calls=args.r_calls,
            threshold=args.threshold,
        ),
        encoding="utf-8",
    )

    print(f"oracle: wrote {len(oracle)} configuration rows to {out_path}")
    print(f"oracle: human-readable table at {md_path}")
    print(f"oracle: risk gate {'PASS' if passed else 'FAIL'} -> {gate_path}")
    columns = [
        "workload",
        "dominant_warm_factor",
        "dominance_warm",
        "median_warm_gain_vs_u1",
        "median_warm_gain_vs_best_competitor",
        "dominant_total_factor_R",
        "median_total_gain_vs_u1",
        "useful_choice",
    ]
    print(pd.DataFrame(evidence)[columns].to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", nargs="+", default=None, help="one or more raw sweep CSV paths"
    )
    parser.add_argument("--out", default=None, help="oracle CSV output path")
    parser.add_argument("--md-out", default=None, help="oracle Markdown output path")
    parser.add_argument("--gate-out", default=None, help="risk-gate Markdown output path")
    parser.add_argument("--r-calls", type=int, default=100)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="minimum fractional gain over factor 1 for a useful choice",
    )
    parser.add_argument("--date", default=None, help="date suffix for default outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
