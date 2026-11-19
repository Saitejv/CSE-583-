"""Audit the retained static-bound fori_loop pilot. No fitting or new timings."""
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.corpus import pilot
from benchmarks.corpus.registry import SPECS, build_workload, related_groups
from benchmarks.corpus.static_experiment import design
from benchmarks.corpus.static_matched import SPECS as MATCHED_SPECS
from benchmarks.oracle import load_raw
from benchmarks.recovery_data import dump, require_cpu, sha
from jax_unroll.features import extract_features_from_workload
from jax_unroll.model import candidate_features
from jax_unroll.schema import FACTORS
from jax_unroll.static_features import extract_static_features

RAW = pilot.PILOT / "primary_1023_raw.csv"
FEATURES = Path("results/static_features/pilot_features.csv")
RATIOS = pilot.PILOT / "warm_ratios.csv"
OUT = Path("results/fori_dev/audit.json")


def audit_body(raw, retained, ratios):
    """Reconstruct one primary configuration from its factor × process blocks."""
    warm = raw.pivot(index="process_repeat", columns="factor", values="warm_median_us")
    cold = raw.pivot(index="process_repeat", columns="factor", values="cold_total_ms")
    if (set(warm.index) != set(range(pilot.REPEATS)) or set(warm.columns) != set(FACTORS)
            or not np.isfinite([warm, cold]).all() or (warm <= 0).any().any() or (cold <= 0).any().any()):
        raise ValueError("requires a complete positive factor by repeat matrix")
    cold_sum = raw[["trace_ms", "lower_ms", "compile_ms"]].sum(axis=1)
    np.testing.assert_allclose(raw.cold_total_ms, cold_sum, atol=2e-6, rtol=0)
    retained = retained.set_index("factor").loc[list(FACTORS)]
    np.testing.assert_allclose(np.log(cold.median()), retained.log_cold_ms, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.log(warm.median()), retained.log_warm_us, atol=1e-12, rtol=0)
    effects = np.log(warm.median() / warm[1].median())
    observed = ratios.set_index("factor").loc[list(FACTORS[1:])]
    np.testing.assert_allclose(effects.loc[list(FACTORS[1:])], observed.measured_log_ratio, atol=1e-12, rtol=0)
    paired = np.log(warm.div(warm[1], axis=0))
    by_factor = {}
    for u in FACTORS[1:]:
        values = paired[u].to_numpy()
        by_factor[u] = {
            "log_ratio_of_warm_medians": float(effects[u]),
            "paired_repeat_log_ratios": values.tolist(),
            "consistent_sign": bool(np.all(np.sign(values) == np.sign(values[0]))),
            "repeat_log_range": float(np.ptp(values)),
        }
    return {
        "relative_warm": by_factor,
        "repeat_cold_log_ranges": np.ptp(np.log(cold), axis=0).tolist(),
        "repeat_warm_log_ranges": np.ptp(np.log(warm), axis=0).tolist(),
        "repeat_warm_minimizers": warm.idxmin(axis=1).tolist(),
        "offline_factor8_vs_1": {
            "optimized_hlo_text_size_ratio": float(retained.hlo_bytes_median[8] / retained.hlo_bytes_median[1]),
            "warm_time_ratio": float(warm[8].median() / warm[1].median()),
        },
    }


def check_features(spec, retained):
    """Regenerate deterministic inference fields from just the public factor-1 loop."""
    workload = build_workload(spec.name, pilot.TRIP, 1, spec.primary_variant)
    one = extract_features_from_workload(workload)
    one.pop("extraction_ms")
    if one["loop_primitive"] != "scan" or one["num_carry"] != one["num_loop_carried"] + 1:
        raise ValueError("expected rolled static fori_loop, including one internal counter")
    extension = extract_static_features(workload)
    for row in retained.to_dict("records"):
        expected = {**candidate_features(one, int(row["factor"])), **extension}
        for field, value in expected.items():
            actual = row[field]
            if value is None:
                equal = pd.isna(actual)
            elif isinstance(value, float):
                equal = np.isclose(value, actual, rtol=1e-12, atol=1e-14)
            else:
                equal = value == actual
            if not equal:
                raise ValueError(f"{spec.name}, factor {row['factor']}: {field} differs from rolled trace")


def design_diagnostics(frame):
    one = frame.loc[frame.factor.eq(1)].set_index("workload")
    coords = pd.DataFrame({
        "log_work_over_8": np.log1p(one.flops_proxy) / 8,
        "log_bytes_per_work_over_8": np.log1p(one.bytes_moved_proxy / (1 + one.flops_proxy)) / 8,
    })
    design_reports = {}
    folds = pilot.family_components(SPECS, related_groups())
    for design_name in ("warm_refit", "all_static"):
        x = np.array([design(row, design_name) for row in frame.to_dict("records")])
        fold_report = {}
        for fold, members in folds.items():
            test = frame.workload.isin(members)
            if not test.any():
                continue
            training = x[~test]
            fold_report[fold] = {
                "train_rank": int(np.linalg.matrix_rank(training)),
                "train_bodies": int(frame.loc[~test, "workload"].nunique()),
            }
        singular = np.linalg.svd(x, compute_uv=False)
        rank = int(np.linalg.matrix_rank(x))
        design_reports[design_name] = {
            "columns": x.shape[1], "rank": rank,
            "condition_on_nonzero_subspace": float(singular[0] / singular[rank - 1]),
            "column_l2_norms": np.linalg.norm(x, axis=0).tolist(),
            "folds": fold_report,
        }
    ranges = {}
    for fold, members in folds.items():
        test = coords.index.isin(members)
        if not test.any():
            continue
        train = coords.loc[~test]
        outside = (coords.loc[test] < train.min()) | (coords.loc[test] > train.max())
        ranges[fold] = {
            "training_range": {f: [float(train[f].min()), float(train[f].max())] for f in coords},
            "out_of_range_bodies": {f: list(outside.index[outside[f]]) for f in coords},
            "count_outside_either": int(outside.any(axis=1).sum()),
        }
    return {
        "warm_coordinate_correlation": float(coords.corr().iloc[0, 1]),
        "warm_coordinate_ranges": ranges, "designs": design_reports,
    }


def main():
    require_cpu()
    eligible = {s.name: s for s in SPECS if s.loop_kind == "fori_loop"}
    source = load_raw(RAW)
    raw, dropped, _ = pilot.validate_acquisition(source.loc[source.workload.isin(eligible)])
    if dropped.keys() & eligible.keys():
        raise ValueError("eligible body missing from retained primary matrix")
    retained = pd.read_csv(FEATURES).query("loop_kind == 'fori_loop'")
    if set(retained.workload) != set(eligible):
        raise ValueError("retained feature scope differs from public corpus registry")
    ratios = pd.read_csv(RATIOS)
    body_audits = {}
    for name, spec in eligible.items():
        rows = retained.loc[retained.workload.eq(name)]
        if len(rows) != len(FACTORS) or set(rows.factor) != set(FACTORS):
            raise ValueError("expected exactly one complete primary configuration per body")
        check_features(spec, rows)
        body_audits[name] = audit_body(raw.loc[raw.workload.eq(name)], rows, ratios.loc[ratios.body.eq(name)])
    effects = [r for a in body_audits.values() for r in a["relative_warm"].values()]
    report = {
        "input_sha256": {str(p): sha(p) for p in (RAW, FEATURES, RATIOS, Path("benchmarks/corpus/manifest.json"))},
        "public_fori_bodies": len(eligible), "raw_repeats": len(raw), "aggregate_labels_checked": len(retained),
        "matched_public_fori_bodies": [s.name for s in MATCHED_SPECS if s.loop_kind == "fori_loop"],
        "body_audits": body_audits, **design_diagnostics(retained),
        "repeat_summary": {
            "effect_comparisons": len(effects), "consistent_direction": sum(r["consistent_sign"] for r in effects),
            "median_log_range": float(np.median([r["repeat_log_range"] for r in effects])),
            "median_absolute_aggregate_effect": float(np.median([abs(r["log_ratio_of_warm_medians"]) for r in effects])),
            "median_abs_aggregation_difference": float(np.median([
                abs(np.median(r["paired_repeat_log_ratios"]) - r["log_ratio_of_warm_medians"]) for r in effects
            ])),
            "median_absolute_cold_repeat_log_range": float(np.median([a["repeat_cold_log_ranges"] for a in body_audits.values()])),
            "median_absolute_warm_repeat_log_range": float(np.median([a["repeat_warm_log_ranges"] for a in body_audits.values()])),
            "bodies_same_warm_minimizer_all_repeats": sum(len(set(a["repeat_warm_minimizers"])) == 1 for a in body_audits.values()),
        },
    }
    dump(OUT, report)
    print(f"Audited {len(raw)} raw repeats, {len(retained)} aggregate labels and {len(effects)} warm ratios")


if __name__ == "__main__":
    main()
