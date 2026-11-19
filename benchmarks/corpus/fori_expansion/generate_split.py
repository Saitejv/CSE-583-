"""Generate and validate the official split for the fori_loop expansion corpus."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from benchmarks.corpus.fori_expansion.registry import (
    CONVERTED_WORKLOAD_SPECS,
    HISTORICAL_FORI_SPECS,
    NEW_INDEPENDENT_SPECS,
    SPECS,
)
from benchmarks.corpus.fori_expansion.split import (
    compute_split_hash,
    validate_split_records,
)

FRESH_EVAL_NAMES = frozenset({
    "sig_pll_carrier",
    "sig_hilbert_analytic",
    "opt_mirror_descent",
    "opt_anderson_acceleration",
    "sp_ellpack_gemv",
    "sp_ilu0_precond_apply",
    "graph_mesh_flux_assembly",
    "graph_greedy_coloring",
    "dp_banded_smith_waterman",
    "dp_knapsack_bounded",
    "sim_verlet_symplectic",
    "sim_shallow_water_1d",
    "tiled_online_softmax",
    "tiled_cholesky_update",
    "idx_permutation_cycle_walk",
    "idx_circular_fifo_buffer",
    "stream_hyperloglog_step",
    "sched_fsm_traffic_controller",
})

HISTORICAL_EVAL_NAMES = frozenset({
    # 6 historical fori_loop holdouts
    "jacobi_solver",
    "ring_feedback",
    "stencil_anisotropic_2d",
    "stencil_diffusion_1d_fori",
    "stencil_five_point_2d",
    "stencil_nine_point_2d",
    # 8 converted scan holdouts
    "conv_indexed_update",
    "conv_bucket_max",
    "conv_masked_rows",
    "conv_strided_mix",
    "conv_segment_carry_ema",
    "conv_segment_max_reset",
    "conv_segment_sum_reset",
    "conv_segment_weighted_mean",
})


def build_split_records() -> list[dict[str, Any]]:
    records = []
    historical_fori_set = {s.name for s in HISTORICAL_FORI_SPECS}
    converted_scan_set = {s.name for s in CONVERTED_WORKLOAD_SPECS}

    for spec in SPECS:
        name = spec.name
        if name in historical_fori_set:
            origin = "historical_fori"
            partition = "historical_eval" if name in HISTORICAL_EVAL_NAMES else "train"
            reason = "Historical fori_loop preserving original split lineage"
        elif name in converted_scan_set:
            origin = "converted_scan"
            partition = "historical_eval" if name in HISTORICAL_EVAL_NAMES else "train"
            reason = "Converted historical scan inheriting historical partition lineage"
        else:
            origin = "new_independent"
            if name in FRESH_EVAL_NAMES:
                partition = "fresh_eval"
                reason = "Reserved fresh evaluation body from unfamiliar family / unobserved mechanism"
            else:
                partition = "train"
                reason = "New independent body allocated to development / training"

        rec = {
            "workload": spec.name,
            "family": spec.family,
            "related_group": spec.related_group,
            "partition": partition,
            "mechanism": spec.mechanism,
            "tier": spec.tier,
            "origin": origin,
            "reason": reason,
        }
        records.append(rec)
    return records


def export_split(
    csv_path: str = "benchmarks/corpus/fori_expansion/split.csv",
    manifest_path: str = "benchmarks/corpus/fori_expansion/manifest.json",
) -> tuple[Path, Path, str]:
    records = build_split_records()

    # Load historical split to pass to validator
    hist_split_file = Path("results/all_workloads/split.csv")
    hist_map = {}
    if hist_split_file.exists():
        with hist_split_file.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                hist_map[row["workload"]] = row["split"]

    # Validate
    validation_summary = validate_split_records(records, historical_split=hist_map, enforce_quotas=True)
    split_hash = compute_split_hash(records)

    # Save split.csv
    out_csv = Path(csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["workload", "family", "related_group", "partition", "mechanism", "tier", "origin", "reason"],
        )
        writer.writeheader()
        writer.writerows(records)

    # Save manifest.json
    out_manifest = Path(manifest_path)
    manifest_data = {
        "version": 1,
        "split_hash": split_hash,
        "summary": validation_summary,
        "records": records,
    }
    out_manifest.write_text(json.dumps(manifest_data, indent=2, sort_keys=True))

    return out_csv, out_manifest, split_hash


if __name__ == "__main__":
    c, m, h = export_split()
    print(f"Exported split to {c} and {m}")
    print(f"Deterministic split hash: {h}")
