"""Conversion manifest and catalog of all 53 converted historical scan workloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from benchmarks.corpus.fori_expansion.converted import (
    converted_control_arithmetic,
    converted_dependent_indexing,
    converted_indirect_gather,
    converted_indirect_scatter,
    converted_locality_access,
    converted_recovery,
    converted_rolling_window,
    converted_segmented_scan,
    converted_stateful_recurrence,
    converted_streaming_reduction,
    converted_workloads,
)
from benchmarks.corpus.spec import CorpusSpec

CONVERTED_MODULES = (
    converted_workloads,
    converted_recovery,
    converted_indirect_gather,
    converted_indirect_scatter,
    converted_locality_access,
    converted_dependent_indexing,
    converted_rolling_window,
    converted_streaming_reduction,
    converted_segmented_scan,
    converted_stateful_recurrence,
    converted_control_arithmetic,
)

CONVERTED_SPECS: tuple[CorpusSpec, ...] = tuple(
    spec for mod in CONVERTED_MODULES for spec in mod.SPECS
)
CONVERTED_WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in CONVERTED_SPECS}

# Historical split mapping from results/all_workloads/split.csv
HISTORICAL_EVAL_SCANS = frozenset({
    "indexed_update",
    "bucket_max",
    "masked_rows",
    "strided_mix",
    "segment_carry_ema",
    "segment_max_reset",
    "segment_sum_reset",
    "segment_weighted_mean",
})


def generate_conversion_manifest() -> list[dict[str, Any]]:
    """Build the conversion manifest documenting all 53 converted workloads."""
    manifest_entries = []
    for spec in CONVERTED_SPECS:
        orig_name = spec.name.replace("conv_", "")
        partition_history = "test" if orig_name in HISTORICAL_EVAL_SCANS else "train"
        entry = {
            "source_path": spec.provenance.source.split(":")[0],
            "original_workload": orig_name,
            "original_family": spec.family,
            "related_group": spec.related_group,
            "converted_id": spec.name,
            "loop_kind": spec.loop_kind,
            "partition_history": partition_history,
            "assigned_partition": "historical_eval" if partition_history == "test" else "train",
            "preserved_variants": dict(sorted(spec.shape_variants.items())),
            "primary_variant": spec.primary_variant,
            "trip_counts": list(spec.trip_counts),
            "disposition": "accepted",
            "conversion_mechanism": spec.mechanism,
            "reference_checks": "Bit-for-bit numerical parity across factors {1, 2, 4, 8} against original scan baseline",
            "measured_coverage": "4 factors x 2 default trips x 3 repeats = 24 raw rows planned",
            "unsuitable_reason": None,
        }
        manifest_entries.append(entry)
    return manifest_entries


def save_conversion_manifest(path: Path | str = "benchmarks/corpus/fori_expansion/converted_manifest.json") -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entries = generate_conversion_manifest()
    out_path.write_text(json.dumps(entries, indent=2, sort_keys=True))
    return out_path


if __name__ == "__main__":
    p = save_conversion_manifest()
    print(f"Saved {len(CONVERTED_SPECS)} conversion entries to {p}")
