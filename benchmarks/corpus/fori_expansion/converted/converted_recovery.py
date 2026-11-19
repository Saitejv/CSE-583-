"""Converted recovery scan workloads from benchmarks.recovery_workloads."""

from __future__ import annotations

import benchmarks.recovery_workloads as rw
from benchmarks.corpus.fori_expansion.converted.base import convert_scan_spec
from benchmarks.corpus.spec import CorpusSpec

SPECS: tuple[CorpusSpec, ...] = (
    convert_scan_spec(rw.RECOVERY_WORKLOADS["bucket_max"], related_group="bucket_max", source_path="benchmarks/recovery_workloads.py"),
    convert_scan_spec(rw.RECOVERY_WORKLOADS["gather_lookup"], related_group="gather_lookup", source_path="benchmarks/recovery_workloads.py"),
    convert_scan_spec(rw.RECOVERY_WORKLOADS["gather_rows"], related_group="gather_rows", source_path="benchmarks/recovery_workloads.py"),
    convert_scan_spec(rw.RECOVERY_WORKLOADS["masked_rows"], related_group="masked_rows", source_path="benchmarks/recovery_workloads.py"),
    convert_scan_spec(rw.RECOVERY_WORKLOADS["scatter_accumulate"], related_group="scatter_accumulate", source_path="benchmarks/recovery_workloads.py"),
    convert_scan_spec(rw.RECOVERY_WORKLOADS["scatter_blend"], related_group="scatter_blend", source_path="benchmarks/recovery_workloads.py"),
    convert_scan_spec(rw.RECOVERY_WORKLOADS["strided_mix"], related_group="strided_mix", source_path="benchmarks/recovery_workloads.py"),
    convert_scan_spec(rw.RECOVERY_WORKLOADS["window_read"], related_group="window_read", source_path="benchmarks/recovery_workloads.py"),
    convert_scan_spec(rw.RECOVERY_WORKLOADS["window_replace"], related_group="window_replace", source_path="benchmarks/recovery_workloads.py"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
