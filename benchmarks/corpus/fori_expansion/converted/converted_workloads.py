"""Converted Phase 1 core scan workloads from benchmarks.workloads."""

from __future__ import annotations

import benchmarks.workloads as bw
from benchmarks.corpus.fori_expansion.converted.base import convert_scan_spec
from benchmarks.corpus.spec import CorpusSpec

SPECS: tuple[CorpusSpec, ...] = (
    convert_scan_spec(bw.WORKLOADS["dense_matvec"], related_group="dense_matvec", source_path="benchmarks/workloads.py"),
    convert_scan_spec(bw.WORKLOADS["elementwise_sine"], related_group="elementwise_sine", source_path="benchmarks/workloads.py"),
    convert_scan_spec(bw.WORKLOADS["indexed_update"], related_group="indexed_update", source_path="benchmarks/workloads.py"),
    convert_scan_spec(bw.WORKLOADS["reduction_vector"], related_group="reduction_vector", source_path="benchmarks/workloads.py"),
    convert_scan_spec(bw.WORKLOADS["scalar_recurrence"], related_group="scalar_recurrence", source_path="benchmarks/workloads.py"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
