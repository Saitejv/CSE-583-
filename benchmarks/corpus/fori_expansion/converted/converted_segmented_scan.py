"""Converted segmented scan workloads."""

from __future__ import annotations

import benchmarks.corpus.registry as cr
from benchmarks.corpus.fori_expansion.converted.base import convert_scan_spec
from benchmarks.corpus.spec import CorpusSpec

SPECS: tuple[CorpusSpec, ...] = (
    convert_scan_spec(cr.CORPUS_WORKLOADS["segment_sum_reset"], related_group="segmented_reset", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["segment_max_reset"], related_group="segmented_reset", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["segment_weighted_mean"], related_group="segmented_reset", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["segment_carry_ema"], related_group="segmented_carry", source_path="benchmarks/corpus/structured.py"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
