"""Converted streaming reduction scan workloads."""

from __future__ import annotations

import benchmarks.corpus.registry as cr
from benchmarks.corpus.fori_expansion.converted.base import convert_scan_spec
from benchmarks.corpus.spec import CorpusSpec

SPECS: tuple[CorpusSpec, ...] = (
    convert_scan_spec(cr.CORPUS_WORKLOADS["streaming_running_mean"], related_group="streaming_moments", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["streaming_welford_var"], related_group="streaming_moments", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["streaming_decayed_extrema"], related_group="streaming_extrema", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["streaming_logsumexp"], related_group="streaming_normalizers", source_path="benchmarks/corpus/structured.py"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
