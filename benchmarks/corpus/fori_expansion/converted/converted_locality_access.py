"""Converted locality access scan workloads."""

from __future__ import annotations

import benchmarks.corpus.registry as cr
from benchmarks.corpus.fori_expansion.converted.base import convert_scan_spec
from benchmarks.corpus.spec import CorpusSpec

SPECS: tuple[CorpusSpec, ...] = (
    convert_scan_spec(cr.CORPUS_WORKLOADS["locality_stream_contiguous"], related_group="sequential_locality", source_path="benchmarks/corpus/indirect.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["locality_stride_sample"], related_group="strided_locality", source_path="benchmarks/corpus/indirect.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["locality_random_window"], related_group="random_window_locality", source_path="benchmarks/corpus/indirect.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["locality_permutation_read"], related_group="permutation_memory", source_path="benchmarks/corpus/indirect.py"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
