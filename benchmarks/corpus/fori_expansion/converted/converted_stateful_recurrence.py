"""Converted stateful recurrence scan workloads."""

from __future__ import annotations

import benchmarks.corpus.registry as cr
from benchmarks.corpus.fori_expansion.converted.base import convert_scan_spec
from benchmarks.corpus.spec import CorpusSpec

SPECS: tuple[CorpusSpec, ...] = (
    convert_scan_spec(cr.CORPUS_WORKLOADS["stateful_scalar_iir"], related_group="iir_filters", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["stateful_vector_iir"], related_group="iir_filters", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["stateful_gru_gate"], related_group="gated_recurrences", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["stateful_lstm_cell"], related_group="gated_recurrences", source_path="benchmarks/corpus/structured.py"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
