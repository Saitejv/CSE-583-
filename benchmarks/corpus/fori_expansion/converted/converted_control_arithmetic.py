"""Converted control arithmetic scan workloads."""

from __future__ import annotations

import benchmarks.corpus.registry as cr
from benchmarks.corpus.fori_expansion.converted.base import convert_scan_spec
from benchmarks.corpus.spec import CorpusSpec

SPECS: tuple[CorpusSpec, ...] = (
    convert_scan_spec(cr.CORPUS_WORKLOADS["control_polynomial_forcing"], related_group="elementwise_control", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["control_transcendental"], related_group="elementwise_control", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["control_bisection_bracket"], related_group="root_finding_control", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["control_lowrank_update"], related_group="lowrank_control", source_path="benchmarks/corpus/structured.py"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
