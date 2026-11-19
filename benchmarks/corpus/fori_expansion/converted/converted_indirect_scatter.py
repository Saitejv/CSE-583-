"""Converted indirect scatter scan workloads."""

from __future__ import annotations

import benchmarks.corpus.registry as cr
import benchmarks.corpus.static_matched as sm
from benchmarks.corpus.fori_expansion.converted.base import convert_scan_spec
from benchmarks.corpus.spec import CorpusSpec

SPECS: tuple[CorpusSpec, ...] = (
    convert_scan_spec(cr.CORPUS_WORKLOADS["scatter_unique_rows"], related_group="unique_update", source_path="benchmarks/corpus/indirect.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["scatter_collision_add"], related_group="collision_memory", source_path="benchmarks/corpus/indirect.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["scatter_two_tables"], related_group="split_table_memory", source_path="benchmarks/corpus/indirect.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["scatter_permutation"], related_group="permutation_memory", source_path="benchmarks/corpus/indirect.py"),
    convert_scan_spec(sm.WORKLOADS["matched_scatter_add"], related_group="collision_memory", source_path="benchmarks/corpus/static_matched.py"),
    convert_scan_spec(sm.WORKLOADS["matched_scatter_min"], related_group="collision_memory", source_path="benchmarks/corpus/static_matched.py"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
