"""Converted indirect gather scan workloads."""

from __future__ import annotations

import benchmarks.corpus.registry as cr
import benchmarks.corpus.static_matched as sm
from benchmarks.corpus.fori_expansion.converted.base import convert_scan_spec
from benchmarks.corpus.spec import CorpusSpec

SPECS: tuple[CorpusSpec, ...] = (
    convert_scan_spec(cr.CORPUS_WORKLOADS["gather_scalar_table"], related_group="random_scalar_gather", source_path="benchmarks/corpus/indirect.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["gather_row_pool"], related_group="random_row_gather", source_path="benchmarks/corpus/indirect.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["gather_multi_table"], related_group="select_gather", source_path="benchmarks/corpus/indirect.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["gather_duplicate_index"], related_group="collision_memory", source_path="benchmarks/corpus/indirect.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["gather_sorted_blocks"], related_group="ordered_gather", source_path="benchmarks/corpus/indirect.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["gather_masked_window"], related_group="masked_memory", source_path="benchmarks/corpus/indirect.py"),
    convert_scan_spec(sm.WORKLOADS["matched_gather"], related_group="random_scalar_gather", source_path="benchmarks/corpus/static_matched.py"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
