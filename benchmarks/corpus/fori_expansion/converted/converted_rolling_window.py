"""Converted rolling window scan workloads."""

from __future__ import annotations

import benchmarks.corpus.registry as cr
from benchmarks.corpus.fori_expansion.converted.base import convert_scan_spec
from benchmarks.corpus.spec import CorpusSpec

SPECS: tuple[CorpusSpec, ...] = (
    convert_scan_spec(cr.CORPUS_WORKLOADS["rolling_mean_shift"], related_group="rolling_moments", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["rolling_conv1d_fir"], related_group="rolling_weighted_windows", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["rolling_extreme_max"], related_group="rolling_extrema", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["rolling_variance_moments"], related_group="rolling_moments", source_path="benchmarks/corpus/structured.py"),
    convert_scan_spec(cr.CORPUS_WORKLOADS["rolling_dot_two_streams"], related_group="rolling_weighted_windows", source_path="benchmarks/corpus/structured.py"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
