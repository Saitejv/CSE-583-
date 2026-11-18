"""Expanded Phase 3 development corpus of independent JAX loop bodies.

The corpus is deliberately separate from ``benchmarks.workloads`` and
``benchmarks.recovery_workloads`` so the frozen Week 3 split and the
one-shot recovery confirmation records stay byte-for-byte unchanged.
"""

from benchmarks.corpus.registry import (
    CORPUS_FAMILIES,
    CORPUS_WORKLOADS,
    build_workload,
    get_spec,
    iter_specs,
    manifest,
    tier_specs,
)

__all__ = [
    "CORPUS_FAMILIES",
    "CORPUS_WORKLOADS",
    "build_workload",
    "get_spec",
    "iter_specs",
    "manifest",
    "tier_specs",
]
