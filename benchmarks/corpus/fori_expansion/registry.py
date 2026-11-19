"""Unified expansion registry for all 136 fori_loop workloads.

Includes:
- 53 converted historical scan workloads
- 13 historical fori_loop workloads
- 70 newly authored independent fori_loop workloads across 10 workstreams
Total: 136 distinct fori_loop bodies.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

import benchmarks.workloads as bw
import benchmarks.recovery_workloads as rw
import benchmarks.corpus.registry as cr
from benchmarks.corpus.fori_expansion import (
    dynamic_programming,
    graph_mesh,
    index_memory,
    numerical_simulation,
    optimization_estimation,
    scheduling_control,
    signal_processing,
    sparse_linear_algebra,
    streaming_compression,
    tiled_tensor,
)
from benchmarks.corpus.fori_expansion.converted.manifest import CONVERTED_SPECS
from benchmarks.corpus.spec import (
    CORPUS_FACTORS,
    CORPUS_REPEATS,
    CORPUS_TRIP_COUNTS,
    SMOKE_REPEATS,
    SMOKE_TRIP_COUNTS,
    CorpusSpec,
    Provenance,
)
from benchmarks.workloads import Workload, WorkloadSpec

# 1. 53 Converted scan specs
CONVERTED_WORKLOAD_SPECS: tuple[CorpusSpec, ...] = CONVERTED_SPECS

# 2. 13 Historical fori_loop specs (wrapped into CorpusSpec)
HISTORICAL_FORI_NAMES: tuple[str, ...] = (
    "jacobi_solver",
    "ring_feedback",
    "dep_bucket_stateful",
    "dep_linked_list",
    "dep_pointer_chase",
    "dep_two_level_indirection",
    "locality_block_reuse",
    "scatter_min_update",
    "stateful_coupled_oscillator",
    "stencil_anisotropic_2d",
    "stencil_diffusion_1d_fori",
    "stencil_five_point_2d",
    "stencil_nine_point_2d",
)


def _wrap_historical_fori(name: str) -> CorpusSpec:
    if name in cr.CORPUS_WORKLOADS:
        return cr.CORPUS_WORKLOADS[name]
    if name == "jacobi_solver":
        s = bw.WORKLOADS[name]
        return CorpusSpec(
            name=s.name, family=s.family, description=s.description,
            mechanism="iterative_solver", structural_difference="Historical 2D Jacobi iterative relaxation",
            loop_kind="fori_loop", shape_variants=s.shape_variants, primary_variant=s.primary_variant,
            factory=s.factory, seed=583_006, provenance=Provenance("original", "benchmarks/workloads.py", "Internal", "Phase 1 holdout solver"),
            tier="full", related_group="jacobi_solver", trip_counts=(16, 64, 100, 256, 1000, 1024),
        )
    if name == "ring_feedback":
        s = rw.RECOVERY_WORKLOADS[name]
        return CorpusSpec(
            name=s.name, family=s.family, description=s.description,
            mechanism="ring_feedback", structural_difference="Historical ring buffer with dynamic update slice",
            loop_kind="fori_loop", shape_variants=s.shape_variants, primary_variant=s.primary_variant,
            factory=s.factory, seed=583_204, provenance=Provenance("original", "benchmarks/recovery_workloads.py", "Internal", "Phase 2 confirmation buffer"),
            tier="full", related_group="ring_feedback", trip_counts=(16, 64, 100, 256, 1000, 1024),
        )
    raise KeyError(name)


HISTORICAL_FORI_SPECS: tuple[CorpusSpec, ...] = tuple(
    _wrap_historical_fori(name) for name in HISTORICAL_FORI_NAMES
)

# 3. 70 New Independent fori_loop specs across 10 workstream modules
NEW_INDEPENDENT_SPECS: tuple[CorpusSpec, ...] = (
    signal_processing.SPECS
    + optimization_estimation.SPECS
    + sparse_linear_algebra.SPECS
    + graph_mesh.SPECS
    + dynamic_programming.SPECS
    + numerical_simulation.SPECS
    + tiled_tensor.SPECS
    + index_memory.SPECS
    + streaming_compression.SPECS
    + scheduling_control.SPECS
)

# Combined 136 specs
ALL_FORI_SPECS: tuple[CorpusSpec, ...] = (
    CONVERTED_WORKLOAD_SPECS + HISTORICAL_FORI_SPECS + NEW_INDEPENDENT_SPECS
)
FORI_REGISTRY: dict[str, CorpusSpec] = {spec.name: spec for spec in ALL_FORI_SPECS}

# Export SPECS alias
SPECS: tuple[CorpusSpec, ...] = ALL_FORI_SPECS


def get_spec(name: str) -> CorpusSpec:
    try:
        return FORI_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown fori_expansion workload {name!r}; available: {sorted(FORI_REGISTRY)}"
        ) from None


def build_workload(
    name: str,
    trip_count: int,
    factor: int,
    variant: str | None = None,
) -> Workload:
    """Public workload builder expected by sweep harness and feature extractors."""
    return get_spec(name).build(trip_count, factor, variant)


def iter_specs(families: Iterable[str] | None = None) -> list[CorpusSpec]:
    if families is None:
        return list(SPECS)
    family_set = set(families)
    return [spec for spec in SPECS if spec.family in family_set]


def tier_specs(tier: str) -> list[CorpusSpec]:
    if tier == "smoke":
        return [spec for spec in SPECS if spec.tier == "smoke"]
    if tier == "full":
        return list(SPECS)
    raise ValueError(f"unknown tier {tier!r}")


def manifest(hashes: Mapping[str, str] | None = None) -> dict[str, object]:
    entries = []
    for spec in SPECS:
        entry = spec.manifest_entry()
        if hashes and spec.name in hashes:
            entry["body_hash"] = hashes[spec.name]
        entries.append(entry)
    return {
        "version": 1,
        "total_workloads": len(SPECS),
        "historical_fori": len(HISTORICAL_FORI_SPECS),
        "converted_scans": len(CONVERTED_WORKLOAD_SPECS),
        "new_independent": len(NEW_INDEPENDENT_SPECS),
        "workloads": entries,
    }
