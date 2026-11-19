"""Unified registry for all 48 targeted fori_loop workloads.

Includes:
- 12 existing development (train) bodies reused from fori_expansion
- 12 new development (train) bodies
- 24 new sealed evaluation bodies
Total: 48 distinct fori_loop bodies across 6 workstreams.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any, Iterable, Mapping, Sequence

from benchmarks.corpus.spec import CorpusSpec
from benchmarks.workloads import Workload

from benchmarks.corpus.fori_targeted import (
    dependency_parallelism,
    locality_indexing,
    writes_state,
    dense_reductions,
    branches_masks,
    application_kernels,
)

# Collect all SPECS from the six domain modules
_DOMAIN_MODULES = (
    dependency_parallelism,
    locality_indexing,
    writes_state,
    dense_reductions,
    branches_masks,
    application_kernels,
)

ALL_SPECS: tuple[CorpusSpec, ...] = ()
for _mod in _DOMAIN_MODULES:
    ALL_SPECS = ALL_SPECS + _mod.SPECS

SPECS: dict[str, CorpusSpec] = {s.name: s for s in ALL_SPECS}

# Partition assignments based on naming convention
# eval_* bodies go to sealed evaluation; others go to development (train)
TRAIN_NAMES: tuple[str, ...] = tuple(s.name for s in ALL_SPECS if not s.name.startswith("eval_"))
EVAL_NAMES: tuple[str, ...] = tuple(s.name for s in ALL_SPECS if s.name.startswith("eval_"))

TRAIN_SPECS: tuple[CorpusSpec, ...] = tuple(s for s in ALL_SPECS if s.name in TRAIN_NAMES)
EVAL_SPECS: tuple[CorpusSpec, ...] = tuple(s for s in ALL_SPECS if s.name in EVAL_NAMES)

assert len(TRAIN_SPECS) == 24, f"Expected 24 train specs, got {len(TRAIN_SPECS)}"
assert len(EVAL_SPECS) == 24, f"Expected 24 eval specs, got {len(EVAL_SPECS)}"
assert len(SPECS) == 48, f"Expected 48 total specs, got {len(SPECS)}"


def get_spec(name: str) -> CorpusSpec:
    """Look up a spec by workload name."""
    if name not in SPECS:
        raise KeyError(f"Unknown workload {name!r}; known: {sorted(SPECS)[:5]}...")
    return SPECS[name]


def build_workload(
    name: str,
    trip_count: int,
    factor: int,
    variant: str | None = None,
) -> Workload:
    """Public workload builder expected by sweep harness and feature extractors."""
    return get_spec(name).build(trip_count, factor, variant)


_SPECIALIZED = {
    "write_multitap_delay_ring": (32, {"blocked": True}),
    "eval_dense_attention_block_step": (4, {"precompute": True}),
    "eval_app_ct_backprojection_step": (8, {"precompute": True}),
    "eval_write_waveguide_mesh": (4, {"paired": True}),
    "eval_dense_horner_matrix_poly": (32, {"blocked": True}),
    "eval_write_pic_charge_deposit": (256, {"blocked": True}),
}
SPECIALIZED_NAMES = tuple(_SPECIALIZED)


def build_specialized_workload(
    name: str,
    trip_count: int,
    variant: str | None = None,
    *,
    factor: int | None = None,
) -> Workload:
    """Build the opt-in unrolled kernels for the pinned CPU experiments."""
    spec = get_spec(name)
    variant = variant or spec.primary_variant
    default_factor, options = _SPECIALIZED[name]
    if name == "eval_app_ct_backprojection_step" and variant == "large" and trip_count <= 256:
        default_factor = 4
    return replace(spec, factory=partial(spec.factory, **options)).build(
        trip_count, default_factor if factor is None else factor, variant,
    )


def iter_specs(families: Iterable[str] | None = None) -> list[CorpusSpec]:
    if families is None:
        return list(ALL_SPECS)
    family_set = set(families)
    return [spec for spec in ALL_SPECS if spec.family in family_set]
