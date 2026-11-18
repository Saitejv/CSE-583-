"""Shared metadata contract for independent corpus loop bodies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from benchmarks.workloads import Workload, WorkloadSpec, _Built

CORPUS_FACTORS: tuple[int, ...] = (1, 2, 4, 8)
CORPUS_TRIP_COUNTS: tuple[int, ...] = (64, 100, 256, 1023)
SMOKE_TRIP_COUNTS: tuple[int, ...] = (64, 100)
CORPUS_REPEATS = 3
SMOKE_REPEATS = 1

INDIRECT_FAMILIES: tuple[str, ...] = (
    "indirect_gather",
    "indirect_scatter",
    "locality_access",
    "dependent_indexing",
)
STRUCTURED_FAMILIES: tuple[str, ...] = (
    "rolling_window",
    "stencil",
    "streaming_reduction",
    "segmented_scan",
    "stateful_recurrence",
    "control_arithmetic",
)
CORPUS_FAMILIES: tuple[str, ...] = INDIRECT_FAMILIES + STRUCTURED_FAMILIES


@dataclass(frozen=True)
class Provenance:
    """Where a body came from and what was reused.

    ``kind`` is ``"original"`` for a body written for this corpus, or
    ``"public-inspired"`` when a public example or paper motivated the
    pattern. Public-inspired entries must name the source and its license;
    the implementation here is independent unless ``adaptation`` says
    otherwise.
    """

    kind: str
    source: str
    license: str
    adaptation: str


@dataclass(frozen=True)
class CorpusSpec:
    """One independent loop body plus the metadata needed to audit it."""

    name: str
    family: str
    description: str
    mechanism: str
    structural_difference: str
    loop_kind: str
    shape_variants: Mapping[str, str]
    primary_variant: str
    factory: Callable[[str, int, int], _Built]
    seed: int
    provenance: Provenance
    tier: str = "full"
    related_group: str = ""
    trip_counts: tuple[int, ...] = CORPUS_TRIP_COUNTS
    atol: float = 1e-4
    rtol: float = 1e-5

    def build(
        self,
        trip_count: int,
        factor: int,
        variant: str | None = None,
    ) -> Workload:
        spec = WorkloadSpec(
            name=self.name,
            family=self.family,
            description=self.description,
            loop_kind=self.loop_kind,
            shape_variants=self.shape_variants,
            primary_variant=self.primary_variant,
            factory=self.factory,
            trip_counts=self.trip_counts,
            atol=self.atol,
            rtol=self.rtol,
        )
        return spec.build(trip_count, factor, variant)

    def manifest_entry(self) -> dict[str, object]:
        return {
            "workload": self.name,
            "family": self.family,
            "description": self.description,
            "mechanism": self.mechanism,
            "structural_difference": self.structural_difference,
            "loop_kind": self.loop_kind,
            "variants": dict(sorted(self.shape_variants.items())),
            "primary_variant": self.primary_variant,
            "trip_counts": list(self.trip_counts),
            "seed": self.seed,
            "tier": self.tier,
            "related_group": self.related_group,
            "provenance": {
                "kind": self.provenance.kind,
                "source": self.provenance.source,
                "license": self.provenance.license,
                "adaptation": self.provenance.adaptation,
            },
            "atol": self.atol,
            "rtol": self.rtol,
        }


def check_family(spec: CorpusSpec) -> None:
    if spec.family not in CORPUS_FAMILIES:
        raise ValueError(
            f"{spec.name}: unknown family {spec.family!r}; "
            f"expected one of {CORPUS_FAMILIES}"
        )


def require_both_trip_kinds(trip_counts: Sequence[int]) -> None:
    if not any(trip % 8 == 0 for trip in trip_counts):
        raise ValueError(f"trip counts {trip_counts} lack a divisible-by-8 trip")
    if not any(trip % 8 != 0 for trip in trip_counts):
        raise ValueError(f"trip counts {trip_counts} lack a remainder trip")
