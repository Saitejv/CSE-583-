"""Registry, grouping, tiers, and manifest for the expanded development corpus."""

from __future__ import annotations

from typing import Iterable, Mapping

from benchmarks.corpus.indirect import SPECS as INDIRECT_SPECS
from benchmarks.corpus.spec import (
    CORPUS_FACTORS,
    CORPUS_FAMILIES,
    CORPUS_REPEATS,
    CORPUS_TRIP_COUNTS,
    SMOKE_REPEATS,
    SMOKE_TRIP_COUNTS,
    CorpusSpec,
    check_family,
    require_both_trip_kinds,
)
from benchmarks.corpus.structured import SPECS as STRUCTURED_SPECS
from benchmarks.corpus.unsupported import UNSUPPORTED_CASES
from benchmarks.recovery_workloads import RECOVERY_WORKLOADS
from benchmarks.workloads import WORKLOADS

CORPUS_VERSION = 1
CORPUS_PINNED_JAX = "0.12.0.1"

SPECS: tuple[CorpusSpec, ...] = INDIRECT_SPECS + STRUCTURED_SPECS
CORPUS_WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
CORPUS_TIER_NAMES: tuple[str, ...] = ("smoke", "full")


def get_spec(name: str) -> CorpusSpec:
    try:
        return CORPUS_WORKLOADS[name]
    except KeyError:
        raise KeyError(
            f"unknown corpus workload {name!r}; available: {sorted(CORPUS_WORKLOADS)}"
        ) from None


def build_workload(name: str, trip_count: int, factor: int, variant: str | None = None):
    return get_spec(name).build(trip_count, factor, variant)


def iter_specs(families: Iterable[str] | None = None) -> list[CorpusSpec]:
    if families is None:
        return list(SPECS)
    family_set = set(families)
    unknown = family_set.difference(CORPUS_FAMILIES)
    if unknown:
        raise ValueError(
            f"unknown corpus families {sorted(unknown)}; available: {list(CORPUS_FAMILIES)}"
        )
    return [spec for spec in SPECS if spec.family in family_set]


def tier_specs(tier: str) -> list[CorpusSpec]:
    if tier == "smoke":
        return [spec for spec in SPECS if spec.tier == "smoke"]
    if tier == "full":
        return list(SPECS)
    raise ValueError(f"unknown tier {tier!r}; expected one of {CORPUS_TIER_NAMES}")


def related_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for spec in SPECS:
        groups.setdefault(spec.related_group, []).append(spec.name)
    return {group: sorted(names) for group, names in sorted(groups.items())}


def _family_summary() -> list[dict[str, object]]:
    rows = []
    for family in CORPUS_FAMILIES:
        members = [spec for spec in SPECS if spec.family == family]
        rows.append(
            {
                "family": family,
                "bodies": len(members),
                "smoke_bodies": sum(spec.tier == "smoke" for spec in members),
                "workloads": sorted(spec.name for spec in members),
            }
        )
    return rows


def _full_measurements() -> int:
    return (
        sum(len(spec.shape_variants) * len(spec.trip_counts) for spec in SPECS)
        * len(CORPUS_FACTORS)
        * CORPUS_REPEATS
    )


def _smoke_measurements() -> int:
    return len(tier_specs("smoke")) * len(SMOKE_TRIP_COUNTS) * len(CORPUS_FACTORS) * SMOKE_REPEATS


def manifest(hashes: Mapping[str, str] | None = None) -> dict[str, object]:
    workloads = []
    for spec in sorted(SPECS, key=lambda item: (item.family, item.name)):
        entry = spec.manifest_entry()
        if hashes is not None:
            entry["body_hash"] = hashes[spec.name]
        workloads.append(entry)
    unsupported = [
        {"case": case.name, "reason": case.reason, "detection": case.detection}
        for case in UNSUPPORTED_CASES
    ]
    return {
        "corpus_version": CORPUS_VERSION,
        "jax_pin": CORPUS_PINNED_JAX,
        "factors": list(CORPUS_FACTORS),
        "families": _family_summary(),
        "family_count": len(CORPUS_FAMILIES),
        "body_count": len(SPECS),
        "workloads": workloads,
        "related_groups": related_groups(),
        "feature_caveats": [
            {
                "workload": "scatter_collision_add",
                "reason": (
                    "Feature schema v1 classifies lax.scatter but not "
                    "scatter-add; eq_gather_scatter is zero for this body."
                ),
            },
            {
                "workload": "scatter_min_update",
                "reason": (
                    "Feature schema v1 classifies lax.scatter but not "
                    "scatter-min; the scatter primitive is not counted."
                ),
            },
        ],
        "tiers": {
            "smoke": {
                "description": "One primary variant per smoke body, divisible and remainder trips.",
                "selection": "tier=smoke, primary variant, trips 64 and 100, all factors, one process repeat",
                "trip_counts": list(SMOKE_TRIP_COUNTS),
                "factors": list(CORPUS_FACTORS),
                "process_repeats": SMOKE_REPEATS,
                "measurements": _smoke_measurements(),
                "command": "make corpus-smoke",
            },
            "full": {
                "description": "Every body and variant at all four trips and factors.",
                "selection": "all bodies, all variants, trips 64/100/256/1023, all factors, three process repeats",
                "trip_counts": list(CORPUS_TRIP_COUNTS),
                "factors": list(CORPUS_FACTORS),
                "process_repeats": CORPUS_REPEATS,
                "measurements": _full_measurements(),
                "command": "make corpus-full",
            },
        },
        "unsupported": unsupported,
    }


def validate_structure() -> dict[str, object]:
    if not 8 <= len(CORPUS_FAMILIES) <= 12:
        raise ValueError(f"expected 8-12 families, found {len(CORPUS_FAMILIES)}")
    if not 40 <= len(SPECS) <= 60:
        raise ValueError(f"expected 40-60 bodies, found {len(SPECS)}")
    names = [spec.name for spec in SPECS]
    if len(set(names)) != len(names):
        raise ValueError("duplicate corpus workload names")
    frozen = set(WORKLOADS) | set(RECOVERY_WORKLOADS)
    collisions = sorted(set(names) & frozen)
    if collisions:
        raise ValueError(f"corpus names collide with frozen registries: {collisions}")
    for spec in SPECS:
        check_family(spec)
        require_both_trip_kinds(spec.trip_counts)
        if spec.loop_kind not in {"scan", "fori_loop"}:
            raise ValueError(f"{spec.name}: unsupported loop_kind {spec.loop_kind!r}")
        if len(spec.shape_variants) < 2:
            raise ValueError(f"{spec.name}: needs at least two shape variants")
        if spec.primary_variant not in spec.shape_variants:
            raise ValueError(f"{spec.name}: primary variant is not in shape_variants")
        if spec.tier not in CORPUS_TIER_NAMES:
            raise ValueError(f"{spec.name}: unknown tier {spec.tier!r}")
        if not spec.related_group:
            raise ValueError(f"{spec.name}: related_group is required")
        if spec.provenance.kind not in {"original", "public-inspired"}:
            raise ValueError(f"{spec.name}: unknown provenance kind")
        if not spec.provenance.source or not spec.provenance.license:
            raise ValueError(f"{spec.name}: provenance source and license are required")
        if spec.provenance.kind == "public-inspired" and not spec.provenance.source.startswith(
            ("http://", "https://")
        ):
            raise ValueError(f"{spec.name}: public-inspired provenance needs a URL")
    smoke_families = {spec.family for spec in tier_specs("smoke")}
    missing = set(CORPUS_FAMILIES) - smoke_families
    if missing:
        raise ValueError(f"smoke tier does not cover families: {sorted(missing)}")
    return {
        "families": len(CORPUS_FAMILIES),
        "bodies": len(SPECS),
        "smoke_bodies": len(tier_specs("smoke")),
        "related_groups": len(related_groups()),
    }


def frozen_body_hashes() -> dict[str, str]:
    from jax_unroll.features import extract_features_from_workload

    hashes = {}
    for prefix, specs in (
        ("workload", WORKLOADS),
        ("recovery", RECOVERY_WORKLOADS),
    ):
        for name, spec in specs.items():
            workload = spec.build(64, 1)
            hashes[f"{prefix}:{name}"] = extract_features_from_workload(workload)[
                "body_hash"
            ]
    return hashes


def validate_environment() -> dict[str, str]:
    import jax
    import jaxlib

    installed = {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
    }
    mismatched = {
        name: value
        for name, value in installed.items()
        if value != CORPUS_PINNED_JAX
    }
    if mismatched:
        raise ValueError(
            f"corpus pin is JAX/jaxlib {CORPUS_PINNED_JAX}; installed {mismatched}"
        )
    return installed


def body_hashes() -> dict[str, str]:
    from jax_unroll.features import extract_features_from_workload

    hashes = {}
    for spec in SPECS:
        workload = spec.build(64, 1)
        hashes[spec.name] = extract_features_from_workload(workload)["body_hash"]
    if len(set(hashes.values())) != len(hashes):
        seen: dict[str, str] = {}
        duplicates = {}
        for name, value in hashes.items():
            if value in seen:
                duplicates.setdefault(value, [seen[value]]).append(name)
            else:
                seen[value] = name
        raise ValueError(f"duplicate Jaxpr bodies: {duplicates}")
    frozen = frozen_body_hashes()
    frozen_by_hash = {value: name for name, value in frozen.items()}
    collisions = {
        name: frozen_by_hash[value]
        for name, value in hashes.items()
        if value in frozen_by_hash
    }
    if collisions:
        raise ValueError(
            f"corpus bodies duplicate frozen Jaxpr bodies: {collisions}"
        )
    return hashes
