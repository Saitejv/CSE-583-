"""Split management, validation, and deterministic hash verification for fori expansion."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

PARTITIONS: tuple[str, ...] = ("train", "historical_eval", "fresh_eval")
DEV_PARTITIONS: frozenset[str] = frozenset({"train"})
EVAL_PARTITIONS: frozenset[str] = frozenset({"historical_eval", "fresh_eval"})

ORIGINS: tuple[str, ...] = ("historical_fori", "converted_scan", "new_independent")
TIERS: tuple[str, ...] = ("smoke", "full")


def compute_split_hash(workloads: Sequence[Mapping[str, Any]]) -> str:
    """Compute the deterministic SHA-256 hash for a set of workload assignments.

    Sorts records by (partition, family, related_group, workload) and computes
    SHA-256 of the canonical CSV-like representation.
    """
    sorted_tuples = sorted(
        (
            str(w["partition"]),
            str(w["family"]),
            str(w["related_group"]),
            str(w["workload"]),
        )
        for w in workloads
    )
    payload = "\n".join(f"{p},{f},{g},{w}" for p, f, g, w in sorted_tuples) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_split_records(
    workloads: Sequence[Mapping[str, Any]],
    historical_split: Mapping[str, str] | None = None,
    enforce_quotas: bool = True,
) -> dict[str, Any]:
    """Validate that split records satisfy all corpus expansion invariants.

    Invariants enforced:
    1. Schema & Required Fields: All records contain required fields with valid enums.
    2. Name Uniqueness: Workload names are globally distinct.
    3. Historical Lineage: Historical workloads strictly match their original partition.
    4. Related-Group Atomicity: Zero cross-boundary leakage of related groups.
    5. Fresh Evaluation Isolation: fresh_eval contains only new_independent workloads.
    6. Quota Bounds (optional): Target counts for new independent bodies and fresh_eval.
    """
    # 1. Schema check
    required_fields = {
        "workload",
        "family",
        "related_group",
        "partition",
        "mechanism",
        "tier",
        "origin",
        "reason",
    }
    seen_names: set[str] = set()
    for w in workloads:
        missing = required_fields - set(w.keys())
        if missing:
            raise ValueError(f"Workload {w.get('workload', '?')} missing fields: {sorted(missing)}")
        if w["workload"] in seen_names:
            raise ValueError(f"Duplicate workload name: {w['workload']}")
        seen_names.add(w["workload"])

        if w["partition"] not in PARTITIONS:
            raise ValueError(f"{w['workload']}: unknown partition {w['partition']!r}; expected {PARTITIONS}")
        if w["origin"] not in ORIGINS:
            raise ValueError(f"{w['workload']}: unknown origin {w['origin']!r}; expected {ORIGINS}")
        if w["tier"] not in TIERS:
            raise ValueError(f"{w['workload']}: unknown tier {w['tier']!r}; expected {TIERS}")

    # 2. Historical Lineage
    if historical_split is not None:
        for w in workloads:
            name = w["workload"]
            if name in historical_split:
                hist_p = historical_split[name]
                expected_p = "train" if hist_p == "train" else "historical_eval"
                if w["partition"] != expected_p:
                    raise ValueError(
                        f"Lineage violation: historical workload {name!r} ({hist_p}) "
                        f"assigned to {w['partition']!r}, expected {expected_p!r}"
                    )
                if w["origin"] not in {"historical_fori", "converted_scan"}:
                    raise ValueError(
                        f"Lineage violation: historical workload {name!r} marked origin={w['origin']!r}"
                    )

    # 3. Fresh Evaluation Isolation
    for w in workloads:
        if w["partition"] == "fresh_eval":
            if w["origin"] != "new_independent":
                raise ValueError(
                    f"Isolation violation: fresh_eval workload {w['workload']!r} has origin {w['origin']!r}"
                )
            if historical_split is not None and w["workload"] in historical_split:
                raise ValueError(
                    f"Contamination: historical workload {w['workload']!r} found in fresh_eval"
                )

    # 4. Related-Group Atomicity (Zero Leakage)
    group_to_partitions: dict[str, set[str]] = {}
    group_to_super: dict[str, set[str]] = {}
    for w in workloads:
        rg = str(w["related_group"])
        p = str(w["partition"])
        sp = "dev" if p in DEV_PARTITIONS else "eval"
        group_to_partitions.setdefault(rg, set()).add(p)
        group_to_super.setdefault(rg, set()).add(sp)

    leaking_groups = {g: sorted(ps) for g, ps in group_to_partitions.items() if len(ps) > 1}
    if leaking_groups:
        raise ValueError(f"Related-group partition leakage: {leaking_groups}")

    leaking_super = {g: sorted(sps) for g, sps in group_to_super.items() if len(sps) > 1}
    if leaking_super:
        raise ValueError(f"Super-partition leakage across dev/eval: {leaking_super}")

    # 5. Quotas check
    counts = {
        "total": len(workloads),
        "train": sum(w["partition"] == "train" for w in workloads),
        "historical_eval": sum(w["partition"] == "historical_eval" for w in workloads),
        "fresh_eval": sum(w["partition"] == "fresh_eval" for w in workloads),
        "new_independent": sum(w["origin"] == "new_independent" for w in workloads),
        "historical_fori": sum(w["origin"] == "historical_fori" for w in workloads),
        "converted_scan": sum(w["origin"] == "converted_scan" for w in workloads),
    }

    if enforce_quotas:
        if not (15 <= counts["fresh_eval"] <= 20):
            raise ValueError(f"fresh_eval count {counts['fresh_eval']} outside target [15, 20]")
        if not (60 <= counts["new_independent"] <= 80):
            raise ValueError(f"new_independent count {counts['new_independent']} outside target [60, 80]")

    split_hash = compute_split_hash(workloads)
    return {
        "status": "passed",
        "counts": counts,
        "split_hash": split_hash,
    }
