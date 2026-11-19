"""Split management, validation, and deterministic hash for targeted fori collection."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

PARTITIONS = ("train", "sealed_eval")


def compute_split_hash(records: Sequence[Mapping[str, Any]]) -> str:
    """Deterministic SHA-256 hash for workload split records."""
    sorted_tuples = sorted(
        (str(r["partition"]), str(r["family"]), str(r["related_group"]), str(r["workload"]))
        for r in records
    )
    payload = "\n".join(f"{p},{f},{g},{w}" for p, f, g, w in sorted_tuples) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_split_csv(specs, output_path: Path) -> list[dict[str, str]]:
    """Generate the machine-readable split CSV from registry specs."""
    records = []
    for s in specs:
        partition = "sealed_eval" if s.name.startswith("eval_") else "train"
        records.append({
            "workload": s.name,
            "family": s.family,
            "related_group": s.related_group,
            "partition": partition,
            "mechanism": s.mechanism,
            "tier": s.tier,
            "origin": "targeted_new",
            "reason": f"Targeted collection {s.description}",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["workload", "family", "related_group", "partition",
                  "mechanism", "tier", "origin", "reason"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(records, key=lambda x: (x["partition"], x["family"], x["workload"])):
            writer.writerow(r)

    return records


def validate_split(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate split integrity."""
    names = [r["workload"] for r in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate workload names in split")

    train = [r for r in records if r["partition"] == "train"]
    sealed = [r for r in records if r["partition"] == "sealed_eval"]

    # Check related group isolation
    train_groups = {r["related_group"] for r in train}
    eval_groups = {r["related_group"] for r in sealed}
    overlap = train_groups & eval_groups
    if overlap:
        raise ValueError(f"Related group leakage across partitions: {overlap}")

    return {
        "status": "passed",
        "total": len(records),
        "train": len(train),
        "sealed_eval": len(sealed),
        "split_hash": compute_split_hash(records),
    }
