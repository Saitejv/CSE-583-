"""Static validation for the expanded development corpus."""

from __future__ import annotations

import argparse
from typing import Sequence

from benchmarks.corpus.registry import (
    CORPUS_FAMILIES,
    SPECS,
    body_hashes,
    manifest,
    tier_specs,
    validate_environment,
    validate_structure,
)
from benchmarks.corpus.unsupported import check_supported


def validate_loops() -> dict[str, object]:
    checked = 0
    for spec in SPECS:
        workload = spec.build(64, 1)
        check_supported(workload.fn, workload.args)
        checked += 1
    return {"static_loop_bodies": checked}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args(argv)
    validate_environment()
    structure = validate_structure()
    loops = validate_loops()
    hashes = {} if args.skip_hashes else body_hashes()
    if hashes and len(set(hashes.values())) != len(hashes):
        raise SystemExit("duplicate Jaxpr body hashes")
    data = manifest(hashes or None)
    print(
        f"corpus valid: {structure['families']} families, "
        f"{structure['bodies']} bodies, {structure['smoke_bodies']} smoke bodies; "
        f"{loops['static_loop_bodies']} static-scan bodies; "
        f"{'hashes checked' if hashes else 'hashes skipped'}; "
        f"smoke rows={data['tiers']['smoke']['measurements']}, "
        f"full rows={data['tiers']['full']['measurements']}"
    )
    print("families: " + ", ".join(CORPUS_FAMILIES))
    print("smoke: " + ", ".join(spec.name for spec in tier_specs("smoke")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
