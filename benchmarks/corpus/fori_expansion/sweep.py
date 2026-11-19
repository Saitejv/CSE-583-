"""Run the expanded fori_loop corpus through the isolated CPU sweep harness."""

from __future__ import annotations

import argparse
import json
from datetime import date
from typing import Sequence

from benchmarks import sweep
from benchmarks.corpus.fori_expansion.registry import (
    ALL_FORI_SPECS,
    manifest,
    tier_specs,
)
from benchmarks.corpus.spec import CORPUS_TRIP_COUNTS, SMOKE_TRIP_COUNTS


def build_parser() -> argparse.ArgumentParser:
    parser = sweep.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--tier",
        choices=("smoke", "full"),
        default="smoke",
        help="smoke uses primary variants and trips 64/100 with one repeat",
    )
    parser.add_argument(
        "--workloads",
        default=None,
        help="comma-separated workload names to measure",
    )
    parser.set_defaults(
        out=None,
        shapes=None,
        trip_counts=None,
        repeats=None,
        families=None,
        workload_module="benchmarks.corpus.fori_expansion.registry",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(manifest(), indent=2, sort_keys=True))
        return 0

    specs = tier_specs(args.tier)
    if args.families:
        families = tuple(
            part.strip() for part in args.families.split(",") if part.strip()
        )
        family_set = set(families)
        specs = [spec for spec in specs if spec.family in family_set]
        args.families = None

    if args.workloads:
        w_names = set(part.strip() for part in args.workloads.split(",") if part.strip())
        specs = [spec for spec in specs if spec.name in w_names]

    today = date.today().isoformat()
    if args.out is None:
        args.out = f"results/fori_expansion/raw_sweep_{args.tier}_{today}.csv"
    if args.shapes is None:
        args.shapes = "primary" if args.tier == "smoke" else "all"
    if args.trip_counts is None:
        trips = SMOKE_TRIP_COUNTS if args.tier == "smoke" else CORPUS_TRIP_COUNTS
        args.trip_counts = ",".join(str(trip) for trip in trips)
    if args.repeats is None:
        args.repeats = 1 if args.tier == "smoke" else 3

    print(
        f"fori_expansion sweep: tier={args.tier}, bodies={len(specs)}, "
        f"shapes={args.shapes}, trips={args.trip_counts}, repeats={args.repeats}"
    )
    return sweep.run(args, specs=specs)


if __name__ == "__main__":
    raise SystemExit(main())
