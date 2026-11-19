"""Correctness and factor-invariance validator for fori_loop expansion workloads."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import jax
import numpy as np

from benchmarks.corpus.fori_expansion.registry import ALL_FORI_SPECS, FORI_REGISTRY, get_spec
from benchmarks.sweep import _all_finite, _allclose, _max_abs_diff
from jax_unroll.features import extract_loop_body


def validate_workload(
    name: str,
    trip_counts: Sequence[int] = (64, 100),
    factors: Sequence[int] = (1, 2, 4, 8),
    verbose: bool = False,
) -> dict[str, object]:
    spec = get_spec(name)
    var = spec.primary_variant
    results: list[dict[str, object]] = []

    # 1. Tracing check & feature extraction check at factor 1
    wl1 = spec.build(trip_counts[0], 1, var)
    loop_body = extract_loop_body(wl1.fn, wl1.args, loop_kind="fori_loop")
    if loop_body is None:
        raise ValueError(f"{name}: could not extract loop body Jaxpr")

    for trip in trip_counts:
        # Reference execution at factor 1
        ref_wl = spec.build(trip, 1, var)
        ref_out = ref_wl.fn(*ref_wl.args)

        if not _all_finite(ref_out):
            raise ValueError(f"{name}: non-finite output at trip={trip}, factor=1")

        for u in factors:
            if u == 1:
                continue
            cand_wl = spec.build(trip, u, var)
            cand_out = cand_wl.fn(*cand_wl.args)

            if not _all_finite(cand_out):
                raise ValueError(f"{name}: non-finite output at trip={trip}, factor={u}")

            diff = _max_abs_diff(ref_out, cand_out)
            close = _allclose(ref_out, cand_out, atol=spec.atol, rtol=spec.rtol)
            if not close:
                raise ValueError(
                    f"{name}: factor mismatch at trip={trip}, factor={u}: "
                    f"max_abs_diff={diff:.6e} > atol={spec.atol}"
                )
            results.append({"trip": trip, "factor": u, "max_diff": diff, "passed": True})

    if verbose:
        print(f"PASS: {name} (trips={trip_counts}, factors={factors})")

    return {"workload": name, "status": "passed", "checks": results}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=str, default=None, help="Specific workload to validate")
    parser.add_argument("--tier", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.workload:
        specs = [get_spec(args.workload)]
    else:
        specs = [s for s in ALL_FORI_SPECS if (s.tier == "smoke" if args.tier == "smoke" else True)]

    trips = (64, 100) if args.tier == "smoke" else (64, 100, 256)
    print(f"Validating {len(specs)} workloads (tier={args.tier})...")

    passed = 0
    failed = []
    for spec in specs:
        try:
            validate_workload(spec.name, trip_counts=trips, verbose=args.verbose)
            passed += 1
        except Exception as exc:
            failed.append((spec.name, str(exc)))
            print(f"FAIL: {spec.name}: {exc}")

    print(f"\nValidation complete: {passed}/{len(specs)} passed.")
    if failed:
        print("Failures:")
        for name, err in failed:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
