"""Freeze proposal-inspired settings and record complete paired native sweeps."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from benchmarks.corpus.fori_proposal.workloads import settings
from benchmarks.corpus.fori_regret.acquire import REPEATS, measure, run_workers

OUT = Path("results/fori_proposal")
HORIZONS = (1, 300, 5_000)
BUDGET_MS = 125
MANIFEST = OUT / "matrix.json"


def freeze(create=False):
    selected = settings()
    matrix = {
        "source_sha256": sha256(Path(__file__).with_name("workloads.py").read_bytes()).hexdigest(),
        "configurations": [
            {"workload": c.spec.name, "family": c.spec.family, "variant": c.variant,
             "trip_count": c.trip, "cohort": c.cohort, "live": c.live} for c in selected
        ],
        "direct_horizons": list(HORIZONS), "process_repeats": REPEATS, "cold_budget_ms": BUDGET_MS,
    }
    if create and not MANIFEST.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open("x") as file:
            file.write(json.dumps(matrix, indent=2) + "\n")
    if json.loads(MANIFEST.read_text()) != matrix:
        raise ValueError("proposal workload matrix differs from its frozen manifest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "sweep"))
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    freeze(create=args.stage == "freeze")
    if args.worker is not None:
        targets = [(c.spec, c.variant, c.trip) for c in settings()]
        print(json.dumps(measure(targets, args.worker)))
    elif args.stage == "sweep":
        run_workers(__spec__.name, ["sweep"], REPEATS, OUT / "native_runs.json")


if __name__ == "__main__":
    main()
