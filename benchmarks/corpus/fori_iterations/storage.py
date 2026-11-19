"""Cycle 1. Compare compressed and uncompressed exports of identical incumbent trees."""

import argparse
from itertools import product
import json
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

from benchmarks.corpus.fori_proposal.compact import MODEL
from benchmarks.corpus.fori_regret.calibrate import MODELS

OUT = Path("results/fori_iterations")
UNCOMPRESSED = OUT / "incumbent_uncompressed.npz"


def worker(path):
    from benchmarks.sweep import _warm_backend

    _warm_backend()
    start = perf_counter_ns()
    from jax_unroll.compact_fori import CompactForiPredictor

    CompactForiPredictor.load(path)
    return (perf_counter_ns() - start) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(worker(args.worker))
        return

    from jax_unroll.compact_fori import CompactForiPredictor

    OUT.mkdir(parents=True, exist_ok=True)
    CompactForiPredictor.export(MODELS["incumbent"], UNCOMPRESSED, compressed=False)
    compressed = CompactForiPredictor.load(MODEL)
    alternate = CompactForiPredictor.load(UNCOMPRESSED)
    data = json.loads(Path("results/fori_proposal/native_runs.json").read_text())["runs"][0]
    for row in data:
        assert compressed.predict_candidates(row["features"]) == alternate.predict_candidates(row["features"])
    models = {"compressed": MODEL, "uncompressed": UNCOMPRESSED}
    order = list(product(models, range(7)))
    random.Random(583_801).shuffle(order)
    times = {name: [] for name in models}
    for name, _ in order:
        command = [sys.executable, "-m", __spec__.name, "--worker", str(models[name])]
        times[name].append(float(subprocess.check_output(command, text=True)))
    result = {name: {"load_ms": values, "median_ms": median(values),
                     "bytes": models[name].stat().st_size} for name, values in times.items()}
    data = json.dumps(result, indent=2) + "\n"
    (OUT / "01_storage.json").write_text(data)
    print(data)


if __name__ == "__main__":
    main()
