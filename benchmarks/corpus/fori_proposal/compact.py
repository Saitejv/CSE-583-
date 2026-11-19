"""Export the incumbent and measure compact-model overhead on development loops."""

import argparse
from dataclasses import asdict
import json

from benchmarks.corpus.fori_proposal.protocol import OUT
from benchmarks.corpus.fori_regret.acquire import REPEATS, run_workers
from benchmarks.corpus.fori_regret.calibrate import (
    CALIBRATION, MODELS, QUANTILE, SHARED_SPECIALIZATIONS, worker as measure_overhead,
)
from jax_unroll.compact_fori import CompactForiPredictor
from jax_unroll.selector import PayoffPolicy

MODEL = OUT / "incumbent.npz"
POLICY = OUT / "compact_payoff.json"


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(measure_overhead("compact", loader=lambda _: CompactForiPredictor.load(MODEL))))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    CompactForiPredictor.export(MODELS["incumbent"], MODEL)
    model = CompactForiPredictor.load(MODEL)
    old = json.loads(CALIBRATION.read_text())["models"]["incumbent"]
    if model.source_version != old["version"]:
        raise ValueError("compact model does not match the calibrated incumbent")
    runs = run_workers(__spec__.name, [], REPEATS, OUT / "compact_overhead.json")
    decision_ms = float(np.quantile(
        [row["decision_ms"] for run in runs for row in run["rows"]], QUANTILE, method="higher",
    ))
    load_ms = float(np.median([run["load_ms"] for run in runs]))
    policy = PayoffPolicy(**{
        **old["policy"], "decision_ms": decision_ms,
        "model_load_ms": load_ms / SHARED_SPECIALIZATIONS,
    })
    POLICY.write_text(json.dumps({
        "version": model.version, "source_version": model.source_version,
        "cost_error_calibration": str(CALIBRATION), "model_load_shared_across": SHARED_SPECIALIZATIONS,
        "median_load_ms": load_ms, "policy": asdict(policy),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
