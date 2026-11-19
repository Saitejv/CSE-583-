"""Freeze application inputs, full holdout, horizons and success criterion before fitting."""

import json
import sys
from hashlib import sha256
from pathlib import Path

import jax
import numpy as np

from benchmarks.corpus.fori_agent_cases.dataset import settings as agent_settings
from benchmarks.corpus.fori_application.applications import settings
from benchmarks.corpus.fori_regret.acquire import REPEATS, measure, run_workers

OUT = Path("results/fori_application")
HORIZONS = (300, 5000)
ARMS = ("native", "warm", "previous", "candidate", "oracle")


def freeze():
    apps = []
    for spec, size, trip in settings():
        args = spec.build(trip, 1, size).args
        hashes = [sha256(np.asarray(x).tobytes()).hexdigest() for x in jax.tree.leaves(args)]
        apps.append(dict(name=spec.name, variant=size, trip=trip, input_hashes=hashes))
    return dict(
        applications=apps,
        application_code=sha256(Path(__file__).with_name("applications.py").read_bytes()).hexdigest(),
        holdout=[(s.name, v, t) for s, v, t in agent_settings("eval")],
        agent_split=sha256(Path("results/fori_agent_cases/split.json").read_bytes()).hexdigest(),
        primary_calls=5000,
        horizons=HORIZONS,
        arms=ARMS,
        repeats=REPEATS,
        success="At least 10% pooled savings vs native and faster in all three repeats on real applications; report each case and full separate agent holdout.",
    )


def check(create=False):
    path = OUT / "manifest.json"
    data = json.dumps(freeze(), indent=2) + "\n"
    if create and not path.exists():
        with path.open("x") as f:
            f.write(data)
    if path.read_text() != data:
        raise ValueError("frozen application matrix changed")


if __name__ == "__main__":
    stage, *extra = sys.argv[1:]
    if stage not in ("freeze", "sweep"):
        raise ValueError(stage)
    check(create=stage == "freeze")
    if extra:
        print(json.dumps(measure(settings(), int(extra[1]))))
    elif stage == "sweep":
        run_workers(__spec__.name, ["sweep"], REPEATS, OUT / "application_runs.json")
