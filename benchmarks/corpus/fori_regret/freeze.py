"""Record and verify the reviewed families and workload source before timing."""

from hashlib import sha256
import json
from pathlib import Path

from benchmarks.corpus.fori_regret.acquire import OUT, REPEATS, SIZES, TRIPS
from benchmarks.corpus.fori_regret.workloads import CASES

MANIFEST = OUT / "split.json"


def snapshot():
    folder = Path(__file__).parent
    return {
        "workloads": [
            {
                "workload": case.name, "family": case.family, "partition": case.partition,
                "factory": case.factory.__name__, "source": case.factory.__module__.split(".")[-1],
                "atol": case.atol, "rtol": case.rtol,
            }
            for case in CASES
        ],
        "sha256": {
            name: sha256((folder / name).read_bytes()).hexdigest()
            for name in ("agy_cases.py", "opencode_cases.py", "workloads.py")
        },
        "sizes": list(SIZES), "trips": list(TRIPS), "process_repeats": REPEATS,
    }


def check():
    if json.loads(MANIFEST.read_text()) != snapshot():
        raise ValueError("the acquisition matrix or its workload sources changed after the split was frozen")


def main():
    if MANIFEST.exists():
        check()
        return
    train = {case.family for case in CASES if case.partition == "train"}
    holdout = {case.family for case in CASES if case.partition == "holdout"}
    if not train or not holdout or train & holdout or len({case.name for case in CASES}) != len(CASES):
        raise ValueError("expected distinct whole families and unique workload names")
    OUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("x") as file:
        json.dump(snapshot(), file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
