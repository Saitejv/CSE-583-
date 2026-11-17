"""Run the pinned recovery matrix through the original isolated CPU harness."""

import argparse
import os
from pathlib import Path

import jax
import jaxlib

from benchmarks import sweep
from benchmarks.recovery_workloads import RECOVERY_WORKLOADS
from jax_unroll.recovery_split import PARTITIONS, PINNED_JAX, REPEATS, TRIPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition", choices=PARTITIONS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite an existing sweep")
    if (jax.__version__, jaxlib.__version__) != (PINNED_JAX, PINNED_JAX):
        parser.error(f"requires JAX and jaxlib {PINNED_JAX}")
    if args.partition == "confirmation":
        from benchmarks.recovery_confirm import LOG, require_frozen

        if LOG.exists():
            parser.error("fresh confirmation has already started")
        require_frozen()
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    os.environ["JAX_ENABLE_X64"] = "false"
    options = sweep.build_parser().parse_args([
        "--backend", "cpu", "--out", str(args.out), "--shapes", "all",
        "--trip-counts", ",".join(map(str, TRIPS)), "--repeats", str(REPEATS),
    ])
    return sweep.run(options, specs=[RECOVERY_WORKLOADS[name] for name in PARTITIONS[args.partition]])


if __name__ == "__main__":
    raise SystemExit(main())
