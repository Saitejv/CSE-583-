"""Runtime-index controls for the development-only static-feature experiment.

Variants change only index argument values. They never specialize the function,
close over indices, or claim that Jaxpr analysis can infer their locality.
Run primary (72 rows) and divisible (48 rows) separately with the existing
fresh-process sweep harness. This module contains no online policy integration.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks import sweep
from benchmarks.corpus.common import built
from benchmarks.workloads import WorkloadSpec

GATHER_WIDTH = 65_536
SCATTER_WIDTH = 4_096
INDEX_WIDTH = 256
MATCHED_COMPONENT = "indirect_gather+indirect_scatter+locality_access"


def _gather(variant: str, trip: int, factor: int):
    rng = np.random.default_rng(583_701)
    table = jnp.asarray(rng.uniform(-0.1, 0.1, GATHER_WIDTH), dtype=jnp.float32)
    offsets = np.arange(trip, dtype=np.int32)[:, None]
    lanes = np.arange(INDEX_WIDTH, dtype=np.int32)[None, :]
    if variant == "clustered":
        idx = (offsets + lanes % 64) % 64
    else:
        idx = (offsets * 997 + lanes * 251) % GATHER_WIDTH

    def fn(init, table_arg, indices_arg):
        def step(carry, indices):
            return carry * jnp.float32(0.99) + jnp.mean(table_arg[indices]), None

        return lax.scan(step, init, indices_arg, unroll=factor)[0]

    return built(fn, jnp.float32(0), table, jnp.asarray(idx))


def _scatter(variant: str, trip: int, factor: int, *, minimum: bool):
    rng = np.random.default_rng(583_702)
    lanes = np.arange(INDEX_WIDTH, dtype=np.int32)[None, :]
    offsets = np.arange(trip, dtype=np.int32)[:, None]
    if variant == "unique":
        idx = (offsets * 17 + lanes * 13) % SCATTER_WIDTH
    else:
        idx = (offsets + lanes) % 8
    # Binary-exact add increments avoid amplified rounding under collision.
    updates = jnp.asarray(rng.integers(-8, 9, (trip, INDEX_WIDTH)), dtype=jnp.float32) / jnp.float32(65_536)
    init = jnp.zeros((SCATTER_WIDTH,), dtype=jnp.float32)

    def fn(carry_arg, indices_arg, updates_arg):
        def step(carry, values):
            indices, update = values
            if minimum:
                result = carry.at[indices].min(update)
            else:
                result = carry.at[indices].add(update)
            return result, None

        return lax.scan(step, carry_arg, (indices_arg, updates_arg), unroll=factor)[0]

    return built(fn, init, jnp.asarray(idx), updates)


def _scatter_add(variant, trip, factor):
    return _scatter(variant, trip, factor, minimum=False)


def _scatter_min(variant, trip, factor):
    return _scatter(variant, trip, factor, minimum=True)


SPECS = (
    WorkloadSpec("matched_gather", "indirect_gather", "Same gather body; clustered/dispersed runtime reads.",
                 "scan", {"clustered": "64-entry cluster", "dispersed": "strided dispersed reads"}, "clustered", _gather,
                 trip_counts=(1023, 1024)),
    WorkloadSpec("matched_scatter_add", "indirect_scatter", "Same additive update body; unique/colliding runtime indices.",
                 "scan", {"unique": "256 distinct indices per iteration", "colliding": "8 buckets per iteration"}, "unique", _scatter_add,
                 trip_counts=(1023, 1024)),
    WorkloadSpec("matched_scatter_min", "indirect_scatter", "Same min update body; unique/colliding runtime indices.",
                 "scan", {"unique": "256 distinct indices per iteration", "colliding": "8 buckets per iteration"}, "unique", _scatter_min,
                 trip_counts=(1023,)),
)
WORKLOADS = {spec.name: spec for spec in SPECS}


def build_workload(name, trip_count, factor, variant=None):
    return WORKLOADS[name].build(trip_count, factor, variant)


def fold_assignments():
    """All matched inputs, trips, factors and related pilot bodies share a fold."""
    return {spec.name: MATCHED_COMPONENT for spec in SPECS}


def matrix_configs(matrix="primary", *, repeats=3):
    if matrix not in {"primary", "divisible"}:
        raise ValueError("matrix must be primary or divisible")
    return sweep.enumerate_configs(
        specs=SPECS if matrix == "primary" else SPECS[:2], shapes="all",
        trip_counts=(1023 if matrix == "primary" else 1024,), repeats=repeats,
    )


def main(argv: Sequence[str] | None = None):
    parser = sweep.build_parser()
    parser.add_argument("--matrix", choices=("primary", "divisible"), default="primary")
    parser.set_defaults(shapes="all", workload_module=__name__ if __name__ != "__main__" else "benchmarks.corpus.static_matched")
    args = parser.parse_args(argv)
    args.trip_counts = "1023" if args.matrix == "primary" else "1024"
    return sweep.run(args, specs=SPECS if args.matrix == "primary" else SPECS[:2])


if __name__ == "__main__":
    raise SystemExit(main())
