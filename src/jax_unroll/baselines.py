"""Baseline unroll-factor policies for comparison against the learned selector.

The module provides the trivial policies (factor 1 and fixed factors 2/4/8) and
an LLVM ``UnrollThreshold``-style static size heuristic. The heuristic estimates
the per-iteration body size as the number of Jaxpr equations and selects the
largest candidate factor whose fully unrolled estimated size fits a fixed
equation-count threshold, mirroring LLVM's "unroll only if the fully unrolled
body is small enough" rule. It does not use runtime feedback.

Assumptions and limits:

- ``num_equations`` is a coarse static proxy for machine-level code size; it
  ignores operation width, dtype, constants, and instruction selection.
- Remainder/tail handling is ignored, so factors that do not divide the trip
  count are treated like exact divisors.
- Dependences, memory traffic, register pressure, and occupancy are not
  modelled, so the heuristic cannot predict whether unrolling helps at runtime.
- Selection uses only static features and never traces, lowers, compiles, or
  runs the loop.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Mapping, Sequence

from jax_unroll.schema import FACTORS

__all__ = [
    "BASELINES",
    "factor_one",
    "fixed_factor",
    "llvm_style_heuristic",
]


def factor_one(features: Mapping[str, Any]) -> int:
    """Return the JAX default unroll factor of one."""
    return 1


def fixed_factor(features: Mapping[str, Any], factor: int) -> int:
    """Return ``factor`` after checking that it is a supported fixed factor."""
    if factor not in FACTORS:
        raise ValueError(f"factor must be one of {FACTORS}, got {factor!r}")
    return factor


def llvm_style_heuristic(
    features: Mapping[str, Any],
    *,
    threshold: int = 128,
    max_factor: int = 8,
    factors: Sequence[int] = FACTORS,
) -> int:
    """Return the largest factor whose fully unrolled body fits ``threshold``.

    The body size is ``int(features["num_equations"])``. A candidate ``u`` is
    considered only when ``1 <= u <= max_factor``, ``u <= trip_count``, and
    ``u`` appears in ``factors``. If no candidate satisfies
    ``u * body_size <= threshold``, the function returns 1.
    """
    trip_count = int(features["trip_count"])
    body_size = int(features["num_equations"])

    if threshold < 1:
        raise ValueError(f"threshold must be >= 1, got {threshold}")
    if max_factor < 1:
        raise ValueError(f"max_factor must be >= 1, got {max_factor}")
    if trip_count < 1:
        raise ValueError(f"trip_count must be >= 1, got {trip_count}")
    if body_size < 0:
        raise ValueError(f"num_equations must be >= 0, got {body_size}")

    best = 1
    for candidate in factors:
        if candidate < 1 or candidate > max_factor or candidate > trip_count:
            continue
        if candidate * body_size <= threshold and candidate > best:
            best = candidate
    return best


BASELINES: Mapping[str, Callable[[Mapping[str, Any]], int]] = {
    "factor_1": factor_one,
    "factor_2": partial(fixed_factor, factor=2),
    "factor_4": partial(fixed_factor, factor=4),
    "factor_8": partial(fixed_factor, factor=8),
    "llvm_size": llvm_style_heuristic,
}
