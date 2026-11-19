"""Opt-in, per-signature tuning for frequently reused, pure JAX kernels."""
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter_ns

import jax
import numpy as np

from jax_unroll.schema import FACTORS


def _times(kernel, args, calls):
    start = perf_counter_ns()
    for _ in range(calls):
        jax.block_until_ready(kernel(*args))
    return (perf_counter_ns() - start) / calls / 1000


def _paired_choice(times):
    ratios = times / times[:, :1]
    # Three percent is a practical noise margin. Require consistency across rounds.
    stable = (ratios < .97).sum(axis=0) >= 5
    medians = np.where(stable, np.median(ratios, axis=0), np.inf)
    medians[0] = 1.
    return int(np.argmin(medians))


@dataclass(frozen=True)
class TunedKernel:
    kernel: Callable
    factor: int
    setup_ms: float


def tune_fori_kernel(build_kernel, args, *, seed=583):
    """Compile and time equivalent factor variants; reuse the returned jitted callable.

    Supply a pure, non-donating kernel factory and representative device-resident inputs.
    Run once outside jit for each important input signature and distribution.
    """
    start = perf_counter_ns()
    kernels = [jax.jit(build_kernel(factor)) for factor in FACTORS]
    for kernel in kernels:
        jax.block_until_ready(kernel(*args))
    pilot_us = _times(kernels[0], args, 3)
    calls = int(np.clip(1500 / max(pilot_us, 1), 2, 128))
    rng = np.random.default_rng(seed)
    times = np.empty((7, len(kernels)))
    for round_times in times:
        for i in rng.permutation(len(kernels)):
            round_times[i] = _times(kernels[i], args, calls)
    i = _paired_choice(times)
    return TunedKernel(kernels[i], FACTORS[i], (perf_counter_ns() - start) / 1e6)
