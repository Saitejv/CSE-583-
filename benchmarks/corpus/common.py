"""Small construction helpers shared by corpus workload modules."""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.workloads import _Built, _shape_json


def built(fn: Callable[..., Any], *args: Any) -> _Built:
    return _Built(
        fn,
        args,
        _shape_json(**{f"arg{index}": arg.shape for index, arg in enumerate(args)}),
    )


def scan(step: Callable[..., Any], factor: int) -> Callable[..., Any]:
    def fn(init: Any, *streams: Any) -> Any:
        return lax.scan(step, init, streams, unroll=factor)[0]

    return fn


def fori(body: Callable[..., Any], factor: int) -> Callable[..., Any]:
    def fn(init: Any, trip: int) -> Any:
        return lax.fori_loop(0, trip, body, init, unroll=factor)

    return fn


def floats(rng: np.random.Generator, *shape: int, scale: float = 0.5) -> Any:
    return jnp.asarray(rng.uniform(-scale, scale, size=shape), dtype=jnp.float32)


def indices(rng: np.random.Generator, high: int, *shape: int) -> Any:
    return jnp.asarray(rng.integers(high, size=shape, dtype=np.int32))


def permutation(rng: np.random.Generator, size: int) -> Any:
    return jnp.asarray(rng.permutation(size), dtype=jnp.int32)
