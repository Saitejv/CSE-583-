"""Opt-in unroll selection for project-owned JAX loops."""

from __future__ import annotations

from operator import index
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import errors, lax
from jax.extend.core import concrete_or_error

from jax_unroll.selector import DecisionEngine, HorizonPolicy


def auto_scan(
    f: Callable[..., Any],
    init: Any,
    xs: Any,
    *,
    B: float | None = None,
    R: int = 1,
    force_factor: int | None = None,
    disable: bool = False,
    **scan_kwargs: Any,
) -> Any:
    """Reserved scan interface."""
    raise NotImplementedError(
        "auto_scan is deferred beyond Phase 4. Use jax.lax.scan with unroll=."
    )


def auto_fori_loop(
    lower: Any,
    upper: Any,
    body_fun: Callable[..., Any],
    init_val: Any,
    *,
    B: float | None = None,
    R: int = 1,
    force_factor: int | None = None,
    disable: bool = False,
    unroll: int | bool | None = None,
    engine: DecisionEngine | None = None,
    throughput_only: bool = False,
) -> Any:
    """Run a native fori_loop, optionally specializing its unroll factor.

    `disable=True` always takes the native default. `unroll` passes through
    JAX's own option and validation. `force_factor` overrides the experimental
    engine for static, nonempty loops and is ignored for dynamic or empty loops.
    Pass a shared engine to opt into selection using budget `B` in milliseconds
    and invocation count `R`. An engine configured with a PayoffPolicy requires
    predicted savings over factor 1 after setup and uncertainty. With
    `throughput_only=True`, rank feasible factors by predicted warm latency.
    Supply an engine with `model_loader` to defer loading until a decision
    requires predictions. An engine with HorizonPolicy can bypass selection
    below a known call-count crossover and use a calibrated matrix decision.
    Under an enclosing jit, selection happens on trace and the executable is
    reused by JAX on cache hits.
    """
    if disable or (force_factor is None and engine is None):
        return lax.fori_loop(lower, upper, body_fun, init_val, unroll=None if disable else unroll)
    if unroll is not None:
        if force_factor is not None:
            raise ValueError("specify only one of force_factor and unroll")
        return lax.fori_loop(lower, upper, body_fun, init_val, unroll=unroll)

    try:
        start = concrete_or_error(int, lower, "static fori_loop lower bound")
        stop = concrete_or_error(int, upper, "static fori_loop upper bound")
    except (TypeError, errors.InconclusiveDimensionOperation):
        return lax.fori_loop(lower, upper, body_fun, init_val)
    if stop <= start:
        return lax.fori_loop(lower, upper, body_fun, init_val)

    if force_factor is not None:
        factor = index(force_factor)
        if isinstance(force_factor, bool) or factor < 1:
            raise ValueError("force_factor must be a positive integer")
    else:
        def rolled(carry):
            return lax.fori_loop(lower, upper, body_fun, carry)

        hints = {"fori_carry": init_val} if isinstance(getattr(engine, "payoff", None), HorizonPolicy) else {}
        factor = engine.decide(
            rolled, (init_val,), trip_count=stop - start, loop_kind="fori_loop", B=B, R=R,
            throughput_only=throughput_only, **hints,
        ).factor
    return lax.fori_loop(lower, upper, body_fun, init_val, unroll=factor)


def precompute_fori_loop(
    lower: int,
    upper: int,
    prepare: Callable[..., Any],
    step: Callable[..., Any],
    init_val: Any,
    *,
    unroll: int | bool = 4,
    max_precompute_bytes: int = 8 * 1024 * 1024,
) -> Any:
    """Batch carry-independent work before a static, partially unrolled loop.

    This is equivalent to applying `step(carry, prepare(i))` in ascending index
    order. Both functions must be pure and `prepare` must support `jax.vmap`.
    It receives a scalar JAX integer. Its result can be a pytree of arrays.
    Include the index in that pytree if `step` also needs it.

    The byte limit covers the materialized results of `prepare`, excluding
    compiler temporaries. Above the limit, evaluate `prepare` within a native
    fori_loop with the same unroll setting. Bounds must be static integers.
    """
    start, stop = index(lower), index(upper)
    limit = index(max_precompute_bytes)
    if limit < 0:
        raise ValueError("max_precompute_bytes must be nonnegative")
    length = max(stop - start, 0)
    dtype = jax.dtypes.canonicalize_dtype(int)

    def body(i, carry):
        return step(carry, prepare(jnp.asarray(i, dtype=dtype)))

    if length == 0 or limit == 0:
        return lax.fori_loop(start, stop, body, init_val, unroll=unroll)

    prepared_shape = jax.eval_shape(prepare, jax.ShapeDtypeStruct((), dtype))
    bytes_per_step = sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(prepared_shape))
    if length * bytes_per_step > limit:
        return lax.fori_loop(start, stop, body, init_val, unroll=unroll)

    prepared = jax.vmap(prepare)(jnp.arange(start, stop, dtype=dtype))

    def advance(carry, inputs):
        return step(carry, inputs), None

    return lax.scan(advance, init_val, prepared, length=length, unroll=unroll)[0]
