"""Out-of-scope loop constructs, kept separate from the performance corpus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core

CONTROL_PRIMITIVES = frozenset({"scan", "while", "cond"})
EFFECT_PRIMITIVES = frozenset(
    {
        "debug_print",
        "host_callback",
        "io_callback",
        "pure_callback",
    }
)


class UnsupportedWorkloadError(ValueError):
    """Raised when a loop is outside the static-scan performance domain."""


def _jaxprs_in(value: Any) -> Iterator[core.Jaxpr]:
    if isinstance(value, core.ClosedJaxpr):
        yield value.jaxpr
    elif isinstance(value, core.Jaxpr):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _jaxprs_in(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _jaxprs_in(item)


def _equations(jaxpr: core.Jaxpr) -> Iterator[core.JaxprEqn]:
    for eqn in jaxpr.eqns:
        yield eqn
        for jaxpr_value in _jaxprs_in(eqn.params):
            yield from _equations(jaxpr_value)


def check_supported(fn: Callable[..., Any], args: tuple[Any, ...]) -> core.JaxprEqn:
    """Require exactly one static-length top-level scan and no nested control."""

    try:
        closed = jax.make_jaxpr(fn)(*args)
    except Exception as exc:
        raise UnsupportedWorkloadError(
            f"body is not traceable as a static JAX loop ({type(exc).__name__})"
        ) from exc

    equations = list(_equations(closed.jaxpr))
    control = [eqn for eqn in equations if eqn.primitive.name in CONTROL_PRIMITIVES]
    effects = [eqn for eqn in equations if eqn.primitive.name in EFFECT_PRIMITIVES]
    if effects:
        names = sorted({eqn.primitive.name for eqn in effects})
        raise UnsupportedWorkloadError(f"effectful primitives are out of scope: {names}")
    if not control:
        raise UnsupportedWorkloadError("no JAX loop primitive found")
    if len(control) != 1:
        names = sorted({eqn.primitive.name for eqn in control})
        raise UnsupportedWorkloadError(
            f"nested or multiple control-flow primitives are out of scope: {names}"
        )
    loop = control[0]
    if loop.primitive.name != "scan":
        raise UnsupportedWorkloadError(
            f"dynamic or conditional loop {loop.primitive.name!r} is out of scope"
        )
    length = loop.params.get("length")
    if not isinstance(length, int):
        raise UnsupportedWorkloadError("scan length is not a static integer")
    return loop


@dataclass(frozen=True)
class UnsupportedCase:
    name: str
    reason: str
    build: Callable[[], tuple[Callable[..., Any], tuple[Any, ...]]]
    detection: str = "jaxpr"


def _dynamic_trip() -> tuple[Callable[..., Any], tuple[Any, ...]]:
    def fn(init, upper):
        return lax.fori_loop(0, upper, lambda _i, carry: carry + 1.0, init)

    return fn, (jnp.float32(0.0), jnp.int32(8))


def _while_pointer() -> tuple[Callable[..., Any], tuple[Any, ...]]:
    def fn(init):
        return lax.while_loop(lambda carry: carry < 4.0, lambda carry: carry + 1.0, init)

    return fn, (jnp.float32(0.0),)


def _nested_scan() -> tuple[Callable[..., Any], tuple[Any, ...]]:
    def fn(init, xs):
        inner = jnp.ones((4,), dtype=jnp.float32)

        def outer(carry, x):
            subtotal, _ = lax.scan(lambda acc, y: (acc + y, None), x, inner)
            return carry + subtotal, None

        return lax.scan(outer, init, xs)[0]

    return fn, (jnp.float32(0.0), jnp.ones((8,), dtype=jnp.float32))


def _nested_cond() -> tuple[Callable[..., Any], tuple[Any, ...]]:
    def fn(init, xs):
        def outer(carry, x):
            updated = lax.cond(x > 0.0, lambda value: carry + value, lambda value: carry - value, x)
            return updated, None

        return lax.scan(outer, init, xs)[0]

    return fn, (jnp.float32(0.0), jnp.ones((8,), dtype=jnp.float32))


def _dynamic_scan_length() -> tuple[Callable[..., Any], tuple[Any, ...]]:
    def fn(init, xs, length):
        return lax.scan(lambda carry, x: (carry + x, None), init, xs, length=length)[0]

    return fn, (
        jnp.float32(0.0),
        jnp.ones((8,), dtype=jnp.float32),
        jnp.int32(8),
    )


def _dynamic_shape() -> tuple[Callable[..., Any], tuple[Any, ...]]:
    def fn(init, xs):
        def outer(carry, x):
            nonzero = jnp.nonzero(xs > x)[0]
            return carry + jnp.sum(nonzero.astype(jnp.float32)), None

        return lax.scan(outer, init, xs)[0]

    return fn, (jnp.float32(0.0), jnp.ones((8,), dtype=jnp.float32))


def _debug_print() -> tuple[Callable[..., Any], tuple[Any, ...]]:
    def fn(init, xs):
        def outer(carry, x):
            jax.debug.print("x={x}", x=x)
            return carry + x, None

        return lax.scan(outer, init, xs)[0]

    return fn, (jnp.float32(0.0), jnp.ones((8,), dtype=jnp.float32))


UNSUPPORTED_CASES: tuple[UnsupportedCase, ...] = (
    UnsupportedCase(
        name="dynamic_trip_fori",
        reason="fori_loop bounds must be static integers",
        build=_dynamic_trip,
    ),
    UnsupportedCase(
        name="while_pointer_chase",
        reason="lax.while_loop has a data-dependent trip count",
        build=_while_pointer,
    ),
    UnsupportedCase(
        name="nested_scan",
        reason="a scan inside a scan is nested control flow",
        build=_nested_scan,
    ),
    UnsupportedCase(
        name="nested_cond",
        reason="lax.cond inside a scan is nested control flow",
        build=_nested_cond,
    ),
    UnsupportedCase(
        name="dynamic_scan_length",
        reason="scan length must be a static Python integer",
        build=_dynamic_scan_length,
    ),
    UnsupportedCase(
        name="dynamic_nonzero_shape",
        reason="dynamic output shapes are not jittable in this corpus",
        build=_dynamic_shape,
    ),
    UnsupportedCase(
        name="debug_print_effect",
        reason="effectful debug printing is outside the timing domain",
        build=_debug_print,
    ),
)
