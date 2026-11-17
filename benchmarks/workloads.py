"""Fixed-trip-count JAX loop families used by the unroll sweeps.

Each family is a separate held-out unit for leave-one-family-out evaluation
(T2.6). Shapes are static and every loop uses ``jax.lax.scan`` or
``jax.lax.fori_loop`` with an explicit ``unroll`` factor. The registry is pure
Python configuration; it does not trace JAX until a harness builds a workload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jax_unroll.schema import FACTORS, TRIP_COUNTS

Array = jax.Array


def _shape_json(**shapes: Sequence[int]) -> str:
    return json.dumps(
        {name: list(shape) for name, shape in shapes.items()},
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class Workload:
    """A fully configured, jittable loop instance."""

    name: str
    family: str
    variant: str
    loop_kind: str
    trip_count: int
    factor: int
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    shape: str
    dtype: str = "float32"
    atol: float = 1e-4
    rtol: float = 1e-5

    @property
    def divides_trip(self) -> bool:
        return self.trip_count % self.factor == 0

    def signature(self) -> dict[str, Any]:
        return {
            "workload": self.name,
            "family": self.family,
            "variant": self.variant,
            "loop_kind": self.loop_kind,
            "trip_count": self.trip_count,
            "factor": self.factor,
            "shape": self.shape,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class _Built:
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    shape: str


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    family: str
    description: str
    loop_kind: str
    shape_variants: Mapping[str, str]
    primary_variant: str
    factory: Callable[[str, int, int], _Built]
    trip_counts: tuple[int, ...] = TRIP_COUNTS
    atol: float = 1e-4
    rtol: float = 1e-5

    def build(
        self,
        trip_count: int,
        factor: int,
        variant: str | None = None,
    ) -> Workload:
        if trip_count <= 0:
            raise ValueError(f"trip_count must be positive, got {trip_count}")
        if factor <= 0:
            raise ValueError(f"factor must be positive, got {factor}")
        variant = variant or self.primary_variant
        if variant not in self.shape_variants:
            raise ValueError(
                f"unknown variant {variant!r} for {self.name}; "
                f"expected one of {sorted(self.shape_variants)}"
            )
        built = self.factory(variant, trip_count, factor)
        return Workload(
            name=self.name,
            family=self.family,
            variant=variant,
            loop_kind=self.loop_kind,
            trip_count=trip_count,
            factor=factor,
            fn=built.fn,
            args=built.args,
            shape=built.shape,
            atol=self.atol,
            rtol=self.rtol,
        )


def _scalar_recurrence(variant: str, trip: int, factor: int) -> _Built:
    width = {"scalar": (), "vector8": (8,)}[variant]
    rng = np.random.default_rng(583_001)
    init = jnp.asarray(rng.normal(size=width), dtype=jnp.float32)
    xs = jnp.asarray(rng.normal(size=(trip, *width)), dtype=jnp.float32)

    def step(carry: Array, x: Array) -> tuple[Array, None]:
        return carry * jnp.float32(0.9) + x, None

    def fn(init_val: Array, xs_val: Array) -> Array:
        carry, _ = lax.scan(step, init_val, xs_val, unroll=factor)
        return carry

    return _Built(fn, (init, xs), _shape_json(carry=width, xs=(trip, *width)))


def _elementwise_sine(variant: str, trip: int, factor: int) -> _Built:
    width = {"n256": 256, "n1024": 1024, "n4096": 4096}[variant]
    rng = np.random.default_rng(583_002)
    init = jnp.asarray(rng.normal(size=(width,)), dtype=jnp.float32)
    xs = jnp.asarray(
        rng.normal(scale=0.01, size=(trip, width)), dtype=jnp.float32
    )

    def step(carry: Array, x: Array) -> tuple[Array, None]:
        return jnp.sin(carry + x), None

    def fn(init_val: Array, xs_val: Array) -> Array:
        carry, _ = lax.scan(step, init_val, xs_val, unroll=factor)
        return carry

    return _Built(fn, (init, xs), _shape_json(carry=(width,), xs=(trip, width)))


def _reduction_vector(variant: str, trip: int, factor: int) -> _Built:
    width = {"n256": 256, "n1024": 1024}[variant]
    rng = np.random.default_rng(583_003)
    init = jnp.asarray(rng.normal(), dtype=jnp.float32)
    weights = jnp.asarray(
        rng.normal(scale=1.0 / np.sqrt(width), size=(width,)), dtype=jnp.float32
    )
    xs = jnp.asarray(rng.normal(size=(trip, width)), dtype=jnp.float32)

    def step(carry: Array, x: Array) -> tuple[Array, None]:
        return carry + jnp.sum(x * weights), None

    def fn(init_val: Array, xs_val: Array) -> Array:
        carry, _ = lax.scan(step, init_val, xs_val, unroll=factor)
        return carry

    return _Built(fn, (init, xs), _shape_json(carry=(), xs=(trip, width)))


def _dense_matvec(variant: str, trip: int, factor: int) -> _Built:
    width = {"n32": 32, "n64": 64, "n128": 128}[variant]
    rng = np.random.default_rng(583_004)
    matrix = jnp.asarray(
        rng.normal(scale=1.0 / np.sqrt(width), size=(width, width)),
        dtype=jnp.float32,
    )
    init = jnp.asarray(rng.normal(size=(width,)), dtype=jnp.float32)
    xs = jnp.asarray(
        rng.normal(scale=0.1, size=(trip, width)), dtype=jnp.float32
    )

    def step(carry: Array, x: Array) -> tuple[Array, None]:
        return matrix @ carry + x, None

    def fn(init_val: Array, xs_val: Array) -> Array:
        carry, _ = lax.scan(step, init_val, xs_val, unroll=factor)
        return carry

    return _Built(
        fn,
        (init, xs),
        _shape_json(matrix=(width, width), carry=(width,), xs=(trip, width)),
    )


def _indexed_update(variant: str, trip: int, factor: int) -> _Built:
    width, updates = {
        "n256": (256, 16),
        "n1024": (1024, 64),
        "n4096": (4096, 256),
    }[variant]
    rng = np.random.default_rng(583_005)
    indices = jnp.asarray(rng.permutation(width)[:updates], dtype=jnp.int32)
    init = jnp.asarray(rng.normal(scale=0.1, size=(width,)), dtype=jnp.float32)
    xs = jnp.asarray(
        rng.normal(scale=0.01, size=(trip, updates)), dtype=jnp.float32
    )

    def step(carry: Array, x: Array) -> tuple[Array, None]:
        gathered = carry[indices]
        return carry.at[indices].set(gathered * jnp.float32(0.5) + x), None

    def fn(init_val: Array, xs_val: Array) -> Array:
        carry, _ = lax.scan(step, init_val, xs_val, unroll=factor)
        return carry

    return _Built(
        fn,
        (init, xs),
        _shape_json(carry=(width,), indices=(updates,), xs=(trip, updates)),
    )


def _jacobi_solver(variant: str, trip: int, factor: int) -> _Built:
    size = {"n16": 16, "n32": 32, "n64": 64}[variant]
    rng = np.random.default_rng(583_006)
    right_hand_side = jnp.asarray(
        rng.normal(scale=0.01, size=(size, size)), dtype=jnp.float32
    )
    init = jnp.asarray(rng.normal(scale=0.01, size=(size, size)), dtype=jnp.float32)

    def body(_i: Array, carry: Array) -> Array:
        up = jnp.roll(carry, 1, axis=0)
        down = jnp.roll(carry, -1, axis=0)
        left = jnp.roll(carry, 1, axis=1)
        right = jnp.roll(carry, -1, axis=1)
        return (up + down + left + right + right_hand_side) * jnp.float32(0.25)

    def fn(init_val: Array) -> Array:
        return lax.fori_loop(0, trip, body, init_val, unroll=factor)

    return _Built(
        fn,
        (init,),
        _shape_json(carry=(size, size), rhs=(size, size)),
    )


WORKLOADS: dict[str, WorkloadSpec] = {
    "scalar_recurrence": WorkloadSpec(
        name="scalar_recurrence",
        family="recurrence",
        description="Sequential scalar recurrence carry = 0.9*carry + x_t.",
        loop_kind="scan",
        shape_variants={
            "scalar": "scalar carry, scalar xs",
            "vector8": "8-element vector carry",
        },
        primary_variant="scalar",
        factory=_scalar_recurrence,
    ),
    "elementwise_sine": WorkloadSpec(
        name="elementwise_sine",
        family="elementwise",
        description="Independent elementwise sine scan carry = sin(carry + x_t).",
        loop_kind="scan",
        shape_variants={
            "n256": "256-element vectors",
            "n1024": "1024-element vectors",
            "n4096": "4096-element vectors",
        },
        primary_variant="n1024",
        factory=_elementwise_sine,
    ),
    "reduction_vector": WorkloadSpec(
        name="reduction_vector",
        family="reduction",
        description="Scalar accumulator with a weighted vector reduction per step.",
        loop_kind="scan",
        shape_variants={
            "n256": "256-element reduction",
            "n1024": "1024-element reduction",
        },
        primary_variant="n1024",
        factory=_reduction_vector,
    ),
    "dense_matvec": WorkloadSpec(
        name="dense_matvec",
        family="dense_linear_algebra",
        description="Dense matrix-vector recurrence carry = W @ carry + x_t.",
        loop_kind="scan",
        shape_variants={
            "n32": "32x32 dense matrix",
            "n64": "64x64 dense matrix",
            "n128": "128x128 dense matrix",
        },
        primary_variant="n64",
        factory=_dense_matvec,
    ),
    "indexed_update": WorkloadSpec(
        name="indexed_update",
        family="indexed_memory",
        description="Gather/scatter update of a vector at fixed unique indices.",
        loop_kind="scan",
        shape_variants={
            "n256": "256-element tensor, 16 updates",
            "n1024": "1024-element tensor, 64 updates",
            "n4096": "4096-element tensor, 256 updates",
        },
        primary_variant="n1024",
        factory=_indexed_update,
    ),
    "jacobi_solver": WorkloadSpec(
        name="jacobi_solver",
        family="iterative_solver",
        description="Application-style 2D Jacobi stencil iteration via fori_loop.",
        loop_kind="fori_loop",
        shape_variants={
            "n16": "16x16 grid",
            "n32": "32x32 grid",
            "n64": "64x64 grid",
        },
        primary_variant="n32",
        factory=_jacobi_solver,
    ),
}

FAMILIES: tuple[str, ...] = tuple(dict.fromkeys(spec.family for spec in WORKLOADS.values()))


def get_spec(name: str) -> WorkloadSpec:
    try:
        return WORKLOADS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown workload {name!r}; available: {sorted(WORKLOADS)}"
        ) from exc


def iter_specs(families: Iterable[str] | None = None) -> list[WorkloadSpec]:
    if families is None:
        return list(WORKLOADS.values())
    family_set = set(families)
    unknown = family_set.difference(FAMILIES)
    if unknown:
        raise ValueError(
            f"unknown families {sorted(unknown)}; available: {list(FAMILIES)}"
        )
    return [spec for spec in WORKLOADS.values() if spec.family in family_set]


def build_workload(
    name: str,
    trip_count: int,
    factor: int,
    variant: str | None = None,
) -> Workload:
    return get_spec(name).build(trip_count, factor, variant)


def registry_table() -> list[dict[str, Any]]:
    return [
        {
            "workload": spec.name,
            "family": spec.family,
            "loop_kind": spec.loop_kind,
            "description": spec.description,
            "primary_variant": spec.primary_variant,
            "variants": sorted(spec.shape_variants),
            "factors": list(FACTORS),
            "trip_counts": list(spec.trip_counts),
        }
        for spec in WORKLOADS.values()
    ]
