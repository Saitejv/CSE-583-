"""Base utilities for converting historical scan workloads to fori_loop."""

from __future__ import annotations

from typing import Any, Callable
import jax
from jax import lax

from benchmarks.corpus.spec import CorpusSpec, Provenance
from benchmarks.workloads import WorkloadSpec, _Built


def make_fori_fn(orig_fn: Callable[..., Any], trip: int, factor: int) -> Callable[..., Any]:
    """Wrap an existing scan function with a lax.fori_loop execution."""

    def fori_fn(*args: Any) -> Any:
        def scan_as_fori(step: Callable[..., Any], init: Any, xs: Any, length: int | None = None, reverse: bool = False, unroll: int = 1) -> tuple[Any, None]:
            def body(i: int, carry: Any) -> Any:
                idx = trip - 1 - i if reverse else i
                if xs is None or (isinstance(xs, (tuple, list)) and len(xs) == 0):
                    next_carry, _ = step(carry, None)
                else:
                    slice_i = jax.tree.map(lambda leaf: leaf[idx], xs)
                    next_carry, _ = step(carry, slice_i)
                return next_carry

            final_carry = lax.fori_loop(0, trip, body, init, unroll=factor)
            return (final_carry, None)

        orig_lax_scan = lax.scan
        lax.scan = scan_as_fori
        try:
            return orig_fn(*args)
        finally:
            lax.scan = orig_lax_scan

    return fori_fn


def convert_scan_spec(
    orig_spec: Any,
    *,
    related_group: str,
    source_path: str,
    converted_id: str | None = None,
) -> CorpusSpec:
    """Create a fori_loop CorpusSpec from an existing scan WorkloadSpec or CorpusSpec."""
    name = converted_id or f"conv_{orig_spec.name}"
    orig_factory = orig_spec.factory

    def factory(variant: str, trip: int, factor: int) -> _Built:
        orig_built = orig_factory(variant, trip, factor)
        fori_fn = make_fori_fn(orig_built.fn, trip, factor)
        return _Built(fori_fn, orig_built.args, orig_built.shape)

    prov = Provenance(
        kind="converted_scan",
        source=f"{source_path}:{orig_spec.name}",
        license="Internal project baseline",
        adaptation=f"Faithful translation of historical scan {orig_spec.name} to lax.fori_loop with unroll={orig_spec.name}",
    )

    tier = getattr(orig_spec, "tier", "full")
    seed = getattr(orig_spec, "seed", 583_000)
    trip_counts = getattr(orig_spec, "trip_counts", (64, 100, 256, 1023))

    return CorpusSpec(
        name=name,
        family=orig_spec.family,
        description=f"Converted fori_loop translation of historical scan {orig_spec.name}: {orig_spec.description}",
        mechanism="converted_scan_fori",
        structural_difference=f"Identical state transition to {orig_spec.name} rewritten from lax.scan to static-bound lax.fori_loop",
        loop_kind="fori_loop",
        shape_variants=orig_spec.shape_variants,
        primary_variant=orig_spec.primary_variant,
        factory=factory,
        seed=seed,
        provenance=prov,
        tier=tier,
        related_group=related_group,
        trip_counts=trip_counts,
        atol=orig_spec.atol,
        rtol=orig_spec.rtol,
    )
