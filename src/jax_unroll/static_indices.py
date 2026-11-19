"""Experimental index-kind/provenance block, separate from frozen features v1.

Provenance describes SSA input dependency, not actual index values. ``constant``
means loop invariant (including runtime arguments closed over by the body), not
known at compile time. A carried value may be deterministic or data dependent;
we conservatively call either ``carry``. No inference of locality, uniqueness,
or collisions is attempted. Mixed dynamic origins increment each source count.
"""
from __future__ import annotations

from typing import Any

from jax.extend import core

from jax_unroll.features import LoopBody

KINDS = ("gather", "scatter", "scatter_add", "scatter_min", "scatter_other")
SOURCES = ("constant", "counter", "scanned", "carry", "unknown")
INDEX_COLUMNS = tuple(f"index_{kind}_count" for kind in KINDS) + tuple(
    f"index_from_{source}" for source in SOURCES
) + ("index_mixed_sources_count", "index_opaque_call_count")


def index_features(loop: LoopBody) -> dict[str, int]:
    """Count indexed operations and conservative transitive index origins.

    ``jit``/``pjit`` calls are interpreted positionally through their Jaxpr.
    Unsupported control-flow regions are traversed with unknown origins for
    counting, and their outputs retain an unknown marker. Only an explicitly
    declared fori_loop gives the leading carry the counter interpretation.
    """
    counts = dict.fromkeys(INDEX_COLUMNS, 0)

    def merge(origins):
        result = frozenset().union(*origins)
        if len(result) > 1:
            result = result - {"constant"}
        return result or frozenset({"constant"})

    def nested_jaxprs(value: Any):
        if hasattr(value, "jaxpr"):
            yield value.jaxpr
        elif isinstance(value, core.Jaxpr):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from nested_jaxprs(item)

    def walk(jaxpr, inputs):
        env = dict(zip(jaxpr.invars, inputs, strict=True))
        env.update({var: frozenset({"constant"}) for var in jaxpr.constvars})

        def source(var):
            if isinstance(var, core.Var):
                return env.get(var, frozenset({"unknown"}))
            return frozenset({"constant"})

        for eqn in jaxpr.eqns:
            name = str(eqn.primitive)
            origins = [source(var) for var in eqn.invars]
            if name == "gather" or name.startswith("scatter"):
                kind = name.replace("-", "_")
                if kind not in KINDS:
                    kind = "scatter_other"
                counts[f"index_{kind}_count"] += 1
                index_source = origins[1] if len(origins) > 1 else frozenset({"unknown"})
                for category in index_source:
                    counts[f"index_from_{category}"] += 1
                counts["index_mixed_sources_count"] += int(len(index_source) > 1)
            if name in {"jit", "pjit"} and "jaxpr" in eqn.params:
                inner = eqn.params["jaxpr"]
                inner = getattr(inner, "jaxpr", inner)
                outputs = walk(inner, origins)
            else:
                opaque = [
                    inner
                    for key, value in eqn.params.items()
                    if not (name.startswith("scatter") and key == "update_jaxpr")
                    for inner in nested_jaxprs(value)
                ]
                combined = merge(origins)
                if opaque:
                    counts["index_opaque_call_count"] += 1
                    for inner in opaque:
                        walk(inner, [frozenset({"unknown"})] * len(inner.invars))
                    combined = merge([combined, frozenset({"unknown"})])
                outputs = [combined] * len(eqn.outvars)
            for var, origin in zip(eqn.outvars, outputs, strict=True):
                if isinstance(var, core.Var):
                    env[var] = origin
        return [source(var) for var in jaxpr.outvars]

    inputs = []
    for position, _ in enumerate(loop.jaxpr.invars):
        if position < loop.num_consts:
            category = "constant"
        elif position < loop.num_consts + loop.num_carry:
            category = "counter" if loop.loop_kind == "fori_loop" and position == loop.num_consts else "carry"
        else:
            category = "scanned"
        inputs.append(frozenset({category}))
    walk(loop.jaxpr, inputs)
    return counts
