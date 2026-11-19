"""Experimental static structure block, schema ``static-structure-v1``.

These are SSA graph proxies, not compiler traffic estimates. Static slices of
carry-derived values expose neighborhood-style structure, but do not establish
an exact stencil radius, cache locality, aliasing, or bytes physically copied.
Only jit/pjit wrappers are inlined. Other nested regions are conservatively
opaque: their outputs may depend on every input and count as one path step.
No lowering, compilation, execution, or runtime array inspection occurs here.
"""
from __future__ import annotations

from typing import Any

from jax.extend import core

from jax_unroll.features import LoopBody, _as_jaxpr, _aval_bytes

STRUCTURE_VERSION = "static-structure-v1"
STRUCTURE_COLUMNS = (
    "slice_count", "concatenate_count", "carry_slice_count",
    "carry_slice_bytes", "carry_concatenate_count", "carry_updated_bytes",
    "carry_dependency_bytes", "carry_dependency_pairs",
    "carry_to_carry_max_depth", "structural_opaque_count",
)


def structure_features(loop: LoopBody) -> dict[str, int]:
    """Characterize explicit slice/concatenate structure and carry feedback.

    ``carry_dependency_pairs`` counts distinct (input carry, next carry) paths;
    ``carry_dependency_bytes`` counts each participating input carry once.
    Path depth counts primitive equations after jit inlining; directly forwarded
    carry has depth zero. The explicitly identified fori counter is excluded.
    ``carry_updated_bytes`` counts abstract output bytes with a different SSA
    identity from the corresponding input, including replacement by scanned data.
    It does not claim that every element changes or that a copy is performed.
    """
    result = dict.fromkeys(STRUCTURE_COLUMNS, 0)
    start = loop.num_consts + int(loop.loop_kind == "fori_loop" and loop.num_carry > 0)
    stop = loop.num_consts + loop.num_carry
    carries = loop.jaxpr.invars[start:stop]
    state: dict[int, dict[int, int]] = {
        id(value): {index: 0} for index, value in enumerate(carries)
    }

    def roots(value: Any, env: dict[int, dict[int, int]]) -> dict[int, int]:
        return env.get(id(value), {}) if isinstance(value, core.Var) else {}

    def walk(jaxpr: core.Jaxpr, env: dict[int, dict[int, int]]) -> list[dict[int, int]]:
        for eqn in jaxpr.eqns:
            name = str(eqn.primitive)
            if name in {"jit", "pjit"} and "jaxpr" in eqn.params:
                inner = _as_jaxpr(eqn.params["jaxpr"])
                inner_env = {
                    id(var): roots(value, env)
                    for var, value in zip(inner.invars, eqn.invars, strict=True)
                }
                outputs = walk(inner, inner_env)
                for var, deps in zip(eqn.outvars, outputs, strict=True):
                    env[id(var)] = deps
                continue
            parents: dict[int, int] = {}
            for value in eqn.invars:
                for source, depth in roots(value, env).items():
                    parents[source] = max(parents.get(source, 0), depth + 1)
            if name == "slice":
                result["slice_count"] += 1
                if parents:
                    result["carry_slice_count"] += 1
                    result["carry_slice_bytes"] += sum(_aval_bytes(v.aval) for v in eqn.outvars)
            elif name == "concatenate":
                result["concatenate_count"] += 1
                if parents:
                    result["carry_concatenate_count"] += 1
            if any(hasattr(value, "jaxpr") or isinstance(value, core.Jaxpr)
                   or (isinstance(value, (list, tuple)) and any(
                       hasattr(child, "jaxpr") or isinstance(child, core.Jaxpr)
                       for child in value)) for value in eqn.params.values()):
                result["structural_opaque_count"] += 1
            for value in eqn.outvars:
                env[id(value)] = parents
        return [roots(value, env) for value in jaxpr.outvars]

    output_dependencies = walk(loop.jaxpr, state)
    output_start = int(loop.loop_kind == "fori_loop" and loop.source == "loop" and loop.num_carry > 0)
    output_stop = output_start + len(carries)
    outputs = loop.jaxpr.outvars[output_start:output_stop]
    dependencies = output_dependencies[output_start:output_stop]
    used: set[int] = set()
    for carry, output, deps in zip(carries, outputs, dependencies, strict=True):
        if output is not carry:
            result["carry_updated_bytes"] += _aval_bytes(output.aval)
        used.update(deps)
        result["carry_dependency_pairs"] += len(deps)
        result["carry_to_carry_max_depth"] = max(
            result["carry_to_carry_max_depth"], max(deps.values(), default=0)
        )
    result["carry_dependency_bytes"] = sum(_aval_bytes(carries[index].aval) for index in used)
    return result
