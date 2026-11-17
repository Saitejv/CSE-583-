"""Static Jaxpr feature extraction for fixed-trip-count loops.

The public entry point is :func:`extract_features`. It traces a jitted loop
function (or a bare loop body) with ``jax.make_jaxpr``, locates the loop
primitive, and derives a flat, JSON/CSV-serializable feature dictionary from the
body Jaxpr. It never lowers, compiles, or benchmarks a candidate factor.

JAX 0.12 traces ``lax.fori_loop`` with Python-integer bounds to the ``scan``
primitive, so a ``fori_loop`` body is handled by the same code path as
``scan``; its integer counter is excluded from the loop-carried state features.
Dynamic-bound ``fori_loop`` / ``while_loop`` bodies trace to ``while`` and are
recognized but not fully characterized (see ``docs/features.md``). The fori
counter is stripped only when the caller requests ``loop_kind="fori_loop"``;
with no requested kind, the traced primitive is reported as ``loop_kind``
(``scan`` or ``while``) and the carry is counted as-is.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np
from jax.extend import core

from jax_unroll.schema import FEATURE_COLUMNS

__all__ = [
    "FEATURE_COLUMNS",
    "FeatureExtractionError",
    "LoopBody",
    "extract_features",
    "extract_features_from_workload",
    "extract_loop_body",
]

_LOOP_PRIMITIVES = frozenset({"scan", "while"})
_INLINED_PRIMITIVES = frozenset({"jit", "pjit"})

_ELEMENTWISE_PRIMITIVES = frozenset(
    {
        "abs",
        "acos",
        "acosh",
        "add",
        "and",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "bessel_i0e",
        "bessel_i1e",
        "betainc",
        "cbrt",
        "ceil",
        "clamp",
        "complex",
        "conj",
        "cos",
        "cosh",
        "digamma",
        "div",
        "eq",
        "erf",
        "erfc",
        "erf_inv",
        "exp",
        "expm1",
        "floor",
        "ge",
        "gt",
        "igamma",
        "igammac",
        "imag",
        "integer_pow",
        "le",
        "lgamma",
        "log",
        "log1p",
        "logistic",
        "lt",
        "max",
        "min",
        "mul",
        "ne",
        "neg",
        "not",
        "or",
        "pow",
        "real",
        "reciprocal",
        "relu",
        "rem",
        "round",
        "rsqrt",
        "select",
        "select_n",
        "shift_left",
        "shift_right_arithmetic",
        "shift_right_logical",
        "sign",
        "sin",
        "sinh",
        "softplus",
        "sqrt",
        "square",
        "sub",
        "tan",
        "tanh",
        "xor",
    }
)
_DOT_PRIMITIVES = frozenset({"dot", "dot_general"})
_GATHER_SCATTER_PRIMITIVES = frozenset(
    {
        "dynamic_index_in_dim",
        "dynamic_slice",
        "dynamic_slice_in_dim",
        "dynamic_update_slice",
        "dynamic_update_slice_in_dim",
        "gather",
        "index_in_dim",
        "scatter",
        "slice",
        "static_index_in_dim",
    }
)
_CONVERT_PRIMITIVES = frozenset({"bitcast_convert_type", "convert_element_type"})


class FeatureExtractionError(RuntimeError):
    """Raised when a loop body cannot be traced or located."""


@dataclass(frozen=True)
class LoopBody:
    """A traced loop body plus the static structure needed for features."""

    loop_kind: str
    loop_primitive: str
    jaxpr: core.Jaxpr
    num_consts: int
    num_carry: int
    num_xs: int
    unroll: int | None
    source: str


def _is_var(value: Any) -> bool:
    return isinstance(value, core.Var)


def _prod(shape: Sequence[int]) -> int:
    return int(math.prod(int(dim) for dim in shape))


def _aval_bytes(aval: Any) -> int:
    try:
        return _prod(aval.shape) * int(np.dtype(aval.dtype).itemsize)
    except Exception:
        return 0


def _aval_elements(aval: Any) -> int:
    try:
        return _prod(aval.shape)
    except Exception:
        return 0


def _aval_dtype_bits(aval: Any) -> int:
    try:
        return int(np.dtype(aval.dtype).itemsize) * 8
    except Exception:
        return 0


def _json_shapes(avals: Sequence[Any]) -> str:
    import json

    return json.dumps(
        [list(aval.shape) for aval in avals],
        separators=(",", ":"),
    )


def _iter_loop_candidates(
    jaxpr: core.Jaxpr,
) -> list[tuple[core.JaxprEqn, str]]:
    """Collect loop equations, looking through jit/pjit wrappers.

    A jitted loop function traces to a top-level ``jit`` equation whose nested
    Jaxpr contains the ``scan``/``while``; descending into those wrappers is
    required for ``extract_features(jax.jit(fn), ...)`` to work. Nested loops
    inside a loop body are not searched once a loop equation is found.
    """
    candidates: list[tuple[core.JaxprEqn, str]] = []
    for eqn in jaxpr.eqns:
        primitive = str(eqn.primitive)
        if primitive in _LOOP_PRIMITIVES:
            candidates.append((eqn, primitive))
        elif primitive in _INLINED_PRIMITIVES and "jaxpr" in eqn.params:
            candidates.extend(_iter_loop_candidates(_as_jaxpr(eqn.params["jaxpr"])))
    return candidates


def _find_loop_eqn(
    jaxpr: core.Jaxpr,
    requested_kind: str | None,
) -> tuple[core.JaxprEqn, str]:
    candidates = _iter_loop_candidates(jaxpr)
    if not candidates:
        raise FeatureExtractionError(
            "no scan/while loop found in traced function; pass body_fn=... for a "
            "bare loop body"
        )
    if len(candidates) > 1:
        raise FeatureExtractionError(
            f"found {len(candidates)} scan/while loops in the traced function; "
            "feature extraction supports exactly one top-level loop per function"
        )
    if requested_kind == "scan":
        for eqn, primitive in candidates:
            if primitive == "scan":
                return eqn, "scan"
        raise FeatureExtractionError(
            "requested loop_kind='scan' but the traced function contains only a "
            "while loop; dynamic trip counts are out of scope, so pass "
            "loop_kind='fori_loop' or use a static-bound loop"
        )
    if requested_kind == "fori_loop":
        for eqn, primitive in candidates:
            if primitive == "scan":
                return eqn, "scan"
        for eqn, primitive in candidates:
            if primitive == "while":
                return eqn, "while"
    if requested_kind == "while":
        for eqn, primitive in candidates:
            if primitive == "while":
                return eqn, "while"
        raise FeatureExtractionError(
            "requested loop_kind='while' but the traced function contains only a "
            "scan loop; use loop_kind='scan' or 'fori_loop'"
        )
    return candidates[0]


def _as_jaxpr(value: Any) -> core.Jaxpr:
    return value.jaxpr if hasattr(value, "jaxpr") else value


def _parse_scan(eqn: core.JaxprEqn, loop_kind: str) -> LoopBody:
    params = eqn.params
    jaxpr = _as_jaxpr(params["jaxpr"])
    if "ft_in" in params:
        consts_tree, carry_tree, xs_tree = params["ft_in"].unpack()
        num_consts = len(consts_tree)
        num_carry = len(carry_tree)
        num_xs = len(xs_tree)
    else:  # older JAX scan params
        num_consts = int(params["num_consts"])
        num_carry = int(params["num_carry"])
        num_xs = len(jaxpr.invars) - num_consts - num_carry
    return LoopBody(
        loop_kind=loop_kind,
        loop_primitive="scan",
        jaxpr=jaxpr,
        num_consts=num_consts,
        num_carry=num_carry,
        num_xs=num_xs,
        unroll=params.get("unroll"),
        source="loop",
    )


def _parse_while(eqn: core.JaxprEqn, loop_kind: str) -> LoopBody:
    params = eqn.params
    jaxpr = _as_jaxpr(params["body_jaxpr"])
    num_consts = int(params.get("body_nconsts", 0))
    num_carry = max(len(jaxpr.invars) - num_consts, 0)
    return LoopBody(
        loop_kind=loop_kind,
        loop_primitive="while",
        jaxpr=jaxpr,
        num_consts=num_consts,
        num_carry=num_carry,
        num_xs=0,
        unroll=None,
        source="loop",
    )


def extract_loop_body(
    fn: Callable[..., Any] | None = None,
    args: Sequence[Any] = (),
    *,
    loop_kind: str | None = None,
    body_fn: Callable[..., Any] | None = None,
    body_args: Sequence[Any] = (),
    body_num_consts: int = 0,
    body_num_xs: int = 0,
) -> LoopBody:
    """Trace ``fn`` (or ``body_fn``) and return its loop-body structure.

    ``fn`` must contain exactly one ``scan`` or ``while`` loop; loops behind
    ``jit``/``pjit`` wrappers are found by descending into the wrapper, and a
    function with more than one loop raises ``FeatureExtractionError``. When a
    bare body is supplied through ``body_fn``, the caller states how many
    leading body inputs are closed-over constants via ``body_num_consts`` and
    how many trailing body inputs are scanned values via ``body_num_xs``; the
    remaining inputs are loop-carried state.

    With ``loop_kind=None`` the reported kind is the traced primitive
    (``scan`` or ``while``) and the carry is counted as-is. Pass
    ``loop_kind="fori_loop"`` to mark a loop whose carry starts with the
    ``fori_loop`` counter; only then is that counter stripped from the
    loop-carried state and output features.
    """

    if loop_kind not in {None, "scan", "fori_loop", "while"}:
        raise ValueError(
            f"loop_kind must be 'scan', 'fori_loop', 'while', or None, got {loop_kind!r}"
        )

    if body_fn is not None:
        if fn is not None:
            raise ValueError("pass either fn or body_fn, not both")
        closed = jax.make_jaxpr(body_fn)(*body_args)
        jaxpr = closed.jaxpr
        kind = loop_kind or "scan"
        num_consts = int(body_num_consts)
        num_xs = int(body_num_xs)
        if num_consts < 0 or num_consts > len(jaxpr.invars):
            raise ValueError("body_num_consts must be within the body input count")
        if num_xs < 0 or num_consts + num_xs > len(jaxpr.invars):
            raise ValueError(
                "body_num_consts + body_num_xs must be within the body input count"
            )
        num_carry = len(jaxpr.invars) - num_consts - num_xs
        return LoopBody(
            loop_kind=kind,
            loop_primitive="body",
            jaxpr=jaxpr,
            num_consts=num_consts,
            num_carry=num_carry,
            num_xs=num_xs,
            unroll=None,
            source="body",
        )

    if fn is None:
        raise ValueError("fn or body_fn is required")
    closed = jax.make_jaxpr(fn)(*args)
    eqn, primitive = _find_loop_eqn(closed.jaxpr, loop_kind)
    # With no requested kind, report the traced primitive verbatim ("scan" or
    # "while") and count the carry as-is. Counter stripping in
    # ``extract_features`` only applies to an explicitly requested
    # ``loop_kind="fori_loop"``, whose loop carries the counter first.
    kind = loop_kind or primitive
    if primitive == "scan":
        return _parse_scan(eqn, kind)
    return _parse_while(eqn, kind)


def _classify(primitive: str) -> str:
    if primitive in _DOT_PRIMITIVES:
        return "dot"
    if primitive in _GATHER_SCATTER_PRIMITIVES:
        return "gather_scatter"
    if primitive in _CONVERT_PRIMITIVES:
        return "convert"
    if primitive.startswith("reduce_") or primitive in {"argmax", "argmin"}:
        return "reduce"
    if primitive in _ELEMENTWISE_PRIMITIVES:
        return "elementwise"
    return "other"


def _dot_flops(eqn: core.JaxprEqn) -> int:
    try:
        lhs = eqn.invars[0].aval
        rhs = eqn.invars[1].aval
        dimension_numbers = eqn.params.get("dimension_numbers")
        if dimension_numbers is None:
            return 0
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
        lhs_batch = tuple(lhs_batch)
        rhs_batch = tuple(rhs_batch)
        lhs_contract = tuple(lhs_contract)
        rhs_contract = tuple(rhs_contract)
        batch = _prod([lhs.shape[i] for i in lhs_batch])
        m = _prod(
            [
                lhs.shape[i]
                for i in range(len(lhs.shape))
                if i not in set(lhs_contract) | set(lhs_batch)
            ]
        )
        n = _prod(
            [
                rhs.shape[i]
                for i in range(len(rhs.shape))
                if i not in set(rhs_contract) | set(rhs_batch)
            ]
        )
        k = _prod([lhs.shape[i] for i in lhs_contract])
        return 2 * batch * m * n * k
    except Exception:
        return 0


def _eqn_flops(eqn: core.JaxprEqn, category: str) -> int:
    try:
        if category == "dot":
            return _dot_flops(eqn)
        if category == "reduce":
            return _aval_elements(eqn.invars[0].aval)
        if category == "elementwise":
            return _aval_elements(eqn.outvars[0].aval)
        return 0
    except Exception:
        return 0


def _body_counts(jaxpr: core.Jaxpr) -> tuple[int, dict[str, int], int, int]:
    """Count leaf equations after inlining jit wrappers, plus flops and max tensor."""

    counts = {
        "elementwise": 0,
        "dot": 0,
        "reduce": 0,
        "gather_scatter": 0,
        "convert": 0,
        "other": 0,
    }

    def walk(eqns: Sequence[core.JaxprEqn]) -> tuple[int, int, int]:
        n_eqns = 0
        flops = 0
        max_bytes = 0
        for eqn in eqns:
            primitive = str(eqn.primitive)
            if primitive in _INLINED_PRIMITIVES and "jaxpr" in eqn.params:
                inner_n, inner_flops, inner_max = walk(
                    _as_jaxpr(eqn.params["jaxpr"]).eqns
                )
                n_eqns += inner_n
                flops += inner_flops
                max_bytes = max(max_bytes, inner_max)
                continue
            category = _classify(primitive)
            counts[category] += 1
            n_eqns += 1
            flops += _eqn_flops(eqn, category)
            for value in (*eqn.invars, *eqn.outvars):
                max_bytes = max(max_bytes, _aval_bytes(value.aval))
        return n_eqns, flops, max_bytes

    n_eqns, flops, max_bytes = walk(jaxpr.eqns)
    return n_eqns, counts, flops, max_bytes


def _dependence_features(jaxpr: core.Jaxpr) -> dict[str, float]:
    eqns = list(jaxpr.eqns)
    producer: dict[int, int] = {}
    for index, eqn in enumerate(eqns):
        for outvar in eqn.outvars:
            producer[id(outvar)] = index

    depth: dict[int, int] = {}
    edge_count = 0
    uses: dict[int, int] = {}
    for index, eqn in enumerate(eqns):
        parents: list[int] = []
        for invar in eqn.invars:
            if _is_var(invar) and id(invar) in producer:
                parents.append(producer[id(invar)])
                uses[id(invar)] = uses.get(id(invar), 0) + 1
        edge_count += len(parents)
        depth[index] = 1 + max((depth[parent] for parent in parents), default=0)

    return {
        "longest_path": int(max(depth.values(), default=0)),
        "max_out_degree": int(max(uses.values(), default=0)),
        "num_feed_forward_values": int(len(uses)),
        "dependence_density": float(edge_count / len(eqns)) if eqns else 0.0,
    }


def _sum_avals(avals: Sequence[Any]) -> tuple[int, int]:
    return sum(_aval_bytes(aval) for aval in avals), sum(
        _aval_elements(aval) for aval in avals
    )


def _validate_like(name: str, value: Any) -> None:
    if name in {"trip_count", "factor"} and not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")


def extract_features(
    fn: Callable[..., Any] | None = None,
    args: Sequence[Any] = (),
    *,
    loop_kind: str | None = None,
    trip_count: int,
    factor: int,
    B: float | None = None,
    R: int | None = None,
    backend: str | None = None,
    device: str | None = None,
    body_fn: Callable[..., Any] | None = None,
    body_args: Sequence[Any] = (),
    body_num_consts: int = 0,
    body_num_xs: int = 0,
) -> dict[str, Any]:
    """Return static features for one ``(loop, trip_count, factor)`` configuration.

    ``fn`` is traced with :func:`jax.make_jaxpr`; no candidate factor is
    compiled or timed. ``fn`` may be plain or wrapped in ``jax.jit``; the loop
    is located through ``jit``/``pjit`` wrappers. When ``body_fn`` is given, it
    is traced as a bare loop body instead of locating a ``scan``/``while``
    equation inside ``fn``.
    """

    _validate_like("trip_count", trip_count)
    _validate_like("factor", factor)
    if trip_count <= 0:
        raise ValueError(f"trip_count must be positive, got {trip_count}")
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    if B is not None and B < 0:
        raise ValueError(f"B must be non-negative, got {B}")
    if R is not None and R < 0:
        raise ValueError(f"R must be non-negative, got {R}")

    started = time.perf_counter()
    loop = extract_loop_body(
        fn,
        args,
        loop_kind=loop_kind,
        body_fn=body_fn,
        body_args=body_args,
        body_num_consts=body_num_consts,
        body_num_xs=body_num_xs,
    )
    jaxpr = loop.jaxpr

    invars = list(jaxpr.invars)
    outvars = list(jaxpr.outvars)
    const_avals = [value.aval for value in invars[: loop.num_consts]]
    carry_avals = [
        value.aval
        for value in invars[loop.num_consts : loop.num_consts + loop.num_carry]
    ]
    state_carry_avals = (
        carry_avals[1:] if loop.loop_kind == "fori_loop" else carry_avals
    )
    xs_avals = [
        value.aval
        for value in invars[loop.num_consts + loop.num_carry :]
    ]
    if loop.loop_kind == "fori_loop":
        state_outvars = outvars[1:] if loop.source == "loop" else outvars
    else:
        state_outvars = outvars
    out_avals = [value.aval for value in state_outvars]
    all_input_avals = [value.aval for value in invars]

    carry_bytes, carry_elements = _sum_avals(state_carry_avals)
    xs_bytes, _ = _sum_avals(xs_avals)
    const_bytes, _ = _sum_avals(const_avals)
    input_bytes, input_elements = _sum_avals(all_input_avals)
    output_bytes, output_elements = _sum_avals(out_avals)

    output_ids = {id(value) for value in outvars}
    intermediate_bytes = 0
    all_avals = [*all_input_avals, *out_avals]
    for eqn in jaxpr.eqns:
        for outvar in eqn.outvars:
            all_avals.append(outvar.aval)
            if id(outvar) not in output_ids:
                intermediate_bytes += _aval_bytes(outvar.aval)

    num_equations, counts, flops_proxy, max_inline_bytes = _body_counts(jaxpr)
    max_tensor_bytes = max(
        (*(_aval_bytes(aval) for aval in all_avals), max_inline_bytes),
        default=0,
    )
    dependence = _dependence_features(jaxpr)
    dtypes = sorted({str(aval.dtype) for aval in all_avals})
    dtype_bits = max((_aval_dtype_bits(aval) for aval in all_avals), default=0)
    remainder = trip_count % factor
    bytes_moved_proxy = input_bytes + output_bytes + intermediate_bytes
    body_hash = hashlib.sha256(str(jaxpr).encode("utf-8")).hexdigest()[:16]

    features: dict[str, Any] = {
        "loop_kind": loop.loop_kind,
        "loop_primitive": loop.loop_primitive,
        "trip_count": int(trip_count),
        "factor": int(factor),
        "full_chunks": int(trip_count // factor),
        "remainder": int(remainder),
        "remainder_ratio": float(remainder / trip_count),
        "divides_evenly": bool(remainder == 0),
        "has_tail": bool(remainder > 0),
        "factor_over_trip": float(factor / trip_count),
        "backend": backend if backend is not None else jax.default_backend(),
        "device": device if device is not None else str(jax.devices()[0]),
        "dtype_bits": int(dtype_bits),
        "B": None if B is None else float(B),
        "R": None if R is None else int(R),
        "num_consts": int(loop.num_consts),
        "num_carry": int(loop.num_carry),
        "num_loop_carried": int(len(state_carry_avals)),
        "num_xs": int(loop.num_xs),
        "num_invars": int(len(invars)),
        "num_outvars": int(len(outvars)),
        "num_equations": int(num_equations),
        "unrolled_equations": int(num_equations * factor),
        "eq_elementwise": int(counts["elementwise"]),
        "eq_dot": int(counts["dot"]),
        "eq_reduce": int(counts["reduce"]),
        "eq_gather_scatter": int(counts["gather_scatter"]),
        "eq_convert": int(counts["convert"]),
        "eq_other": int(counts["other"]),
        "carry_bytes": int(carry_bytes),
        "xs_bytes": int(xs_bytes),
        "const_bytes": int(const_bytes),
        "input_bytes": int(input_bytes),
        "output_bytes": int(output_bytes),
        "intermediate_bytes": int(intermediate_bytes),
        "max_tensor_bytes": int(max_tensor_bytes),
        "bytes_moved_proxy": int(bytes_moved_proxy),
        "input_elements": int(input_elements),
        "output_elements": int(output_elements),
        "carry_elements": int(carry_elements),
        "max_rank": int(max((len(aval.shape) for aval in all_avals), default=0)),
        "carry_shapes": _json_shapes(state_carry_avals),
        "const_shapes": _json_shapes(const_avals),
        "input_shapes": _json_shapes(all_input_avals),
        "output_shapes": _json_shapes(out_avals),
        "dtypes": str(list(dtypes)).replace("'", '"'),
        "longest_path": dependence["longest_path"],
        "max_out_degree": dependence["max_out_degree"],
        "num_feed_forward_values": dependence["num_feed_forward_values"],
        "dependence_density": float(dependence["dependence_density"]),
        "flops_proxy": int(flops_proxy),
        "arithmetic_intensity_proxy": (
            float(flops_proxy / bytes_moved_proxy) if bytes_moved_proxy > 0 else 0.0
        ),
        "equations_per_iteration": int(num_equations),
        "body_hash": body_hash,
        "extraction_ms": float((time.perf_counter() - started) * 1e3),
    }
    missing = set(FEATURE_COLUMNS).difference(features)
    if missing:
        raise FeatureExtractionError(
            f"feature extractor omitted columns: {sorted(missing)}"
        )
    return {column: features[column] for column in FEATURE_COLUMNS}


def extract_features_from_workload(
    workload: Any,
    *,
    B: float | None = None,
    R: int | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for a ``benchmarks.workloads.Workload``-like object."""

    return extract_features(
        workload.fn,
        workload.args,
        loop_kind=workload.loop_kind,
        trip_count=int(workload.trip_count),
        factor=int(workload.factor),
        B=B,
        R=R,
    )
