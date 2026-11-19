"""Opt-in kernels for pinned M2 CPU deployments; choose the factor at trace time."""
import jax
import jax.numpy as jnp
import jaxlib
from jax.extend import core

_JAX_VERSION = "0.12.0.1"
SUBTRACT_SHAPES = frozenset({(128, 8), (257, 1), (512, 8), (769, 16), (1024, 8)})
SEGMENTED_SHAPES = frozenset({(256, 4), (513, 7), (769, 13), (1024, 1), (1024, 16)})
SUBTRACT_FACTOR = 8
SEGMENTED_FACTOR = 4


def _pinned_cpu():
    return jax.__version__ == jaxlib.__version__ == _JAX_VERSION and jax.default_backend() == "cpu"


def _with_unroll(operation, args, *, trip_count, factor):
    """Specialize the one static fori_loop in these pinned JAX array operations."""
    original = jax.make_jaxpr(operation)(*args)
    matches = []

    def specialize(graph):
        equations = []
        for eq in graph.jaxpr.eqns:
            if eq.primitive.name in ("jit", "pjit") and "jaxpr" in eq.params:
                inner = specialize(eq.params["jaxpr"])
                if inner is not eq.params["jaxpr"]:
                    eq = eq.replace(params={**eq.params, "jaxpr": inner})
            elif eq.primitive.name == "scan":
                # JAX represents the two known static fori_loops as scans.
                matches.append((eq.params["length"], eq.params["unroll"]))
                eq = eq.replace(params={**eq.params, "unroll": factor})
            equations.append(eq)
        if all(new is old for new, old in zip(equations, graph.jaxpr.eqns)):
            return graph
        return graph.replace(jaxpr=graph.jaxpr.replace(eqns=equations))

    candidate = specialize(original)
    graph = candidate if matches == [(trip_count, 1)] else original
    result, = core.jaxpr_as_fun(graph)(*args)
    return result


@jax.jit
def subtract_reduce_rows(x):
    """Left-to-right jnp.subtract.reduce(x, axis=0) for narrow float32 matrices."""
    if x.shape not in SUBTRACT_SHAPES or x.dtype != jnp.float32 or not _pinned_cpu():
        return jnp.subtract.reduce(x, axis=0)

    return _with_unroll(jnp.subtract.reduce, (x,), trip_count=x.shape[0] - 1, factor=SUBTRACT_FACTOR)


@jax.jit
def add_reduceat_rows(x, starts):
    """Axis-zero add.reduceat; specialize narrow float32 inputs with few segments."""
    if (x.ndim != 2 or x.dtype != jnp.float32 or x.shape[1] != 1 or starts.ndim != 1
            or starts.dtype != jnp.int32 or (x.shape[0], starts.size) not in SEGMENTED_SHAPES
            or not _pinned_cpu()):
        return jnp.add.reduceat(x, starts, axis=0)

    return _with_unroll(jnp.add.reduceat, (x, starts), trip_count=x.shape[0], factor=SEGMENTED_FACTOR)
