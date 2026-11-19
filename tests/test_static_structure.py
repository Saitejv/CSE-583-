from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from jax_unroll.features import extract_loop_body
from jax_unroll.static_structure import STRUCTURE_COLUMNS, structure_features


def features(body, args, **kwargs):
    return structure_features(extract_loop_body(body_fn=body, body_args=args, **kwargs))


def test_forwarded_replaced_and_cross_carry_dependencies():
    init = jnp.zeros(4)
    f = features(lambda a, b, x: (a, a + x), (init, init, init), body_num_xs=1)
    assert tuple(f) == STRUCTURE_COLUMNS
    assert f["carry_updated_bytes"] == 16
    assert f["carry_dependency_bytes"] == 16
    assert f["carry_dependency_pairs"] == 2
    assert f["carry_to_carry_max_depth"] == 1
    replacement = features(lambda a, x: x, (init, init), body_num_xs=1)
    assert replacement["carry_updated_bytes"] == 16
    assert replacement["carry_dependency_bytes"] == 0


def test_jit_per_output_binding_does_not_taint_independent_carry():
    @jax.jit
    def inner(a, x):
        return a * 2, x + 1

    f = features(lambda a, b, x: inner(a, x), (jnp.zeros(4), jnp.zeros(4), jnp.ones(4)), body_num_xs=1)
    assert f["carry_dependency_pairs"] == 1
    assert f["carry_dependency_bytes"] == 16
    assert f["carry_to_carry_max_depth"] == 1
    assert f["structural_opaque_count"] == 0


def test_roll_exposes_slices_and_concatenate_behind_jit():
    def fn(init):
        return lax.fori_loop(0, 16, lambda i, c: c + jnp.roll(c, 1), init)

    loop = extract_loop_body(fn, (jnp.zeros(8),), loop_kind="fori_loop")
    f = structure_features(loop)
    assert f["slice_count"] == f["carry_slice_count"] == 2
    assert f["concatenate_count"] == f["carry_concatenate_count"] == 1
    assert f["carry_slice_bytes"] == 32
    assert f["carry_updated_bytes"] == f["carry_dependency_bytes"] == 32
    assert f["carry_dependency_pairs"] == 1
    assert f["carry_to_carry_max_depth"] == 3


def test_slices_of_scanned_values_are_not_carry_neighborhoods():
    f = features(lambda c, x: c + x[1:], (jnp.zeros(3), jnp.zeros(4)), body_num_xs=1)
    assert f["slice_count"] == 1
    assert f["carry_slice_count"] == f["carry_slice_bytes"] == 0


def test_fori_counter_is_excluded_even_if_used_in_carry():
    def fn(init):
        return lax.fori_loop(0, 16, lambda i, c: c + i, init)

    f = structure_features(extract_loop_body(fn, (jnp.float32(0),), loop_kind="fori_loop"))
    assert f["carry_updated_bytes"] == f["carry_dependency_bytes"] == 4
    assert f["carry_dependency_pairs"] == 1
    assert f["carry_to_carry_max_depth"] == 1


def test_empty_carry_has_zero_feedback():
    f = features(lambda x: (), (jnp.zeros(4),), body_num_xs=1)
    assert f["carry_updated_bytes"] == f["carry_dependency_bytes"] == 0
    assert f["carry_dependency_pairs"] == f["carry_to_carry_max_depth"] == 0


def test_bare_fori_body_returns_state_without_counter():
    f = features(lambda i, c: c + i, (jnp.int32(0), jnp.zeros(8)), loop_kind="fori_loop")
    assert f["carry_updated_bytes"] == f["carry_dependency_bytes"] == 32
    assert f["carry_dependency_pairs"] == 1


def test_cond_is_marked_opaque_and_conservative():
    g = features(lambda a, x: lax.cond(x > 0, lambda: a, lambda: x),
                 (jnp.float32(1), jnp.float32(3)), body_num_xs=1)
    assert g["structural_opaque_count"] == 1
    assert g["carry_dependency_pairs"] == 1
