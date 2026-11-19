"""Tests for factor invariance across all 48 targeted fori_loop workloads."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.corpus.fori_targeted.registry import ALL_SPECS, build_specialized_workload, get_spec
from benchmarks.sweep import _all_finite, _allclose


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_workload_factor_invariance(spec):
    trip = 256
    var = "small"

    w1 = spec.build(trip, 1, var)
    out1 = w1.fn(*w1.args)
    assert _all_finite(out1), f"{spec.name}: non-finite output at factor=1"

    for u in (2, 4, 8):
        wu = spec.build(trip, u, var)
        outu = wu.fn(*wu.args)
        assert _all_finite(outu), f"{spec.name}: non-finite output at factor={u}"
        assert _allclose(out1, outu, atol=spec.atol, rtol=spec.rtol), (
            f"{spec.name}: output at factor={u} differs from factor=1"
        )


@pytest.mark.parametrize("trip", [5, 37, 289])
def test_blocked_delay_ring_wrap_and_short_tail(trip):
    spec = get_spec("write_multitap_delay_ring")
    native = spec.build(trip, 1, "small")
    specialized = build_specialized_workload(spec.name, trip, "small")
    ring, head, sample, stream = native.args
    args = (jnp.arange(ring.size, dtype=ring.dtype) / ring.size,
            jnp.asarray(ring.size - 3, dtype=head.dtype), sample, stream)
    expected = jax.jit(native.fn)(*args)
    actual = jax.jit(specialized.fn)(*args)
    np.testing.assert_array_equal(actual[1], expected[1])
    assert _allclose(expected, actual, atol=spec.atol, rtol=spec.rtol)


def test_blocked_horner_partial_block_and_noncontractive_fallback():
    spec = get_spec("eval_dense_horner_matrix_poly")
    native = spec.build(37, 1, "small")
    optimized = build_specialized_workload(spec.name, 37, "small")
    initial, _, coeffs = native.args
    initial = initial + 0.125
    identity = jnp.eye(initial.shape[0], dtype=initial.dtype)
    matrix = 0.92 * identity + 0.04 * jnp.roll(identity, 1, axis=1)
    run_native, run_optimized = jax.jit(native.fn), jax.jit(optimized.fn)

    expected = run_native(initial, matrix, coeffs)
    actual = run_optimized(initial, matrix, coeffs)
    np.testing.assert_allclose(actual, expected, rtol=spec.rtol, atol=spec.atol)

    noncontractive = identity * 1.01
    np.testing.assert_array_equal(
        run_optimized(initial, noncontractive, coeffs), run_native(initial, noncontractive, coeffs),
    )


@pytest.mark.parametrize("trip", [5, 263])
def test_blocked_pic_preserves_colliding_updates(trip):
    spec = get_spec("eval_write_pic_charge_deposit")
    native = spec.build(trip, 1, "small")
    optimized = build_specialized_workload(spec.name, trip, "small")
    grid, positions, charges = native.args
    positions = jnp.full_like(positions, 3.25)
    args = (grid + 0.25, positions, charges)
    expected = jax.jit(native.fn)(*args)
    actual = jax.jit(optimized.fn)(*args)
    np.testing.assert_allclose(actual, expected, rtol=spec.rtol, atol=spec.atol)


@pytest.mark.parametrize("trip", [3, 4, 1023])
def test_paired_waveguide_with_nonzero_initial_state(trip):
    spec = get_spec("eval_write_waveguide_mesh")
    native = spec.build(trip, 1, "small")
    optimized = build_specialized_workload(spec.name, trip, "small")
    state, excitations = native.args
    # Mix a constant pressure mode with an asymmetric disturbance at the wrap boundary.
    state = (state + 0.2).at[0, 0, -1].set(-0.4)
    args = state, excitations
    expected = jax.jit(native.fn)(*args)
    actual = jax.jit(optimized.fn)(*args)
    np.testing.assert_allclose(actual, expected, rtol=spec.rtol, atol=spec.atol)


def test_ct_partial_block_adds_angles_in_order():
    spec = get_spec("eval_app_ct_backprojection_step")
    native = spec.build(19, 1, "small")
    optimized = build_specialized_workload(spec.name, 19, "small")
    image, sinogram, cos, sin, gx, gy = native.args
    # The first unit contribution rounds away only if added to the carry
    # before the large negative contribution in the partial block.
    projections = jnp.asarray([0] * 15 + [2**24, 1, -(2**24), 1], dtype=sinogram.dtype)
    angles = jnp.broadcast_to(projections[:, None], sinogram.shape)
    args = image, angles, cos, sin, jnp.zeros_like(gx), jnp.zeros_like(gy)
    expected = jax.jit(native.fn)(*args)
    actual = jax.jit(optimized.fn)(*args)
    np.testing.assert_array_equal(actual, expected)
