import jax
import jax.numpy as jnp
import numpy as np

from jax_unroll.autotune import _paired_choice, tune_fori_kernel


def test_consistent_material_gains_required():
    times = np.tile([10., 9.8, 8., 9.], (7, 1))
    times[:3, 2] = 11  # Factor 4 only wins on four of seven rounds.
    assert _paired_choice(times) == 3
    times[:, 3] = 10
    assert _paired_choice(times) == 0


def test_tuner_returns_reusable_kernel():
    def build(factor):
        return lambda x: jax.lax.fori_loop(0, 8, lambda i, c: c + i, x, unroll=factor)

    tuned = tune_fori_kernel(build, (jnp.int32(1),))
    assert tuned.factor in (1, 2, 4, 8)
    assert tuned.setup_ms > 0
    assert int(tuned.kernel(jnp.int32(5))) == 33
