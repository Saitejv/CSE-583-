"""Application confirmation on bundled real data, fixed before policy fitting."""

from functools import lru_cache

from jax import lax
import jax.numpy as jnp
import numpy as np

from benchmarks.corpus.fori_regret.workloads import Case


@lru_cache
def digits():
    from sklearn.datasets import load_digits
    data = load_digits()
    return (data.data / 16).astype(np.float32), data.target.astype(np.int32)


def threshold_curves(size, trip, factor):
    """One-vs-rest confusion counts and precision/recall over 64 score thresholds."""
    from sklearn.naive_bayes import GaussianNB
    x, y = digits()
    model = GaussianNB(var_smoothing=.01).fit(x[:-trip], y[:-trip])
    scores = model.predict_proba(x[-trip:]).astype(np.float32)
    thresholds = np.linspace(.01, .99, 64, dtype=np.float32)
    zero = np.zeros((10, 64), np.int32)

    def run(scores, labels, thresholds, initial):
        classes = jnp.arange(10)
        def step(i, counts):
            tp, fp = counts
            selected = scores[i, :, None] >= thresholds
            positive = (classes == labels[i])[:, None]
            return tp + (selected & positive), fp + (selected & ~positive)
        tp, fp = lax.fori_loop(0, trip, step, (initial, initial), unroll=factor)
        positives = jnp.bincount(labels, length=10)
        return tp / jnp.maximum(tp + fp, 1), tp / jnp.maximum(positives[:, None], 1)

    return run, (scores, y[-trip:], thresholds, zero)


def neighborhood_contrast(size, trip, factor):
    """Grayscale morphological gradient over a 9 by 9 footprint on a flower photograph."""
    from sklearn.datasets import load_sample_image
    rgb = load_sample_image("flower.jpg")[128:224, 180:276]
    gray = np.asarray(rgb @ np.array([.299, .587, .114]), np.float32) / 255
    assert trip == 81

    def run(image):
        padded = jnp.pad(image, ((4,4), (4,4)), mode="edge")
        def step(i, state):
            low, high = state
            neighbor = lax.dynamic_slice(padded, (i // 9, i % 9), image.shape)
            return jnp.minimum(low, neighbor), jnp.maximum(high, neighbor)
        low, high = lax.fori_loop(0, trip, step, (image, image), unroll=factor)
        return high - low

    return run, (gray,)


def online_regression(size, trip, factor):
    """One normalized least-mean-squares pass over the diabetes regression dataset."""
    from sklearn.datasets import load_diabetes
    x, y = load_diabetes(return_X_y=True)
    x = np.column_stack([np.ones(trip), x[:trip] * np.sqrt(len(x))]).astype(np.float32)
    y = ((y[:trip] - y.mean()) / y.std()).astype(np.float32)
    zero = np.zeros(x.shape[1], np.float32)

    def run(x, y, coefficients):
        def step(i, coef):
            row = x[i]
            error = y[i] - jnp.dot(row, coef)
            return coef + (.2 * error / (.1 + jnp.dot(row, row))) * row
        return lax.fori_loop(0, trip, step, coefficients, unroll=factor)

    return run, (x, y, zero)


def class_pixel_peaks(size, trip, factor):
    """Classwise feature envelopes using JAX's actual segmented maximum implementation."""
    from jax._src.lax import control_flow
    from unittest.mock import patch
    x, y = digits()
    order = np.argsort(y[:trip], kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(y[:trip], minlength=10))[:-1]].astype(np.int32)

    def run(sorted_pixels, starts):
        # Adapt only the upstream call site; preserve the installed reduction body.
        def apply(lower, upper, body, carry):
            return lax.fori_loop(lower, upper, body, carry, unroll=factor)
        with patch.object(control_flow, "fori_loop", apply):
            # The enclosing benchmark JIT owns compilation. Bypass the method's
            # nested JIT cache so one factor cannot reuse another factor's trace.
            return type(jnp.maximum).reduceat.__wrapped__(jnp.maximum, sorted_pixels, starts, axis=0)

    return run, (x[order], starts)


def settings():
    return [(Case(fn.__name__, fn.__name__, "application", fn), "real", trip)
            for fn, trip in ((threshold_curves, 1024), (neighborhood_contrast, 81),
                             (online_regression, 442), (class_pixel_peaks, 1024))]
