"""Reviewed workload drafts from OpenCode."""

import jax
import jax.numpy as jnp
import numpy as np


def adaptive_unmixing(variant, trip, factor):
    """Adaptive nonnegative spectral unmixing of magnitude spectra via normalized multiplicative matrix updates."""
    rank = 3 if variant == "small" else 6
    n_freq = 64 if variant == "small" else 128
    n_spec = 64 if variant == "small" else 128
    batch = 16 if variant == "small" else 32
    rng = np.random.default_rng(0x5EED_A17)

    # Host-side data: nonnegative magnitude spectra as sums of Gaussian peaks plus a noise floor.
    freq = np.linspace(0.0, 1.0, n_freq, dtype=np.float64)
    centers = rng.uniform(0.0, 1.0, size=(n_spec, rank))
    widths = rng.uniform(0.03, 0.12, size=(n_spec, rank))
    amps = rng.uniform(0.4, 1.5, size=(n_spec, rank))
    X = np.zeros((n_spec, n_freq), dtype=np.float64)
    for k in range(rank):
        X += amps[:, k:k + 1] * np.exp(
            -((freq[None, :] - centers[:, k:k + 1]) ** 2) / (2.0 * widths[:, k:k + 1] ** 2)
        )
    X += rng.uniform(0.0, 0.05, size=(n_spec, n_freq))
    X = np.maximum(X, 1e-6).astype(np.float32)

    W0 = (np.abs(rng.normal(0.0, 1.0, size=(n_freq, rank))) + 0.1).astype(np.float32)
    W0 /= np.linalg.norm(W0, axis=0, keepdims=True).astype(np.float32)
    H0 = (np.abs(rng.normal(0.0, 1.0, size=(batch, rank))) + 0.1).astype(np.float32)
    e0 = np.zeros((), dtype=np.float32)

    eps = np.float32(1e-6)

    def run(X, W0, H0, e0):
        def step(i, carry):
            W, H, energy = carry
            idx = (i + jnp.arange(batch, dtype=jnp.int32)) % n_spec
            Xb = jnp.take(X, idx, axis=0)  # changing microbatch: (batch, n_freq)
            # Regularized multiplicative update of abundances, then of the basis.
            H = H * ((W.T @ Xb.T) / ((W.T @ W) @ H.T + eps)).T
            W = W * ((Xb.T @ H) / (W @ (H.T @ H) + eps))
            # Column-normalize the basis and rescale abundances so the reconstruction is unchanged.
            wn = jnp.sqrt(jnp.sum(W * W, axis=0)) + eps
            W = W / wn
            H = H * wn
            energy = energy + jnp.mean((Xb - H @ W.T) ** 2)
            return (W, H, energy)

        return jax.lax.fori_loop(0, trip, step, (W0, H0, e0), unroll=factor)

    args_tuple = (X, W0, H0, e0)
    return run, args_tuple
