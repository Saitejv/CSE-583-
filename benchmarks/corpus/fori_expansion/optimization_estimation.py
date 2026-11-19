"""Optimization and estimation independent fori_loop workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats
from benchmarks.corpus.spec import CorpusSpec, Provenance

# 1. opt_coordinate_descent: Lasso coordinate descent with soft thresholding
def _coord_descent_factory(variant: str, trip: int, factor: int):
    dim = {"dim16": 16, "dim32": 32, "dim64": 64}[variant]
    n_samples = dim * 2
    rng = np.random.default_rng(583_811)
    X = floats(rng, n_samples, dim, scale=0.5)
    y = floats(rng, n_samples, scale=1.0)
    # precompute X column norms
    col_norm_sq = jnp.sum(X ** 2, axis=0) + jnp.float32(1e-5)
    lmbda = jnp.float32(0.1)
    init_coef = jnp.zeros((dim,), dtype=jnp.float32)
    init_resid = -y

    def fn(coef_init: jax.Array, resid_init: jax.Array, X_mat: jax.Array, y_vec: jax.Array, col_sq: jax.Array, lam: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            coef, resid = carry
            j = i % dim
            xj = X_mat[:, j]
            old_cj = coef[j]
            rho = jnp.dot(xj, resid) + old_cj * col_sq[j]
            # soft-thresholding
            new_cj = jnp.sign(rho) * jnp.maximum(jnp.float32(0.0), jnp.abs(rho) - lam) / col_sq[j]
            diff = new_cj - old_cj
            new_resid = resid + diff * xj
            new_coef = coef.at[j].set(new_cj)
            return (new_coef, new_resid)

        return lax.fori_loop(0, trip, body, (coef_init, resid_init), unroll=factor)

    return built(fn, init_coef, init_resid, X, y, col_norm_sq, lmbda)


# 2. opt_primal_dual_chambolle: Chambolle-Pock First-Order Primal-Dual TV Solver
def _primal_dual_factory(variant: str, trip: int, factor: int):
    n = {"n64": 64, "n128": 128, "n256": 256}[variant]
    rng = np.random.default_rng(583_812)
    g = floats(rng, n, scale=1.0)
    tau = jnp.float32(0.2)
    sigma = jnp.float32(0.2)
    theta = jnp.float32(1.0)
    init_x = jnp.zeros((n,), dtype=jnp.float32)
    init_y = jnp.zeros((n,), dtype=jnp.float32)

    def fn(x_init: jax.Array, y_init: jax.Array, obs: jax.Array, t_step: jax.Array, s_step: jax.Array, th: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            x, y, x_bar = carry
            # dual update: y = clip(y + sigma * K(x_bar), -1, 1)
            grad_x = jnp.roll(x_bar, -1) - x_bar
            new_y = jnp.clip(y + s_step * grad_x, -1.0, 1.0)
            # primal update: x_next = (x - tau * K^T(y) + tau * obs) / (1 + tau)
            div_y = new_y - jnp.roll(new_y, 1)
            new_x = (x + t_step * div_y + t_step * obs) / (jnp.float32(1.0) + t_step)
            # extrapolation
            new_x_bar = new_x + th * (new_x - x)
            return (new_x, new_y, new_x_bar)

        return lax.fori_loop(0, trip, body, (x_init, y_init, x_init), unroll=factor)

    return built(fn, init_x, init_y, g, tau, sigma, theta)


# 3. opt_recursive_least_squares: RLS Sherman-Morrison Covariance Tracking
def _rls_factory(variant: str, trip: int, factor: int):
    p_dim = {"p4": 4, "p8": 8, "p16": 16}[variant]
    rng = np.random.default_rng(583_813)
    x_stream = floats(rng, trip, p_dim, scale=0.5)
    d_stream = floats(rng, trip, scale=1.0)
    lam = jnp.float32(0.99)
    init_w = jnp.zeros((p_dim,), dtype=jnp.float32)
    init_P = jnp.eye(p_dim, dtype=jnp.float32) * jnp.float32(10.0)

    def fn(w_init: jax.Array, P_init: jax.Array, xs: jax.Array, ds: jax.Array, l_factor: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            w, P = carry
            x = xs[i]
            d = ds[i]
            Px = P @ x
            denom = l_factor + jnp.dot(x, Px)
            k = Px / denom
            alpha = d - jnp.dot(x, w)
            new_w = w + k * alpha
            new_P = (P - jnp.outer(k, Px)) / l_factor
            return (new_w, new_P)

        return lax.fori_loop(0, trip, body, (w_init, P_init), unroll=factor)

    return built(fn, init_w, init_P, x_stream, d_stream, lam)


# 4. opt_kalman_diagonal: Diagonalized State-Space Kalman Estimator
def _kalman_factory(variant: str, trip: int, factor: int):
    dim = {"dim4": 4, "dim8": 8, "dim16": 16}[variant]
    rng = np.random.default_rng(583_814)
    obs = floats(rng, trip, dim, scale=1.0)
    F_diag = jnp.asarray(0.95 + 0.04 * rng.uniform(size=dim), dtype=jnp.float32)
    Q_diag = jnp.full((dim,), 0.01, dtype=jnp.float32)
    R_diag = jnp.full((dim,), 0.1, dtype=jnp.float32)
    init_x = jnp.zeros((dim,), dtype=jnp.float32)
    init_P = jnp.ones((dim,), dtype=jnp.float32)

    def fn(x_init: jax.Array, P_init: jax.Array, z_stream: jax.Array, F: jax.Array, Q: jax.Array, R: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            x, P = carry
            z = z_stream[i]
            # Predict
            x_pred = F * x
            P_pred = F * F * P + Q
            # Update
            K = P_pred / (P_pred + R)
            x_upd = x_pred + K * (z - x_pred)
            P_upd = (jnp.float32(1.0) - K) * P_pred
            return (x_upd, P_upd)

        return lax.fori_loop(0, trip, body, (x_init, P_init), unroll=factor)

    return built(fn, init_x, init_P, obs, F_diag, Q_diag, R_diag)


# 5. opt_mirror_descent: Entropic Mirror Descent on Simplex [fresh_eval]
def _mirror_descent_factory(variant: str, trip: int, factor: int):
    dim = {"dim16": 16, "dim32": 32, "dim64": 64}[variant]
    rng = np.random.default_rng(583_815)
    grads = floats(rng, trip, dim, scale=0.5)
    eta = jnp.float32(0.05)
    init_w = jnp.full((dim,), 1.0 / dim, dtype=jnp.float32)

    def fn(w_init: jax.Array, grad_stream: jax.Array, step_eta: jax.Array):
        def body(i: int, carry: jax.Array):
            w = carry
            g = grad_stream[i]
            # w_{t+1} \propto w_t * exp(-eta * g)
            log_w = jnp.log(jnp.maximum(w, 1e-12)) - step_eta * g
            log_w = log_w - jnp.max(log_w)
            unnorm = jnp.exp(log_w)
            return unnorm / jnp.sum(unnorm)

        return lax.fori_loop(0, trip, body, w_init, unroll=factor)

    return built(fn, init_w, grads, eta)


# 6. opt_anderson_acceleration: Anderson Depth-2 Fixed-Point Acceleration [fresh_eval]
def _anderson_factory(variant: str, trip: int, factor: int):
    dim = {"dim8": 8, "dim16": 16, "dim32": 32}[variant]
    rng = np.random.default_rng(583_816)
    A = floats(rng, dim, dim, scale=0.8 / np.sqrt(dim))
    b = floats(rng, dim, scale=0.5)
    init_x = jnp.zeros((dim,), dtype=jnp.float32)
    init_r = init_x - (A @ init_x + b)

    def fn(x_init: jax.Array, r_init: jax.Array, A_mat: jax.Array, b_vec: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array]):
            x_k, x_prev, r_k, r_prev = carry
            delta_r = r_k - r_prev
            denom = jnp.dot(delta_r, delta_r) + jnp.float32(1e-6)
            gamma = jnp.dot(r_k, delta_r) / denom
            gamma = jnp.clip(gamma, -2.0, 2.0)
            x_accel = x_k - gamma * (x_k - x_prev)
            r_accel = r_k - gamma * delta_r
            x_next = (A_mat @ x_accel + b_vec)
            r_next = x_next - (A_mat @ x_next + b_vec)
            return (x_next, x_k, r_next, r_k)

        return lax.fori_loop(0, trip, body, (x_init, x_init, r_init, r_init), unroll=factor)

    return built(fn, init_x, init_r, A, b)


# 7. opt_projected_subgradient_box: Projected Subgradient Descent with Box Bounds
def _projected_subgrad_factory(variant: str, trip: int, factor: int):
    dim = {"dim16": 16, "dim32": 32, "dim64": 64}[variant]
    rng = np.random.default_rng(583_817)
    subgrads = floats(rng, trip, dim, scale=0.5)
    gamma = jnp.float32(0.02)
    lower = jnp.full((dim,), -0.5, dtype=jnp.float32)
    upper = jnp.full((dim,), 0.5, dtype=jnp.float32)
    init_x = jnp.zeros((dim,), dtype=jnp.float32)

    def fn(x_init: jax.Array, g_stream: jax.Array, lr: jax.Array, lb: jax.Array, ub: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            x, total_dist = carry
            g = g_stream[i]
            unconstrained = x - lr * g
            projected = jnp.clip(unconstrained, lb, ub)
            step_norm = jnp.linalg.norm(projected - x)
            return (projected, total_dist + step_norm)

        return lax.fori_loop(0, trip, body, (x_init, jnp.float32(0.0)), unroll=factor)

    return built(fn, init_x, subgrads, gamma, lower, upper)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec("opt_coordinate_descent", "optimization_estimation", "Coordinate Descent for Lasso / Elastic Net",
               "opt_coordinate_descent", "Soft-thresholding coordinate update and rank-1 residual downdate",
               "fori_loop", {"dim16": "dim 16", "dim32": "dim 32", "dim64": "dim 64"}, "dim32",
               _coord_descent_factory, 583_811, Provenance("public-inspired", "Friedman et al. (2010) JSS; scikit-learn", "BSD-3-Clause", "Lasso coordinate descent with residual tracking"),
               tier="full", related_group="opt_sparse_reg"),
    CorpusSpec("opt_primal_dual_chambolle", "optimization_estimation", "Chambolle-Pock First-Order Primal-Dual TV Solver",
               "opt_primal_dual", "Coupled primal-dual updates with Linf projection and extrapolation",
               "fori_loop", {"n64": "n 64", "n128": "n 128", "n256": "n 256"}, "n128",
               _primal_dual_factory, 583_812, Provenance("public-inspired", "Chambolle & Pock (2011) JMIV", "Public Domain", "1D total variation primal dual solver"),
               tier="full", related_group="opt_primal_dual"),
    CorpusSpec("opt_recursive_least_squares", "optimization_estimation", "RLS Sherman-Morrison Covariance Tracking",
               "opt_rls_covariance", "Rank-1 inverse covariance tracking via Sherman-Morrison formula",
               "fori_loop", {"p4": "dim 4", "p8": "dim 8", "p16": "dim 16"}, "p8",
               _rls_factory, 583_813, Provenance("public-inspired", "Ljung (1999) System Identification", "Public Domain", "Recursive least squares filtering"),
               tier="full", related_group="opt_recursive"),
    CorpusSpec("opt_kalman_diagonal", "optimization_estimation", "Diagonalized State-Space Kalman Estimator",
               "opt_kalman_filtering", "Predict-update cycle with Kalman gain and variance contraction",
               "fori_loop", {"dim4": "dim 4", "dim8": "dim 8", "dim16": "dim 16"}, "dim8",
               _kalman_factory, 583_814, Provenance("public-inspired", "Kalman (1960) J. Basic Eng.", "Public Domain", "Diagonalized state space Kalman filter"),
               tier="smoke", related_group="opt_kalman"),
    CorpusSpec("opt_mirror_descent", "optimization_estimation", "Entropic Mirror Descent on Simplex",
               "opt_mirror_descent", "Multiplicative weight update on probability simplex with log-sum-exp normalization",
               "fori_loop", {"dim16": "dim 16", "dim32": "dim 32", "dim64": "dim 64"}, "dim32",
               _mirror_descent_factory, 583_815, Provenance("public-inspired", "Beck & Teboulle (2003) Oper. Res. Lett.", "Public Domain", "Simplex entropic mirror descent"),
               tier="full", related_group="opt_mirror_eval"),
    CorpusSpec("opt_anderson_acceleration", "optimization_estimation", "Anderson Depth-2 Fixed-Point Acceleration",
               "opt_anderson_accel", "Multi-history residual regression to compute optimal mixing coefficient",
               "fori_loop", {"dim8": "dim 8", "dim16": "dim 16", "dim32": "dim 32"}, "dim16",
               _anderson_factory, 583_816, Provenance("public-inspired", "Walker & Ni (2011) SIAM J. Numer. Anal.", "Public Domain", "Anderson acceleration fixed point solver"),
               tier="full", related_group="opt_anderson_eval"),
    CorpusSpec("opt_projected_subgradient_box", "optimization_estimation", "Projected Subgradient Descent with Box Bounds",
               "opt_subgradient_projection", "Vector box projection clip(x - gamma*g, l, u) coupled with distance tracking",
               "fori_loop", {"dim16": "dim 16", "dim32": "dim 32", "dim64": "dim 64"}, "dim32",
               _projected_subgrad_factory, 583_817, Provenance("public-inspired", "Shor (1985) Minimization Methods", "Public Domain", "Projected subgradient with box constraints"),
               tier="full", related_group="opt_subgrad"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
