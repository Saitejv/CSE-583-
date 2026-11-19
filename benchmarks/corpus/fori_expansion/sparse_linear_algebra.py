"""Sparse linear algebra independent fori_loop workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats, indices
from benchmarks.corpus.spec import CorpusSpec, Provenance

# 1. sp_csr_matvec_row: Segmented CSR Sparse Matvec
def _csr_gemv_factory(variant: str, trip: int, factor: int):
    # variant controls rows and fixed degree per row
    rows, deg = {"r64k4": (64, 4), "r128k4": (128, 4), "r256k8": (256, 8)}[variant]
    rng = np.random.default_rng(583_821)
    cols = indices(rng, rows, rows, deg)
    vals = floats(rng, rows, deg, scale=0.5)
    x = floats(rng, rows, scale=1.0)
    init_y = jnp.zeros((rows,), dtype=jnp.float32)

    def fn(y_init: jax.Array, col_mat: jax.Array, val_mat: jax.Array, x_vec: jax.Array):
        def body(i: int, carry: jax.Array):
            r = i % rows
            c = col_mat[r]
            v = val_mat[r]
            row_prod = jnp.sum(v * x_vec[c])
            return carry.at[r].add(row_prod * 0.1)

        return lax.fori_loop(0, trip, body, y_init, unroll=factor)

    return built(fn, init_y, cols, vals, x)


# 2. sp_triangular_solve: Forward Substitution Sparse Lower Triangular Solver
def _tri_solve_factory(variant: str, trip: int, factor: int):
    n = {"n32": 32, "n64": 64, "n128": 128}[variant]
    rng = np.random.default_rng(583_822)
    # Banded lower triangular matrix: diagonal + 3 subdiagonals
    diag = jnp.asarray(1.5 + rng.uniform(0.5, 1.0, size=n), dtype=jnp.float32)
    sub1 = floats(rng, n - 1, scale=0.3)
    sub2 = floats(rng, n - 2, scale=0.2)
    sub3 = floats(rng, n - 3, scale=0.1)
    b = floats(rng, n, scale=1.0)
    init_x = jnp.zeros((n,), dtype=jnp.float32)

    def fn(x_init: jax.Array, d: jax.Array, s1: jax.Array, s2: jax.Array, s3: jax.Array, rhs: jax.Array):
        def body(i: int, carry: jax.Array):
            k = i % n
            # forward substitution reading previous components
            prev1 = jnp.where(k >= 1, carry[jnp.maximum(0, k - 1)] * s1[jnp.maximum(0, k - 1)], 0.0)
            prev2 = jnp.where(k >= 2, carry[jnp.maximum(0, k - 2)] * s2[jnp.maximum(0, k - 2)], 0.0)
            prev3 = jnp.where(k >= 3, carry[jnp.maximum(0, k - 3)] * s3[jnp.maximum(0, k - 3)], 0.0)
            val = (rhs[k] - prev1 - prev2 - prev3) / d[k]
            return carry.at[k].set(val)

        return lax.fori_loop(0, trip, body, x_init, unroll=factor)

    return built(fn, init_x, diag, sub1, sub2, sub3, b)


# 3. sp_bcr_relaxation: Block-Coordinate Gauss-Seidel Relaxation
def _bcr_factory(variant: str, trip: int, factor: int):
    blocks, bsize = {"b8s4": (8, 4), "b16s4": (16, 4)}[variant]
    total_dim = blocks * bsize
    rng = np.random.default_rng(583_823)
    # Inverse diagonal blocks: (blocks, bsize, bsize)
    D_inv_np = np.zeros((blocks, bsize, bsize), dtype=np.float32)
    for b in range(blocks):
        m = rng.normal(size=(bsize, bsize)).astype(np.float32)
        sym_pos = m @ m.T + 2.0 * np.eye(bsize, dtype=np.float32)
        D_inv_np[b] = np.linalg.inv(sym_pos)
    D_inv = jnp.asarray(D_inv_np, dtype=jnp.float32)
    coupling = floats(rng, blocks, bsize, bsize, scale=0.1)
    rhs = floats(rng, total_dim, scale=1.0)
    init_x = jnp.zeros((total_dim,), dtype=jnp.float32)

    def fn(x_init: jax.Array, inv_blocks: jax.Array, coup: jax.Array, b_vec: jax.Array):
        def body(i: int, carry: jax.Array):
            blk = i % blocks
            start = blk * bsize
            nbr = (blk + 1) % blocks
            nbr_start = nbr * bsize
            x_nbr = lax.dynamic_slice_in_dim(carry, nbr_start, bsize)
            rhs_blk = lax.dynamic_slice_in_dim(b_vec, start, bsize)
            res_blk = rhs_blk - coup[blk] @ x_nbr
            new_blk = inv_blocks[blk] @ res_blk
            return lax.dynamic_update_slice_in_dim(carry, new_blk, start, 0)

        return lax.fori_loop(0, trip, body, x_init, unroll=factor)

    return built(fn, init_x, D_inv, coupling, rhs)


# 4. sp_residual_propagation: Sparse Graph Residual Mass Diffusion
def _res_prop_factory(variant: str, trip: int, factor: int):
    nodes = {"n64": 64, "n128": 128, "n256": 256}[variant]
    rng = np.random.default_rng(583_824)
    src_nodes = indices(rng, nodes, trip)
    dst_nodes = indices(rng, nodes, trip)
    alpha = jnp.float32(0.85)
    init_res = floats(rng, nodes, scale=1.0)
    init_accum = jnp.zeros((nodes,), dtype=jnp.float32)

    def fn(res_init: jax.Array, acc_init: jax.Array, srcs: jax.Array, dsts: jax.Array, a: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            res, acc = carry
            u = srcs[i]
            v = dsts[i]
            push_mass = res[u] * (1.0 - a)
            new_res = res.at[u].add(-push_mass).at[v].add(push_mass * a)
            new_acc = acc.at[v].add(push_mass * (1.0 - a))
            return (new_res, new_acc)

        return lax.fori_loop(0, trip, body, (res_init, acc_init), unroll=factor)

    return built(fn, init_res, init_accum, src_nodes, dst_nodes, alpha)


# 5. sp_ellpack_gemv: ELLPACK Formatted Fixed-Width Sparse Matvec [fresh_eval]
def _ellpack_factory(variant: str, trip: int, factor: int):
    rows, width = {"r64w8": (64, 8), "r128w8": (128, 8), "r256w16": (256, 16)}[variant]
    rng = np.random.default_rng(583_825)
    col_idx = indices(rng, rows, rows, width)
    coef = floats(rng, rows, width, scale=0.5)
    x = floats(rng, rows, scale=1.0)
    init_y = jnp.zeros((rows,), dtype=jnp.float32)

    def fn(y_init: jax.Array, cols: jax.Array, vals: jax.Array, x_vec: jax.Array):
        def body(i: int, carry: jax.Array):
            r = i % rows
            gathered = x_vec[cols[r]]
            dot_prod = jnp.dot(vals[r], gathered)
            return carry.at[r].set(dot_prod)

        return lax.fori_loop(0, trip, body, y_init, unroll=factor)

    return built(fn, init_y, col_idx, coef, x)


# 6. sp_ilu0_precond_apply: Incomplete LU Zero-Fill Preconditioner Solve [fresh_eval]
def _ilu0_factory(variant: str, trip: int, factor: int):
    n = {"n32": 32, "n64": 64, "n128": 128}[variant]
    rng = np.random.default_rng(583_826)
    # L lower off-diagonal, U upper off-diagonal, D diagonal
    L_sub = floats(rng, n - 1, scale=0.3)
    U_sup = floats(rng, n - 1, scale=0.3)
    D_inv = jnp.asarray(1.0 / (1.5 + rng.uniform(0.5, 1.0, size=n)), dtype=jnp.float32)
    rhs = floats(rng, n, scale=1.0)
    init_z = jnp.zeros((n,), dtype=jnp.float32)

    def fn(z_init: jax.Array, l_sub: jax.Array, u_sup: jax.Array, d_inv: jax.Array, r_vec: jax.Array):
        def body(i: int, carry: jax.Array):
            k = i % n
            # Forward solve: L y = r
            y_prev = jnp.where(k > 0, carry[jnp.maximum(0, k - 1)], 0.0)
            y_k = (r_vec[k] - l_sub[jnp.maximum(0, k - 1)] * y_prev) * d_inv[k]
            # Backward step: U z = y
            back_k = n - 1 - k
            z_next = jnp.where(back_k < n - 1, carry[jnp.minimum(n - 1, back_k + 1)], 0.0)
            z_val = y_k - u_sup[jnp.minimum(n - 2, back_k)] * z_next
            return carry.at[k].set(z_val)

        return lax.fori_loop(0, trip, body, z_init, unroll=factor)

    return built(fn, init_z, L_sub, U_sup, D_inv, rhs)


# 7. sp_jacobi_sparse_row: Sparse Jacobi Iterative Relaxation Step
def _jacobi_sparse_factory(variant: str, trip: int, factor: int):
    n, k_deg = {"n64k4": (64, 4), "n128k4": (128, 4), "n256k8": (256, 8)}[variant]
    rng = np.random.default_rng(583_827)
    nbr_idx = indices(rng, n, n, k_deg)
    nbr_w = floats(rng, n, k_deg, scale=0.2 / k_deg)
    diag_inv = jnp.asarray(1.0 / (1.5 + rng.uniform(0.5, 1.0, size=n)), dtype=jnp.float32)
    rhs = floats(rng, n, scale=1.0)
    init_x = jnp.zeros((n,), dtype=jnp.float32)

    def fn(x_init: jax.Array, nbrs: jax.Array, weights: jax.Array, d_inv: jax.Array, b_vec: jax.Array):
        def body(i: int, carry: jax.Array):
            node = i % n
            off_diag = jnp.dot(weights[node], carry[nbrs[node]])
            x_new = (b_vec[node] - off_diag) * d_inv[node]
            return carry.at[node].set(x_new)

        return lax.fori_loop(0, trip, body, x_init, unroll=factor)

    return built(fn, init_x, nbr_idx, nbr_w, diag_inv, rhs)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec("sp_csr_matvec_row", "sparse_linear_algebra", "Segmented CSR Sparse Matrix-Vector Product",
               "sparse_csr_gemv", "Row-sliced sparse accumulation with dynamic nonzeros",
               "fori_loop", {"r64k4": "64 rows 4 nnz", "r128k4": "128 rows 4 nnz", "r256k8": "256 rows 8 nnz"}, "r128k4",
               _csr_gemv_factory, 583_821, Provenance("public-inspired", "Saad (2003) Iterative Methods for Sparse Linear Systems", "Public Domain", "CSR segmented row matvec"),
               tier="full", related_group="sp_csr"),
    CorpusSpec("sp_triangular_solve", "sparse_linear_algebra", "Forward Substitution Sparse Lower Triangular Solver",
               "sparse_triangular_solve", "Causal recurrent dependency where row i reads solved unknowns x0..x_{i-1}",
               "fori_loop", {"n32": "n 32", "n64": "n 64", "n128": "n 128"}, "n64",
               _tri_solve_factory, 583_822, Provenance("public-inspired", "Golub & Van Loan (2013) Matrix Computations", "Public Domain", "Forward triangular substitution"),
               tier="full", related_group="sp_triangular"),
    CorpusSpec("sp_bcr_relaxation", "sparse_linear_algebra", "Block-Coordinate Gauss-Seidel Relaxation",
               "sparse_block_relaxation", "Block-coordinate sweep solving local linear block systems",
               "fori_loop", {"b8s4": "8 blocks size 4", "b16s4": "16 blocks size 4"}, "b8s4",
               _bcr_factory, 583_823, Provenance("public-inspired", "Hackbusch (1994) Iterative Solution", "Public Domain", "Block coordinate relaxation"),
               tier="full", related_group="sp_block_rel"),
    CorpusSpec("sp_residual_propagation", "sparse_linear_algebra", "Sparse Graph Residual Mass Diffusion",
               "sparse_residual_diffusion", "Mass diffusion pushing mass from source to destination while subtracting from source",
               "fori_loop", {"n64": "n 64", "n128": "n 128", "n256": "n 256"}, "n128",
               _res_prop_factory, 583_824, Provenance("public-inspired", "Page et al. (1999) Stanford InfoLab", "Public Domain", "Mass residual propagation on graph"),
               tier="smoke", related_group="sp_diffusion"),
    CorpusSpec("sp_ellpack_gemv", "sparse_linear_algebra", "ELLPACK Formatted Fixed-Width Sparse Matvec",
               "sparse_ellpack_gemv", "Fixed-width 2D column gathers avoiding pointer indirection",
               "fori_loop", {"r64w8": "64 rows width 8", "r128w8": "128 rows width 8", "r256w16": "256 rows width 16"}, "r128w8",
               _ellpack_factory, 583_825, Provenance("public-inspired", "Grimes et al. (1979) ELLPACK Format", "Public Domain", "ELLPACK sparse matrix vector multiply"),
               tier="full", related_group="sp_ellpack_eval"),
    CorpusSpec("sp_ilu0_precond_apply", "sparse_linear_algebra", "Incomplete LU Zero-Fill Preconditioner Solve",
               "sparse_ilu_preconditioner", "Coupled forward and backward triangular sweep within fixed iteration loop",
               "fori_loop", {"n32": "n 32", "n64": "n 64", "n128": "n 128"}, "n64",
               _ilu0_factory, 583_826, Provenance("public-inspired", "Saad (2003) Iterative Methods for Sparse Linear Systems", "Public Domain", "ILU(0) preconditioner application"),
               tier="full", related_group="sp_ilu0_eval"),
    CorpusSpec("sp_jacobi_sparse_row", "sparse_linear_algebra", "Sparse Jacobi Iterative Relaxation Step",
               "sparse_jacobi_relaxation", "Unstructured irregular sparse graph relaxation step",
               "fori_loop", {"n64k4": "64 nodes deg 4", "n128k4": "128 nodes deg 4", "n256k8": "256 nodes deg 8"}, "n128k4",
               _jacobi_sparse_factory, 583_827, Provenance("public-inspired", "Demmel (1997) Applied Numerical Linear Algebra", "Public Domain", "Sparse graph Jacobi relaxation"),
               tier="full", related_group="sp_jacobi"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
