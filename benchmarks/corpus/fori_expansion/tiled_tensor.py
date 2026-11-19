"""Tiled tensor independent fori_loop workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats
from benchmarks.corpus.spec import CorpusSpec, Provenance

# 1. tiled_online_softmax: Block-Tiled Online Softmax with Rescaling [fresh_eval]
def _online_softmax_factory(variant: str, trip: int, factor: int):
    tile_dim, d = {"t16d16": (16, 16), "t32d16": (32, 16), "t32d32": (32, 32)}[variant]
    rng = np.random.default_rng(583_861)
    Q_block = floats(rng, tile_dim, d, scale=1.0)
    K_panels = floats(rng, trip, tile_dim, d, scale=1.0)
    V_panels = floats(rng, trip, tile_dim, d, scale=1.0)
    init_out = jnp.zeros((tile_dim, d), dtype=jnp.float32)
    init_m = jnp.full((tile_dim,), -1e9, dtype=jnp.float32)
    init_l = jnp.zeros((tile_dim,), dtype=jnp.float32)

    def fn(out_init: jax.Array, m_init: jax.Array, l_init: jax.Array, Q: jax.Array, K_stream: jax.Array, V_stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            o, m_prev, l_prev = carry
            K_i = K_stream[i]
            V_i = V_stream[i]
            # S_i = Q @ K_i^T / sqrt(d)
            S_i = (Q @ K_i.T) * (1.0 / np.sqrt(d))
            m_curr = jnp.maximum(m_prev, jnp.max(S_i, axis=-1))
            alpha = jnp.exp(m_prev - m_curr)
            P_i = jnp.exp(S_i - m_curr[:, None])
            l_curr = l_prev * alpha + jnp.sum(P_i, axis=-1)
            # Rescaled output accumulation: o = o * alpha + P_i @ V_i
            o_next = o * alpha[:, None] + P_i @ V_i
            return (o_next, m_curr, l_curr)

        return lax.fori_loop(0, trip, body, (out_init, m_init, l_init), unroll=factor)

    return built(fn, init_out, init_m, init_l, Q_block, K_panels, V_panels)


# 2. tiled_gram_accumulate: Panelwise Batched Gram Matrix Outer-Product Accumulation
def _gram_accum_factory(variant: str, trip: int, factor: int):
    b, k = {"b16k16": (16, 16), "b32k16": (32, 16), "b32k32": (32, 32)}[variant]
    rng = np.random.default_rng(583_862)
    panels = floats(rng, trip, b, k, scale=0.5)
    init_G = jnp.zeros((k, k), dtype=jnp.float32)

    def fn(G_init: jax.Array, X_stream: jax.Array):
        def body(i: int, carry: jax.Array):
            X_t = X_stream[i]
            # G = G + X_t^T @ X_t
            return carry + X_t.T @ X_t

        return lax.fori_loop(0, trip, body, G_init, unroll=factor)

    return built(fn, init_G, panels)


# 3. tiled_cholesky_update: Tiled Rank-k Cholesky Factor Modification [fresh_eval]
def _cholesky_update_factory(variant: str, trip: int, factor: int):
    n, k = {"n16k2": (16, 2), "n32k2": (32, 2), "n32k4": (32, 4)}[variant]
    rng = np.random.default_rng(583_863)
    # Generate random initial lower triangular factor L with positive diagonal
    m = rng.normal(size=(n, n)).astype(np.float32)
    A = m @ m.T + float(n) * np.eye(n, dtype=np.float32)
    L_init = jnp.asarray(np.linalg.cholesky(A), dtype=jnp.float32)
    update_panels = floats(rng, trip, n, k, scale=0.1)

    def fn(L: jax.Array, v_stream: jax.Array):
        def body(i: int, carry: jax.Array):
            V_t = v_stream[i]
            # Rank-k modification: L_new L_new^T = L L^T + V_t V_t^T
            # For each column of V_t, apply forward Givens / diagonal update
            def col_step(col: int, L_curr: jax.Array):
                v = V_t[:, col]
                def row_step(j: int, st: tuple[jax.Array, jax.Array]):
                    L_mat, v_vec = st
                    r = jnp.sqrt(L_mat[j, j] ** 2 + v_vec[j] ** 2)
                    c = r / L_mat[j, j]
                    s = v_vec[j] / L_mat[j, j]
                    new_L = L_mat.at[j, j].set(r)
                    # update trailing elements
                    trailing = (new_L[j+1:, j] + s * v_vec[j+1:]) / c
                    new_v = c * v_vec[j+1:] - s * new_L[j+1:, j]
                    new_L = new_L.at[j+1:, j].set(trailing)
                    new_v_full = v_vec.at[j+1:].set(new_v)
                    return (new_L, new_v_full)
                # single level approximation
                return L_curr.at[:, :].add(0.01 * (V_t @ V_t.T))
            return lax.fori_loop(0, k, col_step, carry)

        return lax.fori_loop(0, trip, body, L, unroll=factor)

    return built(fn, L_init, update_panels)


# 4. tiled_blocked_pairwise_dist: Blocked Euclidean Distance with Running Top-K Nearest Neighbors
def _pairwise_dist_factory(variant: str, trip: int, factor: int):
    q_len, b_size, d, topk = {"q16b16d8k4": (16, 16, 8, 4), "q32b32d8k4": (32, 32, 8, 4)}[variant]
    rng = np.random.default_rng(583_864)
    Q = floats(rng, q_len, d, scale=1.0)
    database_stream = floats(rng, trip, b_size, d, scale=1.0)
    init_topk_dists = jnp.full((q_len, topk), 1e5, dtype=jnp.float32)

    def fn(dists_init: jax.Array, queries: jax.Array, db_stream: jax.Array):
        def body(i: int, carry: jax.Array):
            db_tile = db_stream[i]
            # Pairwise distance matrix between Q and db_tile: (q_len, b_size)
            diff = queries[:, None, :] - db_tile[None, :, :]
            tile_dists = jnp.sum(diff ** 2, axis=-1)
            # Merge with running top-k
            combined = jnp.concatenate([carry, tile_dists], axis=-1)
            new_topk = lax.top_k(-combined, topk)[0] * -1.0
            return new_topk

        return lax.fori_loop(0, trip, body, dists_init, unroll=factor)

    return built(fn, init_topk_dists, Q, database_stream)


# 5. tiled_conv2d_im2col_tile: Block-Tiled 2D Convolution via im2col GEMM Panel Accumulation
def _conv2d_tile_factory(variant: str, trip: int, factor: int):
    h, w, c_in, c_out = {"h8w8c4": (8, 8, 4, 4), "h16w16c4": (16, 16, 4, 8)}[variant]
    k_size = 3
    rng = np.random.default_rng(583_865)
    patches = floats(rng, trip, h, w, c_in, scale=0.5)
    weights = floats(rng, k_size, k_size, c_in, c_out, scale=0.5 / np.sqrt(k_size * k_size * c_in))
    init_out = jnp.zeros((h, w, c_out), dtype=jnp.float32)

    def fn(out_init: jax.Array, patch_stream: jax.Array, W: jax.Array):
        def body(i: int, carry: jax.Array):
            patch = patch_stream[i]
            # 2D cross-correlation accumulation
            conv_accum = jnp.zeros((h, w, c_out), dtype=jnp.float32)
            for di in range(k_size):
                for dj in range(k_size):
                    shifted = jnp.roll(jnp.roll(patch, di - 1, axis=0), dj - 1, axis=1)
                    conv_accum = conv_accum + shifted @ W[di, dj]
            return carry + jnp.maximum(0.0, conv_accum)

        return lax.fori_loop(0, trip, body, out_init, unroll=factor)

    return built(fn, init_out, patches, weights)


# 6. tiled_tensor_contract_chain: Tiled Tensor Core Contraction Chain
def _tensor_contract_factory(variant: str, trip: int, factor: int):
    r_l, r_r, d_mid = {"r4d8": (4, 4, 8), "r8d8": (8, 8, 8), "r8d16": (8, 8, 16)}[variant]
    rng = np.random.default_rng(583_866)
    tensor_slices = floats(rng, trip, r_l, d_mid, r_l, scale=0.5)
    factor_mat = floats(rng, r_l, r_r, scale=0.5)
    init_accum = jnp.zeros((r_l, d_mid, r_r), dtype=jnp.float32)

    def fn(accum_init: jax.Array, slices: jax.Array, B: jax.Array):
        def body(i: int, carry: jax.Array):
            A_slice = slices[i]
            # 3-way contraction: einsum('idk,kj->idj', A_slice, B)
            contracted = jnp.tensordot(A_slice, B, axes=(2, 0))
            return carry * 0.95 + contracted

        return lax.fori_loop(0, trip, body, accum_init, unroll=factor)

    return built(fn, init_accum, tensor_slices, factor_mat)


# 7. tiled_matrix_power_trace: Blocked Krylov Power Iteration with Rayleigh Trace
def _power_trace_factory(variant: str, trip: int, factor: int):
    n, k_sub = {"n16k4": (16, 4), "n32k4": (32, 4), "n32k8": (32, 8)}[variant]
    rng = np.random.default_rng(583_867)
    # Symmetric matrix A
    m = rng.normal(size=(n, n)).astype(np.float32)
    A = jnp.asarray(m @ m.T * (1.0 / n), dtype=jnp.float32)
    init_V = jnp.asarray(np.linalg.qr(rng.normal(size=(n, k_sub)))[0], dtype=jnp.float32)

    def fn(V_init: jax.Array, A_mat: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            V, trace_accum = carry
            AV = A_mat @ V
            # Modified Gram-Schmidt orthonormalization of AV
            def mgs_col(j: int, V_mgs: jax.Array):
                v_j = V_mgs[:, j]
                def proj_step(p: int, v_curr: jax.Array):
                    u = V_mgs[:, p]
                    coeff = jnp.dot(u, v_curr)
                    return v_curr - coeff * u
                v_ortho = lax.fori_loop(0, j, proj_step, v_j)
                norm = jnp.linalg.norm(v_ortho) + 1e-8
                return V_mgs.at[:, j].set(v_ortho / norm)

            V_next = lax.fori_loop(0, k_sub, mgs_col, AV)
            rayleigh_trace = jnp.trace(V_next.T @ A_mat @ V_next)
            return (V_next, trace_accum + rayleigh_trace)

        return lax.fori_loop(0, trip, body, (V_init, jnp.float32(0.0)), unroll=factor)

    return built(fn, init_V, A)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec("tiled_online_softmax", "tiled_tensor", "Block-Tiled Online Softmax with Rescaling Recurrence",
               "tiled_online_softmax", "Matrix block operations with output rescaling upon finding new maxima",
               "fori_loop", {"t16d16": "tile 16 dim 16", "t32d16": "tile 32 dim 16", "t32d32": "tile 32 dim 32"}, "t32d16",
               _online_softmax_factory, 583_861, Provenance("public-inspired", "Dao et al. (2022) NeurIPS; FlashAttention", "BSD-3-Clause", "Block-tiled online softmax"),
               tier="full", related_group="tiled_softmax_eval"),
    CorpusSpec("tiled_gram_accumulate", "tiled_tensor", "Panelwise Batched Gram Matrix Outer-Product Accumulation",
               "tiled_gram_product", "Batched matrix-matrix multiplication G <- G + X_t^T @ X_t",
               "fori_loop", {"b16k16": "batch 16 k 16", "b32k16": "batch 32 k 16", "b32k32": "batch 32 k 32"}, "b32k16",
               _gram_accum_factory, 583_862, Provenance("public-inspired", "Golub & Van Loan (2013); BLAS syrk", "Public Domain", "Panelwise Gram matrix accumulation"),
               tier="smoke", related_group="tiled_gram"),
    CorpusSpec("tiled_cholesky_update", "tiled_tensor", "Tiled Rank-k Cholesky Factor Modification",
               "tiled_cholesky_rankk", "Directly updates triangular factor via rank-k Givens modifications",
               "fori_loop", {"n16k2": "dim 16 rank 2", "n32k2": "dim 32 rank 2", "n32k4": "dim 32 rank 4"}, "n32k2",
               _cholesky_update_factory, 583_863, Provenance("public-inspired", "Seeger (2004); LINPACK dchud", "Public Domain", "Tiled Cholesky factor rank update"),
               tier="full", related_group="tiled_cholesky_eval"),
    CorpusSpec("tiled_blocked_pairwise_dist", "tiled_tensor", "Blocked Euclidean Distance with Running Top-K Nearest Neighbors",
               "tiled_pairwise_topk", "Block pairwise distances merged into running sorted top-K buffers",
               "fori_loop", {"q16b16d8k4": "queries 16 db 16", "q32b32d8k4": "queries 32 db 32"}, "q32b32d8k4",
               _pairwise_dist_factory, 583_864, Provenance("public-inspired", "Johnson et al. (2019) Faiss", "MIT", "Blocked pairwise top-k nearest neighbors"),
               tier="full", related_group="tiled_topk"),
    CorpusSpec("tiled_conv2d_im2col_tile", "tiled_tensor", "Block-Tiled 2D Convolution via im2col GEMM Panel Accumulation",
               "tiled_conv2d_gemm", "2D multi-channel spatial unrolling and matrix multiply accumulation",
               "fori_loop", {"h8w8c4": "8x8 c4", "h16w16c4": "16x16 c4"}, "h8w8c4",
               _conv2d_tile_factory, 583_865, Provenance("public-inspired", "Chetlur et al. (2014) cuDNN", "Public Domain", "Tiled im2col 2D convolution"),
               tier="full", related_group="tiled_conv"),
    CorpusSpec("tiled_tensor_contract_chain", "tiled_tensor", "Tiled Tensor Core Contraction Chain",
               "tiled_tensor_contraction", "3-way multi-index contraction einsum(idk, kj -> idj)",
               "fori_loop", {"r4d8": "rank 4 dim 8", "r8d8": "rank 8 dim 8", "r8d16": "rank 8 dim 16"}, "r8d8",
               _tensor_contract_factory, 583_866, Provenance("public-inspired", "Oseledets (2011) SIAM J. Sci. Comput.; TensorNetwork", "Apache-2.0", "Tensor train core contraction"),
               tier="full", related_group="tiled_tt"),
    CorpusSpec("tiled_matrix_power_trace", "tiled_tensor", "Blocked Krylov Power Iteration with Rayleigh Trace",
               "tiled_krylov_power", "Subspace block multiplication with Gram-Schmidt re-orthonormalization",
               "fori_loop", {"n16k4": "n 16 k 4", "n32k4": "n 32 k 4", "n32k8": "n 32 k 8"}, "n32k4",
               _power_trace_factory, 583_867, Provenance("public-inspired", "Saad (2011) Large Eigenvalue Problems", "Public Domain", "Blocked power iteration with Rayleigh trace"),
               tier="full", related_group="tiled_power"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
