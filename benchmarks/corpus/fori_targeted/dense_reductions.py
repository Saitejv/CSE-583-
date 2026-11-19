"""Domain 4: Dense arithmetic and reductions workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats
from benchmarks.corpus.spec import CORPUS_TRIP_COUNTS, CorpusSpec, Provenance
from jax_unroll.auto_loop import precompute_fori_loop


# 1. dense_matvec_step (train): Matrix-vector row dot-product reduction
def _dense_matvec_factory(variant: str, trip: int, factor: int):
    dim = 64 if variant == "small" else 128
    rng = np.random.default_rng(583_401)
    matrix = floats(rng, dim, dim, scale=0.5)
    base_x = floats(rng, 1024, dim, scale=0.5)
    x_stream = base_x[:trip]
    init_y = jnp.zeros((dim,), dtype=jnp.float32)

    def fn(y: jax.Array, a: jax.Array, xs: jax.Array):
        def body(i: int, target: jax.Array):
            x = xs[i]
            # Matrix-vector multiply and accumulate
            row = (i * 7) % dim
            val = jnp.dot(a[row], x)
            return target.at[row].add(val)

        return lax.fori_loop(0, trip, body, y, unroll=factor)

    return built(fn, init_y, matrix, x_stream)


# 2. dense_gram_accum (train): Outer-product Gram matrix accumulation G <- G + x x^T
def _dense_gram_factory(variant: str, trip: int, factor: int):
    dim = 16 if variant == "small" else 32
    rng = np.random.default_rng(583_402)
    init_gram = jnp.zeros((dim, dim), dtype=jnp.float32)
    base_x = floats(rng, 1024, dim, scale=0.2)
    x_stream = base_x[:trip]

    def fn(gram: jax.Array, xs: jax.Array):
        def body(i: int, g: jax.Array):
            x = xs[i]
            # Outer product accumulation with slight damping
            outer = jnp.outer(x, x)
            return g * 0.999 + outer

        return lax.fori_loop(0, trip, body, gram, unroll=factor)

    return built(fn, init_gram, x_stream)


# 3. dense_outer_product_cov (train): Streaming sample covariance update with decay
def _outer_product_cov_factory(variant: str, trip: int, factor: int):
    dim = 12 if variant == "small" else 28
    rng = np.random.default_rng(583_403)
    init_cov = jnp.eye(dim, dtype=jnp.float32) * 0.1
    init_mean = jnp.zeros((dim,), dtype=jnp.float32)
    base_x = floats(rng, 1024, dim, scale=0.5)
    x_stream = base_x[:trip]
    decay = jnp.float32(0.98)

    def fn(cov: jax.Array, mean: jax.Array, xs: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            s, m = carry
            x = xs[i]
            # Online Welford / EMA covariance update
            new_m = decay * m + (1.0 - decay) * x
            diff = x - new_m
            new_s = decay * s + (1.0 - decay) * jnp.outer(diff, diff)
            return (new_s, new_m)

        return lax.fori_loop(0, trip, body, (cov, mean), unroll=factor)

    return built(fn, init_cov, init_mean, x_stream)


# 4. dense_givens_qr_step (train): Givens orthogonal plane rotation applied to matrix columns
def _givens_qr_factory(variant: str, trip: int, factor: int):
    cols = 16 if variant == "small" else 48
    rng = np.random.default_rng(583_404)
    # 2 rows of columns: row_a and row_b of length cols
    init_rows = floats(rng, 2, cols, scale=1.0)
    # Stream of angles theta
    base_theta = floats(rng, 1024, scale=np.pi * 0.25)
    thetas = base_theta[:trip]

    def fn(rows: jax.Array, angles: jax.Array):
        def body(i: int, r: jax.Array):
            theta = angles[i]
            c = jnp.cos(theta)
            s = jnp.sin(theta)
            # Plane rotation: [new_r0; new_r1] = [c -s; s c] * [r0; r1]
            r0 = r[0]
            r1 = r[1]
            new_r0 = c * r0 - s * r1
            new_r1 = s * r0 + c * r1
            return jnp.stack([new_r0, new_r1], axis=0)

        return lax.fori_loop(0, trip, body, rows, unroll=factor)

    return built(fn, init_rows, thetas)


# 5. eval_dense_lanczos_tridiag (eval): Lanczos 3-term vector orthogonalization
def _lanczos_tridiag_factory(variant: str, trip: int, factor: int):
    dim = 32 if variant == "small" else 64
    rng = np.random.default_rng(583_405)
    # Symmetric matrix A
    mat = floats(rng, dim, dim, scale=0.5)
    sym_a = 0.5 * (mat + mat.T) + jnp.eye(dim, dtype=jnp.float32) * 2.0
    
    init_v = floats(rng, dim, scale=0.5)
    init_v = init_v / jnp.linalg.norm(init_v)
    init_prev = jnp.zeros((dim,), dtype=jnp.float32)
    init_beta = jnp.float32(0.0)

    def fn(v_curr: jax.Array, v_prev: jax.Array, beta_init: jax.Array, a: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array, jax.Array]):
            v, prev, beta = carry
            # w = A v - beta * prev
            w = jnp.dot(a, v) - beta * prev
            alpha = jnp.dot(w, v)
            w = w - alpha * v
            new_beta = jnp.linalg.norm(w) + 1e-6
            new_v = w / new_beta
            return (new_v, v, new_beta)

        return lax.fori_loop(0, trip, body, (v_curr, v_prev, beta_init), unroll=factor)

    return built(fn, init_v, init_prev, init_beta, sym_a)


# 6. eval_dense_horner_matrix_poly (eval): Horner's nested scheme for matrix polynomial
def _horner_matrix_poly_factory(variant: str, trip: int, factor: int, *, blocked=False):
    dim = 16 if variant == "small" else 32
    rng = np.random.default_rng(583_406)
    mat = floats(rng, dim, dim, scale=0.2)
    # Ensure spectral radius < 1 for numerical stability
    mat = mat / float(dim)
    init_p = jnp.zeros((dim, dim), dtype=jnp.float32)
    
    base_coeffs = floats(rng, 1024, scale=1.0)
    coeffs = base_coeffs[:trip]

    def fn(p: jax.Array, a: jax.Array, c_stream: jax.Array):
        def body(i: int, accum: jax.Array):
            c = c_stream[i]
            # P <- P * A + c * I
            return jnp.dot(accum, a) + c * jnp.eye(dim, dtype=jnp.float32)

        width = min(factor, trip)
        if not blocked or width == 1:
            return lax.fori_loop(0, trip, body, p, unroll=factor)

        def run_native(initial):
            return lax.fori_loop(0, trip, body, initial)

        if c_stream.shape != (trip,) or a.dtype != jnp.float32 or p.dtype != jnp.float32:
            return run_native(p)

        blocks, tail = divmod(trip, width)

        def run_blocked(initial):
            # Expand a block of Horner steps into one multiplication by A**width
            # plus a linear combination of the lower powers for that block.
            def next_power(power, _):
                return power @ a, power

            block_power, lower_powers = lax.scan(next_power, a, None, length=width - 1)
            identity = jnp.eye(dim, dtype=jnp.float32)
            powers = jnp.concatenate((identity[None], lower_powers)).reshape(width, dim * dim)
            coeffs = c_stream[:blocks * width].reshape(blocks, width)[:, ::-1]
            updates = (coeffs @ powers).reshape(blocks, dim, dim)

            def advance(accum, update):
                return accum @ block_power + update, None

            result = lax.scan(advance, initial, updates)[0]
            if tail:
                tail_power = powers[tail].reshape(dim, dim)
                tail_update = (c_stream[-tail:][::-1] @ powers[:tail]).reshape(dim, dim)
                result = result @ tail_power + tail_update
            return result

        # The blocked form reassociates floating-point operations. Use it for
        # the contractive matrices this workload is designed around.
        contractive = jnp.max(jnp.sum(jnp.abs(a), axis=1)) < 1
        return lax.cond(contractive, run_blocked, run_native, p)

    return built(fn, init_p, mat, coeffs)


# 7. eval_dense_attention_block_step (eval): Scaled dot-product attention score and projection
def _attention_block_step_factory(variant: str, trip: int, factor: int, *, precompute=False):
    head_dim = 16 if variant == "small" else 32
    seq_block = 16 if variant == "small" else 32
    rng = np.random.default_rng(583_407)
    
    # Static Keys and Values blocks: (seq_block, head_dim)
    keys = floats(rng, seq_block, head_dim, scale=0.3)
    values = floats(rng, seq_block, head_dim, scale=0.3)
    scale = jnp.float32(1.0 / np.sqrt(head_dim))
    
    base_queries = floats(rng, 1024, head_dim, scale=0.3)
    queries = base_queries[:trip]
    init_context = jnp.zeros((head_dim,), dtype=jnp.float32)

    def fn(context: jax.Array, q_stream: jax.Array, k: jax.Array, v: jax.Array):
        def project(i: int):
            q = q_stream[i]
            # Attention scores: dot(k, q) * scale
            scores = jnp.dot(k, q) * scale
            # Softmax over seq_block
            probs = jax.nn.softmax(scores)
            # Value projection
            return jnp.dot(probs, v)

        def accumulate(ctx, out_vec):
            return ctx * 0.95 + out_vec

        if precompute:
            return precompute_fori_loop(0, trip, project, accumulate, context, unroll=factor)
        return lax.fori_loop(0, trip, lambda i, ctx: accumulate(ctx, project(i)), context, unroll=factor)

    return built(fn, init_context, queries, keys, values)


# 8. eval_dense_fft_radix2_butterfly (eval): Cooley-Tukey radix-2 FFT complex butterfly step
def _fft_radix2_butterfly_factory(variant: str, trip: int, factor: int):
    pairs = 16 if variant == "small" else 64
    rng = np.random.default_rng(583_408)
    
    # State: 2 complex vectors (upper and lower halves) shape (pairs, 2) [real, imag]
    init_upper = floats(rng, pairs, 2, scale=0.5)
    init_lower = floats(rng, pairs, 2, scale=0.5)
    
    # Twiddle factors W_N^k: shape (trip, pairs, 2)
    angles = np.linspace(0, 2.0 * np.pi, pairs, endpoint=False, dtype=np.float32)
    base_twiddle = np.zeros((1024, pairs, 2), dtype=np.float32)
    for t in range(1024):
        shift = t * 0.01
        base_twiddle[t, :, 0] = np.cos(angles + shift)
        base_twiddle[t, :, 1] = -np.sin(angles + shift)
    twiddles = jnp.asarray(base_twiddle[:trip])

    def fn(up_init: jax.Array, lo_init: jax.Array, twiddle_stream: jax.Array):
        def body(i: int, carry: tuple[jax.Array, jax.Array]):
            u, lo = carry
            w = twiddle_stream[i]
            # Complex multiplication: lo * w
            # (lr + i li) * (wr + i wi) = (lr*wr - li*wi) + i (lr*wi + li*wr)
            lr, li = lo[:, 0], lo[:, 1]
            wr, wi = w[:, 0], w[:, 1]
            prod_r = lr * wr - li * wi
            prod_i = lr * wi + li * wr
            prod = jnp.stack([prod_r, prod_i], axis=1)
            # Butterfly: upper = u + prod; lower = u - prod
            new_u = (u + prod) * 0.7071
            new_lo = (u - prod) * 0.7071
            return (new_u, new_lo)

        return lax.fori_loop(0, trip, body, (up_init, lo_init), unroll=factor)

    return built(fn, init_upper, init_lower, twiddles)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec(
        name="dense_matvec_step",
        family="dense_reductions",
        description="Matrix-vector multiplication row dot-product reduction",
        mechanism="dense_matrix_vector_row_reduction",
        structural_difference="Dense row vector inner product reduction and accumulator update",
        loop_kind="fori_loop",
        shape_variants={
            "small": "dim=64 (matrix 64x64 = 16 KB)",
            "large": "dim=128 (matrix 128x128 = 64 KB)",
        },
        primary_variant="small",
        factory=_dense_matvec_factory,
        seed=583_401,
        provenance=Provenance(
            kind="original",
            source="Adapted from conv_dense_matvec historical lineage",
            license="MIT",
            adaptation="Expanded to dual matrix dimensions (64, 128) with deterministic input stream",
        ),
        related_group="dense_matvec",
    ),
    CorpusSpec(
        name="dense_gram_accum",
        family="dense_reductions",
        description="Outer-product Gram matrix accumulation G <- G + x x^T",
        mechanism="dense_outer_product_gram_matrix",
        structural_difference="Quadratic outer product tensor expansion and matrix accumulation",
        loop_kind="fori_loop",
        shape_variants={
            "small": "dim=16 (matrix 16x16 = 1 KB)",
            "large": "dim=32 (matrix 32x32 = 4 KB)",
        },
        primary_variant="small",
        factory=_dense_gram_factory,
        seed=583_402,
        provenance=Provenance(
            kind="original",
            source="Adapted from tiled_gram_accumulate historical lineage",
            license="MIT",
            adaptation="Self-contained outer product Gram accumulation with continuous damping",
        ),
        related_group="dense_gram_accum",
    ),
    CorpusSpec(
        name="dense_outer_product_cov",
        family="dense_reductions",
        description="Streaming sample covariance matrix update with exponential decay",
        mechanism="streaming_covariance_matrix_update",
        structural_difference="Coupled vector mean tracking and rank-1 outer product covariance update",
        loop_kind="fori_loop",
        shape_variants={
            "small": "dim=12 (cov 12x12 = 576 B)",
            "large": "dim=28 (cov 28x28 = 3136 B)",
        },
        primary_variant="small",
        factory=_outer_product_cov_factory,
        seed=583_403,
        provenance=Provenance(
            kind="original",
            source="Online multivariate statistics and adaptive signal processing",
            license="MIT",
            adaptation="Welford-style exponential moving average covariance recurrence",
        ),
        related_group="dense_outer_product_cov",
    ),
    CorpusSpec(
        name="dense_givens_qr_step",
        family="dense_reductions",
        description="Givens orthogonal plane rotation applied to matrix columns",
        mechanism="givens_plane_rotation_qr",
        structural_difference="2x2 orthogonal trigonometric transformation applied across row vectors",
        loop_kind="fori_loop",
        shape_variants={
            "small": "cols=16 (1 KB)",
            "large": "cols=48 (3 KB)",
        },
        primary_variant="small",
        factory=_givens_qr_factory,
        seed=583_404,
        provenance=Provenance(
            kind="original",
            source="Givens, W. (1958). Computation of plane unitary rotations transforming a general matrix to triangular form",
            license="MIT",
            adaptation="Vectorized Givens plane rotation step over row pairs",
        ),
        related_group="dense_givens_qr",
    ),
    CorpusSpec(
        name="eval_dense_lanczos_tridiag",
        family="eval_dense_reductions",
        description="Lanczos three-term recurrence vector orthogonalization",
        mechanism="lanczos_three_term_orthogonalization",
        structural_difference="Matrix-vector multiplication, two dot products, and two-vector orthogonalization carry",
        loop_kind="fori_loop",
        shape_variants={
            "small": "dim=32 (matrix 32x32 = 4 KB)",
            "large": "dim=64 (matrix 64x64 = 16 KB)",
        },
        primary_variant="small",
        factory=_lanczos_tridiag_factory,
        seed=583_405,
        provenance=Provenance(
            kind="public-inspired",
            source="Lanczos, C. (1950). An iteration method for the solution of the eigenvalue problem. J. Res. Natl. Bur. Stand.",
            license="Public domain / academic",
            adaptation="Orthogonal Krylov subspace basis vector generation step",
        ),
        related_group="eval_lanczos_tridiag",
    ),
    CorpusSpec(
        name="eval_dense_horner_matrix_poly",
        family="eval_dense_reductions",
        description="Horner nested multiplication scheme for matrix polynomial evaluation",
        mechanism="horner_matrix_polynomial_evaluation",
        structural_difference="Dense matrix-matrix multiplication and diagonal identity addition per loop trip",
        loop_kind="fori_loop",
        shape_variants={
            "small": "matrix 16x16 (1 KB)",
            "large": "matrix 32x32 (4 KB)",
        },
        primary_variant="small",
        factory=_horner_matrix_poly_factory,
        seed=583_406,
        provenance=Provenance(
            kind="public-inspired",
            source="Higham, N. J. (2008). Functions of Matrices: Theory and Computation. SIAM",
            license="Academic reference",
            adaptation="Horner matrix power evaluation with stable contracted matrices",
        ),
        related_group="eval_horner_matrix_poly",
    ),
    CorpusSpec(
        name="eval_dense_attention_block_step",
        family="eval_dense_reductions",
        description="Scaled dot-product attention score reduction and value projection",
        mechanism="scaled_dot_product_attention_step",
        structural_difference="Dense inner products, vector softmax reduction, and linear value combination",
        loop_kind="fori_loop",
        shape_variants={
            "small": "head_dim=16, seq_block=16 (1 KB)",
            "large": "head_dim=32, seq_block=32 (4 KB)",
        },
        primary_variant="small",
        factory=_attention_block_step_factory,
        seed=583_407,
        provenance=Provenance(
            kind="public-inspired",
            source="Vaswani, A., et al. (2017). Attention is all you need. NeurIPS",
            license="Academic standard",
            adaptation="Single-step query-key attention reduction over block sequence memory",
        ),
        related_group="eval_attention_block_step",
    ),
    CorpusSpec(
        name="eval_dense_fft_radix2_butterfly",
        family="eval_dense_reductions",
        description="Cooley-Tukey radix-2 FFT complex butterfly stage transformation",
        mechanism="radix2_fft_complex_butterfly",
        structural_difference="Complex multiplication with twiddle factor and dual addition/subtraction butterfly",
        loop_kind="fori_loop",
        shape_variants={
            "small": "pairs=16 complex (128 B)",
            "large": "pairs=64 complex (512 B)",
        },
        primary_variant="small",
        factory=_fft_radix2_butterfly_factory,
        seed=583_408,
        provenance=Provenance(
            kind="public-inspired",
            source="Cooley, J. W., & Tukey, J. W. (1965). An algorithm for machine calculation of complex Fourier series. Math. Comp.",
            license="Public domain / academic",
            adaptation="Vectorized radix-2 complex butterfly transformation with streaming twiddle factors",
        ),
        related_group="eval_fft_radix2_butterfly",
    ),
)
