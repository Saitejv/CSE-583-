"""Reviewed OpenCode drafts from the GPT-OSS 120B provider."""

import jax.numpy as jnp
import numpy as np
from jax import lax


def branch_tags(variant, trip, factor):
    """Read and replace tags in a two-way table indexed by carried history."""
    sets = 32 if variant == "small" else 256
    rng = np.random.default_rng(20231118)
    pcs = rng.integers(0, 4 * sets, size=trip, dtype=np.int32)
    init_tags = np.full((sets, 2), -1, dtype=np.int32)

    def run(instructions, tags):
        def body(i, carry):
            table, history, hits = carry
            pc = instructions[i]
            row = (pc ^ history) & (sets - 1)
            hit = jnp.any(table[row] == pc)
            way = history & 1
            table = lax.cond(hit, lambda t: t, lambda t: t.at[row, way].set(pc), table)
            history = ((history << 1) ^ pc ^ hit.astype(jnp.int32)) & 255
            return table, history, hits + hit.astype(jnp.int32)

        return lax.fori_loop(0, trip, body, (tags, jnp.int32(0), jnp.int32(0)), unroll=factor)

    return run, (pcs, init_tags)


def discounted_softmin(variant, trip, factor):
    """Reduce a bank of penalized path costs and feed the result into the next step."""
    lanes = 32 if variant == "small" else 256
    rng = np.random.default_rng(20231119)
    costs = rng.uniform(.1, 5, size=(trip, lanes)).astype(np.float32)
    penalties = rng.uniform(.1, .8, size=lanes).astype(np.float32)

    def run(stream, penalty):
        def body(i, estimate):
            adjusted = stream[i] + penalty * jnp.tanh(estimate) ** 2
            smooth_min = -jnp.log(jnp.mean(jnp.exp(-adjusted)))
            return .875 * estimate + .125 * smooth_min

        return lax.fori_loop(0, trip, body, jnp.float32(0), unroll=factor)

    return run, (costs, penalties)
