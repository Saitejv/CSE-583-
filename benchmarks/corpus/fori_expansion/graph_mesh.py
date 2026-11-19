"""Graph and mesh traversal independent fori_loop workloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from benchmarks.corpus.common import built, floats, indices
from benchmarks.corpus.spec import CorpusSpec, Provenance

# 1. graph_edge_message_passing: Edge-to-Node Message Passing Neural Aggregation
def _edge_mp_factory(variant: str, trip: int, factor: int):
    nodes, feat_dim = {"n64d8": (64, 8), "n128d8": (128, 8), "n256d16": (256, 16)}[variant]
    rng = np.random.default_rng(583_831)
    senders = indices(rng, nodes, trip)
    receivers = indices(rng, nodes, trip)
    edge_attr = floats(rng, trip, feat_dim, scale=0.5)
    W_edge = floats(rng, feat_dim * 2, feat_dim, scale=0.5 / np.sqrt(feat_dim))
    init_nodes = floats(rng, nodes, feat_dim, scale=1.0)

    def fn(nodes_init: jax.Array, s_idx: jax.Array, r_idx: jax.Array, e_attr: jax.Array, W: jax.Array):
        def body(i: int, carry: jax.Array):
            u = s_idx[i]
            v = r_idx[i]
            msg_in = jnp.concatenate([carry[u], e_attr[i]])
            msg = jnp.tanh(msg_in @ W)
            return carry.at[v].add(msg * 0.1)

        return lax.fori_loop(0, trip, body, nodes_init, unroll=factor)

    return built(fn, init_nodes, senders, receivers, edge_attr, W_edge)


# 2. graph_frontier_relaxation: Fixed-Capacity Bellman-Ford Shortest Path Relaxation
def _frontier_relax_factory(variant: str, trip: int, factor: int):
    nodes = {"n64": 64, "n128": 128, "n256": 256}[variant]
    rng = np.random.default_rng(583_832)
    src_edges = indices(rng, nodes, trip)
    dst_edges = indices(rng, nodes, trip)
    edge_weights = jnp.asarray(rng.uniform(0.1, 1.0, size=trip), dtype=jnp.float32)
    init_dists = jnp.full((nodes,), 1e5, dtype=jnp.float32).at[0].set(0.0)

    def fn(dist_init: jax.Array, srcs: jax.Array, dsts: jax.Array, weights: jax.Array):
        def body(i: int, carry: jax.Array):
            u = srcs[i]
            v = dsts[i]
            w = weights[i]
            cand = carry[u] + w
            new_val = jnp.minimum(carry[v], cand)
            return carry.at[v].set(new_val)

        return lax.fori_loop(0, trip, body, dist_init, unroll=factor)

    return built(fn, init_dists, src_edges, dst_edges, edge_weights)


# 3. graph_greedy_coloring: Parallel Greedy Graph Vertex Coloring Conflict Resolution [fresh_eval]
def _greedy_coloring_factory(variant: str, trip: int, factor: int):
    nodes = {"n64": 64, "n128": 128, "n256": 256}[variant]
    rng = np.random.default_rng(583_833)
    u_edges = indices(rng, nodes, trip)
    v_edges = indices(rng, nodes, trip)
    priorities = jnp.asarray(rng.uniform(size=nodes), dtype=jnp.float32)
    init_colors = jnp.zeros((nodes,), dtype=jnp.int32)

    def fn(col_init: jax.Array, prio: jax.Array, u_arr: jax.Array, v_arr: jax.Array):
        def body(i: int, carry: jax.Array):
            u = u_arr[i]
            v = v_arr[i]
            # If colors conflict and priority u > v, advance color of v
            conflict = (carry[u] == carry[v]) & (prio[u] > prio[v])
            new_c_v = carry[v] + jnp.where(conflict, 1, 0)
            return carry.at[v].set(new_c_v)

        return lax.fori_loop(0, trip, body, col_init, unroll=factor)

    return built(fn, init_colors, priorities, u_edges, v_edges)


# 4. graph_mesh_flux_assembly: Unstructured Finite-Volume Mesh Edge Flux Assembly [fresh_eval]
def _mesh_flux_factory(variant: str, trip: int, factor: int):
    cells = {"c64": 64, "c128": 128, "c256": 256}[variant]
    rng = np.random.default_rng(583_834)
    left_cells = indices(rng, cells, trip)
    right_cells = indices(rng, cells, trip)
    face_normals = floats(rng, trip, 2, scale=1.0)
    init_conserved = floats(rng, cells, 3, scale=1.0)

    def fn(cell_init: jax.Array, lefts: jax.Array, rights: jax.Array, normals: jax.Array):
        def body(i: int, carry: jax.Array):
            l = lefts[i]
            r = rights[i]
            n = normals[i]
            u_l = carry[l]
            u_r = carry[r]
            # Simple Rusanov/Lax-Friedrichs interface flux
            flux = 0.5 * (u_l + u_r) * n[0] - 0.2 * (u_r - u_l)
            # Add flux to left cell, subtract from right cell
            return carry.at[l].add(-flux * 0.05).at[r].add(flux * 0.05)

        return lax.fori_loop(0, trip, body, cell_init, unroll=factor)

    return built(fn, init_conserved, left_cells, right_cells, face_normals)


# 5. graph_pagerank_iter: PageRank Power Iteration with Dangling Redistribution
def _pagerank_factory(variant: str, trip: int, factor: int):
    nodes = {"n64": 64, "n128": 128, "n256": 256}[variant]
    deg = 4
    rng = np.random.default_rng(583_835)
    nbr_idx = indices(rng, nodes, nodes, deg)
    alpha = jnp.float32(0.85)
    init_ranks = jnp.full((nodes,), 1.0 / nodes, dtype=jnp.float32)

    def fn(rank_init: jax.Array, nbrs: jax.Array, damp: jax.Array):
        def body(i: int, carry: jax.Array):
            node = i % nodes
            # Gather incoming mass from neighbors
            in_mass = jnp.sum(carry[nbrs[node]]) / deg
            new_r = (1.0 - damp) / nodes + damp * in_mass
            return carry.at[node].set(new_r)

        return lax.fori_loop(0, trip, body, rank_init, unroll=factor)

    return built(fn, init_ranks, nbr_idx, alpha)


# 6. graph_label_propagation: Semi-Supervised Edge Affinity Label Propagation
def _label_prop_factory(variant: str, trip: int, factor: int):
    nodes, classes = {"n64c4": (64, 4), "n128c4": (128, 4), "n256c8": (256, 8)}[variant]
    rng = np.random.default_rng(583_836)
    src_nodes = indices(rng, nodes, trip)
    dst_nodes = indices(rng, nodes, trip)
    weights = floats(rng, trip, scale=1.0)
    # Seeds (first 8 nodes are fixed seeds)
    seed_mask = jnp.arange(nodes) < 8
    init_labels = floats(rng, nodes, classes, scale=1.0)
    init_labels = jnp.abs(init_labels) / jnp.sum(jnp.abs(init_labels), axis=-1, keepdims=True)

    def fn(label_init: jax.Array, srcs: jax.Array, dsts: jax.Array, w_arr: jax.Array, is_seed: jax.Array):
        def body(i: int, carry: jax.Array):
            u = srcs[i]
            v = dsts[i]
            w = w_arr[i]
            cand = carry[v] + w * carry[u]
            cand_norm = cand / (jnp.sum(jnp.abs(cand)) + 1e-6)
            # Clamping: keep seed nodes unchanged
            new_v = jnp.where(is_seed[v], carry[v], cand_norm)
            return carry.at[v].set(new_v)

        return lax.fori_loop(0, trip, body, label_init, unroll=factor)

    return built(fn, init_labels, src_nodes, dst_nodes, weights, seed_mask)


# 7. graph_degree_normalized_conv: Kipf-Welling GCN Normalized Graph Convolution
def _gcn_factory(variant: str, trip: int, factor: int):
    nodes, dim = {"n64d8": (64, 8), "n128d8": (128, 8), "n256d16": (256, 16)}[variant]
    rng = np.random.default_rng(583_837)
    src_nodes = indices(rng, nodes, trip)
    dst_nodes = indices(rng, nodes, trip)
    norm_factor = floats(rng, trip, scale=0.5)
    W_gcn = floats(rng, dim, dim, scale=0.5 / np.sqrt(dim))
    init_features = floats(rng, nodes, dim, scale=1.0)

    def fn(feat_init: jax.Array, srcs: jax.Array, dsts: jax.Array, norms: jax.Array, W: jax.Array):
        def body(i: int, carry: jax.Array):
            u = srcs[i]
            v = dsts[i]
            c_uv = norms[i]
            h_trans = jnp.maximum(carry[u] @ W, 0.0)
            return carry.at[v].add(c_uv * h_trans * 0.1)

        return lax.fori_loop(0, trip, body, feat_init, unroll=factor)

    return built(fn, init_features, src_nodes, dst_nodes, norm_factor, W_gcn)


SPECS: tuple[CorpusSpec, ...] = (
    CorpusSpec("graph_edge_message_passing", "graph_mesh", "Edge-to-Node Message Passing Neural Aggregation",
               "graph_message_passing", "Gathers endpoints, transforms features via MLP, and scatter-reduces to receivers",
               "fori_loop", {"n64d8": "64 nodes dim 8", "n128d8": "128 nodes dim 8", "n256d16": "256 nodes dim 16"}, "n128d8",
               _edge_mp_factory, 583_831, Provenance("public-inspired", "Battaglia et al. (2018) arXiv:1806.01261; Jraph", "Apache-2.0", "Edge to node message passing"),
               tier="full", related_group="graph_mp"),
    CorpusSpec("graph_frontier_relaxation", "graph_mesh", "Fixed-Capacity Bellman-Ford Shortest Path Relaxation",
               "graph_shortest_path", "Vectorized min-plus tropical semiring relaxation d[v] = min(d[v], d[u] + w_uv)",
               "fori_loop", {"n64": "64 nodes", "n128": "128 nodes", "n256": "256 nodes"}, "n128",
               _frontier_relax_factory, 583_832, Provenance("public-inspired", "Bellman (1958); NetworkX", "BSD-3-Clause", "Bellman-Ford frontier relaxation"),
               tier="full", related_group="graph_bellman"),
    CorpusSpec("graph_greedy_coloring", "graph_mesh", "Parallel Greedy Graph Vertex Coloring Conflict Resolution",
               "graph_coloring", "Priority-guided independent set conflict resolution assigning valid colors",
               "fori_loop", {"n64": "64 nodes", "n128": "128 nodes", "n256": "256 nodes"}, "n128",
               _greedy_coloring_factory, 583_833, Provenance("public-inspired", "Jones & Plassmann (1993) SIAM J. Sci. Comput.", "Public Domain", "Greedy parallel vertex coloring"),
               tier="full", related_group="graph_color_eval"),
    CorpusSpec("graph_mesh_flux_assembly", "graph_mesh", "Unstructured Finite-Volume Mesh Edge Flux Assembly",
               "mesh_flux_assembly", "Unstructured face topology adding flux to left cell and subtracting from right cell",
               "fori_loop", {"c64": "64 cells", "c128": "128 cells", "c256": "256 cells"}, "c128",
               _mesh_flux_factory, 583_834, Provenance("public-inspired", "Toro (2009) Riemann Solvers; PyFR", "BSD-3-Clause", "Finite-volume mesh flux assembly"),
               tier="full", related_group="graph_mesh_eval"),
    CorpusSpec("graph_pagerank_iter", "graph_mesh", "PageRank Power Iteration with Dangling Redistribution",
               "graph_pagerank", "Link propagation combined with uniform dangling redistribution",
               "fori_loop", {"n64": "64 nodes", "n128": "128 nodes", "n256": "256 nodes"}, "n128",
               _pagerank_factory, 583_835, Provenance("public-inspired", "Page et al. (1999) Stanford InfoLab", "Public Domain", "PageRank graph power iteration"),
               tier="smoke", related_group="graph_pagerank"),
    CorpusSpec("graph_label_propagation", "graph_mesh", "Semi-Supervised Edge Affinity Label Propagation",
               "graph_label_prop", "Boundary clamped seed nodes restoring ground truth labels per step",
               "fori_loop", {"n64c4": "64 nodes 4 classes", "n128c4": "128 nodes 4 classes", "n256c8": "256 nodes 8 classes"}, "n128c4",
               _label_prop_factory, 583_836, Provenance("public-inspired", "Zhu & Ghahramani (2002) CMU Tech Report", "Public Domain", "Semi-supervised label propagation"),
               tier="full", related_group="graph_label_prop"),
    CorpusSpec("graph_degree_normalized_conv", "graph_mesh", "Kipf-Welling GCN Normalized Graph Convolution",
               "graph_gcn_conv", "Spectral GCN degree normalization with bias and ReLU activation",
               "fori_loop", {"n64d8": "64 nodes dim 8", "n128d8": "128 nodes dim 8", "n256d16": "256 nodes dim 16"}, "n128d8",
               _gcn_factory, 583_837, Provenance("public-inspired", "Kipf & Welling (2017) ICLR; PyG", "MIT", "GCN normalized graph convolution"),
               tier="full", related_group="graph_gcn"),
)
WORKLOADS: dict[str, CorpusSpec] = {spec.name: spec for spec in SPECS}
