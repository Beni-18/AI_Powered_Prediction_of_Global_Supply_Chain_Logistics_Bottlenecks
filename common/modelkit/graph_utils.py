"""ROLE: ML ENGINEER  |  COMMON MODELKIT
Graph topology and adjacency matrix utilities for Stage 02 DL Spatial GNN.

Owner: Role 1 - ML Engineer
Reads: contracts/network_nodes.csv, contracts/network_edges.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from common.paths import EDGES_CSV, NODES_CSV


def load_canonical_nodes() -> list[str]:
    """Load the 27 registered node IDs in canonical ordering from contracts/network_nodes.csv."""
    df = pd.read_csv(NODES_CSV)
    return df["node_id"].tolist()


def load_adjacency(
    symmetric: bool = True,
    self_loops: bool = True,
    normalise: bool = True,
) -> np.ndarray:
    r"""Construct the normalized adjacency matrix \hat{A} in canonical node order.

    Normalization follows Kipf & Welling GCN:
        \hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}
        where \tilde{A} = A + I_N
    """
    node_ids = load_canonical_nodes()
    node_idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    adj = np.zeros((n, n), dtype=np.float32)

    # Read the frozen edge registry
    edges_df = pd.read_csv(EDGES_CSV)
    for _, row in edges_df.iterrows():
        u = row["src"]
        v = row["dst"]
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            adj[i, j] = 1.0
            if symmetric or row.get("bidirectional", False):
                adj[j, i] = 1.0

    # Add self loops (identity matrix)
    if self_loops:
        adj = adj + np.eye(n, dtype=np.float32)

    # Symmetric degree normalization: D^{-1/2} * A * D^{-1/2}
    if normalise:
        degrees = np.sum(adj, axis=1)
        degrees_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
        degrees_inv_sqrt[degrees == 0] = 0.0
        d_mat = np.diag(degrees_inv_sqrt)
        adj = d_mat @ adj @ d_mat

    return adj


def get_shuffled_adjacency(seed: int = 42) -> np.ndarray:
    """Generate an edge-permuted adjacency matrix for graph ablation experiments."""
    adj = load_adjacency(symmetric=True, self_loops=False, normalise=False)
    rng = np.random.default_rng(seed)
    n = adj.shape[0]

    # Permute node indices
    perm = rng.permutation(n)
    shuffled = adj[perm, :][:, perm]

    # Add self loops and normalize
    shuffled = shuffled + np.eye(n, dtype=np.float32)
    degrees = np.sum(shuffled, axis=1)
    d_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
    d_mat = np.diag(d_inv_sqrt)
    return d_mat @ shuffled @ d_mat
