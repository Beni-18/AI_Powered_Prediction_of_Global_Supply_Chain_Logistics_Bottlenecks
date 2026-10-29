"""ROLE: DATA ENGINEER  |  SHARED HELPER
Turns the trade-lane list into an adjacency matrix for graph models (Stage 2 GNN).

READS : contracts/network_nodes.csv (node order), contracts/network_edges.csv (lanes)
WRITES: nothing - returns a NumPy array
OWNER : Role 3 - Data & Integration Engineer
"""

import numpy as np
import pandas as pd

from common.paths import EDGES_CSV, NODES_CSV


def load_adjacency(symmetric: bool = True, self_loops: bool = True, normalise: bool = True) -> np.ndarray:
    """Return an [n_nodes, n_nodes] matrix: entry [i, j] > 0 when a lane connects node i to node j."""
    nodes = pd.read_csv(NODES_CSV)["node_id"].tolist()
    edges = pd.read_csv(EDGES_CSV)
    index = {node: i for i, node in enumerate(nodes)}

    # Start with a directed 0/1 matrix: row = where the lane starts, column = where it ends.
    adj = np.zeros((len(nodes), len(nodes)))
    for edge in edges.itertuples():
        adj[index[edge.src], index[edge.dst]] = 1.0
        # Some lanes are marked as running both ways in the contract.
        if str(edge.bidirectional).lower() == "true":
            adj[index[edge.dst], index[edge.src]] = 1.0

    # Symmetric = ignore direction (congestion can be felt both up- and downstream).
    if symmetric:
        adj = np.maximum(adj, adj.T)

    # Self-loops let a node keep its own information when a GNN mixes in its neighbours.
    if self_loops:
        np.fill_diagonal(adj, 1.0)

    # Normalise with D^-1/2 A D^-1/2 (the standard GCN scaling), so busy hubs do not dominate.
    if normalise:
        degree = adj.sum(axis=1)
        inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-12)), 0.0)
        adj = inv_sqrt[:, None] * adj * inv_sqrt[None, :]
    return adj
