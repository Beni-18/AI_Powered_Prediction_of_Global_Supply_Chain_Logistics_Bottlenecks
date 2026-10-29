"""ROLE: ML ENGINEER  |  COMMON MODELKIT
Deep learning architectures for Stage 02: Multi-Quantile LSTM and Spatial Topology GNN.

Owner: Role 1 - ML Engineer
"""

from pathlib import Path
from typing import Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HORIZONS = [1, 6, 24, 72, 168]
TARGETS = ["dwell", "throughput"]
QUANTILES = [0.10, 0.50, 0.90]


class PinballLoss(nn.Module):
    """Pinball (quantile) loss for asymmetric prediction interval estimation."""

    def __init__(self, quantiles: list[float] = QUANTILES):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """y_pred: [batch, n_horizons, n_targets, n_quantiles]

        y_true: [batch, n_horizons, n_targets]
        """
        loss = 0.0
        y_true_exp = y_true.unsqueeze(-1)  # [batch, n_horizons, n_targets, 1]

        for q_idx, q in enumerate(self.quantiles):
            error = y_true_exp - y_pred[..., q_idx : q_idx + 1]
            loss += torch.max((q - 1.0) * error, q * error).mean()

        return loss / len(self.quantiles)


class LSTMForecaster(nn.Module):
    """Shared multi-node temporal sequence forecaster with node embeddings."""

    def __init__(
        self,
        n_features: int = 8,
        n_nodes: int = 27,
        node_emb_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_horizons: int = 5,
        n_targets: int = 2,
        n_quantiles: int = 3,
    ):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_targets = n_targets
        self.n_quantiles = n_quantiles

        self.node_embedding = nn.Embedding(n_nodes, node_emb_dim)
        total_input_dim = n_features + node_emb_dim

        self.lstm = nn.LSTM(
            input_size=total_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        out_dim = n_horizons * n_targets * n_quantiles
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, node_idx: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, n_features]

        node_idx: [batch]
        Returns: [batch, n_horizons, n_targets, n_quantiles]
        """
        batch_size, seq_len, _ = x.shape
        emb = self.node_embedding(node_idx).unsqueeze(1).repeat(1, seq_len, 1)
        x_comb = torch.cat([x, emb], dim=-1)

        lstm_out, _ = self.lstm(x_comb)
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim]

        out = self.head(last_hidden)
        return out.view(batch_size, self.n_horizons, self.n_targets, self.n_quantiles)


class GraphConvolution(nn.Module):
    """Dense Spatial Graph Convolution layer: H' = ReLU(A_hat * H * W)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """h: [n_nodes, in_features]

        adj: [n_nodes, n_nodes]
        """
        support = self.linear(h)
        output = torch.matmul(adj, support) + self.bias
        return F.relu(output)


class SpatialTopologyGNN(nn.Module):
    """Spatial Topology GNN combining node-level LSTMs with graph message passing."""

    def __init__(
        self,
        n_features: int = 8,
        n_nodes: int = 27,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_horizons: int = 5,
        n_targets: int = 2,
        n_quantiles: int = 3,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_horizons = n_horizons
        self.n_targets = n_targets
        self.n_quantiles = n_quantiles

        self.temporal_encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.gcn1 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)

        out_dim = n_horizons * n_targets * n_quantiles
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x_all_nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x_all_nodes: [batch, n_nodes, seq_len, n_features]

        adj: [n_nodes, n_nodes]
        Returns: [batch, n_nodes, n_horizons, n_targets, n_quantiles]
        """
        batch_size, n_nodes, seq_len, n_feats = x_all_nodes.shape
        x_flat = x_all_nodes.view(batch_size * n_nodes, seq_len, n_feats)

        lstm_out, _ = self.temporal_encoder(x_flat)
        h_temporal = lstm_out[:, -1, :].view(batch_size, n_nodes, -1)  # [batch, n_nodes, hidden_dim]

        # Apply graph convolution per time slice in batch
        graph_reps = []
        for b in range(batch_size):
            h_b = h_temporal[b]  # [n_nodes, hidden_dim]
            g1 = self.gcn1(h_b, adj)
            g2 = self.gcn2(g1, adj)
            graph_reps.append(g2)

        h_graph = torch.stack(graph_reps, dim=0)  # [batch, n_nodes, hidden_dim]

        # Fuse temporal and spatial topologies
        fused = torch.cat([h_temporal, h_graph], dim=-1)  # [batch, n_nodes, hidden_dim * 2]
        out = self.head(fused)  # [batch, n_nodes, out_dim]
        return out.view(batch_size, n_nodes, self.n_horizons, self.n_targets, self.n_quantiles)


def save_torch_model(model: nn.Module, path: Path | str) -> None:
    """Save PyTorch model weights to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)


def load_torch_model(model: nn.Module, path: Path | str) -> nn.Module:
    """Load PyTorch model weights from disk."""
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
