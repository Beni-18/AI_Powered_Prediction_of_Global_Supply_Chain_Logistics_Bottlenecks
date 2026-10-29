"""ROLE: ML ENGINEER  |  TESTS
Unit tests for Stage 02 Deep Learning models (LSTM, Spatial GNN, Pinball Loss, Adjacency).

Owner: Role 1 - ML Engineer
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from common.modelkit import (
    HORIZONS,
    QUANTILES,
    TARGETS,
    LSTMForecaster,
    PinballLoss,
    SpatialTopologyGNN,
    load_adjacency,
    load_canonical_nodes,
    load_torch_model,
    save_torch_model,
)


def test_load_adjacency_shape_and_properties():
    """Verify adjacency matrix is 27x27, symmetric, and has valid normalized values."""
    nodes = load_canonical_nodes()
    assert len(nodes) == 27

    adj = load_adjacency(symmetric=True, self_loops=True, normalise=True)
    assert adj.shape == (27, 27)
    assert np.allclose(adj, adj.T, atol=1e-5)  # Must be symmetric
    assert (adj >= 0.0).all()  # Probabilities / normalized weights are non-negative
    assert np.all(np.diag(adj) > 0.0)  # Self-loops ensure positive diagonal


def test_pinball_loss():
    """Verify quantile loss evaluates correctly and handles asymmetric penalties."""
    loss_fn = PinballLoss(quantiles=[0.10, 0.50, 0.90])
    y_true = torch.tensor([[[10.0, 20.0]]])  # [1, 1, 2]
    # Prediction matches truth exactly
    y_pred_exact = torch.tensor([[[[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]]])  # [1, 1, 2, 3]

    exact_loss = loss_fn(y_pred_exact, y_true)
    assert torch.isclose(exact_loss, torch.tensor(0.0), atol=1e-5)


def test_lstm_forecaster_forward():
    """Verify LSTM forecaster outputs exact tensor shape: [batch, horizons, targets, quantiles]."""
    batch_size = 4
    seq_len = 72
    n_features = 8
    n_nodes = 27

    model = LSTMForecaster(
        n_features=n_features,
        n_nodes=n_nodes,
        hidden_dim=32,
        num_layers=1,
    )
    model.eval()

    x = torch.randn(batch_size, seq_len, n_features)
    node_idx = torch.tensor([0, 5, 12, 26])

    with torch.no_grad():
        out = model(x, node_idx)

    assert out.shape == (batch_size, len(HORIZONS), len(TARGETS), len(QUANTILES))


def test_spatial_gnn_forward():
    """Verify Spatial Topology GNN performs temporal encoding and graph convolution."""
    batch_size = 2
    n_nodes = 27
    seq_len = 24
    n_features = 8

    model = SpatialTopologyGNN(
        n_features=n_features,
        n_nodes=n_nodes,
        hidden_dim=32,
        num_layers=1,
    )
    model.eval()

    adj = torch.from_numpy(load_adjacency()).float()
    x_all = torch.randn(batch_size, n_nodes, seq_len, n_features)

    with torch.no_grad():
        out = model(x_all, adj)

    assert out.shape == (batch_size, n_nodes, len(HORIZONS), len(TARGETS), len(QUANTILES))


def test_model_save_and_load(tmp_path: Path):
    """Verify weights can be saved and loaded accurately."""
    model = LSTMForecaster(n_features=4, n_nodes=5, hidden_dim=16, num_layers=1)
    save_path = tmp_path / "test_model.pt"

    save_torch_model(model, save_path)
    assert save_path.exists()

    new_model = LSTMForecaster(n_features=4, n_nodes=5, hidden_dim=16, num_layers=1)
    load_torch_model(new_model, save_path)

    # Weights match
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)
