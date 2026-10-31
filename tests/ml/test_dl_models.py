"""ROLE: ML ENGINEER  |  TESTS
Unit tests for the Stage 2 neural forecasters (common/modelkit/dl_models.py): output shapes, ordered quantiles,
the pinball loss, identical size of the graph ablations, and save/load round trips.

Owner: Role 1 - ML Engineer. Rewritten by Role 4 with the new models; Role 1 to review.
"""

import numpy as np
import pandas as pd
import pytest

# These tests need PyTorch (Stage 02 only); skip them on machines and CI runners without it.
torch = pytest.importorskip("torch")

from common.modelkit.dl_data import SIM_TRACK, shuffle_permutation, symmetric_adjacency, transit_edges  # noqa: E402
from common.modelkit.dl_models import (  # noqa: E402
    PinballLoss, build_model, count_parameters, load_model, save_model,
)
from common.paths import EDGES_CSV, NODES_CSV  # noqa: E402

NODES = pd.read_csv(NODES_CSV)["node_id"].tolist()
EDGES = pd.read_csv(EDGES_CSV)
B, N, L, C, H, T = 2, len(NODES), 12, 13, 5, 2


def _inputs():
    """Small random inputs with the real number of nodes (so the real lanes can be used)."""
    g = torch.Generator().manual_seed(0)
    return (torch.randn(B, N, L, C, generator=g), torch.randn(B, N, SIM_TRACK.n_blocks, 8, generator=g),
            torch.randn(B, N, T, generator=g))


def _config(kind: str, variant: str = "graph_full") -> dict:
    """Config for a tiny model; the three graph variants differ only in their lanes."""
    cfg = {"n_inputs": C, "n_nodes": N, "n_horizons": H, "n_targets": T, "hidden": 16, "layers": 1, "input_len": L,
           "n_ctx_channels": 8, "n_blocks": SIM_TRACK.n_blocks}
    if kind == "graph":
        perm = shuffle_permutation(N) if variant == "graph_shuffled" else None
        cfg["adjacency"] = symmetric_adjacency(NODES, EDGES, permutation=perm, identity=variant == "graph_none")
        edges = transit_edges(NODES, EDGES, SIM_TRACK, permutation=perm)
        cfg["edges"] = {k: v[:0] for k, v in edges.items()} if variant == "graph_none" else edges
    return cfg


def test_pinball_loss_matches_hand_calculation():
    """y = 10, quantiles 9/10/12: losses 0.1*1, 0, 0.1*2 -> mean 0.1."""
    loss = PinballLoss()(torch.tensor([[9.0, 10.0, 12.0]]), torch.tensor([10.0]))
    assert loss.item() == pytest.approx((0.1 * 1 + 0 + 0.1 * 2) / 3)


def test_pinball_loss_ignores_masked_targets():
    """A weight of 0 (imputed target hour) removes that target from the loss."""
    pred = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    y = torch.tensor([1.0, 100.0])
    assert PinballLoss()(pred, y, torch.tensor([1.0, 0.0])).item() == pytest.approx(PinballLoss()(pred[:1], y[:1]).item())


@pytest.mark.parametrize("kind", ["lstm", "transformer", "graph"])
def test_every_model_returns_ordered_quantiles(kind):
    """Output is [batch, nodes, horizons, targets, 3] and q10 <= q50 <= q90 everywhere."""
    model = build_model(kind, _config(kind))
    q = model(*_inputs())
    assert q.shape == (B, N, H, T, 3)
    assert (q[..., 0] <= q[..., 1]).all() and (q[..., 1] <= q[..., 2]).all()


def test_graph_ablations_have_identical_parameter_counts():
    """graph_full, graph_shuffled and graph_none differ only in lanes, never in size (fair ablation)."""
    sizes = {v: count_parameters(build_model("graph", _config("graph", v))) for v in ("graph_full", "graph_shuffled", "graph_none")}
    assert len(set(sizes.values())) == 1


def test_graph_none_passes_no_messages_and_lanes_change_the_forecast():
    """Without lanes the message block is all zeros; with the same weights, real lanes change the output."""
    torch.manual_seed(0)
    full = build_model("graph", _config("graph", "graph_full"))
    none = build_model("graph", _config("graph", "graph_none"))
    # Copy the weights (not the lane buffers) so only the graph differs.
    none.load_state_dict({k: v for k, v in full.state_dict().items() if not k.startswith(("adj", "e_"))}, strict=False)
    x, ctx, anchor = _inputs()
    assert torch.count_nonzero(none.messages(ctx)) == 0
    assert not torch.allclose(full(x, ctx, anchor), none(x, ctx, anchor))


@pytest.mark.parametrize("kind", ["lstm", "transformer", "graph"])
def test_save_and_load_gives_the_same_forecast(tmp_path, kind):
    """A saved model (weights + config + lanes) reloads to exactly the same outputs."""
    torch.manual_seed(0)
    cfg = _config(kind)
    model = build_model(kind, cfg).eval()
    save_model(model, tmp_path / "m.pt", kind, cfg, meta={"model_version": "test"})
    loaded, bundle = load_model(tmp_path / "m.pt")
    with torch.no_grad():
        assert torch.allclose(model(*_inputs()), loaded(*_inputs()), atol=1e-6)
    assert bundle["meta"]["model_version"] == "test"


def test_transformer_attention_maps_are_probabilities():
    """Each attention row sums to 1 over the input hours (Role 2 plots these as attention maps)."""
    model = build_model("transformer", _config("transformer"))
    maps = model.attention_maps(_inputs()[0])
    assert len(maps) == 1 and maps[0].shape == (B * N, L, L)
    assert np.allclose(maps[0].sum(dim=-1).numpy(), 1.0, atol=1e-5)
