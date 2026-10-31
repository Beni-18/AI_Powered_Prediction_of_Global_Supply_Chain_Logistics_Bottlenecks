"""ROLE: ML ENGINEER  |  COMMON MODELKIT
The Stage 2 neural forecasters: an LSTM, a Transformer encoder and the cascade-aware graph model, all in plain PyTorch.

Owner: Role 1 - ML Engineer. Rewritten by Role 4 at the team lead's request; Role 1 to review.

Shared design (all three models):
  * Input  x [batch, nodes, steps, channels], ctx [batch, nodes, blocks, channels], anchor [batch, nodes, targets].
  * Output [batch, nodes, horizons, targets, 3] = the 10th, 50th and 90th percentile in SCALED units.
  * The median is anchor + a learnt change, so an untrained model starts as the persistence forecast.
  * The 10th/90th percentiles are the median minus/plus a positive gap (softplus), so they can never cross.
  * Loss = pinball (quantile) loss, which teaches the 10/50/90 percentiles directly.

GraphForecaster (the problem statement's "spatial topology GNN") adds two graph mechanisms to the LSTM:
  1. two graph-convolution layers H' = H + ReLU(A_hat H W) over the trade lanes (A_hat = D^-1/2 (A+I) D^-1/2);
  2. TRANSIT-DELAYED messages: congestion that left node s L hours ago reaches node d now. For each lane and
     horizon the model reads s's state from the moment whose congestion will arrive at d at t + h.
The ablations (graph_none / graph_shuffled) keep the SAME weights and parameter count and only change which
lanes exist (none, or relabelled at random), so any gain of graph_full is due to the real topology.

READS/WRITES: nothing (artefacts are saved by save_model / load_model below)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Kept for older imports: the default Stage 2 horizons, targets and quantile levels.
HORIZONS = [1, 6, 24, 72, 168]
TARGETS = ["dwell", "throughput"]
QUANTILES = [0.10, 0.50, 0.90]


class PinballLoss(nn.Module):
    """Pinball (quantile) loss averaged over every quantile, horizon, target and node."""

    def __init__(self, quantiles=QUANTILES):
        super().__init__()
        self.register_buffer("q", torch.tensor(list(quantiles), dtype=torch.float32))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        """y_pred [..., Q], y_true [...]; `weight` [...] (0/1) can switch off target hours that were imputed."""
        diff = y_true.unsqueeze(-1) - y_pred
        # Under-forecasting the 90th percentile costs 0.9 per unit, over-forecasting it costs only 0.1, and so on.
        loss = torch.maximum(self.q * diff, (self.q - 1.0) * diff).mean(dim=-1)
        if weight is None:
            return loss.mean()
        return (loss * weight).sum() / weight.sum().clamp(min=1.0)


class QuantileHead(nn.Module):
    """Turns a feature vector per node and horizon into ordered 10/50/90 percentiles around the anchor."""

    def __init__(self, in_dim: int, hidden: int, n_targets: int, dropout: float):
        super().__init__()
        self.n_targets = n_targets
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_targets * 3))

    def forward(self, feats: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """feats [B, N, H, F], anchor [B, N, n_targets] -> [B, N, H, n_targets, 3]."""
        raw = self.net(feats).view(*feats.shape[:3], self.n_targets, 3)
        median = anchor.unsqueeze(2) + raw[..., 1]
        lower = median - F.softplus(raw[..., 0]) - 1e-3
        upper = median + F.softplus(raw[..., 2]) + 1e-3
        return torch.stack([lower, median, upper], dim=-1)


class _Base(nn.Module):
    """Common parts: node identity embedding, horizon embedding and the quantile head."""

    def __init__(self, n_inputs: int, n_nodes: int, n_horizons: int, n_targets: int, hidden: int, dropout: float,
                 head_in: int):
        super().__init__()
        self.n_horizons = n_horizons
        # A small learnt "fingerprint" per node (dim 8, role file) lets one shared model treat ports differently.
        self.node_emb = nn.Embedding(n_nodes, 8)
        self.horizon_emb = nn.Embedding(n_horizons, 8)
        self.head = QuantileHead(head_in + 8, hidden, n_targets, dropout)

    def _with_node_id(self, x: torch.Tensor) -> torch.Tensor:
        """Append the node embedding to every time step: [B, N, L, C] -> [B, N, L, C + 8]."""
        B, N, L, _ = x.shape
        emb = self.node_emb.weight[None, :, None, :].expand(B, N, L, -1)
        return torch.cat([x, emb], dim=-1)

    def _predict(self, node_feats: torch.Tensor, anchor: torch.Tensor, per_horizon: torch.Tensor | None = None) -> torch.Tensor:
        """Repeat node features for every horizon, add the horizon embedding (+ optional extras), apply the head."""
        B, N, _ = node_feats.shape
        feats = node_feats.unsqueeze(2).expand(B, N, self.n_horizons, -1)
        h_emb = self.horizon_emb.weight[None, None].expand(B, N, -1, -1)
        parts = [feats, h_emb] if per_horizon is None else [feats, per_horizon, h_emb]
        return self.head(torch.cat(parts, dim=-1), anchor)


class LSTMForecaster(_Base):
    """One LSTM shared by all nodes (2 layers, hidden 64, dropout 0.2 in the role file), reading 72 hours."""

    def __init__(self, n_inputs: int, n_nodes: int, n_horizons: int, n_targets: int, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.2, **_):
        super().__init__(n_inputs, n_nodes, n_horizons, n_targets, hidden, dropout, head_in=hidden)
        self.lstm = nn.LSTM(n_inputs + 8, hidden, num_layers=layers, dropout=dropout if layers > 1 else 0.0, batch_first=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the LSTM over each node's history and keep its last hidden state: [B, N, hidden]."""
        B, N, L, _ = x.shape
        out, _ = self.lstm(self._with_node_id(x).reshape(B * N, L, -1))
        return out[:, -1].reshape(B, N, -1)

    def forward(self, x, ctx, anchor):
        """Forecast quantiles [B, N, H, targets, 3] (ctx is not used by this model)."""
        return self._predict(self.encode(x), anchor)


class _AttnLayer(nn.Module):
    """One Transformer encoder layer that also remembers its attention weights (for Role 2's attention maps)."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.last_attention = None

    def forward(self, h: torch.Tensor, keep_attention: bool = False) -> torch.Tensor:
        """Self-attention + feed-forward, each with a residual connection (pre-norm)."""
        a, w = self.attn(self.norm1(h), self.norm1(h), self.norm1(h), need_weights=keep_attention,
                         average_attn_weights=True)
        if keep_attention:
            self.last_attention = w.detach()  # [batch*nodes, steps, steps]
        h = h + self.drop(a)
        return h + self.drop(self.ff(self.norm2(h)))


class TransformerForecaster(_Base):
    """Transformer encoder (2 layers, 4 heads, d = 64 in the role file) over each node's 72-hour history."""

    def __init__(self, n_inputs: int, n_nodes: int, n_horizons: int, n_targets: int, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.1, heads: int = 4, input_len: int = 72, **_):
        super().__init__(n_inputs, n_nodes, n_horizons, n_targets, hidden, dropout, head_in=hidden)
        self.proj = nn.Linear(n_inputs + 8, hidden)
        # Learnt position embedding: tells attention which hour is "1 hour ago" and which is "3 days ago".
        self.pos = nn.Parameter(torch.zeros(1, input_len, hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([_AttnLayer(hidden, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)

    def encode(self, x: torch.Tensor, keep_attention: bool = False) -> torch.Tensor:
        """Attend over the history and summarise it with the representation of the latest hour: [B, N, hidden]."""
        B, N, L, _ = x.shape
        h = self.proj(self._with_node_id(x).reshape(B * N, L, -1)) + self.pos[:, -L:]
        for layer in self.layers:
            h = layer(h, keep_attention)
        return self.norm(h[:, -1]).reshape(B, N, -1)

    def forward(self, x, ctx, anchor):
        """Forecast quantiles [B, N, H, targets, 3] (ctx is not used by this model)."""
        return self._predict(self.encode(x), anchor)

    def attention_maps(self, x: torch.Tensor) -> list:
        """Attention weights of every layer for these inputs: list of [B*N, steps, steps] (rows = query step)."""
        self.eval()
        with torch.no_grad():
            self.encode(x, keep_attention=True)
        return [layer.last_attention for layer in self.layers]


class GraphForecaster(_Base):
    """LSTM encoder + long-context encoder + graph convolution + transit-delayed messages along the lanes."""

    def __init__(self, n_inputs: int, n_nodes: int, n_horizons: int, n_targets: int, n_ctx_channels: int,
                 n_blocks: int, adjacency: np.ndarray, edges: dict, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.2, gcn_layers: int = 2, msg_dim: int = 32, **_):
        # The head reads [LSTM state h, graph state g, lane messages] (role file: "concatenate [h_i, graph_i]").
        super().__init__(n_inputs, n_nodes, n_horizons, n_targets, hidden, dropout, head_in=2 * hidden + msg_dim)
        self.lstm = nn.LSTM(n_inputs + 8, hidden, num_layers=layers, dropout=dropout if layers > 1 else 0.0, batch_first=True)
        # Long context (e.g. 28 daily averages x 8 channels) squeezed into 32 numbers per node. Dropout on its input:
        # in the first version this path memorised season-specific levels (train loss well below val loss).
        self.ctx_enc = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_blocks * n_ctx_channels, 32), nn.ReLU())
        self.mix = nn.Linear(hidden + 32, hidden)
        self.gcn = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(gcn_layers)])
        self.gcn_norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(gcn_layers)])
        # One shared message network for every lane: input = the source's state + a "nowcast" flag + the delay.
        self.msg = nn.Sequential(nn.Linear(n_ctx_channels + 2, msg_dim), nn.ReLU(), nn.Linear(msg_dim, msg_dim))
        self.n_nodes, self.msg_dim = n_nodes, msg_dim
        # Graph structure is stored as buffers: saved with the weights, but never trained.
        self.register_buffer("adj", torch.as_tensor(adjacency, dtype=torch.float32))
        self.register_buffer("e_src", torch.as_tensor(edges["src"], dtype=torch.long))
        self.register_buffer("e_dst", torch.as_tensor(edges["dst"], dtype=torch.long))
        self.register_buffer("e_block", torch.as_tensor(edges["block"], dtype=torch.long))       # [E, H]
        self.register_buffer("e_future", torch.as_tensor(edges["future"], dtype=torch.float32))   # [E, H]
        lag = np.asarray(edges["lag_steps"], dtype=np.float32)
        self.register_buffer("e_lag", torch.as_tensor(np.log1p(lag) / 7.0, dtype=torch.float32))  # ~0..1 scale

    def messages(self, ctx: torch.Tensor) -> torch.Tensor:
        """Average transit-delayed message arriving at every node for every horizon: [B, N, H, msg_dim]."""
        B = ctx.shape[0]
        H = self.n_horizons
        out = ctx.new_zeros(B, self.n_nodes, H, self.msg_dim)
        if self.e_src.numel() == 0:
            return out  # graph_none: no lanes, so no messages
        E = self.e_src.numel()
        # Source state at the right moment in the past: ctx[b, src, block(e, h), :]  -> [B, E, H, C]
        src_ctx = ctx[:, self.e_src]                                          # [B, E, D, C]
        idx = self.e_block[None, :, :, None].expand(B, E, H, ctx.shape[-1])
        state = torch.gather(src_ctx, 2, idx)
        extra = torch.stack([self.e_future, self.e_lag[:, None].expand(E, H)], dim=-1)[None].expand(B, E, H, 2)
        m = self.msg(torch.cat([state, extra], dim=-1))                       # [B, E, H, msg_dim]
        # Sum the messages per destination node, then divide by how many lanes arrive there (a mean).
        out.index_add_(1, self.e_dst, m)
        counts = torch.bincount(self.e_dst, minlength=self.n_nodes).clamp(min=1).to(ctx.dtype)
        return out / counts[None, :, None, None]

    def forward(self, x, ctx, anchor):
        """Forecast quantiles [B, N, H, targets, 3]."""
        B, N, L, _ = x.shape
        seq, _ = self.lstm(self._with_node_id(x).reshape(B * N, L, -1))
        h = seq[:, -1].reshape(B, N, -1)
        c = self.ctx_enc(ctx.reshape(B, N, -1))
        z = torch.relu(self.mix(torch.cat([h, c], dim=-1)))
        # Graph convolution: every node mixes in its lane neighbours' state (residual keeps its own).
        for lin, norm in zip(self.gcn, self.gcn_norm):
            z = norm(z + torch.relu(torch.einsum("ij,bjf->bif", self.adj, lin(z))))
        # The LSTM state goes to the head directly too, so the graph can only ADD information.
        return self._predict(torch.cat([h, z], dim=-1), anchor, per_horizon=self.messages(ctx))


MODEL_CLASSES = {"lstm": LSTMForecaster, "transformer": TransformerForecaster, "graph": GraphForecaster}


def build_model(kind: str, config: dict) -> nn.Module:
    """Create a model of kind 'lstm' / 'transformer' / 'graph' from a plain config dict."""
    return MODEL_CLASSES[kind](**config)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable weights (reported in the model card and the ablation table)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def save_model(model: nn.Module, path: Path | str, kind: str, config: dict, meta: dict | None = None) -> None:
    """Save weights + the exact config needed to rebuild the model (so Role 2 and the API can load it)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in config.items() if k not in ("adjacency", "edges")}
    torch.save({"kind": kind, "config": cfg, "state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_model(path: Path | str) -> tuple[nn.Module, dict]:
    """Rebuild a saved forecaster on CPU in evaluation mode. Returns (model, saved bundle without weights)."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(bundle["config"])
    state = bundle["state_dict"]
    if bundle["kind"] == "graph":
        # The graph structure lives in the saved buffers; rebuild the model around it.
        cfg["adjacency"] = state["adj"].numpy()
        cfg["edges"] = {"src": state["e_src"].numpy(), "dst": state["e_dst"].numpy(), "block": state["e_block"].numpy(),
                        "future": state["e_future"].numpy(),
                        "lag_steps": np.expm1(state["e_lag"].numpy() * 7.0).round().astype(np.int64)}
    model = build_model(bundle["kind"], cfg)
    model.load_state_dict(state)
    model.eval()
    return model, {k: v for k, v in bundle.items() if k != "state_dict"}


def save_torch_model(model: nn.Module, path: Path | str) -> None:
    """Save only the weights (older helper, kept for compatibility)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_torch_model(model: nn.Module, path: Path | str) -> nn.Module:
    """Load only the weights into an already-built model (older helper, kept for compatibility)."""
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
