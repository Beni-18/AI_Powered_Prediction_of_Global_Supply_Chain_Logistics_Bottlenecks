"""ROLE: ML ENGINEER  |  COMMON MODELKIT
Load a saved Stage 2 forecaster and run it on new hours, with EXACTLY the preprocessing used in training.

Owner: Role 1 - ML Engineer. Written by Role 4 at the team lead's request; Role 1 to review.
Used by: Stage02_DL/05_integration_engineer.py (the /forecast API) and Role 2's attribution work, e.g.

    from common.modelkit.dl_infer import Stage2Forecaster, sim_window
    f = Stage2Forecaster.from_artifact("lstm_forecaster")
    data, origin = sim_window(grid, f.node_ids, f.scaler, as_of="2025-12-01T00:00:00Z")
    result = f.forecast(data, origin)                    # original units, intervals included
    batch = data.batch([origin])                         # the exact tensors the model reads (for attribution)

READS : Stage02_DL/artifacts/<name>.pt (weights + config + interval margins), Stage02_DL/data/scaler.json
WRITES: nothing
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from common.dataio.hourly_grid import CHANNELS
from common.modelkit.dl_data import SIM_TRACK, ForecastData
from common.modelkit.dl_models import load_model
from common.paths import REPO_ROOT, stage_path


class Stage2Forecaster:
    """A trained forecaster plus everything needed to turn raw hours into its inputs and its outputs into hours."""

    def __init__(self, model, bundle: dict, scaler: dict):
        self.model, self.bundle, self.scaler = model, bundle, scaler
        self.meta = bundle.get("meta", {})
        self.kind = bundle["kind"]
        self.node_ids = self.meta.get("node_ids")
        self.margins = np.asarray(self.meta.get("conformal_margin_scaled", 0.0))

    @classmethod
    def from_artifact(cls, name: str) -> "Stage2Forecaster":
        """Load Stage02_DL/artifacts/<name>.pt and the scaler it was trained with."""
        model, bundle = load_model(stage_path(2, "artifacts", f"{name}.pt"))
        scaler_file = bundle.get("meta", {}).get("scaler_file", "Stage02_DL/data/scaler.json")
        scaler = json.loads((REPO_ROOT / scaler_file).read_text())
        return cls(model, bundle, scaler)

    @property
    def needs_network(self) -> bool:
        """Graph models read every node's history and 4 weeks of context; LSTM/Transformer read one node's 72 h."""
        return self.kind == "graph"

    def forecast(self, data: ForecastData, origin: int) -> dict:
        """Quantile forecasts for every node at one origin: {"q": [N, H, T, 3] original units, "horizons_h": [...]}."""
        batch = data.batch(np.array([origin]))
        with torch.no_grad():
            q = self.model(torch.from_numpy(batch["x"]), torch.from_numpy(batch["ctx"]), torch.from_numpy(batch["anchor"])).numpy()
        # Same split-conformal widening as in the predictions file, then back to hours / TEU per hour.
        q[..., 0] = np.minimum(q[..., 0] - self.margins[None, None], q[..., 1])
        q[..., 2] = np.maximum(q[..., 2] + self.margins[None, None], q[..., 1])
        q = np.maximum(data.to_original(q), 0.0)[0]
        hours = [h * data.spec.step_hours for h in data.spec.horizons]
        return {"q": q, "horizons_h": hours, "targets": data.target_names, "origin_ts": data.times[origin]}


def sim_window(grid: pd.DataFrame, node_ids: list, scaler: dict, as_of, override: pd.DataFrame | None = None,
               override_node: str | None = None) -> tuple[ForecastData, int]:
    """Cut the hours a forecaster needs (4 weeks of context + 72 h) ending at `as_of` from the hourly grid.

    `override` (72 hourly rows for `override_node`) replaces that node's last 72 hours with an operator's own
    readings, so the API forecasts from the data it was sent. Returns (ForecastData, index of the origin).
    """
    spec = SIM_TRACK
    g = grid.copy()
    g["hour_ts"] = pd.to_datetime(g["hour_ts"], utc=True)
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    span = spec.block_len * spec.n_blocks + spec.input_len
    g = g[(g["hour_ts"] > as_of - pd.Timedelta(hours=span)) & (g["hour_ts"] <= as_of)]
    if override is not None:
        o = override.copy()
        o["hour_ts"] = pd.to_datetime(o["hour_ts"], utc=True)
        o["node_id"] = override_node
        g = g[~((g["node_id"] == override_node) & g["hour_ts"].isin(o["hour_ts"]))]
        g = pd.concat([g, o[["hour_ts", "node_id", *CHANNELS, "is_imputed"]]], ignore_index=True)
    g["is_imputed"] = g["is_imputed"].astype(str).str.lower().eq("true")
    wide = g.pivot(index="hour_ts", columns="node_id")
    expected = pd.date_range(as_of - pd.Timedelta(hours=len(wide) - 1), as_of, freq="h")
    if len(wide) < spec.input_len or not wide.index.equals(expected) or wide[list(CHANNELS)].isna().any().any():
        raise ValueError("network history is incomplete for this hour")
    values = np.stack([wide[c][node_ids].to_numpy(dtype=np.float64) for c in CHANNELS], axis=-1)
    imputed = wide["is_imputed"][node_ids].to_numpy(dtype=bool)
    # Pad with empty future hours so ForecastData can be cut at the last row (targets are unknown here).
    pad = max(spec.horizons)
    values = np.concatenate([values, np.repeat(values[-1:], pad, axis=0)])
    imputed = np.concatenate([imputed, np.ones((pad, len(node_ids)), dtype=bool)])
    times = pd.date_range(wide.index[0], periods=len(values), freq="h")
    data = ForecastData(values, imputed, times, node_ids, spec, scaler)
    return data, len(wide) - 1


def single_node_window(records: pd.DataFrame, node_id: str, node_ids: list, scaler: dict) -> tuple[ForecastData, int]:
    """Inputs for a one-node model (LSTM / Transformer) from exactly 72 hourly records of one node.

    Other nodes are filled with that node's scaled median (zeros after scaling): these models read each node
    separately, so the other rows do not change this node's forecast.
    """
    spec = SIM_TRACK
    rec = records.sort_values("hour_ts")
    med = np.array([[scaler["nodes"][nid][c]["median"] for c in CHANNELS] for nid in node_ids])
    values = np.repeat(med[None], spec.input_len + max(spec.horizons), axis=0)
    values[:spec.input_len, node_ids.index(node_id)] = rec[list(CHANNELS)].to_numpy(dtype=np.float64)
    imputed = np.zeros(values.shape[:2], dtype=bool)
    imputed[:spec.input_len, node_ids.index(node_id)] = rec.get("is_imputed", pd.Series(False, index=rec.index)).astype(bool).to_numpy()
    imputed[spec.input_len:] = True
    start = pd.Timestamp(rec["hour_ts"].iloc[0])
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    times = pd.date_range(start, periods=len(values), freq="h")
    return ForecastData(values, imputed, times, node_ids, spec, scaler), spec.input_len - 1
