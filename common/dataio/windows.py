"""ROLE: DATA ENGINEER  |  SHARED HELPER
Cuts the hourly node table into sliding windows for the Stage 2 deep-learning forecasters.

Each window = the last `input_len` hours of every node (X) and the values `horizons` hours later (Y).
Windows get the SAME train/val/test periods as Stage 1 (Stage01_ML/data/split_boundaries.json), and a
window is only kept when its inputs AND its future targets sit fully inside one period, so the purge
gap really separates train from val/test.

READS : an hourly_node_telemetry DataFrame (complete grid: every node x every hour)
        Stage01_ML/data/split_boundaries.json (unless boundaries are passed in)
WRITES: nothing - returns a WindowSet
OWNER : Role 3 - Data & Integration Engineer
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.paths import NODES_CSV, stage_path

DEFAULT_FEATURES = ("avg_dwell_time_hours", "throughput_teu_per_hr", "capacity_utilization_pct",
                    "customs_hold_hours", "weather_severity_index", "labor_deficit_ratio",
                    "equipment_availability_pct", "inventory_level_teu")


@dataclass
class WindowSet:
    """Model-ready windows. Shapes: X [n, input_len, n_nodes, n_features], Y [n, n_horizons, n_nodes, n_targets]."""

    X: np.ndarray
    Y: np.ndarray
    origin_ts: np.ndarray   # the last input hour of each window ("now" for the forecast)
    split: np.ndarray       # "train" / "val" / "test"
    node_ids: list
    feature_cols: tuple
    target_cols: tuple
    horizons: tuple


def load_boundaries() -> dict:
    """Read the split dates written by Stage01_ML/01_data_engineer.py (one split, one place)."""
    return json.loads(stage_path(1, "data", "split_boundaries.json").read_text())


def build_windows(hourly_df: pd.DataFrame, input_len: int = 72, horizons=(1, 6, 24, 72, 168),
                  feature_cols=DEFAULT_FEATURES,
                  target_cols=("avg_dwell_time_hours", "throughput_teu_per_hr"),
                  boundaries: dict | None = None, stride: int = 1) -> WindowSet:
    """Build sliding windows from a complete node-hour grid. `stride` = hours between window starts."""
    horizons, feature_cols, target_cols = tuple(horizons), tuple(feature_cols), tuple(target_cols)
    boundaries = boundaries or load_boundaries()
    node_ids = pd.read_csv(NODES_CSV)["node_id"].tolist()

    # Step 1: reshape the long table into a cube [hours, nodes, columns] in the registry's node order.
    df = hourly_df.copy()
    df["hour_ts"] = pd.to_datetime(df["hour_ts"], utc=True)
    cols = list(dict.fromkeys(feature_cols + target_cols))
    wide = df.pivot(index="hour_ts", columns="node_id", values=cols).sort_index()
    hours = wide.index
    # A gap in the grid would silently glue two different days together, so we refuse it.
    expected = pd.date_range(hours[0], hours[-1], freq="h")
    if len(hours) != len(expected) or wide.isna().any().any():
        raise ValueError("hourly_df must be a complete grid (every node, every hour, no NaN). "
                         "Run Stage02_DL/01_data_engineer.py first.")
    cube = np.stack([wide[c][node_ids].to_numpy(dtype=np.float32) for c in cols], axis=-1)

    # Step 2: every possible window "origin" t needs input_len hours before it and max(horizons) after it.
    max_h = max(horizons)
    origins = np.arange(input_len - 1, len(hours) - max_h, stride)
    origin_ts = hours[origins]
    target_end_ts = hours[origins + max_h]
    input_start_ts = hours[origins - input_len + 1]

    # Step 3: give each window a split only if inputs and targets stay inside ONE period.
    b = {k: pd.Timestamp(v) for k, v in boundaries.items() if k.endswith(("_end", "_start"))}
    split = np.full(len(origins), "", dtype=object)
    split[target_end_ts < b["train_end"]] = "train"
    split[(input_start_ts >= b["val_start"]) & (target_end_ts < b["val_end"])] = "val"
    split[input_start_ts >= b["test_start"]] = "test"
    keep = split != ""

    # Step 4: collect inputs and targets for the windows we keep.
    origins, origin_ts, split = origins[keep], origin_ts[keep], split[keep]
    f_idx = [cols.index(c) for c in feature_cols]
    t_idx = [cols.index(c) for c in target_cols]
    X = np.stack([cube[o - input_len + 1:o + 1][:, :, f_idx] for o in origins])
    Y = np.stack([cube[[o + h for h in horizons]][:, :, t_idx] for o in origins])
    return WindowSet(X, Y, np.asarray(origin_ts), split, node_ids, feature_cols, target_cols, horizons)
