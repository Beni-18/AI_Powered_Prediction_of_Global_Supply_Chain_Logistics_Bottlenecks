"""Rebuilds the complete hourly node grid for Stage 2 and scales it with numbers learnt on TRAIN hours only.

Owner: Role 3 (Data & Integration Engineer). Written by Role 4 at the team lead's request; Role 3 to review.

Why this file exists: a sequence model reads "the last 72 hours of every node". If an hour is missing, the
window silently glues two different days together. The simulator drops about 1% of hours on purpose, so we put
every missing hour back, mark it `is_imputed=True`, and fill it in a way that never looks into the future:
    1. carry the last real reading forward for at most 6 hours (a short sensor gap);
    2. otherwise use the node's typical value for that hour of the week, learnt on TRAIN hours only.
The robust scaler (median / IQR per node and channel) is also fitted on TRAIN hours only, so the model never
learns anything about the val/test period from the scaling.

READS : a raw hourly node DataFrame (data/raw/sim/node_hourly_raw.csv.gz has the right columns)
WRITES: nothing - returns DataFrames / dicts; Stage02_DL/01_data_engineer.py saves them
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The eight measured channels every node reports each hour (contracts/schemas/hourly_node_telemetry.schema.json).
CHANNELS = ("avg_dwell_time_hours", "throughput_teu_per_hr", "capacity_utilization_pct", "customs_hold_hours",
            "weather_severity_index", "labor_deficit_ratio", "equipment_availability_pct", "inventory_level_teu")
FFILL_LIMIT_HOURS = 6   # role file (Day 2): forward-fill at most 6 hours, then use the hour-of-week median
MIN_IQR_SHARE = 0.05    # an IQR smaller than 5% of the median (or zero) is too small to divide by; see fit_robust_scaler


def complete_grid(raw: pd.DataFrame, node_ids: list, train_end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Return one row per node per hour (no gaps) plus a log of what was filled and how.

    `train_end` limits the hour-of-week medians to training hours, so filling a test hour never uses test data.
    """
    df = raw.copy()
    df["hour_ts"] = pd.to_datetime(df["hour_ts"], utc=True).dt.floor("h")
    # Two raw rows in the same node-hour would make the grid ambiguous; keep the last one and count them.
    n_dupes = int(df.duplicated(["node_id", "hour_ts"]).sum())
    df = df.drop_duplicates(["node_id", "hour_ts"], keep="last")

    # Step 1: the full grid = every registered node x every hour from the first to the last hour seen.
    hours = pd.date_range(df["hour_ts"].min(), df["hour_ts"].max(), freq="h")
    full = pd.MultiIndex.from_product([node_ids, hours], names=["node_id", "hour_ts"])
    grid = df.set_index(["node_id", "hour_ts"]).reindex(full)
    missing = grid[list(CHANNELS)].isna().any(axis=1)
    grid["is_imputed"] = missing.to_numpy()

    # Step 2: short gaps - carry the node's last real reading forward (only the past is used).
    filled_short = grid.groupby(level="node_id")[list(CHANNELS)].ffill(limit=FFILL_LIMIT_HOURS)
    after_ffill_missing = filled_short.isna().any(axis=1)
    grid[list(CHANNELS)] = filled_short

    # Step 3: longer gaps - the node's median for that hour of the week, learnt on TRAIN hours only.
    flat = grid.reset_index()
    flat["how"] = flat["hour_ts"].dt.day_of_week * 24 + flat["hour_ts"].dt.hour  # hour of week, 0..167
    train_rows = flat[(flat["hour_ts"] < train_end) & ~flat["is_imputed"]]
    how_median = train_rows.groupby(["node_id", "how"])[list(CHANNELS)].median()
    node_median = train_rows.groupby("node_id")[list(CHANNELS)].median()
    for col in CHANNELS:
        gap = flat[col].isna()
        if gap.any():
            keys = pd.MultiIndex.from_frame(flat.loc[gap, ["node_id", "how"]])
            fill = how_median[col].reindex(keys).to_numpy()
            # A node-hour-of-week never seen in train (very rare) falls back to the node's overall train median.
            fallback = node_median[col].reindex(flat.loc[gap, "node_id"]).to_numpy()
            flat.loc[gap, col] = np.where(np.isnan(fill), fallback, fill)

    # Step 4: bookkeeping columns. A filled hour had no raw records, so n_records = 0 there.
    flat["n_records"] = flat["n_records"].fillna(0).astype(int)
    if "vessels_waiting" in flat:
        flat["vessels_waiting"] = flat.groupby("node_id")["vessels_waiting"].ffill().bfill().round().astype(int)
    flat["data_origin"] = flat["data_origin"].fillna("SYNTHETIC_CALIBRATED")
    flat["scenario_id"] = flat["scenario_id"].fillna("NOMINAL")
    flat = flat.drop(columns="how").sort_values(["hour_ts", "node_id"]).reset_index(drop=True)

    log = {
        "hours": len(hours), "nodes": len(node_ids), "grid_rows": len(flat),
        "raw_rows": int(len(raw)), "duplicate_raw_rows_dropped": n_dupes,
        "imputed_rows": int(missing.sum()),
        "filled_by_forward_fill_max_6h": int(missing.sum() - after_ffill_missing.sum()),
        "filled_by_train_hour_of_week_median": int(after_ffill_missing.sum()),
        "imputed_rows_by_split_period": {},
    }
    return flat, log


def fit_robust_scaler(grid: pd.DataFrame, train_end: pd.Timestamp, channels=CHANNELS) -> dict:
    """Per node and channel: median and IQR of REAL (not imputed) training hours. Returns a JSON-ready dict.

    Why robust: a strike or blockage creates huge values; a mean/std scaler would be dragged by them, the
    median and the inter-quartile range are not. Why per node: Shanghai moves 10x more boxes than Colombo,
    so one global scale would make small ports look flat.
    """
    train = grid[(pd.to_datetime(grid["hour_ts"], utc=True) < train_end) & ~grid["is_imputed"].astype(bool)]
    out = {"method": "robust (x - median) / IQR, per node and channel, fitted on TRAIN hours only",
           "train_end": str(train_end), "channels": list(channels), "nodes": {}}
    for node, g in train.groupby("node_id"):
        stats = {}
        for col in channels:
            med = float(g[col].median())
            iqr = float(g[col].quantile(0.75) - g[col].quantile(0.25))
            # A channel that barely moves in train (e.g. customs at a chokepoint) would explode when divided by a
            # tiny IQR; use the standard deviation instead, and 1.0 if even that is zero.
            if iqr < MIN_IQR_SHARE * max(abs(med), 1e-6):
                iqr = float(g[col].std()) or 1.0
            stats[col] = {"median": round(med, 6), "iqr": round(iqr, 6)}
        out["nodes"][node] = stats
    return out


def scale_array(values: np.ndarray, scaler: dict, node_ids: list, channels=CHANNELS, inverse: bool = False) -> np.ndarray:
    """Scale (or un-scale) an array whose last two axes are [nodes, channels] with the saved robust scaler."""
    med = np.array([[scaler["nodes"][n][c]["median"] for c in channels] for n in node_ids], dtype=np.float64)
    iqr = np.array([[scaler["nodes"][n][c]["iqr"] for c in channels] for n in node_ids], dtype=np.float64)
    if inverse:
        return values * iqr + med
    return (values - med) / iqr
