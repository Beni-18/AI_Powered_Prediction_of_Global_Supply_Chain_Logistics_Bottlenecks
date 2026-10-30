"""Node context from the hourly node table: what each port, hub or chokepoint looked like JUST BEFORE a shipment.

Owner: Role 3 (Data & Integration Engineer). Written by Role 4 at the team lead's request; Role 3 to review.

Why this file exists: a shipment's delay depends on how congested its node is about to be. The hourly node table
(queue of waiting vessels, yard inventory, utilisation) describes that congestion, but it is a separate file.
These helpers attach the node's state from the last COMPLETE hour before the snapshot, so nothing from the
snapshot's own hour (or later) can leak in. Used by Stage01_ML/01 (data cleaning), 03 (features) and 05 (API).

READS : data/raw/sim/node_hourly_raw.csv.gz
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from common.paths import REPO_ROOT

HOURLY_FILE = REPO_ROOT / "data" / "raw" / "sim" / "node_hourly_raw.csv.gz"
MAX_GAP_HOURS = 6   # the hourly table has ~1% missing hours; reuse a reading at most 6 hours old, else leave it missing

# The node-history features the Stage 1 model uses (all named lag_ because they come from EARLIER hours).
NODE_HISTORY_FEATURES = [
    "lag_capacity_utilization_pct", "lag_avg_dwell_time_hours", "lag_vessels_waiting", "lag_inventory_level_teu",
    "lag_dest_capacity_utilization_pct", "lag_dest_vessels_waiting", "lag_util_change_24h", "lag_queue_change_24h",
]


@lru_cache(maxsize=1)
def load_hourly() -> pd.DataFrame:
    """The hourly node table with parsed UTC hours, read once per process."""
    hourly = pd.read_csv(HOURLY_FILE)
    hourly["hour_ts"] = pd.to_datetime(hourly["hour_ts"], utc=True)
    return hourly.sort_values("hour_ts").reset_index(drop=True)


@lru_cache(maxsize=1)
def _hourly_by_node() -> dict:
    """The hourly table split per node, with plain UTC times ready for binary search (built once per process)."""
    return _index(load_hourly())


def _index(hourly: pd.DataFrame) -> dict:
    """node -> (sorted hour times as datetime64[ns], that node's rows)."""
    return {node: (g["hour_ts"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]"), g.reset_index(drop=True))
            for node, g in hourly.groupby("node_id", sort=False)}


def node_lag(df: pd.DataFrame, cols: list, node_col: str = "current_node_id", lag_h: int = 1,
             prefix: str = "lag_", hourly: pd.DataFrame | None = None) -> pd.DataFrame:
    """For each row: the node's hourly values `lag_h` hours before the snapshot's hour (lag_h=1 = last complete hour).

    Same rule as a backward as-of join per node: the latest hourly row at or before that hour, if it is at most
    MAX_GAP_HOURS old; otherwise the value stays missing. Done with a binary search per node so a single API
    request costs microseconds instead of re-joining the whole table.
    """
    index = _hourly_by_node() if hourly is None else _index(hourly)
    t = (pd.to_datetime(df["timestamp"], utc=True).dt.floor("h") - pd.Timedelta(hours=lag_h)).dt.tz_convert(None)
    t = t.to_numpy(dtype="datetime64[ns]")
    nodes = df[node_col].to_numpy()
    out = np.full((len(df), len(cols)), np.nan)
    max_age = np.timedelta64(MAX_GAP_HOURS, "h")
    for node in pd.unique(nodes):
        if node not in index:
            continue
        rows = np.flatnonzero(nodes == node)
        times, table = index[node]
        pos = np.searchsorted(times, t[rows], side="right") - 1   # last hour at or before the wanted hour
        found = pos >= 0
        fresh = np.zeros(len(rows), dtype=bool)
        fresh[found] = (t[rows][found] - times[pos[found]]) <= max_age
        out[rows[fresh]] = table.loc[pos[fresh], cols].to_numpy(dtype=float)
    return pd.DataFrame(out, columns=[f"{prefix}{c}" for c in cols], index=df.index)


def node_history_features(df: pd.DataFrame, hourly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add NODE_HISTORY_FEATURES: current node's congestion and queue, their 24 h change, and the destination's."""
    now = node_lag(df, ["capacity_utilization_pct", "avg_dwell_time_hours", "vessels_waiting", "inventory_level_teu"],
                   hourly=hourly)
    day_ago = node_lag(df, ["capacity_utilization_pct", "vessels_waiting"], lag_h=25, prefix="lag24_", hourly=hourly)
    dest = node_lag(df, ["capacity_utilization_pct", "vessels_waiting"], node_col="destination_port",
                    prefix="lag_dest_", hourly=hourly)
    out = pd.concat([df, now, dest], axis=1)
    # Is the node getting busier or calmer? Positive = more congested than a day earlier.
    out["lag_util_change_24h"] = now["lag_capacity_utilization_pct"] - day_ago["lag24_capacity_utilization_pct"]
    out["lag_queue_change_24h"] = now["lag_vessels_waiting"] - day_ago["lag24_vessels_waiting"]
    return out
