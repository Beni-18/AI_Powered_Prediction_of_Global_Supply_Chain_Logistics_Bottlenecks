"""Real daily traffic per node from IMF PortWatch: the REAL-DATA track for the Stage 2 forecasters.

Owner: Role 3 (Data & Integration Engineer). Written by Role 4 at the team lead's request; Role 3 to review.

Why this file exists: everything else in Stage 2 is calibrated simulation. PortWatch publishes, for every day
since 2019, how many container ships called at each port and how many passed each chokepoint (counted from
satellite AIS signals). That is a real, public time series on the same network nodes, with real disruptions in
it (Ever Given 2021, Shanghai lockdown 2022, Red Sea diversions 2024). Training and testing the same model
architectures on it shows the DL design is not just fitting our simulator.

Mapping rules (decided here, documented in Stage02_DL/HANDOFF/01_data_engineer.md):
  * ports: container port calls per day; chokepoints: container-ship transits per day (both "activity");
  * Los Angeles and Long Beach are ONE port in PortWatch, so the real track keeps PORT_US_LAX only
    (a copy under PORT_US_LGB would be a fake duplicate series);
  * the rail hub and the two distribution centres are not in PortWatch and are left out of the real track.

READS : data/raw/real/portwatch_ports_daily.csv, portwatch_chokepoints_daily.csv
        (downloaded by scripts/fetch_real_data.py; git-ignored because the bulk file is large)
WRITES: nothing - returns a DataFrame
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from common.paths import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw" / "real"
PORTS_FILE = RAW_DIR / "portwatch_ports_daily.csv"
CHOKES_FILE = RAW_DIR / "portwatch_chokepoints_daily.csv"
SHARED_PORT_DROPPED = "PORT_US_LGB"   # same PortWatch series as PORT_US_LAX (see module docstring)
MAX_FILL_DAYS = 3                      # a missing day is carried forward at most 3 days, longer gaps stay flagged
REAL_TRAIN_END = pd.Timestamp("2024-01-01")   # real-track train = 2019-2023 (see real_split_boundaries)


def _fetch_module():
    """Load scripts/fetch_real_data.py (not a package) to reuse its node -> PortWatch id table."""
    spec = importlib.util.spec_from_file_location("fetch_real_data", REPO_ROOT / "scripts" / "fetch_real_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_files_present() -> bool:
    """True when both PortWatch downloads exist on this laptop."""
    return PORTS_FILE.exists() and CHOKES_FILE.exists()


def load_real_daily(node_order: list) -> tuple[pd.DataFrame, dict]:
    """Return a complete daily grid (node_id, date, activity, is_imputed, source_id, data_origin) and a log.

    `node_order` is the registry order; nodes without PortWatch data (and the LA/LB duplicate) are skipped.
    """
    frd = _fetch_module()
    ports = pd.read_csv(PORTS_FILE)
    chokes = pd.read_csv(CHOKES_FILE)
    # Role 3's own helper builds the tidy table: one row per node and day with an `activity` count.
    daily = frd.node_daily_activity(ports, chokes)
    daily = daily[daily["node_id"] != SHARED_PORT_DROPPED]
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None).dt.normalize()

    # Step 1: complete grid over the dates EVERY node covers (ports and chokepoints are published on different
    # schedules; padding the shorter files with months of filled values would invent data), in registry order.
    nodes = [n for n in node_order if n in set(daily["node_id"])]
    spans = daily.groupby("node_id")["date"].agg(["min", "max"])
    days = pd.date_range(spans["min"].max(), spans["max"].min(), freq="D")
    daily = daily[daily["date"].between(days[0], days[-1])]
    full = pd.MultiIndex.from_product([nodes, days], names=["node_id", "date"])
    grid = daily.set_index(["node_id", "date"])[["activity", "source_id"]].reindex(full).reset_index()
    grid["is_imputed"] = grid["activity"].isna()

    # Step 2: a short gap (<= 3 days) takes the last known day (past only); a negative count is impossible.
    grid["activity"] = grid.groupby("node_id")["activity"].ffill(limit=MAX_FILL_DAYS)
    still_missing = grid["activity"].isna()
    # Longer gaps (or the very first days of a series) take the node's TRAIN-years median and stay flagged.
    # Train years only, so a filled day never carries information from the val/test years.
    train_median = grid[(grid["date"] < REAL_TRAIN_END) & ~grid["is_imputed"]].groupby("node_id")["activity"].median()
    grid["activity"] = grid["activity"].fillna(grid["node_id"].map(train_median))
    grid["activity"] = grid["activity"].clip(lower=0)
    grid["source_id"] = grid.groupby("node_id")["source_id"].transform(lambda s: s.ffill().bfill())
    grid["data_origin"] = "REAL"

    log = {
        "source": "IMF PortWatch Daily_Ports_Data (portcalls_container) and Daily_Chokepoints_Data (n_container)",
        "nodes": nodes, "n_nodes": len(nodes),
        "nodes_without_real_data": [n for n in node_order if n not in nodes],
        "first_day": str(days[0].date()), "last_day": str(days[-1].date()), "n_days": len(days),
        "imputed_days": int(grid["is_imputed"].sum()),
        "imputed_longer_than_3_days_filled_with_median": int(still_missing.sum()),
        "zero_activity_days": int((grid["activity"] == 0).sum()),
    }
    return grid, log


def real_split_boundaries(days: pd.DatetimeIndex, gap_days: int = 7) -> dict:
    """Calendar-year split for the real track: train up to 2023, val = 2024, test = 2025 onwards, 7-day purge.

    Why calendar years and not 70/15/15: the real track spans ~7 years; whole years keep every season in every
    split and make the story simple ("trained on 2019-2023, tuned on 2024, tested on 2025-26").
    """
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    train_end = REAL_TRAIN_END
    val_end = pd.Timestamp("2025-01-01")
    gap = pd.Timedelta(days=gap_days)
    return {
        "train_end": train_end.strftime(fmt), "val_start": (train_end + gap).strftime(fmt),
        "val_end": val_end.strftime(fmt), "test_start": (val_end + gap).strftime(fmt),
        "purge_gap_hours": gap_days * 24,
        "first_day": str(days[0].date()), "last_day": str(days[-1].date()),
        "rule": "a forecast window belongs to a split only if its inputs and targets sit fully inside that period",
    }


def activity_log_transform(values: np.ndarray) -> np.ndarray:
    """log(1 + count): daily ship counts are skewed (busy ports call 50+ ships, small ones 2-5)."""
    return np.log1p(np.clip(values, 0, None))
