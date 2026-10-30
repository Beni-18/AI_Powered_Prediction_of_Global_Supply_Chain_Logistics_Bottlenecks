"""Real-world drivers for the simulator, read from the calibration files (data/calibration/).

Owner: Role 3 (Data & Integration Engineer). Written by Role 4 at the team lead's request; Role 3 to review.

Why this file exists: the simulator used made-up weather, a made-up peak season and a sine wave for global
supply-chain pressure. These helpers replace them with REAL data, aligned to the simulator's hours and nodes:
  * weather         - Open-Meteo hourly wind gusts + rain at every node (weather_hourly.csv.gz)
  * month_factor    - IMF PortWatch: each node's real traffic in each month relative to its yearly mean
  * relative_size   - IMF PortWatch: each node's real daily traffic relative to the average node
  * gscpi           - NY Fed Global Supply Chain Pressure Index, monthly
Shipment outcomes are still simulated from these drivers (data_origin stays SYNTHETIC_CALIBRATED).

READS : data/calibration/weather_hourly.csv.gz, activity_profile.csv, gscpi_monthly.csv
WRITES: nothing (the files are built by scripts/fetch_real_data.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.paths import REPO_ROOT

CAL_DIR = REPO_ROOT / "data" / "calibration"


def real_drivers(hours: pd.DatetimeIndex, node_ids: list) -> dict:
    """Real drivers as arrays shaped like the simulator's: (hours x nodes) or (nodes,) or (hours,)."""
    # Weather: one value per node per hour. Every simulated hour must be covered, or we stop loudly.
    w = pd.read_csv(CAL_DIR / "weather_hourly.csv.gz")
    w["hour_ts"] = pd.to_datetime(w["hour_ts"], utc=True)
    grid = w.pivot(index="hour_ts", columns="node_id", values="weather_severity_index").reindex(index=hours, columns=node_ids)
    if grid.isna().any(axis=None):
        missing = int(grid.isna().sum().sum())
        raise ValueError(f"real weather is missing {missing} node-hours; run scripts/fetch_real_data.py --weather-only")
    weather = grid.to_numpy(dtype=float)

    # PortWatch activity: month-of-year traffic factor and relative size per node (1.0 = no data, i.e. neutral).
    profile = pd.read_csv(CAL_DIR / "activity_profile.csv").set_index("node_id")
    months = hours.month.to_numpy()
    month_cols = [f"month_{m:02d}" for m in range(1, 13)]
    factors = profile.reindex(node_ids)[month_cols].fillna(1.0).to_numpy()      # (nodes x 12)
    month_factor = factors[:, months - 1].T                                       # (hours x nodes)
    relative_size = profile.reindex(node_ids)["relative_size"].fillna(1.0).to_numpy()

    # GSCPI: the month's index value for every hour (standard deviations from its long-run average).
    g = pd.read_csv(CAL_DIR / "gscpi_monthly.csv").set_index("month")["gscpi"]
    keys = hours.strftime("%Y-%m")
    if not set(keys) <= set(g.index):
        raise ValueError(f"GSCPI has no value for {sorted(set(keys) - set(g.index))}; refresh gscpi_monthly.csv")
    gscpi = g.reindex(keys).to_numpy(dtype=float)

    return {"weather": weather, "month_factor": month_factor, "relative_size": relative_size, "gscpi": gscpi,
            "has_profile": profile.reindex(node_ids)["relative_size"].notna().to_numpy()}
