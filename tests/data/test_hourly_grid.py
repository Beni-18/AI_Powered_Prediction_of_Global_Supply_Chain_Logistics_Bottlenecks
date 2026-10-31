"""Tests for common/dataio/hourly_grid.py: every missing hour comes back flagged, filling never uses the future,
and the scaler is learnt on training hours only.

Owner: Role 3 (Data & Integration Engineer). Written by Role 4 at the team lead's request; Role 3 to review.
"""

import numpy as np
import pandas as pd

from common.dataio.hourly_grid import CHANNELS, complete_grid, fit_robust_scaler, scale_array

NODES = ["A", "B"]


def _raw(hours: int = 24 * 21) -> pd.DataFrame:
    """Three weeks of hourly rows for two nodes; value = hour number, so filled values are easy to trace."""
    times = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    rows = []
    for n in NODES:
        for i, t in enumerate(times):
            rows.append({"hour_ts": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "node_id": n, **{c: float(i) for c in CHANNELS},
                         "vessels_waiting": 2, "n_records": 8, "data_origin": "SYNTHETIC_CALIBRATED", "scenario_id": "NOMINAL"})
    return pd.DataFrame(rows)


def test_short_gap_is_forward_filled_and_flagged():
    """A 2-hour hole is filled with the last real value (from the past) and marked is_imputed."""
    raw = _raw()
    hole = raw["hour_ts"].isin(["2025-01-02T05:00:00Z", "2025-01-02T06:00:00Z"]) & (raw["node_id"] == "A")
    grid, log = complete_grid(raw[~hole], NODES, pd.Timestamp("2025-01-15", tz="UTC"))
    a = grid[grid["node_id"] == "A"].set_index("hour_ts")
    assert len(grid) == len(raw)
    assert log["imputed_rows"] == 2 and log["filled_by_forward_fill_max_6h"] == 2
    assert a.loc[pd.Timestamp("2025-01-02T06:00:00Z"), "avg_dwell_time_hours"] == 28.0  # hour 28 = last real value
    assert a.loc[pd.Timestamp("2025-01-02T06:00:00Z"), "is_imputed"]
    assert a.loc[pd.Timestamp("2025-01-02T06:00:00Z"), "n_records"] == 0


def test_long_gap_uses_the_train_hour_of_week_median_not_the_future():
    """A 10-hour hole after the 6-hour forward-fill limit takes the node's TRAIN median for that hour of week."""
    raw = _raw()
    gap_hours = pd.date_range("2025-01-17T00:00:00Z", periods=10, freq="h").strftime("%Y-%m-%dT%H:%M:%SZ")
    hole = raw["hour_ts"].isin(gap_hours) & (raw["node_id"] == "B")
    grid, log = complete_grid(raw[~hole], NODES, pd.Timestamp("2025-01-15", tz="UTC"))
    assert log["filled_by_train_hour_of_week_median"] == 4
    b = grid[grid["node_id"] == "B"].set_index("hour_ts")
    filled = b.loc[pd.Timestamp("2025-01-17T09:00:00Z"), "avg_dwell_time_hours"]
    # Train hours with the same hour of week as 17 Jan 09:00: 3 Jan and 10 Jan 09:00 -> hours 57 and 225.
    assert filled == np.median([57.0, 225.0])


def test_scaler_ignores_hours_after_train_end():
    """Median/IQR are learnt on train hours only: huge values after train_end do not move them."""
    raw = _raw()
    grid, _ = complete_grid(raw, NODES, pd.Timestamp("2025-01-08", tz="UTC"))
    late = pd.to_datetime(grid["hour_ts"], utc=True) >= pd.Timestamp("2025-01-08", tz="UTC")
    spiked = grid.copy()
    spiked.loc[late, list(CHANNELS)] = 1e6
    assert fit_robust_scaler(grid, pd.Timestamp("2025-01-08", tz="UTC")) == fit_robust_scaler(spiked, pd.Timestamp("2025-01-08", tz="UTC"))


def test_scaling_round_trip():
    """scale then un-scale gives back the original numbers."""
    raw = _raw()
    grid, _ = complete_grid(raw, NODES, pd.Timestamp("2025-01-08", tz="UTC"))
    scaler = fit_robust_scaler(grid, pd.Timestamp("2025-01-08", tz="UTC"))
    x = np.random.default_rng(0).normal(size=(5, len(NODES), len(CHANNELS)))
    assert np.allclose(scale_array(scale_array(x, scaler, NODES), scaler, NODES, inverse=True), x)
