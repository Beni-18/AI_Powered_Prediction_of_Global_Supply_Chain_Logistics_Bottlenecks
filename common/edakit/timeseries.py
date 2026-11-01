"""ROLE: EDA ENGINEER | COMMON EDA TOOLKIT - TIME SERIES
Small time-series helpers for the Stage 2 analysis: weekly seasonal profiles fitted on train hours only,
past-only residuals, the noise-floor rule that separates seasonal peaks from shocks and structural
bottlenecks, lagged cross-correlation between linked nodes, and run lengths of flagged hours.

OWNER : Role 2 - EDA Engineer
READS : nothing (pure functions on arrays / pandas objects)
WRITES: nothing
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def hour_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    """Hour of the week (0 = Monday 00:00 ... 167 = Sunday 23:00) for every timestamp."""
    return np.asarray(index.day_of_week * 24 + index.hour)


def weekly_profile(frame: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    """Median value per hour of the week and column, fitted on TRAIN rows only (168 rows)."""
    # Only train hours decide what "a normal Tuesday 14:00" looks like, so later periods cannot leak in.
    train = frame[train_mask]
    return train.groupby(hour_of_week(train.index)).median()


def deseasonalise(frame: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """Subtract the weekly profile from every row, matching each row to its hour of the week."""
    # Look up the profile row for each timestamp and subtract it column by column.
    seasonal = profile.reindex(hour_of_week(frame.index)).to_numpy()
    return frame - seasonal


def trailing_residual(deseasonalised: pd.DataFrame, window_hours: int = 336, min_hours: int = 24) -> pd.DataFrame:
    """Deviation from the trailing (past-only) median level: what is NEW in this hour, compared with the last 2 weeks.

    The level uses hours up to and including t only, so the same rule can run live (no future data).
    """
    level = deseasonalised.rolling(window_hours, min_periods=min_hours).median()
    return deseasonalised - level


def mad_sigma(values: Sequence[float]) -> float:
    """Robust spread: median absolute deviation x 1.4826 (equals the standard deviation for bell-shaped noise)."""
    v = pd.Series(values).dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return 0.0
    return float(np.median(np.abs(v - np.median(v))) * 1.4826)


def run_lengths(flags: Sequence[bool]) -> List[tuple]:
    """List of (start_index, length) for every unbroken run of True values."""
    f = np.asarray(flags, dtype=bool)
    runs, i = [], 0
    while i < len(f):
        if not f[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(f) and f[j + 1]:
            j += 1
        runs.append((i, j - i + 1))
        i = j + 1
    return runs


def classify_excursion(residual: Sequence[float], raw: Sequence[float], noise_floor: float, alert_line: float,
                       k: float = 3.0, structural_hours: int = 24) -> Dict[str, object]:
    """Apply the team-huddle rule to one window of hours and say which kind of delay it is.

    Rule (role file, Day 2 item 3), checked in this order:
      - STRUCTURAL candidate: the residual stays above k x noise floor for >= `structural_hours` hours in a row.
      - TRANSIENT shock     : the residual goes above k x noise floor, but for fewer hours than that.
      - SEASONAL peak       : the raw value crosses the alert line while the residual never passes k x noise floor
                              (the rise is slow enough for the trailing level to follow it).
      - NORMAL              : none of the above.
    """
    r = np.asarray(residual, dtype=float)
    x = np.asarray(raw, dtype=float)
    above = np.nan_to_num(r, nan=0.0) > k * noise_floor
    runs = run_lengths(above)
    longest = max((length for _, length in runs), default=0)
    crosses_alert = bool(np.nanmax(x) > alert_line) if len(x) else False
    if longest >= structural_hours:
        label = "STRUCTURAL"
    elif longest > 0:
        label = "TRANSIENT"
    elif crosses_alert:
        label = "SEASONAL_PEAK"
    else:
        label = "NORMAL"
    return {"label": label, "longest_run_above_h": int(longest), "hours_above": int(above.sum()),
            "max_residual_over_floor": float(np.nanmax(r) / noise_floor) if noise_floor > 0 and len(r) else float("nan"),
            "crosses_alert_line": crosses_alert}


def lagged_correlation(a: Sequence[float], b: Sequence[float], max_lag: int) -> np.ndarray:
    """Correlation of a(t) with b(t + lag) for lag = 0..max_lag (a leads b when the peak is at a positive lag)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # Standardise both series once, so each lag is a plain average of products.
    a = (a - a.mean()) / (a.std() or 1.0)
    b = (b - b.mean()) / (b.std() or 1.0)
    n = len(a)
    return np.array([np.dot(a[: n - lag], b[lag:]) / (n - lag) for lag in range(max_lag + 1)])


def remove_common_factor(frame: pd.DataFrame) -> pd.DataFrame:
    """Subtract the network-wide average of every hour, so shared drivers (global stress, peak season) do not look like propagation."""
    return frame.sub(frame.mean(axis=1), axis=0)


def iqr_scale(frame: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    """Divide every column by its train inter-quartile range, so big and small nodes weigh the same."""
    train = frame[train_mask]
    iqr = (train.quantile(0.75) - train.quantile(0.25)).replace(0, np.nan)
    return frame / iqr.fillna(1.0)
