"""Lead time: how many hours BEFORE a bottleneck begins does the model first raise an alert?

Why this file exists: the problem statement asks us to "measure how far in advance the model predicts
bottlenecks". Definitions (agreed with the EDA role, see docs/roles/ROLE_4_EVALUATION_ENGINEER.md, Day 2):
  * episode onset  = first hour of >= 6 consecutive hours above the node's alert threshold,
                     with episodes at least 24 h apart;
  * lead time      = onset - first alert inside the 168 h before the onset;
  * false alarm    = an alert with no onset in the following 168 h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = np.timedelta64(1, "h")


def _hours(times) -> np.ndarray:
    """Convert timestamps to float hours since 1970 so differences are plain numbers."""
    arr = pd.to_datetime(pd.Series(list(times))).to_numpy(dtype="datetime64[ns]")
    return arr.astype("int64") / 3.6e12  # nanoseconds -> hours


def episodes_from_series(times, values, threshold: float, min_hours: int = 6, min_gap_hours: int = 24) -> list:
    """Return the onset timestamp of every episode in one node's hourly series.

    A run counts when the value stays above `threshold` for at least `min_hours` consecutive hours.
    Runs closer than `min_gap_hours` (end of one to start of the next) are merged into one episode.
    """
    t = pd.to_datetime(pd.Series(list(times))).reset_index(drop=True)
    v = np.asarray(values, dtype=float)
    order = np.argsort(t.to_numpy())
    t, v = t.iloc[order].reset_index(drop=True), v[order]
    above = v > threshold

    runs = []  # (start_hour, end_hour, start_timestamp) of qualifying runs
    h = _hours(t)
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i
        # Extend while the next row is the next hour AND still above the threshold.
        while j + 1 < len(above) and above[j + 1] and abs(h[j + 1] - h[j] - 1.0) < 1e-6:
            j += 1
        if (h[j] - h[i]) + 1.0 >= min_hours:
            runs.append((h[i], h[j], t.iloc[i]))
        i = j + 1

    onsets, last_end = [], None
    for start_h, end_h, start_ts in runs:
        if last_end is None or start_h - last_end >= min_gap_hours:
            onsets.append(start_ts)
        last_end = end_h  # a merged run extends the current episode
    return onsets


def lead_time(alert_times, onset_times, max_lookback_h: int = 168) -> dict:
    """Score alerts against onsets for one node (or one pooled set of nodes with unique times).

    Returns the per-episode lead times (NaN = missed), episode recall, the false alarm count,
    and the median and quartiles of the lead time over caught episodes.
    """
    alerts = np.sort(_hours(alert_times)) if len(alert_times) else np.array([])
    onsets = np.sort(_hours(onset_times)) if len(onset_times) else np.array([])

    leads = []
    for onset in onsets:
        # Alerts that fired inside the allowed look-back window up to the onset itself.
        in_window = alerts[(alerts >= onset - max_lookback_h) & (alerts <= onset)]
        leads.append(onset - in_window.min() if len(in_window) else np.nan)
    leads = np.array(leads, dtype=float)

    # An alert is a false alarm when no onset follows within the look-ahead window.
    false_alarms = 0
    for a in alerts:
        if not ((onsets >= a) & (onsets <= a + max_lookback_h)).any():
            false_alarms += 1

    caught = leads[~np.isnan(leads)]
    return {
        "n_episodes": int(len(onsets)),
        "n_caught": int(len(caught)),
        "episode_recall": float(len(caught) / len(onsets)) if len(onsets) else float("nan"),
        "lead_times_h": leads,
        "median_lead_h": float(np.median(caught)) if len(caught) else float("nan"),
        "q25_lead_h": float(np.percentile(caught, 25)) if len(caught) else float("nan"),
        "q75_lead_h": float(np.percentile(caught, 75)) if len(caught) else float("nan"),
        "n_alerts": int(len(alerts)),
        "false_alarms": int(false_alarms),
    }
