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



def _hours(times) -> np.ndarray:
    """Convert timestamps to float hours since 1970 so differences are plain numbers."""
    arr = pd.to_datetime(pd.Series(list(times))).to_numpy(dtype="datetime64[ns]")
    return arr.astype("int64") / 3.6e12  # nanoseconds -> hours


def episodes_from_series(times, values, threshold: float, min_hours: int = 6, min_gap_hours: int = 24, bridge_hours: int = 3) -> list:
    """Return the onset timestamp of every episode in one node's hourly series.

    A run counts when the value stays above `threshold` for at least `min_hours` consecutive hours.
    Runs closer than `min_gap_hours` (end of one to start of the next) are merged into one episode.
    The raw hourly grid has about 1% missing hours, so a gap of up to `bridge_hours` missing hours inside
    a run is bridged with the last known value instead of cutting a real episode in two.
    """
    series = pd.Series(np.asarray(values, dtype=float), index=pd.to_datetime(pd.Series(list(times))).dt.floor("h"))
    # One value per hour, in time order; if two rows share an hour keep the last.
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        return []
    grid = series.reindex(pd.date_range(series.index[0], series.index[-1], freq="h"))
    above = (grid.ffill(limit=bridge_hours) > threshold).to_numpy()
    stamps = grid.index

    runs, i = [], 0  # (start_index, end_index) of runs long enough to count
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(above) and above[j + 1]:
            j += 1
        if j - i + 1 >= min_hours:
            runs.append((i, j))
        i = j + 1

    onsets, last_end = [], None
    for start, end in runs:
        if last_end is None or start - last_end >= min_gap_hours:
            onsets.append(stamps[start])
        last_end = end  # a merged run extends the current episode
    return onsets


def lead_time(alert_times, onset_times, max_lookback_h: int = 168) -> dict:
    """Score alerts against onsets for ONE node. For many nodes use lead_time_by_node (never pool raw times).

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


def lead_time_by_node(alerts: pd.DataFrame, onsets: pd.DataFrame, max_lookback_h: int = 168) -> dict:
    """Score alerts against onsets node by node, then pool the results.

    Both frames need columns `node_id` and `time`. An alert only counts for an onset at the SAME node, so a
    warning at one port is never credited to another port's bottleneck.
    """
    nodes = sorted(set(alerts["node_id"]) | set(onsets["node_id"]))
    leads, false_alarms, n_alerts = [], 0, 0
    for node in nodes:
        res = lead_time(list(alerts.loc[alerts["node_id"] == node, "time"]), list(onsets.loc[onsets["node_id"] == node, "time"]), max_lookback_h)
        leads.extend(res["lead_times_h"])
        false_alarms += res["false_alarms"]
        n_alerts += res["n_alerts"]
    leads = np.array(leads, dtype=float)
    caught = leads[~np.isnan(leads)]
    return {
        "n_episodes": int(len(leads)), "n_caught": int(len(caught)),
        "episode_recall": float(len(caught) / len(leads)) if len(leads) else float("nan"),
        "lead_times_h": leads,
        "median_lead_h": float(np.median(caught)) if len(caught) else float("nan"),
        "q25_lead_h": float(np.percentile(caught, 25)) if len(caught) else float("nan"),
        "q75_lead_h": float(np.percentile(caught, 75)) if len(caught) else float("nan"),
        "n_alerts": n_alerts, "false_alarms": false_alarms,
    }
