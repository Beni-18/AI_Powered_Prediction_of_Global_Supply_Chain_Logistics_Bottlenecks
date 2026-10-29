"""Calibration: when a model says "80% critical", is it right about 80% of the time?

Why this file exists: operators act on the percentage, not just the label. A well-ranked but badly
calibrated model gives misleading percentages, so we measure and show the gap (ECE and the reliability table).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """For each probability bin: mean predicted probability, observed rate of the event, and row count.

    y is 0/1 (event happened or not); p is the predicted probability of the event.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # The last edge is inclusive so p == 1.0 lands in the top bin.
    which = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = which == b
        n = int(mask.sum())
        rows.append({
            "bin": b,
            "bin_low": edges[b],
            "bin_high": edges[b + 1],
            "n": n,
            "mean_predicted": float(p[mask].mean()) if n else np.nan,
            "observed_rate": float(y[mask].mean()) if n else np.nan,
        })
    return pd.DataFrame(rows)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: average |predicted - observed| over bins, weighted by bin size."""
    table = reliability_table(y, p, n_bins)
    used = table[table["n"] > 0]
    return float((used["n"] * (used["mean_predicted"] - used["observed_rate"]).abs()).sum() / used["n"].sum())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Binary Brier score: mean squared error of the probability (lower is better)."""
    return float(((p - y) ** 2).mean())
