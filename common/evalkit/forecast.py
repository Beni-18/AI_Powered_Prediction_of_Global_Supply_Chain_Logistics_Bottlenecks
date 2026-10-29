"""Forecast scoring used on Day 2: skill versus a baseline, interval quality, pinball loss.

Why this file exists: a forecast number means nothing alone. "MAE 3.1 hours" is good or bad only
compared with just repeating the last value (persistence). Skill score makes that comparison explicit.
"""

from __future__ import annotations

import numpy as np


def skill_score(err_model: float, err_baseline: float) -> float:
    """1 - model error / baseline error. 0 = no better than the baseline, 1 = perfect, negative = worse."""
    if err_baseline == 0:
        return float("nan")
    return float(1.0 - err_model / err_baseline)


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.abs(y - yhat).mean())


def coverage_and_width(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple:
    """Share of true values inside [lo, hi], and the average interval width.

    For an 80% interval the coverage should be near 0.80: much lower means over-confident,
    much higher means the interval is wastefully wide.
    """
    inside = (y >= lo) & (y <= hi)
    return float(inside.mean()), float((hi - lo).mean())


def pinball_loss(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    """Quantile (pinball) loss: under-forecasts and over-forecasts are penalised by the quantile level q."""
    diff = y - yhat
    return float(np.maximum(q * diff, (q - 1) * diff).mean())
