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


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root mean squared error: like MAE but punishes big misses more."""
    return float(np.sqrt(((y - yhat) ** 2).mean()))


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Symmetric mean absolute percentage error in percent (0 = perfect, 200 = worst); safe when both are 0."""
    denom = np.abs(y) + np.abs(yhat)
    ratio = np.where(denom > 0, 2 * np.abs(y - yhat) / np.where(denom > 0, denom, 1), 0.0)
    return float(100 * ratio.mean())


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.2) -> float:
    """Winkler interval score for a (1 - alpha) interval: width + a penalty for every miss. Lower is better.

    It rewards narrow intervals AND punishes intervals that miss, so a model cannot win by being over-confident.
    """
    below = (lo - y) * (y < lo)
    above = (y - hi) * (y > hi)
    return float(((hi - lo) + (2 / alpha) * (below + above)).mean())


def block_skill_ci(err_model: np.ndarray, err_base: np.ndarray, blocks, n: int = 1000, seed: int = 42,
                   alpha: float = 0.05) -> tuple:
    """Skill = 1 - sum|err_model| / sum|err_base| with a block-bootstrap (1 - alpha) interval. Returns (skill, lo, hi).

    Both error arrays must be for the SAME forecasts (same rows). Whole blocks (e.g. forecast days) are resampled,
    because forecasts made on the same day share weather and congestion and are not independent.
    """
    err_model, err_base = np.abs(np.asarray(err_model, float)), np.abs(np.asarray(err_base, float))
    codes, block_idx = np.unique(np.asarray(blocks), return_inverse=True)
    # Sum the errors inside each block once; a resample then only adds up block totals (fast).
    sum_m = np.bincount(block_idx, weights=err_model, minlength=len(codes))
    sum_b = np.bincount(block_idx, weights=err_base, minlength=len(codes))
    point = skill_score(sum_m.sum(), sum_b.sum())
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(codes), size=(n, len(codes)))
    draws = 1.0 - sum_m[pick].sum(axis=1) / np.maximum(sum_b[pick].sum(axis=1), 1e-12)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, max_lag: int = 0) -> dict:
    """Diebold-Mariano test: do forecasts A and B have the same expected loss? (A better -> negative statistic.)

    `loss_a`, `loss_b` = one loss per forecast time, in time order. Forecasts h steps ahead made every step overlap,
    so the variance uses a Newey-West correction with `max_lag` autocovariances (about h / origin spacing).
    Returns the mean loss difference, the statistic and a two-sided p-value (normal approximation).
    """
    from scipy import stats

    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 10:
        return {"mean_diff": float("nan"), "dm_stat": float("nan"), "p_value": float("nan"), "n": n}
    dc = d - d.mean()
    var = (dc @ dc) / n
    for k in range(1, max_lag + 1):
        weight = 1 - k / (max_lag + 1)  # Bartlett weights keep the variance positive
        var += 2 * weight * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(max(var, 1e-12) / n)
    return {"mean_diff": float(d.mean()), "dm_stat": float(stat), "p_value": float(2 * stats.norm.sf(abs(stat))), "n": n}


def exceedance_probability(lo: np.ndarray, mid: np.ndarray, hi: np.ndarray, threshold, coverage: float = 0.80) -> np.ndarray:
    """P(value >= threshold) from a forecast's median and central interval, assuming a bell-shaped spread.

    The [lo, hi] interval holds `coverage` of the outcomes, so sigma = (hi - lo) / (2 z), z = 1.2816 for 80%.
    Used for the probability version of the alert rule and for the API's alert_prob_24h.
    """
    from scipy import stats

    z = stats.norm.ppf(0.5 + coverage / 2)
    sigma = np.maximum((np.asarray(hi, float) - np.asarray(lo, float)) / (2 * z), 1e-6)
    return stats.norm.sf((np.asarray(threshold, float) - np.asarray(mid, float)) / sigma)
