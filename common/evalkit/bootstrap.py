"""Block bootstrap confidence intervals.

Why this file exists: shipments on the same day share weather, congestion and events, so they are not
independent. Resampling single rows would give confidence intervals that are too narrow and make us
over-claim. We resample whole blocks (default: calendar days) instead.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from common.seed import SEED


def _block_index_lists(blocks) -> list:
    """Group row positions by block label, e.g. all rows of one calendar day."""
    blocks = np.asarray(blocks)
    order = np.argsort(blocks, kind="mergesort")
    sorted_blocks = blocks[order]
    cut = np.flatnonzero(sorted_blocks[1:] != sorted_blocks[:-1]) + 1
    return np.split(order, cut)


def _resample_indices(groups: list, rng: np.random.Generator) -> np.ndarray:
    """Draw as many blocks as exist, with replacement, and join their row positions."""
    pick = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[i] for i in pick])


def block_bootstrap_ci(metric_fn: Callable, *arrays, blocks, n: int = 1000, seed: int = SEED, alpha: float = 0.05) -> tuple:
    """Return (point estimate, lower, upper) of metric_fn(*arrays) with a (1-alpha) interval.

    Resamples whole blocks with replacement n times. Same seed gives the same interval on every laptop.
    """
    arrays = [np.asarray(a) for a in arrays]
    groups = _block_index_lists(blocks)
    rng = np.random.default_rng(seed)
    point = float(metric_fn(*arrays))
    draws = []
    for _ in range(n):
        idx = _resample_indices(groups, rng)
        value = metric_fn(*[a[idx] for a in arrays])
        # A resample can lack a class entirely; skip undefined values instead of poisoning the interval.
        if value == value:
            draws.append(float(value))
    if not draws:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_diff(metric_fn: Callable, arrays_a: tuple, arrays_b: tuple, blocks, n: int = 1000, seed: int = SEED, alpha: float = 0.05) -> dict:
    """Compare two models on the SAME rows: CI of metric(A) - metric(B) using shared block resamples.

    Pairing removes the noise both models share, so a small real gain can still be detected.
    `p_a_not_better` is the share of resamples where A did not beat B (a rough one-sided p-value).
    """
    arrays_a = [np.asarray(a) for a in arrays_a]
    arrays_b = [np.asarray(a) for a in arrays_b]
    groups = _block_index_lists(blocks)
    rng = np.random.default_rng(seed)
    point = float(metric_fn(*arrays_a) - metric_fn(*arrays_b))
    diffs = []
    for _ in range(n):
        idx = _resample_indices(groups, rng)
        d = metric_fn(*[a[idx] for a in arrays_a]) - metric_fn(*[a[idx] for a in arrays_b])
        if d == d:
            diffs.append(float(d))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": point, "lo": float(lo), "hi": float(hi), "p_a_not_better": float((diffs <= 0).mean())}
