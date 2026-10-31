"""Training-time augmentation helpers for the Stage 2 forecasters: jitter, scaling and window-warp.

Owner: Role 3 (Data & Integration Engineer). Role 1 may call these; nothing in the pipeline calls them by itself.

Why this file exists: the simulated year gives only ~6,000 training origins per node, and the neural models can
memorise them. Small, realistic changes to the TRAINING inputs (sensor noise, a busier or quieter node, a
slightly faster or slower build-up) give the model more variety without inventing new outcomes.

Leakage rule: every helper takes the split label of each window and refuses anything that is not "train".
Val and test windows must stay exactly as measured, or the evaluation would score an altered past.

Shapes follow common/modelkit/dl_data.py ForecastData.batch():
    x      [B, N, L, C_in]   inputs; the first `n_measured` channels are the scaled sensor channels, the rest
                             (is_imputed flag, calendar, extra) are bookkeeping and are never changed
    anchor [B, N, n_targets] the scaled target channels at the origin (the "last value")
    y      [B, N, H, n_targets] the scaled future targets
READS : nothing.  WRITES: nothing - returns new arrays, the originals are never modified.
"""

from __future__ import annotations

import numpy as np

JITTER_SIGMA = 0.03      # noise in robust-scaled units (1 unit = one IQR), so 3% of a node's normal spread
SCALE_SIGMA = 0.10       # magnitude factor ~ N(1, 0.1): a node 10% busier or quieter than measured
WARP_FACTORS = (0.8, 1.2)   # one slice of the window is played 20% slower or faster


def _check_train_only(splits) -> None:
    """Stop with a clear error if any window is not a training window (the leakage rule above)."""
    splits = np.asarray(splits)
    if splits.size == 0 or not np.all(splits == "train"):
        bad = sorted(set(splits.tolist()) - {"train"}) if splits.size else ["<empty>"]
        raise ValueError(f"augmentation is for TRAIN windows only; got {bad}")


def jitter(batch: dict, splits, rng: np.random.Generator, n_measured: int, sigma: float = JITTER_SIGMA) -> dict:
    """Add small Gaussian noise to the measured input channels (like a noisier sensor). Targets are unchanged."""
    _check_train_only(splits)
    out = dict(batch)
    x = batch["x"].copy()
    x[..., :n_measured] += rng.normal(0.0, sigma, size=x[..., :n_measured].shape).astype(x.dtype)
    out["x"] = x
    return out


def scaling(batch: dict, splits, rng: np.random.Generator, n_measured: int, target_idx,
            sigma: float = SCALE_SIGMA) -> dict:
    """Multiply each node's measured channels by one random factor per window, inputs AND targets alike.

    The same factor goes on the past, the anchor and the future of a channel, so the model still learns a
    true relationship ("this node, 10% busier") instead of a made-up jump between input and target.
    """
    _check_train_only(splits)
    x = batch["x"]
    factor = rng.normal(1.0, sigma, size=(x.shape[0], x.shape[1], 1, n_measured)).astype(x.dtype)  # [B, N, 1, C]
    out = dict(batch)
    x = x.copy()
    x[..., :n_measured] *= factor
    out["x"] = x
    # The targets are some of the measured channels; scale them with the factor of their own channel.
    target_factor = factor[:, :, 0, list(target_idx)]                     # [B, N, n_targets]
    out["anchor"] = batch["anchor"] * target_factor
    if "y" in batch:
        out["y"] = batch["y"] * target_factor[:, :, None, :]
    return out


def window_warp(batch: dict, splits, rng: np.random.Generator, n_measured: int, factors=WARP_FACTORS) -> dict:
    """Stretch or squeeze one slice in the OLDER half of the window, then resample back to the same length.

    The last hour of the window (the origin) never moves, so the anchor and the targets stay correct.
    """
    _check_train_only(splits)
    x = batch["x"].copy()
    n_windows, n_nodes, length, _ = x.shape
    grid = np.arange(length, dtype=np.float64)
    for b in range(n_windows):
        # Step 1: pick a slice in the older half of the window and a speed for it.
        start = int(rng.integers(0, max(1, length // 2 - length // 4)))
        stop = start + max(2, length // 4)
        speed = float(rng.choice(factors))
        # Step 2: build the warped time axis (slice played at `speed`, the rest shifted), then rescale it so it
        # still runs from the first to the last hour of the window.
        steps = np.ones(length - 1)
        steps[start:stop] = speed
        warped = np.concatenate([[0.0], np.cumsum(steps)])
        warped = warped / warped[-1] * (length - 1)
        # Step 3: read every measured channel of every node at the warped times (linear interpolation).
        for n in range(n_nodes):
            for c in range(n_measured):
                x[b, n, :, c] = np.interp(warped, grid, x[b, n, :, c])
    out = dict(batch)
    out["x"] = x
    return out
