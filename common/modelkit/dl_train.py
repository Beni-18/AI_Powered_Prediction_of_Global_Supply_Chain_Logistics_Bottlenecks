"""ROLE: ML ENGINEER  |  COMMON MODELKIT
Training loop, baselines, interval calibration and the long-format output used by Stage02_DL/03_dl_engineer.py.

Owner: Role 1 - ML Engineer. Written by Role 4 at the team lead's request; Role 1 to review.

Rules this file enforces (AGENTS.md R5, ROLE_1 section 4):
  * fit only on TRAIN origins; early-stop and calibrate intervals on VAL origins; never look at TEST numbers;
  * every baseline is a real, stated method (no invented offsets):
      persistence     = the last observed value;
      seasonal_naive  = the value one week before the target time (168 h, or 7 days on the real track);
      rolling_mean    = the mean of the last day (24 h) or week (7 days);
      xgb_lag         = XGBoost on lag features, one model per horizon and target, 3 quantiles each;
  * baseline intervals come from the 10th/90th percentile of their own TRAIN errors, per node, horizon, target;
  * every model (baseline or neural) then gets the same split-conformal correction on VAL, so interval
    widths are compared fairly.

READS/WRITES: nothing (the stage script loads data and saves outputs)
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch

from common.modelkit.dl_data import ForecastData
from common.modelkit.dl_models import PinballLoss
from common.seed import SEED

INTERVAL_COVERAGE = 0.80  # the 10th-90th percentile band should hold the truth 80% of the time


# ------------------------------------------------------------------ neural training

def _to_torch(batch: dict) -> tuple:
    """NumPy batch -> float tensors (x, ctx, anchor, y or None)."""
    t = lambda k: torch.from_numpy(batch[k]) if k in batch else None
    return t("x"), t("ctx"), t("anchor"), t("y")


def predict_scaled(model: torch.nn.Module, data: ForecastData, origins: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Quantile forecasts in scaled units, [n_origins, N, H, n_targets, 3]."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(origins), batch_size):
            x, ctx, anchor, _ = _to_torch(data.batch(origins[i:i + batch_size]))
            outs.append(model(x, ctx, anchor).numpy())
    return np.concatenate(outs) if outs else np.zeros((0, data.n_nodes, len(data.spec.horizons), len(data.target_idx), 3))


def _loss_weight(data: ForecastData, origins: np.ndarray) -> torch.Tensor:
    """1 for measured target hours, 0 for imputed ones (we never teach the model to copy our own filling)."""
    w = (~data.target_imputed(origins)).astype(np.float32)          # [B, N, H]
    return torch.from_numpy(np.repeat(w[..., None], len(data.target_idx), axis=-1))


def train_neural(model: torch.nn.Module, data: ForecastData, train_origins: np.ndarray, val_origins: np.ndarray,
                 epochs: int = 30, batch_size: int = 16, lr: float = 1e-3, patience: int = 5, clip: float = 1.0,
                 weight_decay: float = 1e-4, log=print, max_minutes: float | None = None) -> tuple:
    """Train with AdamW + pinball loss; keep the weights of the best VAL epoch (early stopping, patience 5).

    batch_size counts ORIGINS; each origin holds every node, so 16 origins = 432 node histories per step.
    Returns (best model, training curve as a list of dicts).
    """
    loss_fn = PinballLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(SEED)
    best, best_state, bad, curve = float("inf"), None, 0, []
    start = time.time()
    for epoch in range(1, epochs + 1):
        # One pass over the training origins in a fresh random order.
        model.train()
        order = rng.permutation(train_origins)
        total, n = 0.0, 0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            x, ctx, anchor, y = _to_torch(data.batch(idx))
            opt.zero_grad()
            loss = loss_fn(model(x, ctx, anchor), y, _loss_weight(data, idx))
            loss.backward()
            # Gradient clipping (1.0, role file) stops one stormy batch from blowing up the weights.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total += loss.item() * len(idx)
            n += len(idx)

        # Score on validation origins only; the test period is never touched here.
        val_loss = val_pinball(model, data, val_origins, loss_fn)
        curve.append({"epoch": epoch, "train_pinball": round(total / max(n, 1), 5), "val_pinball": round(val_loss, 5),
                      "seconds": round(time.time() - start, 1)})
        log(f"    epoch {epoch:>2}  train {total / max(n, 1):.4f}  val {val_loss:.4f}")
        if val_loss < best - 1e-4:
            best, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                log(f"    early stop: no val improvement for {patience} epochs (best {best:.4f})")
                break
        if max_minutes is not None and time.time() - start > 60 * max_minutes:
            log(f"    time budget of {max_minutes} min reached")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    for row in curve:
        row["best_epoch"] = int(min(curve, key=lambda r: r["val_pinball"])["epoch"])
    return model, curve


def val_pinball(model: torch.nn.Module, data: ForecastData, origins: np.ndarray, loss_fn=None) -> float:
    """Average pinball loss (scaled units) over the given origins, ignoring imputed target hours."""
    loss_fn = loss_fn or PinballLoss()
    pred = torch.from_numpy(predict_scaled(model, data, origins))
    y = torch.from_numpy(data.batch(origins)["y"])
    return float(loss_fn(pred, y, _loss_weight(data, origins)))


# ------------------------------------------------------------------ simple baselines (scaled units)

def naive_point(data: ForecastData, origins: np.ndarray, kind: str) -> np.ndarray:
    """Point forecast [B, N, H, n_targets] of one simple baseline, in scaled units."""
    spec, tidx = data.spec, data.target_idx
    origins = np.asarray(origins)
    if kind == "persistence":
        # The last observed value, repeated for every horizon.
        last = data.scaled[origins][..., tidx]
        return np.repeat(last[:, :, None, :], len(spec.horizons), axis=2)
    if kind == "seasonal_naive":
        # The value one week before the target time; for a 1-week horizon that is the current value.
        cols = []
        for h in spec.horizons:
            src = origins + h - spec.seasonal_period
            src = np.where(src <= origins, src, origins)  # never read the future
            cols.append(data.scaled[np.clip(src, 0, None)][..., tidx])
        return np.stack(cols, axis=2)
    if kind == "rolling_mean":
        # Mean of the last day (hourly track) or week (daily track): smooths out sensor noise.
        steps = origins[:, None] + np.arange(-spec.rolling_len + 1, 1)[None, :]
        mean = data.scaled[np.clip(steps, 0, None)][..., tidx].mean(axis=1)
        return np.repeat(mean[:, :, None, :], len(spec.horizons), axis=2)
    raise ValueError(f"unknown baseline {kind}")


def residual_quantiles(point_train: np.ndarray, y_train: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """10th and 90th percentile of TRAIN errors (truth - forecast) per node, horizon and target: [2, N, H, T]."""
    err = np.where(mask[..., None], y_train - point_train, np.nan)
    return np.nanpercentile(err, [10, 90], axis=0)


def with_residual_band(point: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Stack [point + q10 error, point, point + q90 error] into [B, N, H, T, 3]."""
    return np.stack([point + band[0][None], point, point + band[1][None]], axis=-1)


# ------------------------------------------------------------------ XGBoost on lag features

def lag_features(data: ForecastData, origins: np.ndarray) -> tuple:
    """Tabular features per (origin, node): target lags and rolling stats, other channels, context, clock, node id.

    Returns (X [B*N, F], feature names). Every value comes from steps <= origin (past only).
    """
    spec = data.spec
    origins = np.asarray(origins)
    if spec.step_hours == 1:
        lags, windows = (0, 1, 2, 3, 6, 12, 24, 48, 72, 168), (6, 24, 72, 168)
    else:
        lags, windows = (0, 1, 2, 3, 6, 7, 13, 14, 21, 27), (3, 7, 14, 28)
    cols, names = [], []
    for ti, tname in zip(data.target_idx, data.target_names):
        series = data.scaled[..., ti]                                       # [T, N]
        for lag in lags:
            cols.append(series[np.clip(origins - lag, 0, None)])
            names.append(f"{tname}_lag{lag}")
        for w in windows:
            steps = np.clip(origins[:, None] + np.arange(-w + 1, 1)[None, :], 0, None)
            block = series[steps]                                           # [B, w, N]
            cols += [block.mean(axis=1), block.std(axis=1)]
            names += [f"{tname}_mean{w}", f"{tname}_std{w}"]
    # Other channels: current value and the mean over the last block (e.g. 24 h).
    for ci, cname in enumerate(spec.channels):
        if ci in data.target_idx:
            continue
        steps = np.clip(origins[:, None] + np.arange(-spec.block_len + 1, 1)[None, :], 0, None)
        cols += [data.scaled[origins][..., ci], data.scaled[steps][..., ci].mean(axis=1)]
        names += [f"{cname}_now", f"{cname}_mean_block"]
    # Long context: the target's block averages further back (weekly memory on both tracks).
    ctx = data.batch(origins)["ctx"]                                        # [B, N, D, C]
    for k in range(0, spec.n_blocks, max(1, spec.n_blocks // 7)):
        for ti, tname in zip(data.target_idx, data.target_names):
            cols.append(ctx[:, :, k, ti])
            names.append(f"{tname}_block{k}")
    B, N = len(origins), data.n_nodes
    X = np.stack([c.reshape(B, N) for c in cols], axis=-1)                  # [B, N, F]
    clock = np.broadcast_to(data.calendar[origins][:, None, :], (B, N, 4))
    node = np.broadcast_to(np.arange(N)[None, :, None], (B, N, 1)).astype(np.float32)
    X = np.concatenate([X, clock, node], axis=-1)
    names += ["cal_a_sin", "cal_a_cos", "dow_sin", "dow_cos", "node_index"]
    return X.reshape(B * N, -1).astype(np.float32), names


def train_xgb_lag(data: ForecastData, train_origins: np.ndarray, val_origins: np.ndarray, n_estimators: int = 600,
                  n_jobs: int = 4, log=print) -> dict:
    """One XGBoost model per (horizon, target) predicting the CHANGE from the last value at 3 quantiles.

    Early stopping (50 rounds) uses the VAL origins. Returns {(h_index, t_index): fitted model} plus metadata.
    """
    import xgboost as xgb

    Xtr, names = lag_features(data, train_origins)
    Xva, _ = lag_features(data, val_origins)
    btr, bva = data.batch(train_origins), data.batch(val_origins)
    wtr = ~data.target_imputed(train_origins)                               # [B, N, H]
    wva = ~data.target_imputed(val_origins)
    models = {}
    for hi, h in enumerate(data.spec.horizons):
        for ti, tname in enumerate(data.target_names):
            # Target = change from the anchor, so the trees learn "how much will it move", like the neural models.
            ytr = (btr["y"][:, :, hi, ti] - btr["anchor"][:, :, ti]).reshape(-1)
            yva = (bva["y"][:, :, hi, ti] - bva["anchor"][:, :, ti]).reshape(-1)
            mtr, mva = wtr[:, :, hi].reshape(-1), wva[:, :, hi].reshape(-1)
            model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array([0.1, 0.5, 0.9]),
                                     n_estimators=n_estimators, learning_rate=0.05, max_depth=5, subsample=0.8,
                                     colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                                     early_stopping_rounds=50, random_state=SEED, n_jobs=n_jobs)
            model.fit(Xtr[mtr], ytr[mtr], eval_set=[(Xva[mva], yva[mva])], verbose=False)
            models[(hi, ti)] = model
            log(f"    xgb_lag {tname} h={h}: best iteration {model.best_iteration}")
    return {"models": models, "feature_names": names}


def predict_xgb_lag(bundle: dict, data: ForecastData, origins: np.ndarray) -> np.ndarray:
    """XGBoost quantile forecasts in scaled units, [B, N, H, n_targets, 3], sorted so the quantiles never cross."""
    X, _ = lag_features(data, origins)
    anchor = data.batch(origins)["anchor"]
    B, N = len(origins), data.n_nodes
    out = np.zeros((B, N, len(data.spec.horizons), len(data.target_idx), 3), dtype=np.float32)
    for (hi, ti), model in bundle["models"].items():
        q = np.sort(model.predict(X).reshape(B, N, 3), axis=-1)
        out[:, :, hi, ti, :] = q + anchor[:, :, ti, None]
    return out


# ------------------------------------------------------------------ interval calibration (split conformal)

def conformal_margins(q_val: np.ndarray, y_val: np.ndarray, mask: np.ndarray, coverage: float = INTERVAL_COVERAGE) -> np.ndarray:
    """Conformalised quantile regression margin per (horizon, target), from VAL origins, in scaled units: [H, T].

    Score = how far the truth falls outside [q10, q90] (negative when inside). The margin is the score's
    `coverage` quantile, so widening (or shrinking, if negative) every interval by it gives ~80% coverage.
    """
    H, T = q_val.shape[2], q_val.shape[3]
    margins = np.zeros((H, T))
    for h in range(H):
        for t in range(T):
            lo, hi, y = q_val[:, :, h, t, 0], q_val[:, :, h, t, 2], y_val[:, :, h, t]
            score = np.maximum(lo - y, y - hi)[mask[:, :, h]]
            n = len(score)
            if n == 0:
                continue
            level = min(1.0, np.ceil((n + 1) * coverage) / n)  # finite-sample correction
            margins[h, t] = float(np.quantile(score, level))
    return margins


def apply_margins(q: np.ndarray, margins: np.ndarray) -> np.ndarray:
    """Widen q10/q90 by the margins (never past the median)."""
    out = q.copy()
    out[..., 0] = np.minimum(q[..., 0] - margins[None, None], q[..., 1])
    out[..., 2] = np.maximum(q[..., 2] + margins[None, None], q[..., 1])
    return out


# ------------------------------------------------------------------ output

def long_frame(data: ForecastData, origins: np.ndarray, q_orig: np.ndarray, split: str, model_name: str,
               version: str, decimals: int = 3) -> pd.DataFrame:
    """Turn [B, N, H, T, 3] forecasts (original units) into contract rows: one per node x origin x horizon x target."""
    B, N, H, T, _ = q_orig.shape
    truth = data.truth_original(origins)
    if data.output_transform is not None:
        truth, q_orig = data.output_transform(truth), data.output_transform(q_orig)
    hours = np.asarray(data.spec.horizons) * data.spec.step_hours
    o, n, h, t = np.meshgrid(np.arange(B), np.arange(N), np.arange(H), np.arange(T), indexing="ij")
    stamps = data.times[np.asarray(origins)].strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()
    return pd.DataFrame({
        "node_id": np.asarray(data.node_ids)[n.ravel()],
        "origin_ts": stamps[o.ravel()],
        "horizon_h": hours[h.ravel()],
        "target": np.asarray(data.target_names)[t.ravel()],
        "split": split,
        "y_true": np.round(truth[o, n, h, t].ravel(), decimals),
        "y_pred": np.round(q_orig[..., 1].ravel(), decimals),
        "y_lo": np.round(q_orig[..., 0].ravel(), decimals),
        "y_hi": np.round(q_orig[..., 2].ravel(), decimals),
        "model_name": model_name,
        "model_version": version,
    })
