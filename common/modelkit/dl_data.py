"""ROLE: ML ENGINEER  |  COMMON MODELKIT
Turns a complete node-by-time grid into model-ready forecast batches for Stage 2 (simulated AND real track).

Owner: Role 1 - ML Engineer. Written by Role 4 at the team lead's request; Role 1 to review.

The idea in plain words: every forecast has an ORIGIN t = "now". The model may see
  * x       the last `input_len` steps of every node (scaled channels + an is-imputed flag + calendar),
  * ctx     a long, coarse memory: the average of each channel over the last `n_blocks` blocks
            (28 daily blocks for the hourly track = 4 weeks), so slow cascades along long sea lanes are visible,
  * anchor  the last observed value of each target (the persistence forecast; models predict the change from it),
and must predict the targets at t + h for every horizon h.
Nothing at or after t + 1 is ever an input. tests/ml/test_dl_data.py checks this by changing the future and
proving the inputs stay identical.

Splits follow common/dataio/windows.py exactly (same rule, same dates): a window is train only if its last
target is before train_end; val/test only if its first input step is on/after val_start/test_start.

READS : nothing by itself (callers pass DataFrames and the scaler dict written by Stage02_DL/01_data_engineer.py)
WRITES: nothing
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.dataio.hourly_grid import CHANNELS
from common.seed import SEED


@dataclass(frozen=True)
class TrackSpec:
    """Everything that differs between the hourly simulated track and the daily real track."""

    name: str                 # "sim" or "real"
    step_hours: int           # 1 (hourly grid) or 24 (daily grid)
    input_len: int            # steps of detailed history the model reads
    horizons: tuple           # steps ahead to forecast
    block_len: int            # steps per long-context block
    n_blocks: int             # how many blocks of long context
    seasonal_period: int      # steps in one week (seasonal-naive baseline)
    rolling_len: int          # steps in the rolling-mean baseline
    channels: tuple           # scaled input channels, in order
    targets: tuple            # (channel, contract target name) pairs


# Hourly simulated track: 72 h input, 1/6/24/72/168 h horizons (role file, Day 2), 4 weeks of daily context.
SIM_TRACK = TrackSpec("sim", 1, 72, (1, 6, 24, 72, 168), 24, 28, 168, 24, CHANNELS,
                      (("avg_dwell_time_hours", "dwell"), ("throughput_teu_per_hr", "throughput")))
# Daily real track: 28 days input, 1/3/7 day horizons, 8 weekly blocks of context (= 8 weeks).
REAL_TRACK = TrackSpec("real", 24, 28, (1, 3, 7), 7, 8, 7, 7, ("log_activity",), (("log_activity", "activity"),))


def calendar_features(times: pd.DatetimeIndex, step_hours: int) -> np.ndarray:
    """Sine/cosine clock features [T, 4]: hour of day + day of week (hourly) or day of week + day of year (daily).

    Sine and cosine are used so 23:00 and 00:00 are close numbers, not 23 apart.
    """
    if step_hours == 1:
        a = 2 * np.pi * times.hour.to_numpy() / 24
    else:
        a = 2 * np.pi * (times.day_of_year.to_numpy() - 1) / 365.25
    b = 2 * np.pi * times.day_of_week.to_numpy() / 7
    return np.stack([np.sin(a), np.cos(a), np.sin(b), np.cos(b)], axis=1).astype(np.float32)


def split_origins(times: pd.DatetimeIndex, spec: TrackSpec, boundaries: dict, stride: int = 1,
                  origin_hours: tuple | None = None) -> dict:
    """Return {"train"/"val"/"test": array of origin indices} using the windows.py rule.

    `origin_hours` (e.g. (0, 6, 12, 18)) keeps only origins at those clock hours, so the same forecast times are
    used every run (role file: test origins every 6 h).
    """
    to_utc = lambda v: pd.Timestamp(v).tz_localize("UTC") if pd.Timestamp(v).tzinfo is None else pd.Timestamp(v).tz_convert("UTC")
    t_utc = times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
    b = {k: to_utc(v) for k, v in boundaries.items() if k.endswith(("_end", "_start"))}
    max_h = max(spec.horizons)
    origins = np.arange(spec.input_len - 1, len(times) - max_h, stride)
    if origin_hours is not None:
        origins = origins[np.isin(t_utc[origins].hour, origin_hours)]
    first_in = t_utc[origins - spec.input_len + 1]
    last_target = t_utc[origins + max_h]
    out = {
        "train": origins[last_target < b["train_end"]],
        "val": origins[(first_in >= b["val_start"]) & (last_target < b["val_end"])],
        "test": origins[first_in >= b["test_start"]],
    }
    return out


class ForecastData:
    """A scaled [time, node, channel] cube plus everything needed to cut forecast batches out of it."""

    def __init__(self, values: np.ndarray, imputed: np.ndarray, times: pd.DatetimeIndex, node_ids: list,
                 spec: TrackSpec, scaler: dict, noise_channels: bool = False):
        """values: RAW (unscaled) [T, N, len(spec.channels)]; imputed: [T, N] booleans; scaler: robust scaler dict."""
        self.spec, self.node_ids, self.times = spec, list(node_ids), times
        self.raw = values.astype(np.float64)
        # Robust per-node scaling with the TRAIN-ONLY numbers written by the data engineer.
        med = np.array([[scaler["nodes"][n][c]["median"] for c in spec.channels] for n in node_ids])
        iqr = np.array([[scaler["nodes"][n][c]["iqr"] for c in spec.channels] for n in node_ids])
        self.med, self.iqr = med, iqr
        scaled = ((self.raw - med[None]) / iqr[None]).astype(np.float32)
        self.target_idx = [spec.channels.index(ch) for ch, _ in spec.targets]
        self.target_names = [name for _, name in spec.targets]
        self.scaled = scaled
        self.imputed = imputed.astype(np.float32)
        self.calendar = calendar_features(times, spec.step_hours)
        self.input_channel_names = list(spec.channels) + ["is_imputed", "cal_a_sin", "cal_a_cos", "dow_sin", "dow_cos"]
        # Optional function from model units back to reported units (the real track predicts log(1 + count)).
        self.output_transform = None

        # Optional sanity channels for Role 2's attribution test: pure noise and a time-shuffled copy of the
        # first channel. A healthy model gives them ~zero attribution because they carry no real signal.
        self.extra = None
        if noise_channels:
            rng = np.random.default_rng(SEED)
            noise = rng.standard_normal(scaled.shape[:2]).astype(np.float32)
            shuffled = scaled[rng.permutation(len(times)), :, 0]
            self.extra = np.stack([noise, shuffled], axis=-1)
            self.input_channel_names += ["noise_random", "noise_time_shuffled_ch0"]

        # Cumulative sums let us average any block of time in one subtraction (fast long-context features).
        self._cumsum = np.concatenate([np.zeros((1, *scaled.shape[1:]), dtype=np.float64),
                                       np.cumsum(scaled, axis=0, dtype=np.float64)])

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the grid."""
        return len(self.node_ids)

    @property
    def n_inputs(self) -> int:
        """Number of channels per step in x (scaled channels + imputed flag + 4 calendar + optional noise)."""
        return len(self.input_channel_names)

    def batch(self, origins: np.ndarray) -> dict:
        """Cut the inputs (and, when they exist, the true targets) for a list of origin indices."""
        spec = self.spec
        origins = np.asarray(origins)
        steps = origins[:, None] + np.arange(-spec.input_len + 1, 1)[None, :]          # [B, L] past steps only
        x = self.scaled[steps]                                                           # [B, L, N, C]
        parts = [x, self.imputed[steps][..., None],
                 np.broadcast_to(self.calendar[steps][:, :, None, :], (*x.shape[:3], 4))]
        if self.extra is not None:
            parts.append(self.extra[steps])
        x = np.concatenate(parts, axis=-1).transpose(0, 2, 1, 3)                         # [B, N, L, C_in]

        # Long context: block k covers the steps (t - (k+1)*len, t - k*len]; k = 0 is the most recent block.
        ends = origins[:, None] + 1 - np.arange(spec.n_blocks)[None, :] * spec.block_len  # [B, D]
        starts = np.clip(ends - spec.block_len, 0, None)
        ends = np.clip(ends, 1, None)
        starts = np.minimum(starts, ends - 1)  # the earliest origins reuse the first available step (padding)
        sums = self._cumsum[ends] - self._cumsum[starts]
        ctx = (sums / (ends - starts)[..., None, None]).astype(np.float32).transpose(0, 2, 1, 3)  # [B, N, D, C]

        anchor = self.scaled[origins][:, :, self.target_idx]                            # [B, N, n_targets]
        out = {"x": np.ascontiguousarray(x, dtype=np.float32), "ctx": np.ascontiguousarray(ctx),
               "anchor": np.ascontiguousarray(anchor), "origins": origins}
        future = origins[:, None] + np.asarray(spec.horizons)[None, :]                  # [B, H]
        if future.max() < len(self.times):
            out["y"] = self.scaled[future][..., self.target_idx].transpose(0, 2, 1, 3).copy()  # [B, N, H, n_t]
        return out

    def to_original(self, scaled_targets: np.ndarray) -> np.ndarray:
        """Undo the scaling for target values shaped [..., N, H, n_targets] (or [..., N, H, n_targets, Q])."""
        med = self.med[:, self.target_idx]   # [N, n_targets]
        iqr = self.iqr[:, self.target_idx]
        if scaled_targets.ndim >= 5:         # trailing quantile axis
            return scaled_targets * iqr[:, None, :, None] + med[:, None, :, None]
        return scaled_targets * iqr[:, None, :] + med[:, None, :]

    def truth_original(self, origins: np.ndarray) -> np.ndarray:
        """True target values in original units, [B, N, H, n_targets]."""
        future = np.asarray(origins)[:, None] + np.asarray(self.spec.horizons)[None, :]
        return self.raw[future][..., self.target_idx].transpose(0, 2, 1, 3)

    def target_imputed(self, origins: np.ndarray) -> np.ndarray:
        """True where the TARGET hour was filled in rather than measured, [B, N, H] (evaluation drops these)."""
        future = np.asarray(origins)[:, None] + np.asarray(self.spec.horizons)[None, :]
        return self.imputed[future].transpose(0, 2, 1).astype(bool)


def sim_data_from_grid(grid: pd.DataFrame, node_ids: list, scaler: dict, noise_channels: bool = False,
                       spec: TrackSpec | None = None) -> ForecastData:
    """Build the hourly simulated-track ForecastData from hourly_node_telemetry.csv[.gz].

    `spec` allows overriding the default TrackSpec (e.g. for input_len ablation studies).
    """
    df = grid.copy()
    df["hour_ts"] = pd.to_datetime(df["hour_ts"], utc=True)
    # The flag may arrive as True/False booleans or as "True"/"False" text, depending on how the CSV was read.
    df["is_imputed"] = df["is_imputed"].astype(str).str.lower().eq("true")
    wide = df.pivot(index="hour_ts", columns="node_id")
    times = wide.index
    if len(times) != len(pd.date_range(times[0], times[-1], freq="h")) or wide[list(CHANNELS)].isna().any().any():
        raise ValueError("The hourly grid has gaps; run Stage02_DL/01_data_engineer.py first.")
    values = np.stack([wide[c][node_ids].to_numpy(dtype=np.float64) for c in CHANNELS], axis=-1)
    imputed = wide["is_imputed"][node_ids].to_numpy(dtype=bool)
    track_spec = SIM_TRACK if spec is None else spec
    return ForecastData(values, imputed, times, node_ids, track_spec, scaler, noise_channels)


def real_data_from_grid(grid: pd.DataFrame, scaler: dict) -> ForecastData:
    """Build the daily real-track ForecastData from real_daily_activity.csv[.gz] (log(1 + count) channel)."""
    df = grid.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    df["log_activity"] = np.log1p(df["activity"].clip(lower=0))
    df["is_imputed"] = df["is_imputed"].astype(str).str.lower().eq("true")
    node_ids = [n for n in df["node_id"].drop_duplicates() if n in scaler["nodes"]]
    wide = df.pivot(index="date", columns="node_id")
    values = wide["log_activity"][node_ids].to_numpy(dtype=np.float64)[..., None]
    imputed = wide["is_imputed"][node_ids].to_numpy(dtype=bool)
    data = ForecastData(values, imputed, wide.index, node_ids, REAL_TRACK, scaler)
    data.output_transform = np.expm1  # report ship counts per day, not log counts
    return data


def transit_edges(node_ids: list, edges: pd.DataFrame, spec: TrackSpec, permutation: np.ndarray | None = None) -> dict:
    """Directed lanes as index arrays for the transit-delayed messages of the graph model.

    For lane src -> dst taking L steps and horizon h, the congestion that will ARRIVE at dst at t + h left src at
    t + h - L. If that moment is already in the past (L >= h), the model reads src's long-context block that
    contains it; otherwise it reads src's most recent block and a `future` flag says "this is a nowcast".
    `permutation` relabels the nodes (graph_shuffled ablation): same lanes, same delays, wrong partners.
    """
    index = {n: i for i, n in enumerate(node_ids)}
    src, dst, lag = [], [], []
    for e in edges.itertuples():
        if e.src not in index or e.dst not in index:
            continue  # real track: lanes touching nodes without real data are skipped
        steps = max(1, int(round(float(e.typical_transit_days) * 24 / spec.step_hours)))
        pairs = [(index[e.src], index[e.dst])]
        if str(e.bidirectional).lower() == "true":
            pairs.append((index[e.dst], index[e.src]))
        for s, d in pairs:
            src.append(s), dst.append(d), lag.append(steps)
    src, dst, lag = np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64), np.array(lag, dtype=np.int64)
    if permutation is not None:
        src, dst = permutation[src], permutation[dst]
    back = lag[:, None] - np.asarray(spec.horizons)[None, :]                  # [E, H] steps into the past
    block = np.clip(np.where(back >= 0, back // spec.block_len, 0), 0, spec.n_blocks - 1)
    return {"src": src, "dst": dst, "lag_steps": lag, "block": block.astype(np.int64),
            "future": (back < 0).astype(np.float32)}


def symmetric_adjacency(node_ids: list, edges: pd.DataFrame, permutation: np.ndarray | None = None,
                        identity: bool = False) -> np.ndarray:
    """D^-1/2 (A + I) D^-1/2 over the lanes between `node_ids` (identity matrix for the graph_none ablation)."""
    n = len(node_ids)
    if identity:
        return np.eye(n, dtype=np.float32)
    index = {node: i for i, node in enumerate(node_ids)}
    adj = np.zeros((n, n))
    for e in edges.itertuples():
        if e.src in index and e.dst in index:
            i, j = index[e.src], index[e.dst]
            if permutation is not None:
                i, j = permutation[i], permutation[j]
            adj[i, j] = adj[j, i] = 1.0
    adj += np.eye(n)
    inv_sqrt = 1.0 / np.sqrt(adj.sum(axis=1))
    return (inv_sqrt[:, None] * adj * inv_sqrt[None, :]).astype(np.float32)


def shuffle_permutation(n: int, seed: int = SEED) -> np.ndarray:
    """A node relabelling with no node mapped to itself, so every shuffled lane really is a wrong partner."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
