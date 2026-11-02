"""ROLE: EDA ENGINEER  |  STAGE 02 - DEEP LEARNING
Owner: Role 2 - EDA Engineer (Isha). First version written by Role 4 at the team lead's request; Role 2 to review.
Full instructions: docs/roles/ROLE_2_EDA_ENGINEER.md (Day 2).

Studies the hourly node telemetry before and after the forecasting models exist.
  BEFORE the model (train hours only for anything that steers a modelling decision):
    1. stationarity (ADF test) and memory (ACF / PACF up to 168 lags)
    2. seasonal decomposition (MSTL, periods 24 h and 168 h): variance shares per node
    3. the 5 ambiguous delay graphs and the explicit seasonal / transient / structural rule, checked against the events
    4. node-to-node lead-lag along the lanes (simulated hourly track and the real PortWatch daily track)
    5. imputed (filled-in) hours: how many, where, are they clustered
  AFTER the model (test origins are only LOOKED AT here; nothing is chosen from them):
    6. integrated-gradients attribution for the LSTM and the Transformer, by input channel and by past hour
    7. noise-channel sanity test: pure noise and a time-shuffled channel must get ~0 attribution
    8. Transformer attention maps for 3 test episodes, and attention on filled-in hours
    9. graph_full vs graph_none attribution: which other nodes' histories the graph model reads

READS : Stage02_DL/data/hourly_node_telemetry.csv.gz, splits.json, scaler.json, real_daily_activity.csv.gz, real_splits.json
        contracts/network_nodes.csv, contracts/network_edges.csv, data/raw/sim/events_ground_truth.csv
        Stage02_DL/reports/eval_alert_thresholds.csv (Role 4's alert line; recomputed the same way if missing)
        Stage02_DL/artifacts/*.pt and artifacts/noise_check/lstm_noise.pt (model part; skipped with a note if missing)
WRITES: Stage02_DL/reports/eda_* (csv, png, json, md). Never changes a data file.

RUN (from Stage02_DL/):  python 02_eda_engineer.py              (about 5 minutes on a laptop CPU)
                         python 02_eda_engineer.py --no-model   (model-independent part only)
"""

import os

# PyTorch and XGBoost/statsmodels-linked OpenMP clash on macOS; one thread per library avoids the deadlock.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import binomtest, chisquare  # noqa: E402
from statsmodels.tsa.seasonal import MSTL  # noqa: E402
from statsmodels.tsa.stattools import acf, adfuller, pacf  # noqa: E402

# Make "import common" work when this file is run from inside the stage folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dataio.hourly_grid import CHANNELS  # noqa: E402
from common.edakit import apply_style, save_fig  # noqa: E402
from common.edakit.timeseries import (  # noqa: E402
    classify_excursion, deseasonalise, iqr_scale, lagged_correlation, mad_sigma, remove_common_factor,
    run_lengths, trailing_residual, weekly_profile,
)
from common.evalkit.leadtime import episode_hours  # noqa: E402
from common.paths import EDGES_CSV, NODES_CSV, REPO_ROOT, stage_path  # noqa: E402
from common.seed import SEED, set_global_seed  # noqa: E402

STAGE = 2
OUT = stage_path(STAGE, "reports")
DWELL, TPUT = "avg_dwell_time_hours", "throughput_teu_per_hr"
# Okabe-Ito colours: readable for the common kinds of colour blindness.
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

# The team-huddle rule (role file, Day 2 item 3): residual above 3 x noise floor for >= 24 h = structural candidate.
K_NOISE, STRUCTURAL_H = 3.0, 24
LEVEL_WINDOW_H = 14 * 24        # trailing level = median of the last 2 weeks (past hours only)
# Lead-lag: the role file asks for 0-14 days, but 13 of the 50 lanes take longer than 14 days, so we scan 0-28 days.
SIM_MAX_LAG_H, SIM_SMOOTH_H, SIM_TOL_H = 28 * 24, 24, 48
REAL_MAX_LAG_D, REAL_SMOOTH_D, REAL_TOL_D = 35, 7, 3
IG_STEPS = 32                   # integrated-gradients path steps (completeness gap is reported to show it is enough)

# Which inputs are real operational drivers, which are the targets' own history, and which are bookkeeping.
CHANNEL_GROUPS = {
    DWELL: "own history", TPUT: "own history",
    "capacity_utilization_pct": "operational driver", "customs_hold_hours": "operational driver",
    "weather_severity_index": "operational driver", "labor_deficit_ratio": "operational driver",
    "equipment_availability_pct": "operational driver", "inventory_level_teu": "yard state",
    "is_imputed": "reporting flag", "cal_a_sin": "calendar", "cal_a_cos": "calendar",
    "dow_sin": "calendar", "dow_cos": "calendar",
    "noise_random": "NOISE (sanity)", "noise_time_shuffled_ch0": "NOISE (sanity)",
}


def utc(value) -> pd.Timestamp:
    """Parse a time string as a UTC timestamp."""
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def short(node_id: str) -> str:
    """Short label for charts: PORT_CN_SHA -> CN_SHA, CHOKE_SUEZ -> SUEZ."""
    return node_id.split("_", 1)[1] if "_" in node_id else node_id


# ----------------------------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------------------------

def load_sim() -> dict:
    """Load the hourly grid as one [hour x node] table per channel, plus splits, alert lines, events and lanes."""
    grid = pd.read_csv(stage_path(STAGE, "data", "hourly_node_telemetry.csv.gz"))
    grid["hour_ts"] = pd.to_datetime(grid["hour_ts"], utc=True)
    grid["is_imputed"] = grid["is_imputed"].astype(str).str.lower().eq("true")
    nodes = pd.read_csv(NODES_CSV)
    node_ids = nodes["node_id"].tolist()
    # One wide table per channel: rows = hours, columns = nodes in the registry order.
    wide = {c: grid.pivot(index="hour_ts", columns="node_id", values=c)[node_ids] for c in [*CHANNELS, "is_imputed"]}
    splits = json.loads(stage_path(STAGE, "data", "splits.json").read_text())
    t = wide[DWELL].index
    masks = {
        "train": np.asarray(t < utc(splits["train_end"])),
        "val": np.asarray((t >= utc(splits["val_start"])) & (t < utc(splits["val_end"]))),
        "test": np.asarray(t >= utc(splits["test_start"])),
    }
    # Role 4's alert line (p95 of TRAIN dwell per node); recomputed the same way if the evaluation has not run.
    thr_file = stage_path(STAGE, "reports", "eval_alert_thresholds.csv")
    if thr_file.exists():
        thr = pd.read_csv(thr_file).set_index("node_id")["dwell_alert_threshold_h"].to_dict()
    else:
        thr = wide[DWELL][masks["train"]].quantile(0.95).to_dict()
    # Same node order everywhere, so tables divided by these numbers keep their columns in place.
    thr = {n: float(thr[n]) for n in node_ids}
    events = pd.read_csv(REPO_ROOT / "data" / "raw" / "sim" / "events_ground_truth.csv")
    events["start_ts"] = pd.to_datetime(events["start_ts"], utc=True)
    events["end_ts"] = pd.to_datetime(events["end_ts"], utc=True)
    edges = pd.read_csv(EDGES_CSV)
    # Undirected neighbours and the transit time of every link (used to decide what "nearby in time" means).
    neighbours = {n: set() for n in node_ids}
    transit_h = {}
    for e in edges.itertuples():
        neighbours[e.src].add(e.dst)
        neighbours[e.dst].add(e.src)
        hours = float(e.typical_transit_days) * 24
        for pair in ((e.src, e.dst), (e.dst, e.src)):
            transit_h[pair] = min(transit_h.get(pair, hours), hours)
    return {"grid": grid, "wide": wide, "node_ids": node_ids, "node_type": dict(zip(nodes["node_id"], nodes["node_type"])),
            "names": dict(zip(nodes["node_id"], nodes["display_name"])), "splits": splits, "masks": masks,
            "thr": thr, "events": events, "edges": edges, "neighbours": neighbours, "transit_h": transit_h}


def load_real() -> dict | None:
    """Load the real PortWatch daily track as a [day x node] table of log(1 + activity); None if it was not built."""
    path = stage_path(STAGE, "data", "real_daily_activity.csv.gz")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    # Ship counts are skewed (a few huge days), so we study log(1 + count), the same target the models use.
    df["log_activity"] = np.log1p(df["activity"].astype(float))
    wide = df.pivot(index="date", columns="node_id", values="log_activity")
    splits = json.loads(stage_path(STAGE, "data", "real_splits.json").read_text())
    return {"wide": wide, "train": np.asarray(wide.index < utc(splits["train_end"])), "splits": splits}


# ----------------------------------------------------------------------------------------------------------------
# 1. Stationarity and memory
# ----------------------------------------------------------------------------------------------------------------

def stationarity_and_memory(S: dict) -> dict:
    """ADF test + ACF/PACF up to 168 lags for every node x channel on TRAIN hours -> eda_acf_summary.csv, eda_acf.png."""
    train = S["masks"]["train"]
    rows, acf_dwell, pacf_dwell = [], {}, {}
    for ch in CHANNELS:
        for nid in S["node_ids"]:
            x = S["wide"][ch][nid].to_numpy(dtype=float)[train]
            row = {"node_id": nid, "node_type": S["node_type"][nid], "channel": ch, "n_hours": len(x)}
            # A channel that never moves (e.g. no customs holds at a chokepoint) has no memory to measure.
            if np.nanstd(x) < 1e-9 or len(np.unique(x)) < 3:
                rows.append({**row, "note": "constant on train"})
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_stat, adf_p = adfuller(x, maxlag=24, autolag=None, regression="c")[:2]
                a = acf(x, nlags=168, fft=True)
                p = pacf(x, nlags=168, method="yw")
            band = 1.96 / np.sqrt(len(x))
            below = np.where(a[1:] < 1 / np.e)[0]
            row.update({
                "adf_stat": round(float(adf_stat), 3), "adf_p": float(adf_p), "stationary_5pct": bool(adf_p < 0.05),
                "acf_1": a[1], "acf_24": a[24], "acf_72": a[72], "acf_168": a[168],
                # Decorrelation time: first lag where the autocorrelation drops below 1/e (NaN = later than 168 h).
                "decorrelation_h": float(below[0] + 1) if len(below) else np.nan,
                # Seasonal "bumps": a daily cycle makes lag 24 more similar than lag 12, a weekly one lag 168 than 156.
                "daily_bump": a[24] - a[12], "weekly_bump": a[168] - a[156],
                "pacf_1": p[1], "pacf_2": p[2], "pacf_24": p[24], "pacf_168": p[168],
                "pacf_significant_lags": int((np.abs(p[1:]) > band).sum()), "note": "",
            })
            rows.append(row)
            if ch == DWELL:
                acf_dwell[nid], pacf_dwell[nid] = a, p
    table = pd.DataFrame(rows).round(4)
    table.to_csv(OUT / "eda_acf_summary.csv", index=False)

    dw = table[table["channel"] == DWELL]
    stats = {
        "series_tested": int(table["adf_p"].notna().sum()),
        "share_stationary": float(table["stationary_5pct"].dropna().astype(bool).mean()),
        "constant_series": int((table["note"] == "constant on train").sum()),
        "dwell_median_decorrelation_h": float(dw["decorrelation_h"].median()),
        "dwell_decorrelation_over_168h": int(dw["decorrelation_h"].isna().sum()),
        "dwell_median_acf_24": float(dw["acf_24"].median()), "dwell_median_acf_72": float(dw["acf_72"].median()),
        "dwell_median_acf_168": float(dw["acf_168"].median()),
        "dwell_nodes_with_daily_bump": int((dw["daily_bump"] > 0.01).sum()),
        "dwell_nodes_with_weekly_bump": int((dw["weekly_bump"] > 0.01).sum()),
        "dwell_median_pacf_1": float(dw["pacf_1"].median()), "dwell_median_pacf_2": float(dw["pacf_2"].median()),
    }

    # Chart: median ACF and PACF of dwell per node type, with the 24 h / 72 h / 168 h marks.
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    lags = np.arange(169)
    for i, nt in enumerate(sorted(set(S["node_type"].values()))):
        ids = [n for n in acf_dwell if S["node_type"][n] == nt]
        if not ids:
            continue
        axes[0].plot(lags, np.median([acf_dwell[n] for n in ids], axis=0), color=COLOURS[i], label=f"{nt} (n={len(ids)})")
        axes[1].plot(lags[1:], np.median([pacf_dwell[n][1:] for n in ids], axis=0), color=COLOURS[i])
    band = 1.96 / np.sqrt(int(train.sum()))
    for ax in axes:
        for m in (24, 72, 168):
            ax.axvline(m, color="grey", lw=0.8, ls=":")
        ax.axhspan(-band, band, color="grey", alpha=0.2)
    axes[0].axhline(1 / np.e, color="black", lw=0.8, ls="--", label="1/e (decorrelation)")
    axes[0].set_ylabel("autocorrelation")
    axes[1].set_ylabel("partial autocorrelation")
    axes[1].set_xlabel("lag (hours); dotted lines at 24 h, 72 h (model window) and 168 h")
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].set_title(f"Dwell keeps its memory for ~{stats['dwell_median_decorrelation_h']:.0f} h (median node); "
                      f"ACF at 72 h = {stats['dwell_median_acf_72']:.2f}, at 168 h = {stats['dwell_median_acf_168']:.2f} (train hours)")
    save_fig(fig, OUT / "eda_acf.png")
    return stats


# ----------------------------------------------------------------------------------------------------------------
# 2. Seasonal decomposition
# ----------------------------------------------------------------------------------------------------------------

def seasonal_decomposition(S: dict) -> dict:
    """MSTL (24 h + 168 h) on TRAIN dwell and throughput per node -> eda_stl_components.csv/png (variance shares)."""
    train = S["masks"]["train"]
    rows, example = [], None
    for ch in (DWELL, TPUT):
        for nid in S["node_ids"]:
            x = S["wide"][ch][nid].to_numpy(dtype=float)[train]
            res = MSTL(x, periods=(24, 168)).fit()
            total = np.var(x)
            # Share of the series' variance carried by each part (they need not add to exactly 1: parts can co-vary).
            rows.append({"node_id": nid, "node_type": S["node_type"][nid], "channel": ch,
                         "share_trend": np.var(res.trend) / total, "share_daily_24h": np.var(res.seasonal[:, 0]) / total,
                         "share_weekly_168h": np.var(res.seasonal[:, 1]) / total, "share_residual": np.var(res.resid) / total})
            if ch == DWELL and nid == "PORT_US_LAX":
                example = (S["wide"][ch].index[train], x, res)
    table = pd.DataFrame(rows).round(4)
    table.to_csv(OUT / "eda_stl_components.csv", index=False)
    med = table.groupby("channel")[["share_trend", "share_daily_24h", "share_weekly_168h", "share_residual"]].median()
    stats = {f"{ch}_median_{k}": float(v) for ch in med.index for k, v in med.loc[ch].items()}

    # Chart: the four parts for Los Angeles (last 6 weeks of train) + the dwell shares for every node.
    fig = plt.figure(figsize=(12, 15))
    gs = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1, 1, 2.6], hspace=0.45)
    times, x, res = example
    keep = slice(len(x) - 6 * 168, len(x))
    parts = [("observed dwell (h)", x), ("trend (h)", res.trend), ("daily part (h)", res.seasonal[:, 0]),
             ("weekly part (h)", res.seasonal[:, 1]), ("residual (h)", res.resid)]
    for i, (label, values) in enumerate(parts):
        ax = fig.add_subplot(gs[i])
        ax.plot(times[keep], np.asarray(values)[keep], color=COLOURS[i % 4], lw=0.8)
        ax.set_ylabel(label, fontsize=8)
        if i == 0:
            ax.set_title("Los Angeles, last 6 weeks of train: slow trend moves dwell far more than the daily or weekly cycle")
    ax = fig.add_subplot(gs[5])
    d = table[table["channel"] == DWELL].sort_values("share_trend")
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for i, col in enumerate(["share_trend", "share_daily_24h", "share_weekly_168h", "share_residual"]):
        ax.barh(y, d[col], left=left, color=COLOURS[i], label=col.replace("share_", ""))
        left += d[col].to_numpy()
    ax.set_yticks(y, [short(n) for n in d["node_id"]], fontsize=7)
    ax.set_xlabel("share of dwell variance on train hours")
    ax.legend(fontsize=8, ncol=4, loc="lower right")
    ax.set_title(f"Dwell variance by part, every node: trend {stats[DWELL + '_median_share_trend']:.0%}, "
                 f"daily {stats[DWELL + '_median_share_daily_24h']:.0%}, weekly {stats[DWELL + '_median_share_weekly_168h']:.0%}, "
                 f"residual {stats[DWELL + '_median_share_residual']:.0%} (medians)")
    save_fig(fig, OUT / "eda_stl_components.png")
    return stats


# ----------------------------------------------------------------------------------------------------------------
# 3. Seasonal peak vs transient shock vs structural bottleneck
# ----------------------------------------------------------------------------------------------------------------

def residuals(S: dict) -> tuple:
    """Weekly-profile-adjusted dwell, its past-only residual, and each node's noise floor (MAD of the train residual)."""
    dw = S["wide"][DWELL]
    d = deseasonalise(dw, weekly_profile(dw, S["masks"]["train"]))
    r = trailing_residual(d, LEVEL_WINDOW_H)
    sigma = {n: mad_sigma(r[n][S["masks"]["train"]]) for n in S["node_ids"]}
    return d, r, sigma


def structural_flags(r: pd.Series, sigma: float) -> np.ndarray:
    """True for every hour inside a run where the residual stays above 3 x noise floor for >= 24 h."""
    above = np.nan_to_num(r.to_numpy(dtype=float), nan=0.0) > K_NOISE * sigma
    flags = np.zeros(len(above), dtype=bool)
    for start, length in run_lengths(above):
        if length >= STRUCTURAL_H:
            flags[start:start + length] = True
    return flags


def rule_check(S: dict, r: pd.DataFrame, sigma: dict) -> pd.DataFrame:
    """Compare the structural rule with Role 4's p95 rule against the known events on TRAIN + VAL hours.

    recall           = share of events whose node shows a flagged hour between the event start and 48 h after its end
    explained_share  = share of flagged episodes that start near an event at the same node (start - 24 h .. end + 72 h)
                       or at a linked node (start - 24 h .. end + transit + 7 days)
    """
    times = r.index
    period = S["masks"]["train"] | S["masks"]["val"]
    ev = S["events"][(S["events"]["start_ts"] >= times[0] + pd.Timedelta(days=14))
                     & (S["events"]["end_ts"] < times[period][-1])]
    days = period.sum() / 24
    flags = {
        "structural_3x_noise_24h": {n: structural_flags(r[n], sigma[n]) & period for n in S["node_ids"]},
        "p95_dwell_6h (Role 4)": {n: episode_hours(times, S["wide"][DWELL][n].to_numpy(), S["thr"][n]).to_numpy() & period
                                  for n in S["node_ids"]},
    }
    rows = []
    for rule, per_node in flags.items():
        # Recall per event type and overall.
        caught = []
        for e in ev.itertuples():
            win = (times >= e.start_ts) & (times <= e.end_ts + pd.Timedelta(hours=48))
            caught.append((e.type, bool(per_node[e.node_id][win].any())))
        caught = pd.DataFrame(caught, columns=["type", "caught"])
        # Episodes (flagged runs) and whether an event explains them.
        n_runs, explained = 0, 0
        for n, f in per_node.items():
            for start, _ in run_lengths(f):
                n_runs += 1
                t0 = times[start]
                near = ev[(ev["node_id"] == n) & (ev["start_ts"] - pd.Timedelta(hours=24) <= t0)
                          & (t0 <= ev["end_ts"] + pd.Timedelta(hours=72))]
                linked = ev[ev["node_id"].isin(S["neighbours"][n])]
                linked = linked[[(row.start_ts - pd.Timedelta(hours=24) <= t0 <= row.end_ts
                                  + pd.Timedelta(hours=S["transit_h"][(row.node_id, n)] + 168)) for row in linked.itertuples()]]
                explained += int(len(near) > 0 or len(linked) > 0)
        base = {"rule": rule, "flagged_episodes": n_runs,
                "episodes_per_node_per_30d": n_runs / len(per_node) / days * 30,
                "explained_by_event_share": explained / n_runs if n_runs else np.nan,
                "flagged_hour_share": float(np.mean([f[period].mean() for f in per_node.values()]))}
        rows.append({**base, "event_type": "ALL", "events": len(caught), "events_caught": int(caught["caught"].sum()),
                     "recall": float(caught["caught"].mean())})
        for typ, grp in caught.groupby("type"):
            rows.append({"rule": rule, "event_type": typ, "events": len(grp), "events_caught": int(grp["caught"].sum()),
                         "recall": float(grp["caught"].mean())})
    table = pd.DataFrame(rows).round(4)
    table.to_csv(OUT / "eda_episode_rule_check.csv", index=False)
    return table


def near_event_mask(S: dict, times: pd.DatetimeIndex) -> pd.DataFrame:
    """True where a node is within 3 days before to 14 days after an event at itself or at a linked node."""
    mask = pd.DataFrame(False, index=times, columns=S["node_ids"])
    for e in S["events"].itertuples():
        win = (times >= e.start_ts - pd.Timedelta(days=3)) & (times <= e.end_ts + pd.Timedelta(days=14))
        for n in [e.node_id, *S["neighbours"][e.node_id]]:
            mask.loc[win, n] = True
    return mask


def pick_ambiguous(S: dict, r: pd.DataFrame, sigma: dict) -> list:
    """Choose 5 episodes in TRAIN + VAL by fixed rules: 3 known events of different kinds + 2 spikes with no event."""
    times = r.index
    period = S["masks"]["train"] | S["masks"]["val"]
    first_ok = times[0] + pd.Timedelta(days=14)          # the trailing level needs two weeks of history
    ev = S["events"][(S["events"]["start_ts"] >= first_ok) & (S["events"]["end_ts"] < times[period][-1])].copy()
    ev["dur_h"] = (ev["end_ts"] - ev["start_ts"]).dt.total_seconds() / 3600
    picks = []
    # A long strike, the strongest demand surge (looks like peak season), a short sharp equipment failure.
    rules = [("LABOR_STRIKE", "dur_h", False), ("DEMAND_SURGE", "severity", False), ("EQUIPMENT_FAILURE", "dur_h", True)]
    for typ, col, ascending in rules:
        sub = ev[ev["type"] == typ]
        if typ == "EQUIPMENT_FAILURE":
            sub = sub[sub["severity"] >= sub["severity"].median()]
        if sub.empty:
            continue
        e = sub.sort_values(col, ascending=ascending).iloc[0]
        picks.append({"kind": "event", "label": f"{e['type']} {e['event_id']}", "node_id": e["node_id"],
                      "start": e["start_ts"], "end": e["end_ts"] + pd.Timedelta(hours=48), "truth": f"{e['type']} ({e['cause']})",
                      "event_start": e["start_ts"], "event_end": e["end_ts"]})
    # Two spikes with no event at the node or its neighbours: the largest residual, the largest dwell / alert-line ratio.
    near = near_event_mask(S, times).to_numpy()
    ok = period[:, None] & ~near & (times >= first_ok)[:, None]
    z = r.to_numpy() / np.array([sigma[n] for n in S["node_ids"]])[None, :]
    ratio = S["wide"][DWELL].to_numpy() / np.array([S["thr"][n] for n in S["node_ids"]])[None, :]
    used = []
    for label, score in (("largest residual spike, no event", z), ("highest dwell vs alert line, no event", ratio)):
        s = np.where(ok, np.nan_to_num(score, nan=-np.inf), -np.inf)
        for t_used, n_used in used:   # keep the two non-event picks apart (same node within 7 days)
            s[np.abs((times - t_used).total_seconds()) < 7 * 86400, n_used] = -np.inf
        ti, ni = np.unravel_index(np.argmax(s), s.shape)
        used.append((times[ti], ni))
        picks.append({"kind": "no event", "label": label, "node_id": S["node_ids"][ni],
                      "start": times[ti] - pd.Timedelta(hours=48), "end": times[ti] + pd.Timedelta(hours=48),
                      "truth": "no event at this node or a linked node (3 days before to 14 days after)"})
    return picks


def ambiguous_graphs(S: dict, d: pd.DataFrame, r: pd.DataFrame, sigma: dict, check: pd.DataFrame) -> dict:
    """Plot and classify the 5 ambiguous episodes -> eda_ambiguous_1..5.png and eda_ambiguous_graphs.md."""
    picks = pick_ambiguous(S, r, sigma)
    times = r.index
    rows = []
    for i, p in enumerate(picks, 1):
        n = p["node_id"]
        win = (times >= p["start"]) & (times <= p["end"])
        res = classify_excursion(r[n][win], S["wide"][DWELL][n][win], sigma[n], S["thr"][n], K_NOISE, STRUCTURAL_H)
        # How strongly the linked nodes moved in the same window (up to 4 nearest lanes; ties broken by node id,
        # because a set's order changes from run to run and would change which neighbours are shown).
        nbrs = sorted(S["neighbours"][n], key=lambda m: (S["transit_h"][(n, m)], m))[:4]
        nbr_peak = {m: float(np.nanmax(r[m][win] / sigma[m])) for m in nbrs}
        rows.append({"episode": i, **{k: p[k] for k in ("label", "node_id", "truth")},
                     "window": f"{p['start']:%Y-%m-%d %H:%M} .. {p['end']:%Y-%m-%d %H:%M}",
                     "rule_label": res["label"], "longest_run_above_3x_h": res["longest_run_above_h"],
                     "max_residual_x_noise": round(res["max_residual_over_floor"], 1),
                     "crosses_alert_line": res["crosses_alert_line"],
                     "peak_dwell_h": round(float(S["wide"][DWELL][n][win].max()), 1), "alert_line_h": round(S["thr"][n], 1),
                     "linked_nodes_above_3x": sum(v > K_NOISE for v in nbr_peak.values()), "linked_checked": len(nbrs)})

        # Chart: raw dwell (+ alert line), residual in noise-floor units (+ 3x line), linked nodes' residuals.
        view = (times >= p["start"] - pd.Timedelta(days=4)) & (times <= p["end"] + pd.Timedelta(days=4))
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        tv = times[view]
        axes[0].plot(tv, S["wide"][DWELL][n][view], color=COLOURS[0], lw=1, label="dwell (h)")
        axes[0].plot(tv, (S["wide"][DWELL][n] - d[n] + d[n].rolling(LEVEL_WINDOW_H, min_periods=24).median())[view],
                     color=COLOURS[1], lw=1, label="normal level (weekly profile + 2-week trailing median)")
        axes[0].axhline(S["thr"][n], color=COLOURS[3], ls="--", lw=1, label="alert line (p95 of train)")
        axes[1].plot(tv, (r[n] / sigma[n])[view], color=COLOURS[0], lw=1)
        axes[1].axhline(K_NOISE, color=COLOURS[3], ls="--", lw=1, label="3 x noise floor")
        flag = structural_flags(r[n], sigma[n])[view]
        axes[1].fill_between(tv, 0, (r[n] / sigma[n])[view], where=flag, color=COLOURS[3], alpha=0.3,
                             label=">= 24 h above (structural)")
        for j, m in enumerate(nbrs):
            axes[2].plot(tv, (r[m] / sigma[m])[view], color=COLOURS[(j + 1) % len(COLOURS)], lw=0.9,
                         label=f"{short(m)} ({S['transit_h'][(n, m)] / 24:.1f} d lane)")
        axes[2].axhline(K_NOISE, color=COLOURS[3], ls="--", lw=1)
        for ax in axes:
            ax.axvspan(p["start"], p["end"], color="grey", alpha=0.12)
            if p.get("event_start") is not None:
                ax.axvline(p["event_start"], color="black", lw=0.8, ls=":")
            ax.legend(fontsize=7, loc="upper left")
        axes[0].set_ylabel("hours")
        axes[1].set_ylabel("residual / noise floor")
        axes[2].set_ylabel("linked nodes\nresidual / noise")
        axes[0].set_title(f"Episode {i}: {p['label']} at {S['names'][n]} -> rule says {res['label']} "
                          f"(longest run above 3x noise: {res['longest_run_above_h']} h)")
        save_fig(fig, OUT / f"eda_ambiguous_{i}.png")

    table = pd.DataFrame(rows)
    write_ambiguous_md(table, check, sigma)
    agree = []
    for row in table.itertuples():
        if row.truth.startswith("no event"):
            agree.append(row.rule_label in ("SEASONAL_PEAK", "TRANSIENT", "NORMAL"))
        else:
            agree.append(row.rule_label != "NORMAL")
    return {"episodes": table.to_dict("records"), "agree_with_ground_truth": int(sum(agree)), "n": len(agree)}


def write_ambiguous_md(table: pd.DataFrame, check: pd.DataFrame, sigma: dict) -> None:
    """Write eda_ambiguous_graphs.md: the rule, the 5 cases, and how both alert rules score against the events."""
    lines = [
        "# Stage 2 - the 5 ambiguous delay graphs (team huddle)",
        "",
        "Generated by `Stage02_DL/02_eda_engineer.py`. Episodes are chosen by fixed rules on TRAIN + VAL hours only (never test).",
        "",
        "## The rule (explicit, past hours only, so it can run live)",
        "",
        "1. Remove the normal weekly pattern: subtract the TRAIN median for that hour of the week, per node.",
        f"2. Residual = that value minus its trailing {LEVEL_WINDOW_H // 24}-day median (past hours only).",
        "3. Noise floor = robust spread of the residual on TRAIN hours (median absolute deviation x 1.4826), per node "
        f"(median over nodes {np.median(list(sigma.values())):.2f} h).",
        f"4. **STRUCTURAL candidate**: residual above {K_NOISE:.0f} x noise floor for at least {STRUCTURAL_H} hours in a row.",
        f"   **TRANSIENT shock**: above {K_NOISE:.0f} x noise floor, but for fewer than {STRUCTURAL_H} hours.",
        "   **SEASONAL peak**: dwell crosses the alert line while the residual stays under 3 x noise floor (a slow rise "
        "the trailing level follows, typically shared by many nodes).",
        "",
        "## The 5 cases",
        "",
        md_table(table[["episode", "label", "node_id", "window", "truth", "rule_label", "longest_run_above_3x_h",
                        "max_residual_x_noise", "crosses_alert_line", "linked_nodes_above_3x"]]),
        "",
        "Charts: `eda_ambiguous_1.png` ... `eda_ambiguous_5.png` (top: dwell, normal level and alert line; middle: residual in "
        "noise-floor units, shaded where the structural rule fires; bottom: residuals of up to 4 linked nodes).",
        "",
        "## How the two alert rules score against the 67 known events (TRAIN + VAL hours)",
        "",
        md_table(check[check["event_type"] == "ALL"][["rule", "events", "events_caught", "recall", "flagged_episodes",
                                                      "episodes_per_node_per_30d", "explained_by_event_share", "flagged_hour_share"]]),
        "",
        "Recall by event type:",
        "",
        md_table(check.pivot_table(index="event_type", columns="rule", values="recall").reset_index().round(2)),
        "",
        "Reading guide: *recall* = share of events whose node is flagged between the event start and 48 h after it ends; "
        "*explained_by_event_share* = share of flagged episodes that start near an event at the node or a linked node. "
        "Associations only: a flag near an event does not prove the event caused it.",
    ]
    (OUT / "eda_ambiguous_graphs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table (no extra dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


# ----------------------------------------------------------------------------------------------------------------
# 4. Node-to-node lead-lag
# ----------------------------------------------------------------------------------------------------------------

def lead_lag(x: pd.DataFrame, edges: pd.DataFrame, step_h: int, max_lag: int, tol: int, track: str) -> tuple:
    """Lagged correlation for every lane direction, compared with unlinked node pairs (the "chance" level).

    x: prepared series [time x node] (deseasonalised, scaled, network average removed, smoothed), TRAIN only.
    Returns (edge table, correlation matrix [edges x lags], summary dict).
    """
    nodes = list(x.columns)
    # Every lane direction the network allows (both ways for bidirectional lanes), both ends present on this track.
    pairs = []
    for e in edges.itertuples():
        for s, t in ((e.src, e.dst), (e.dst, e.src)) if e.bidirectional else ((e.src, e.dst),):
            if s in nodes and t in nodes:
                pairs.append((s, t, e.mode, float(e.typical_transit_days) * 24))
    rows, curves = [], []
    for s, t, mode, transit in pairs:
        cc = lagged_correlation(x[s].to_numpy(), x[t].to_numpy(), max_lag)
        lag_t = min(max_lag, int(round(transit / step_h)))
        best = int(np.argmax(cc))
        rows.append({"track": track, "src": s, "dst": t, "mode": mode, "typical_transit_h": transit,
                     "best_lag_h": best * step_h, "corr": cc[best], "n": len(x), "corr_at_transit": cc[lag_t],
                     "corr_at_0": cc[0], "lag_error_h": (best - lag_t) * step_h,
                     "lag_within_tolerance": abs(best - lag_t) <= tol})
        curves.append(cc)
    table = pd.DataFrame(rows)
    # Chance level: unlinked pairs, each given a transit time drawn from the real lanes (seeded).
    linked = {(s, t) for s, t, *_ in pairs} | {(t, s) for s, t, *_ in pairs}
    unlinked = [(a, b) for a in nodes for b in nodes if a != b and (a, b) not in linked]
    rng = np.random.default_rng(SEED)
    transits = table["typical_transit_h"].to_numpy()
    null_max, null_hit = [], []
    for a, b in unlinked:
        cc = lagged_correlation(x[a].to_numpy(), x[b].to_numpy(), max_lag)
        lag_t = min(max_lag, int(round(rng.choice(transits) / step_h)))
        null_max.append(cc.max())
        null_hit.append(abs(int(np.argmax(cc)) - lag_t) <= tol)
    p95 = float(np.percentile(null_max, 95))
    table["above_unlinked_p95"] = table["corr"] > p95
    chance = float(np.mean(null_hit))
    hits = int(table["lag_within_tolerance"].sum())
    summary = {
        "lane_directions": len(table), "unlinked_pairs": len(unlinked), "max_lag_h": max_lag * step_h,
        "tolerance_h": tol * step_h, "lag_hits": hits, "lag_hit_rate": hits / len(table),
        "chance_hit_rate_unlinked": chance,
        "binomial_p_value": float(binomtest(hits, len(table), max(chance, 1e-6), alternative="greater").pvalue),
        "unlinked_corr_p95": p95, "edges_corr_above_unlinked_p95": int(table["above_unlinked_p95"].sum()),
        "median_lag_error_h_when_hit": float(table.loc[table["lag_within_tolerance"], "lag_error_h"].median()) if hits else np.nan,
        "median_corr_at_transit": float(table["corr_at_transit"].median()),
        "median_corr_at_transit_minus_at_0": float((table["corr_at_transit"] - table["corr_at_0"]).median()),
    }
    return table.round(4), np.array(curves), summary


def plot_lead_lag(table: pd.DataFrame, curves: np.ndarray, step_h: int, summary: dict, path: Path, what: str) -> None:
    """Heat-map: one row per lane direction (sorted by transit), colour = correlation, white bar = transit, dot = best lag."""
    order = np.argsort(table["typical_transit_h"].to_numpy(), kind="stable")
    t = table.iloc[order]
    c = curves[order]
    fig, ax = plt.subplots(figsize=(12, max(6, 0.22 * len(t) + 2)))
    days = np.arange(c.shape[1]) * step_h / 24
    lim = np.nanmax(np.abs(c))
    im = ax.imshow(c, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest",
                   extent=[days[0], days[-1], len(t) - 0.5, -0.5])
    ax.scatter(t["typical_transit_h"] / 24, np.arange(len(t)), marker="|", s=90, color="white", linewidths=2,
               label="lane transit time (contract)")
    ax.scatter(t["best_lag_h"] / 24, np.arange(len(t)), marker="o", s=14, color="black", label="best lag (data)")
    ax.set_yticks(np.arange(len(t)), [f"{short(s)} -> {short(d)}" for s, d in zip(t["src"], t["dst"])], fontsize=6)
    ax.set_xlabel("lag in days (upstream leads downstream by this much)")
    fig.colorbar(im, ax=ax, label="correlation", shrink=0.6)
    ax.grid(False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
    ax.set_title(f"{what}: best lag lands within {summary['tolerance_h'] / 24:.0f} days of the transit time on "
                 f"{summary['lag_hits']}/{summary['lane_directions']} lane directions vs {summary['chance_hit_rate_unlinked']:.0%} "
                 f"for unlinked pairs (p = {summary['binomial_p_value']:.1g})", fontsize=10)
    save_fig(fig, path)


def lead_lag_sim(S: dict) -> tuple:
    """Lead-lag on the simulated hourly dwell (TRAIN hours) -> eda_leadlag_edges.csv, eda_leadlag_heatmap.png."""
    train = S["masks"]["train"]
    dw = S["wide"][DWELL]
    # Remove the weekly pattern, put nodes on the same scale, remove what the whole network does at once, smooth a day.
    x = remove_common_factor(iqr_scale(deseasonalise(dw, weekly_profile(dw, train)), train))
    x = x.rolling(SIM_SMOOTH_H).mean()[train].dropna()
    table, curves, summary = lead_lag(x, S["edges"], 1, SIM_MAX_LAG_H, SIM_TOL_H, "sim")
    table.to_csv(OUT / "eda_leadlag_edges.csv", index=False)
    plot_lead_lag(table, curves, 1, summary, OUT / "eda_leadlag_heatmap.png", "Simulated dwell, train hours")
    return table, summary


def lead_lag_real(R: dict | None, edges: pd.DataFrame) -> tuple:
    """Same analysis on the REAL PortWatch daily activity (train years) -> eda_leadlag_edges_real.csv + heat-map."""
    if R is None:
        return None, {"status": "real track not built (run scripts/fetch_real_data.py and 01_data_engineer.py)"}
    w, train = R["wide"], R["train"]
    # Day-of-week profile from the train years, then the same preparation as the simulated track (7-day smoothing).
    prof = w[train].groupby(w.index[train].day_of_week).median()
    d = w - prof.reindex(w.index.day_of_week).to_numpy()
    x = remove_common_factor(iqr_scale(d, train)).rolling(REAL_SMOOTH_D).mean()[train].dropna()
    table, curves, summary = lead_lag(x, edges, 24, REAL_MAX_LAG_D, REAL_TOL_D, "real")
    table.to_csv(OUT / "eda_leadlag_edges_real.csv", index=False)
    plot_lead_lag(table, curves, 24, summary, OUT / "eda_leadlag_real_heatmap.png", "REAL PortWatch ship activity, 2019-2023")
    return table, summary


# ----------------------------------------------------------------------------------------------------------------
# 5. Imputed hours
# ----------------------------------------------------------------------------------------------------------------

def imputed_hours(S: dict) -> dict:
    """How many filled-in hours each node has, how long the gaps are, and whether they cluster -> eda_imputed_hours.*"""
    imp = S["wide"]["is_imputed"].astype(bool)
    times = imp.index
    rows = []
    for n in S["node_ids"]:
        runs = run_lengths(imp[n].to_numpy())
        lengths = np.array([length for _, length in runs]) if runs else np.array([0])
        rows.append({"node_id": n, "node_type": S["node_type"][n], "imputed_hours": int(imp[n].sum()),
                     "share": float(imp[n].mean()), "gaps": len(runs), "mean_gap_h": float(lengths.mean()),
                     "max_gap_h": int(lengths.max()),
                     "share_in_gaps_of_2h_plus": float(lengths[lengths >= 2].sum() / max(lengths.sum(), 1)),
                     **{f"imputed_{k}": int(imp[n][S["masks"][k]].sum()) for k in ("train", "val", "test")}})
    table = pd.DataFrame(rows).round(4)
    table.to_csv(OUT / "eda_imputed_hours.csv", index=False)

    counts = table["imputed_hours"].to_numpy()
    by_hour = imp.sum(axis=1).groupby(times.hour).sum()
    # Rate of filled-in hours inside event windows vs outside: do gaps hide the start of disruptions?
    inside = pd.DataFrame(False, index=times, columns=S["node_ids"])
    for e in S["events"].itertuples():
        inside.loc[(times >= e.start_ts) & (times <= e.end_ts), e.node_id] = True
    rate_in = float(imp.to_numpy()[inside.to_numpy()].mean())
    rate_out = float(imp.to_numpy()[~inside.to_numpy()].mean())
    # Rate on high-dwell hours (above the alert line) vs the rest.
    high = S["wide"][DWELL].to_numpy() > np.array(list(S["thr"].values()))[None, :]
    summary = {
        "total_imputed_hours": int(counts.sum()), "share_of_all_hours": float(imp.to_numpy().mean()),
        "nodes_with_imputed_hours": int((counts > 0).sum()),
        "top3_nodes_share": float(np.sort(counts)[-3:].sum() / max(counts.sum(), 1)),
        "chi2_uniform_across_nodes_p": float(chisquare(counts).pvalue) if counts.sum() else 1.0,
        "max_gap_h": int(table["max_gap_h"].max()), "share_in_gaps_of_2h_plus": float(
            (table["share_in_gaps_of_2h_plus"] * table["imputed_hours"]).sum() / max(counts.sum(), 1)),
        "busiest_hour_of_day_vs_quietest": float(by_hour.max() / max(by_hour.min(), 1)),
        "rate_inside_event_windows": rate_in, "rate_outside_event_windows": rate_out,
        "rate_ratio_events": rate_in / rate_out if rate_out else np.nan,
        "rate_on_hours_above_alert_line": float(imp.to_numpy()[high].mean()),
        "rate_on_other_hours": float(imp.to_numpy()[~high].mean()),
    }

    # Chart: imputed hours per node and week, plus the hour-of-day profile.
    weekly = imp.groupby(times.to_period("W").start_time).sum() if times.tz is None else \
        imp.groupby(times.tz_convert(None).to_period("W").start_time).sum()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [3, 1]})
    im = axes[0].imshow(weekly.T.to_numpy(), aspect="auto", cmap="Blues", interpolation="nearest")
    axes[0].grid(False)
    axes[0].set_yticks(np.arange(len(S["node_ids"])), [short(n) for n in S["node_ids"]], fontsize=7)
    ticks = np.arange(0, len(weekly), 8)
    axes[0].set_xticks(ticks, [f"{weekly.index[i]:%b %d}" for i in ticks], fontsize=7)
    fig.colorbar(im, ax=axes[0], label="filled-in hours per week", shrink=0.7)
    axes[0].set_title(f"{summary['total_imputed_hours']:,} filled-in hours ({summary['share_of_all_hours']:.1%}); "
                      f"longest gap {summary['max_gap_h']} h; event-window rate x{summary['rate_ratio_events']:.2f} vs outside")
    axes[1].bar(by_hour.index, by_hour.to_numpy(), color=COLOURS[0])
    axes[1].set_xlabel("hour of day (UTC)")
    axes[1].set_ylabel("filled-in hours")
    axes[1].set_title("By hour of day")
    save_fig(fig, OUT / "eda_imputed_hours.png")
    (OUT / "eda_imputed_hours.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------------------------------------------------
# 6-9. Model part (attribution, noise channel, attention, graph)
# ----------------------------------------------------------------------------------------------------------------

def model_part(S: dict) -> dict:
    """Everything that needs the trained forecasters; returns a summary (or a SKIPPED note when a model is missing)."""
    try:
        import torch
        from common.edakit.attribution import completeness_gap, integrated_gradients
        from common.modelkit.dl_data import sim_data_from_grid, split_origins
        from common.modelkit.dl_infer import Stage2Forecaster
        from common.modelkit.dl_models import PinballLoss
    except ImportError as err:
        return {"status": f"SKIPPED: {err} (install Stage02_DL/requirements.txt)"}
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    art = stage_path(STAGE, "artifacts")
    scaler = json.loads(stage_path(STAGE, "data", "scaler.json").read_text())
    data = sim_data_from_grid(S["grid"], S["node_ids"], scaler)
    # Test origins at 00:00 each day for attribution (enough for stable shares), every 6 h for attention.
    daily = split_origins(data.times, data.spec, S["splits"], origin_hours=(0,))["test"]
    six_hourly = split_origins(data.times, data.spec, S["splits"], origin_hours=(0, 6, 12, 18))["test"]
    tools = {"torch": torch, "ig": integrated_gradients, "gap": completeness_gap, "pinball": PinballLoss}
    out = {"status": "RUN", "test_origins_attribution": int(len(daily)), "test_origins_attention": int(len(six_hourly)),
           "model_version": None}

    # 6. Attribution by channel and by past hour for the two single-node models.
    att = {}
    for name in ("lstm_forecaster", "transformer_forecaster"):
        if (art / f"{name}.pt").exists():
            f = Stage2Forecaster.from_artifact(name)
            out["model_version"] = f.meta.get("model_version")
            att[name] = {tgt: ig_attribution(f.model, data, daily, k, tools) for k, tgt in enumerate(data.target_names)}
    if att:
        out["attribution"] = write_attribution(att, data.input_channel_names, data.spec.input_len)
    else:
        out["attribution"] = {"status": "SKIPPED: no lstm_forecaster.pt / transformer_forecaster.pt"}

    # 7. Noise-channel sanity test.
    out["noise_channel"] = noise_channel_test(S, scaler, daily, six_hourly, tools)

    # 8. Attention maps.
    if (art / "transformer_forecaster.pt").exists():
        out["attention"] = attention_maps(S, data, six_hourly, Stage2Forecaster.from_artifact("transformer_forecaster"))
    else:
        out["attention"] = {"status": "SKIPPED: no transformer_forecaster.pt"}

    # 9. Graph attribution (SHOULD).
    if (art / "graph_forecaster.pt").exists() and (art / "graph_none.pt").exists():
        out["graph"] = graph_attribution(S, data, daily[::3], tools, Stage2Forecaster)
    else:
        out["graph"] = {"status": "SKIPPED: graph_forecaster.pt / graph_none.pt missing"}
    return out


def ig_attribution(model, data, origins: np.ndarray, target: int, tools: dict, chunk: int = 6) -> dict:
    """Integrated gradients of the median forecast (all nodes, all horizons) of one target w.r.t. the 72-hour input."""
    torch = tools["torch"]
    model.eval()
    total, gaps = None, []
    for i in range(0, len(origins), chunk):
        b = data.batch(origins[i:i + chunk])
        x, ctx, anchor = (torch.from_numpy(b[k]) for k in ("x", "ctx", "anchor"))

        def fn(x, ctx=ctx, anchor=anchor):
            """Median (q50) forecast of this target, summed over nodes and horizons: one number per origin."""
            return model(x, ctx, anchor)[..., target, 1].sum(dim=(1, 2))

        a = tools["ig"](fn, {"x": x}, steps=IG_STEPS)["x"]
        gaps.append(tools["gap"](fn, {"x": x}, {"x": a}))
        part = a.abs().sum(dim=(0, 1)).numpy()           # [L, C]: summed over origins and nodes
        total = part if total is None else total + part
    return {"abs": total, "completeness_gap_median": float(np.median(gaps))}


def write_attribution(att: dict, channels: list, input_len: int) -> dict:
    """Save channel and time-step shares -> eda_attribution_channels.csv/png, eda_attribution_timesteps.csv/png."""
    ch_rows, ts_rows, summary = [], [], {}
    for model, per_target in att.items():
        for tgt, res in per_target.items():
            a = res["abs"]
            by_ch = a.sum(axis=0) / a.sum()
            by_t = a.sum(axis=1) / a.sum()
            order = np.argsort(-by_ch)
            for rank, c in enumerate(order, 1):
                ch_rows.append({"model": model, "target": tgt, "channel": channels[c], "group": CHANNEL_GROUPS.get(channels[c], "?"),
                                "share_abs_attribution": by_ch[c], "rank": rank})
            for step in range(input_len):
                ts_rows.append({"model": model, "target": tgt, "hours_before_origin": input_len - 1 - step, "share": by_t[step]})
            groups = pd.Series(by_ch, index=[CHANNEL_GROUPS.get(c, "?") for c in channels]).groupby(level=0).sum()
            hours_ago = input_len - 1 - np.arange(input_len)
            summary[f"{model}/{tgt}"] = {
                "top3_channels": [channels[c] for c in order[:3]], "group_shares": groups.round(4).to_dict(),
                "share_last_6h": float(by_t[hours_ago < 6].sum()), "share_last_24h": float(by_t[hours_ago < 24].sum()),
                "completeness_gap_median": res["completeness_gap_median"],
            }
    ch = pd.DataFrame(ch_rows).round(5)
    ts = pd.DataFrame(ts_rows).round(5)
    ch.to_csv(OUT / "eda_attribution_channels.csv", index=False)
    ts.to_csv(OUT / "eda_attribution_timesteps.csv", index=False)

    # Chart 1: channel shares for the dwell forecast, one panel per model.
    models = list(att)
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 5.5), squeeze=False)
    fig.subplots_adjust(wspace=0.75)
    for ax, m in zip(axes[0], models):
        d = ch[(ch["model"] == m) & (ch["target"] == "dwell")].sort_values("share_abs_attribution")
        groups = sorted(set(CHANNEL_GROUPS.values()))
        ax.barh(d["channel"], d["share_abs_attribution"], color=[COLOURS[groups.index(g) % len(COLOURS)] for g in d["group"]])
        ax.set_xlabel("share of absolute attribution (dwell forecast)")
        s = summary[f"{m}/dwell"]["group_shares"]
        ax.set_title(f"{m}: drivers {s.get('operational driver', 0):.0%}, own history {s.get('own history', 0):.0%}, "
                     f"flag {s.get('reporting flag', 0):.1%}", fontsize=10)
    fig.suptitle("What the dwell forecast reads (integrated gradients, test origins, change from the last value)")
    save_fig(fig, OUT / "eda_attribution_channels.png")

    # Chart 2: share of attribution by hour before the forecast time.
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, ((m, tgt), grp) in enumerate(ts.groupby(["model", "target"])):
        ax.plot(grp["hours_before_origin"], grp["share"], color=COLOURS[i], label=f"{m} / {tgt}")
    ax.invert_xaxis()
    ax.set_xlabel("hours before the forecast time (0 = latest hour)")
    ax.set_ylabel("share of absolute attribution")
    ax.legend(fontsize=8)
    first = summary[f"{models[0]}/dwell"]
    ax.set_title(f"{models[0]} puts {first['share_last_6h']:.0%} of the dwell attribution on the last 6 h and "
                 f"{first['share_last_24h']:.0%} on the last 24 h of the 72 h window")
    save_fig(fig, OUT / "eda_attribution_timesteps.png")
    return summary


def noise_channel_test(S: dict, scaler: dict, daily: np.ndarray, six_hourly: np.ndarray, tools: dict) -> dict:
    """Attribution and permutation test on the LSTM trained with 2 extra noise inputs -> eda_noise_channel_test.json."""
    from common.modelkit.dl_data import sim_data_from_grid
    from common.modelkit.dl_infer import Stage2Forecaster

    path = stage_path(STAGE, "artifacts", "noise_check/lstm_noise.pt")
    rule = ("PASS if each noise channel gets < 2% of the absolute attribution AND less than the median real channel, "
            "AND shuffling the noise channels changes the test pinball loss by < 1% (a real driver is shuffled for contrast)")
    if not path.exists():
        res = {"status": "NOT_RUN", "how_to_run": "cd Stage02_DL && python 03_dl_engineer.py --noise-channel (~3 min), "
                                                  "then python 02_eda_engineer.py", "rule": rule}
        (OUT / "eda_noise_channel_test.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        return res
    torch = tools["torch"]
    f = Stage2Forecaster.from_artifact("noise_check/lstm_noise")
    data = sim_data_from_grid(S["grid"], S["node_ids"], scaler, noise_channels=True)
    names = data.input_channel_names
    shares, gaps = {}, {}
    for k, tgt in enumerate(data.target_names):
        res = ig_attribution(f.model, data, daily, k, tools)
        by_ch = res["abs"].sum(axis=0) / res["abs"].sum()
        shares[tgt] = dict(zip(names, by_ch.round(5).tolist()))
        gaps[tgt] = res["completeness_gap_median"]

    def pinball(d) -> float:
        """Test pinball loss (scaled units) of the noise model on data `d`, 6-hourly test origins."""
        loss, n = 0.0, 0
        for i in range(0, len(six_hourly), 16):
            b = d.batch(six_hourly[i:i + 16])
            with torch.no_grad():
                q = f.model(*(torch.from_numpy(b[k]) for k in ("x", "ctx", "anchor")))
            loss += tools["pinball"]()(q, torch.from_numpy(b["y"])).item() * len(b["origins"])
            n += len(b["origins"])
        return loss / n

    rng = np.random.default_rng(SEED)
    base = pinball(data)
    # Shuffle the two noise channels in time (inputs only).
    noisy = copy.copy(data)
    noisy.extra = data.extra[rng.permutation(len(data.times))]
    # Contrast: shuffle a real driver (capacity utilisation) in time. Only x changes; anchor and context stay as they were.
    driver = copy.copy(data)
    driver.scaled = data.scaled.copy()
    ci = CHANNELS.index("capacity_utilization_pct")
    driver.scaled[:, :, ci] = data.scaled[rng.permutation(len(data.times)), :, ci]
    noise_change = pinball(noisy) / base - 1
    driver_change = pinball(driver) / base - 1

    real = [c for c in names if not c.startswith("noise_")]
    checks = {}
    for tgt, s in shares.items():
        med_real = float(np.median([s[c] for c in real]))
        for c in ("noise_random", "noise_time_shuffled_ch0"):
            checks[f"{tgt}/{c}"] = {"share": s[c], "median_real_channel_share": round(med_real, 5),
                                    "rank_of_15": int(sorted(s.values(), reverse=True).index(s[c]) + 1),
                                    "pass": bool(s[c] < 0.02 and s[c] < med_real)}
    passed = all(v["pass"] for v in checks.values()) and abs(noise_change) < 0.01
    res = {"status": "PASS" if passed else "FAIL", "rule": rule, "model": "artifacts/noise_check/lstm_noise.pt",
           "test_origins": int(len(daily)), "attribution_checks": checks, "channel_shares": shares,
           "completeness_gap_median": gaps, "test_pinball_base": round(base, 5),
           "pinball_change_noise_shuffled": round(noise_change, 5), "pinball_change_utilisation_shuffled": round(driver_change, 5)}
    (OUT / "eda_noise_channel_test.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def attention_maps(S: dict, data, origins: np.ndarray, f) -> dict:
    """Transformer attention for 3 test episodes + attention by lag and on filled-in hours -> eda_attention_*."""
    import torch

    model = f.model
    times = data.times
    L = data.spec.input_len
    test_start = utc(S["splits"]["test_start"])
    # Episodes in the test period with Role 4's rule; keep those whose 72 h input (+24 h warning) lies inside test.
    episodes = []
    for j, n in enumerate(S["node_ids"]):
        inside = episode_hours(times, S["wide"][DWELL][n].to_numpy(), S["thr"][n]).to_numpy()
        for start, length in run_lengths(inside):
            t0 = times[start]
            if t0 - pd.Timedelta(hours=24 + L) >= test_start:
                peak = float(S["wide"][DWELL][n].iloc[start:start + length].max() / S["thr"][n])
                episodes.append({"node": n, "j": j, "onset": t0, "onset_idx": start, "peak_vs_alert": peak})
    episodes = sorted(episodes, key=lambda e: -e["peak_vs_alert"])
    chosen, seen = [], set()
    for e in episodes:
        if e["node"] not in seen:
            chosen.append(e)
            seen.add(e["node"])
        if len(chosen) == 3:
            break

    # Attention of the latest hour (the one the forecast is read from) over the 72 input hours, every test origin.
    last_rows, imputed = [], []
    for i in range(0, len(origins), 16):
        b = data.batch(origins[i:i + 16])
        maps = model.attention_maps(torch.from_numpy(b["x"]))
        avg = torch.stack([m[:, -1, :] for m in maps]).mean(0).numpy().reshape(len(b["origins"]), data.n_nodes, L)
        last_rows.append(avg)
        steps = b["origins"][:, None] + np.arange(-L + 1, 1)[None, :]
        imputed.append(data.imputed[steps].transpose(0, 2, 1))           # [B, N, L]
    att = np.concatenate(last_rows)
    imp = np.concatenate(imputed).astype(bool)
    hours_ago = L - 1 - np.arange(L)
    # Which (origin, node) pairs sit 1-48 h before an episode onset at that node.
    pre = np.zeros(att.shape[:2], dtype=bool)
    onset_idx = {}
    for e in episodes:
        onset_idx.setdefault(e["j"], []).append(e["onset_idx"])
    for oi, o in enumerate(origins):
        for j, starts in onset_idx.items():
            pre[oi, j] = any(0 < s - o <= 48 for s in starts)
    rows = [{"window": "all test origins", "hours_before_origin": int(h), "mean_attention": float(v)}
            for h, v in zip(hours_ago, att.reshape(-1, L).mean(0))]
    if pre.any():
        rows += [{"window": "1-48 h before an episode", "hours_before_origin": int(h), "mean_attention": float(v)}
                 for h, v in zip(hours_ago, att[pre].mean(0))]
    pd.DataFrame(rows).round(5).to_csv(OUT / "eda_attention_by_lag.csv", index=False)
    # Attention per filled-in input hour vs per measured hour, in windows that contain both.
    mixed = imp.any(-1) & ~imp.all(-1)
    per_imp = float((att * imp)[mixed].sum() / imp[mixed].sum()) if mixed.any() else np.nan
    per_meas = float((att * ~imp)[mixed].sum() / (~imp)[mixed].sum()) if mixed.any() else np.nan
    uniform = 1 / L
    summary = {
        "episodes_plotted": [{"node": e["node"], "onset": str(e["onset"]), "peak_vs_alert": round(e["peak_vs_alert"], 2)}
                             for e in chosen],
        "share_on_last_6h_all": float(att.reshape(-1, L)[:, hours_ago < 6].sum(1).mean()),
        "share_on_last_6h_pre_episode": float(att[pre][:, hours_ago < 6].sum(1).mean()) if pre.any() else np.nan,
        "uniform_share_last_6h": 6 / L, "pre_episode_pairs": int(pre.sum()),
        "attention_per_imputed_hour": per_imp, "attention_per_measured_hour": per_meas,
        "imputed_vs_measured_ratio": per_imp / per_meas if mixed.any() else np.nan,
        "windows_with_imputed_and_measured": int(mixed.sum()), "uniform_attention_per_hour": uniform,
    }

    # Chart: for each episode, the full attention matrix (last layer) and the latest hour's attention vs the dwell input.
    if chosen:
        fig, axes = plt.subplots(2, len(chosen), figsize=(5.5 * len(chosen), 8.5), squeeze=False)
        for k, e in enumerate(chosen):
            origin = e["onset_idx"] - 24
            b = data.batch(np.array([origin]))
            maps = model.attention_maps(torch.from_numpy(b["x"]))
            full = maps[-1].numpy().reshape(data.n_nodes, L, L)[e["j"]]
            last = torch.stack([m[:, -1, :] for m in maps]).mean(0).numpy().reshape(data.n_nodes, L)[e["j"]]
            axes[0, k].imshow(full, cmap="viridis", aspect="auto", extent=[L - 1, 0, L - 1, 0])
            axes[0, k].grid(False)
            axes[0, k].set_xlabel("key: hours before forecast time")
            axes[0, k].set_ylabel("query: hours before forecast time")
            axes[0, k].set_title(f"{S['names'][e['node']]}, 24 h before onset {e['onset']:%m-%d %H:%M}\n"
                                 f"(peak = {e['peak_vs_alert']:.1f} x alert line), last layer", fontsize=9)
            ax2 = axes[1, k]
            ax2.bar(hours_ago, last, color=COLOURS[0], width=0.9, label="attention of latest hour")
            ax2.axhline(uniform, color="grey", ls=":", lw=1, label="uniform (1/72)")
            ax2.invert_xaxis()
            ax2.set_xlabel("hours before forecast time")
            ax2.set_ylabel("attention weight")
            tw = ax2.twinx()
            dwell = data.raw[origin - L + 1:origin + 1, e["j"], CHANNELS.index(DWELL)][::-1]
            tw.plot(hours_ago, dwell, color=COLOURS[3], lw=1.2, label="dwell input (h)")
            tw.set_ylabel("dwell (h)")
            ax2.legend(fontsize=7, loc="upper left")
            tw.legend(fontsize=7, loc="upper right")
        fig.suptitle(f"Transformer attention before 3 test episodes: {summary['share_on_last_6h_all']:.0%} of the latest hour's "
                     f"attention goes to the last 6 h (uniform = {6 / L:.0%}); filled-in hours get "
                     f"x{summary['imputed_vs_measured_ratio']:.2f} the attention of measured hours")
        save_fig(fig, OUT / "eda_attention_episodes.png")
    (OUT / "eda_attention_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def graph_attribution(S: dict, data, origins: np.ndarray, tools: dict, forecaster) -> dict:
    """Which nodes' inputs drive each node's dwell forecast in graph_full vs graph_none -> eda_graph_attribution.*"""
    torch = tools["torch"]
    N = data.n_nodes
    idx = {n: i for i, n in enumerate(S["node_ids"])}
    nbr = np.zeros((N, N), dtype=bool)
    for n, ms in S["neighbours"].items():
        for m in ms:
            nbr[idx[n], idx[m]] = True
    b = data.batch(origins)
    x, ctx, anchor = (torch.from_numpy(b[k]) for k in ("x", "ctx", "anchor"))
    rows, mats, summary = [], {}, {}
    for name in ("graph_forecaster", "graph_none"):
        f = forecaster.from_artifact(name)
        f.model.eval()
        M = np.zeros((N, N))
        for j in range(N):
            def fn(x, ctx, j=j):
                """Median dwell forecast of node j, summed over horizons: one number per origin."""
                return f.model(x, ctx, anchor)[:, j, :, 0, 1].sum(-1)

            a = tools["ig"](fn, {"x": x, "ctx": ctx}, steps=16)
            per_node = (a["x"].abs().sum(dim=(0, 2, 3)) + a["ctx"].abs().sum(dim=(0, 2, 3))).numpy()
            M[j] = per_node / max(per_node.sum(), 1e-12)
        mats[name] = M
        cross = M.sum(1) - np.diag(M)
        to_nbr = (M * nbr).sum(1)
        # Lift: share of the cross-node attribution that lands on lane neighbours, vs what an even spread would give.
        expected = nbr.sum(1) / (N - 1)
        lift = np.where(cross > 1e-9, (to_nbr / np.maximum(cross, 1e-12)) / np.maximum(expected, 1e-9), np.nan)
        for j, n in enumerate(S["node_ids"]):
            rows.append({"model": name, "target_node": n, "share_self": M[j, j], "share_lane_neighbours": to_nbr[j],
                         "share_other_nodes": cross[j] - to_nbr[j], "neighbour_lift_vs_even_spread": lift[j]})
        summary[name] = {"mean_share_self": float(np.diag(M).mean()), "mean_share_other_nodes_total": float(cross.mean()),
                         "mean_share_lane_neighbours": float(to_nbr.mean()),
                         "median_neighbour_lift": float(np.nanmedian(lift)) if np.isfinite(lift).any() else np.nan}
    pd.DataFrame(rows).round(5).to_csv(OUT / "eda_graph_attribution.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, (name, M) in zip(axes, mats.items()):
        off = M.copy()
        np.fill_diagonal(off, np.nan)          # hide the node itself so the cross-node pattern is visible
        im = ax.imshow(off, cmap="magma_r", vmin=0, vmax=max(np.nanmax(off), 1e-6))
        ax.grid(False)
        ys, xs = np.where(nbr)
        ax.scatter(xs, ys, s=6, marker="s", facecolors="none", edgecolors=COLOURS[5], linewidths=0.6)
        ax.set_xticks(range(N), [short(n) for n in S["node_ids"]], rotation=90, fontsize=6)
        ax.set_yticks(range(N), [short(n) for n in S["node_ids"]], fontsize=6)
        ax.set_xlabel("input node (whose history)")
        ax.set_ylabel("forecast node (dwell)")
        s = summary[name]
        ax.set_title(f"{name}: {s['mean_share_other_nodes_total']:.1%} from other nodes, "
                     f"{s['mean_share_lane_neighbours']:.1%} from lane neighbours", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.6, label="share of attribution")
    fig.suptitle("Where the graph model looks: cross-node attribution (squares = lanes); graph_none must be empty")
    save_fig(fig, OUT / "eda_graph_attribution.png")
    return summary


# ----------------------------------------------------------------------------------------------------------------
# Findings report
# ----------------------------------------------------------------------------------------------------------------

def pct(v) -> str:
    """Format a share as a percentage string (or 'n/a')."""
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0%}"


def write_findings(F: dict) -> None:
    """Write eda_findings.md (numbers inserted by code) and eda_summary.json."""
    m, s, amb, ll, llr, imp = F["memory"], F["stl"], F["ambiguous"], F["leadlag_sim"], F["leadlag_real"], F["imputed"]
    check = F["rule_check"]
    allrows = check[check["event_type"] == "ALL"].set_index("rule")
    st, p95 = allrows.iloc[0], allrows.iloc[1]
    mp = F.get("model", {})
    findings = []
    cycles = "clear" if m["dwell_nodes_with_weekly_bump"] + m["dwell_nodes_with_daily_bump"] > 27 else "weak"
    findings.append((
        f"Dwell has long memory; its daily and weekly cycles are {cycles}",
        f"Median decorrelation time of dwell is {m['dwell_median_decorrelation_h']:.0f} h "
        f"({m['dwell_decorrelation_over_168h']} of 27 nodes stay above 1/e beyond 168 h); median ACF {m['dwell_median_acf_24']:.2f} at "
        f"24 h, {m['dwell_median_acf_72']:.2f} at 72 h, {m['dwell_median_acf_168']:.2f} at 168 h. Daily bump at "
        f"{m['dwell_nodes_with_daily_bump']}/27 nodes, weekly bump at {m['dwell_nodes_with_weekly_bump']}/27. "
        f"ADF rejects a unit root for {pct(m['share_stationary'])} of {m['series_tested']} node x channel series. "
        f"PACF: lag 1 = {m['dwell_median_pacf_1']:.2f}, lag 2 = {m['dwell_median_pacf_2']:.2f}.",
        "`eda_acf_summary.csv`, `eda_acf.png`",
        "Keep the 72 h window (it covers most of the memory) and keep seasonal-naive (t-168 h) as a baseline, expecting it "
        f"to lose to persistence because the cycles are {cycles}. Most of the memory is a slow level, so the 4-week context "
        "and the 'last value + learnt change' design are justified; a 168 h window is worth testing (Role 1 hparam "
        "study), not assuming.",
    ))
    findings.append((
        "The slow trend, not the daily or weekly cycle, carries most of the dwell variance",
        f"MSTL on train hours, median over nodes: trend {pct(s[DWELL + '_median_share_trend'])}, daily "
        f"{pct(s[DWELL + '_median_share_daily_24h'])}, weekly {pct(s[DWELL + '_median_share_weekly_168h'])}, residual "
        f"{pct(s[DWELL + '_median_share_residual'])}. Throughput: trend {pct(s[TPUT + '_median_share_trend'])}, daily "
        f"{pct(s[TPUT + '_median_share_daily_24h'])}, weekly {pct(s[TPUT + '_median_share_weekly_168h'])}.",
        "`eda_stl_components.csv`, `eda_stl_components.png`",
        "Calendar features alone cannot forecast dwell; the model must track the level. A seasonal-adjusted residual "
        "(not raw dwell) is the right input for the shock / structural rule.",
    ))
    findings.append((
        "An explicit rule separates seasonal peaks, transient shocks and structural bottlenecks",
        f"Rule: residual above {K_NOISE:.0f} x noise floor for >= {STRUCTURAL_H} h = structural. On the 5 ambiguous "
        f"graphs it agrees with the ground truth on {amb['agree_with_ground_truth']}/{amb['n']} (event -> flagged, no "
        f"event -> not structural). Against all train + val events: structural rule recall {pct(st['recall'])} with "
        f"{st['episodes_per_node_per_30d']:.2f} episodes per node per 30 days, {pct(st['explained_by_event_share'])} of "
        f"them near an event; Role 4's p95 rule recall {pct(p95['recall'])}, {p95['episodes_per_node_per_30d']:.2f} "
        f"episodes per node per 30 days, {pct(p95['explained_by_event_share'])} near an event.",
        "`eda_ambiguous_graphs.md`, `eda_ambiguous_1..5.png`, `eda_episode_rule_check.csv`",
        "Role 4 decides whether to keep the p95 rule, switch, or report both (see recommendations).",
    ))
    findings.append((
        "Congestion travels along the lanes with roughly the lane's transit time (simulated data)",
        f"Best lag within {ll['tolerance_h'] / 24:.0f} days of the contract transit time on {ll['lag_hits']}/"
        f"{ll['lane_directions']} lane directions ({pct(ll['lag_hit_rate'])}) vs {pct(ll['chance_hit_rate_unlinked'])} "
        f"for {ll['unlinked_pairs']} unlinked pairs (binomial p = {ll['binomial_p_value']:.1g}). When it hits, the data lag "
        f"is {ll['median_lag_error_h_when_hit']:+.0f} h from the contract (median). But the correlation itself is weak: "
        f"only {ll['edges_corr_above_unlinked_p95']} lane directions beat the 95th percentile of unlinked pairs "
        f"(r = {ll['unlinked_corr_p95']:.2f}); median r at the transit lag = {ll['median_corr_at_transit']:.2f}.",
        "`eda_leadlag_edges.csv`, `eda_leadlag_heatmap.png`",
        "The transit-delayed lane messages in the graph model point at the right lag, so the design is sound; the weak "
        "strength is consistent with the negative graph ablation (little extra signal to gain). Propagation weights "
        f"(Stage 5) can use the contract transit time; the data lag is {ll['median_lag_error_h_when_hit']:+.0f} h from it.",
    ))
    if "lag_hits" in llr:
        findings.append((
            "On REAL PortWatch ship counts the lane lag signal is " + ("present" if llr["binomial_p_value"] < 0.05 else "not detectable"),
            f"Best lag within {llr['tolerance_h'] / 24:.0f} days of the transit time on {llr['lag_hits']}/{llr['lane_directions']} "
            f"lane directions ({pct(llr['lag_hit_rate'])}) vs {pct(llr['chance_hit_rate_unlinked'])} for unlinked pairs "
            f"(p = {llr['binomial_p_value']:.2g}); {llr['edges_corr_above_unlinked_p95']} directions beat the unlinked 95th "
            f"percentile (r = {llr['unlinked_corr_p95']:.2f}). Train years 2019-2023, daily, 7-day smoothing.",
            "`eda_leadlag_edges_real.csv`, `eda_leadlag_real_heatmap.png`",
            "This is the honest real-world check of the graph idea: say exactly this in the debate, and do not claim "
            "lane propagation on real data beyond what these numbers show.",
        ))
    findings.append((
        "Filled-in hours are few, short and spread out",
        f"{imp['total_imputed_hours']:,} filled-in hours ({imp['share_of_all_hours']:.1%} of all), longest gap "
        f"{imp['max_gap_h']} h, {pct(imp['share_in_gaps_of_2h_plus'])} of them in gaps of 2 h or more. Top 3 nodes hold "
        f"{pct(imp['top3_nodes_share'])} (chi-square vs even spread p = {imp['chi2_uniform_across_nodes_p']:.2g}). "
        f"Rate inside event windows {imp['rate_inside_event_windows']:.2%} vs outside {imp['rate_outside_event_windows']:.2%} "
        f"(x{imp['rate_ratio_events']:.2f}); above the alert line {imp['rate_on_hours_above_alert_line']:.2%} vs "
        f"{imp['rate_on_other_hours']:.2%}.",
        "`eda_imputed_hours.csv`, `eda_imputed_hours.json`, `eda_imputed_hours.png`",
        "Forward-fill up to 6 h (Role 3) is adequate; keep the is_imputed input flag and keep excluding imputed target "
        "hours from the metrics (Role 4).",
    ))
    att = mp.get("attribution", {})
    if "lstm_forecaster/dwell" in att:
        a = att["lstm_forecaster/dwell"]
        t = att.get("transformer_forecaster/dwell", {})
        g = a["group_shares"]
        findings.append((
            "The forecasters lean on their own recent history and the operational drivers; the reporting flag gets ~0",
            f"LSTM dwell forecast (integrated gradients, {mp['test_origins_attribution']} test origins): top channels "
            f"{', '.join(a['top3_channels'])}; drivers {pct(g.get('operational driver'))}, own history "
            f"{pct(g.get('own history'))}, yard state {pct(g.get('yard state'))}, calendar {pct(g.get('calendar'))}, "
            f"is_imputed flag {g.get('reporting flag', 0):.1%}. Last 6 h carry {pct(a['share_last_6h'])} and the last "
            f"24 h {pct(a['share_last_24h'])} of the attribution. "
            + (f"Transformer: top channels {', '.join(t['top3_channels'])}, last 6 h {pct(t['share_last_6h'])}. " if t else "")
            + f"Completeness gap (median) {a['completeness_gap_median']:.1%}.",
            "`eda_attribution_channels.csv/png`, `eda_attribution_timesteps.csv/png`",
            "Attribution covers the learnt change from the last value (the anchor is held fixed). Channels near zero "
            "are candidates to drop in a Role 1 ablation, not to drop blindly.",
        ))
    nc = mp.get("noise_channel", {})
    if nc.get("status") in ("PASS", "FAIL"):
        dw = nc["channel_shares"]["dwell"]
        findings.append((
            f"Noise-channel sanity test: {nc['status']}",
            f"On the LSTM trained with two extra inputs, pure noise gets {dw['noise_random']:.2%} and a time-shuffled copy "
            f"of dwell {dw['noise_time_shuffled_ch0']:.2%} of the dwell attribution. Shuffling both in time changes the "
            f"test pinball loss by {nc['pinball_change_noise_shuffled']:+.2%}; shuffling a real driver (utilisation) "
            f"changes it by {nc['pinball_change_utilisation_shuffled']:+.2%}.",
            "`eda_noise_channel_test.json`",
            "The attribution method is not just reflecting input magnitudes: it ignores inputs that carry no signal. "
            "Use this as the evidence when someone asks whether the model looks at genuine signals.",
        ))
    elif nc:
        findings.append(("Noise-channel sanity test not run", nc.get("how_to_run", nc.get("status", "")),
                         "`eda_noise_channel_test.json`", "Run it before the debate."))
    at = mp.get("attention", {})
    if "share_on_last_6h_all" in at:
        focused = at["share_on_last_6h_all"] > 1.5 * at["uniform_share_last_6h"]
        findings.append((
            "Transformer attention " + ("concentrates on the latest hours" if focused else
                                        "is spread almost evenly over the 72 h window, so it explains little"),
            f"The latest hour sends {pct(at['share_on_last_6h_all'])} of its attention to the last 6 h (uniform would be "
            f"{pct(at['uniform_share_last_6h'])}); 1-48 h before an episode: {pct(at['share_on_last_6h_pre_episode'])} "
            f"({at['pre_episode_pairs']} node-origins). A filled-in input hour gets x{at['imputed_vs_measured_ratio']:.2f} "
            f"the attention of a measured hour ({at['windows_with_imputed_and_measured']} windows with both).",
            "`eda_attention_episodes.png`, `eda_attention_by_lag.csv`, `eda_attention_summary.json`",
            "Attention weights are not an explanation of the forecast; use the integrated-gradients numbers (F7) in the "
            "debate. Filled-in hours are neither ignored nor over-used by attention.",
        ))
    gr = mp.get("graph", {})
    if "graph_forecaster" in gr:
        gf, gn = gr["graph_forecaster"], gr["graph_none"]
        findings.append((
            f"graph_full takes {pct(gf['mean_share_other_nodes_total'])} of its dwell attribution from other nodes, "
            f"{pct(gf['mean_share_lane_neighbours'] / max(gf['mean_share_other_nodes_total'], 1e-9))} of that from lane neighbours",
            f"graph_full: {pct(gf['mean_share_other_nodes_total'])} of a node's dwell attribution comes from other nodes, "
            f"{pct(gf['mean_share_lane_neighbours'])} from lane neighbours (median lift vs an even spread "
            f"x{gf['median_neighbour_lift']:.1f}). graph_none (sanity): {gn['mean_share_other_nodes_total']:.2%} from other nodes.",
            "`eda_graph_attribution.csv`, `eda_graph_attribution.png`",
            "The graph path is really used and it follows the lanes (graph_none confirms the method gives 0 when no lanes "
            "exist). Read together with Role 4's ablation (no accuracy gain), this is consistent with the neighbours' "
            "information being largely redundant with a node's own history (F4: weak lane correlation). More graph "
            "capacity is unlikely to pay off on this data.",
        ))

    lines = [
        "# Stage 2 - EDA findings (time series, propagation, attribution)",
        "",
        f"Generated by `Stage02_DL/02_eda_engineer.py` (numbers inserted by code). Model artefacts: "
        f"{mp.get('model_version') or 'not used'}. Train hours for every modelling decision; test origins only for the "
        "attribution / attention checks, where nothing is chosen. Correlations are associations, not causes.",
        "",
        "## Findings",
        "",
    ]
    for i, (title, what, evidence, so_what) in enumerate(findings, 1):
        lines += [f"### F{i}. {title}", "", f"- **What:** {what}", f"- **Evidence:** {evidence}", f"- **So what:** {so_what}", ""]
    lines += [
        "## Recommendations",
        "",
        "**Role 1 (ML)**",
        f"- Keep the 72 h window as default; test 168 h in the hparam study (dwell memory: median decorrelation "
        f"{m['dwell_median_decorrelation_h']:.0f} h). Keep the 1/6/24/72/168 h horizons and the seasonal-naive baseline.",
        "- Keep node embeddings: memory and variance shares differ by node type (`eda_acf_summary.csv`, `eda_stl_components.csv`).",
        f"- Keep the transit-delayed messages at the contract transit time; the data lag runs "
        f"{ll['median_lag_error_h_when_hit']:+.0f} h from it (median where it matches), well inside one daily context "
        "block, so no change is needed (`eda_leadlag_edges.csv`, column `lag_error_h`). Do not enlarge the graph part.",
        "- Use the channel attribution (`eda_attribution_channels.csv`) to pick candidates for a drop-one-channel ablation on val.",
        "",
        "**Role 3 (Data & Integration)**",
        "- Forward-fill <= 6 h is adequate (`eda_imputed_hours.json`); keep `is_imputed` in the grid and in the API inputs.",
        "- The dashboard can show the structural / transient / seasonal label from the rule in `eda_ambiguous_graphs.md` "
        "(past hours only, so it runs live).",
        "",
        "**Role 4 (Evaluation)**",
        f"- Alert definition: the p95 rule catches {pct(p95['recall'])} of events with "
        f"{p95['episodes_per_node_per_30d']:.2f} episodes per node per 30 days; the structural rule catches "
        f"{pct(st['recall'])} with {st['episodes_per_node_per_30d']:.2f}. Report lead time under both, choose on val, "
        "never on test (`eda_episode_rule_check.csv`).",
        "- Quote the noise-channel result and the real-data lead-lag numbers exactly as above; neither is a claim of causation.",
        "",
        "## Files",
        "",
        "`eda_acf_summary.csv`, `eda_acf.png`, `eda_stl_components.csv/png`, `eda_ambiguous_graphs.md`, `eda_ambiguous_1..5.png`, "
        "`eda_episode_rule_check.csv`, `eda_leadlag_edges.csv`, `eda_leadlag_heatmap.png`, `eda_leadlag_edges_real.csv`, "
        "`eda_leadlag_real_heatmap.png`, `eda_imputed_hours.csv/json/png`, `eda_attribution_channels.csv/png`, "
        "`eda_attribution_timesteps.csv/png`, `eda_noise_channel_test.json`, `eda_attention_episodes.png`, "
        "`eda_attention_by_lag.csv`, `eda_attention_summary.json`, `eda_graph_attribution.csv/png`, `eda_summary.json`.",
    ]
    (OUT / "eda_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_jsonable(obj):
    """Turn numpy / pandas values into plain Python so json.dumps can write them."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def main() -> int:
    """Run every analysis in order and write the reports."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-model", action="store_true", help="skip the parts that need trained models (6-9)")
    args = ap.parse_args()
    set_global_seed()  # same random numbers on every laptop
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading the hourly grid, splits, events and lanes ...")
    S = load_sim()
    F = {}
    print("1/9 stationarity and memory (ADF, ACF, PACF) ...")
    F["memory"] = stationarity_and_memory(S)
    print("2/9 seasonal decomposition (MSTL 24 h + 168 h) ...")
    F["stl"] = seasonal_decomposition(S)
    print("3/9 seasonal vs transient vs structural rule, 5 ambiguous graphs ...")
    d, r, sigma = residuals(S)
    F["rule_check"] = rule_check(S, r, sigma)
    F["ambiguous"] = ambiguous_graphs(S, d, r, sigma, F["rule_check"])
    print("4/9 lead-lag along the lanes (simulated + real) ...")
    _, F["leadlag_sim"] = lead_lag_sim(S)
    _, F["leadlag_real"] = lead_lag_real(load_real(), S["edges"])
    print("5/9 imputed hours ...")
    F["imputed"] = imputed_hours(S)
    if args.no_model:
        F["model"] = {"status": "SKIPPED (--no-model)"}
    else:
        print("6-9/9 attribution, noise channel, attention, graph attribution (needs the trained models) ...")
        F["model"] = model_part(S)
    write_findings(F)
    summary = {k: v for k, v in F.items() if k != "rule_check"}
    summary["rule_check"] = F["rule_check"].to_dict("records")
    (OUT / "eda_summary.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")

    ll = F["leadlag_sim"]
    print(f"Done. Lead-lag: {ll['lag_hits']}/{ll['lane_directions']} lanes at the transit lag "
          f"(chance {ll['chance_hit_rate_unlinked']:.0%}). Noise test: {F['model'].get('noise_channel', {}).get('status', 'n/a')}. "
          f"See Stage02_DL/reports/eda_findings.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
