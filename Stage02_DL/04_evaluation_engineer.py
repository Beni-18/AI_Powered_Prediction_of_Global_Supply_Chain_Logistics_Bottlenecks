"""ROLE: EVALUATION ENGINEER  |  STAGE 02 - DEEP LEARNING
Owner: Role 4 - Evaluation Engineer. Full instructions: docs/roles/ROLE_4_EVALUATION_ENGINEER.md (Day 2).

WHAT THIS SCRIPT DOES
    0. Trust checks first: contract, same forecasts for every model, y_true re-read from Role 3's grid, time order.
    1. Accuracy per model x target x horizon (MAE, RMSE, sMAPE) and SKILL versus persistence and versus the best
       baseline, with block-bootstrap 95% intervals (blocks = forecast days).
    2. Interval quality: coverage of the 80% band, width, Winkler interval score.
    3. LEAD TIME (the Stage 2 headline): how many hours before a dwell bottleneck starts does each model warn?
       Episode = dwell above the node's 95th train percentile for >= 6 h (episodes >= 24 h apart).
       Rule A: alert when any horizon's median forecast crosses the threshold.
       Rule B: alert when P(crossing) >= p*, with p* chosen on VALIDATION by cost (miss 10, false alarm 1).
       Each model then uses whichever rule had the lower VALIDATION cost; test is scored with that rule only.
    4. Graph ablation: graph_full vs graph_shuffled vs graph_none (paired bootstrap on the same forecasts).
    5. Visual prediction benchmark: three test episodes, truth vs persistence / xgb_lag / LSTM / graph model,
       plus the Stage 1 shipment risk score turned into a node alarm ("baseline ML feature detection").
    6. Slices by node type and the worst nodes; Diebold-Mariano tests; leakage and time-order checks.
    7. Real track (IMF PortWatch daily ship counts): skill, intervals and ablation on REAL data.
    8. Champion chosen on VALIDATION (never on test), eval_champion.json, claims ledger rows.

READS : Stage02_DL/predictions/dl_forecasts.csv.gz          (Role 1, contract dl_forecasts)
        Stage02_DL/predictions/dl_real_forecasts.csv.gz     (Role 1, real track; optional)
        Stage02_DL/data/hourly_node_telemetry.csv.gz, splits.json, real_daily_activity.csv.gz   (Role 3)
        Stage02_DL/artifacts/model_card.json                (Role 1: parameter counts)
        Stage01_ML/predictions/ml_predictions.csv, Stage01_ML/data/prepared_dataset.csv.gz,
        Stage01_ML/reports/eval_champion.json               (optional: Stage 1 detection baseline)
        contracts/cost_policy.json, contracts/network_nodes.csv
WRITES: Stage02_DL/reports/eval_*.csv|json|png|md, claims/CLAIMS_LEDGER.md

HOW TO RUN
    python Stage02_DL/04_evaluation_engineer.py                         (real run, ~1-2 min)
    python Stage02_DL/04_evaluation_engineer.py --out-dir /tmp/e2       (dry run: no ledger)
"""

import os
import sys

# PyTorch and XGBoost each bring their own OpenMP library. On macOS the two deadlock in one process unless OpenMP
# itself is limited to one thread. Keep this setting before importing packages that load OpenMP.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

# Make "import common" work when this file is run from inside the stage folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # draw to files only, so it works on a machine with no screen
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import contracts  # noqa: E402
from common.evalkit import (  # noqa: E402
    block_skill_ci, diebold_mariano, episode_hours, episodes_from_series, exceedance_probability, interval_score,
    ledger, load_cost_matrix, rmse, smape, wilson_interval,
)
from common.paths import NODES_CSV, stage_path  # noqa: E402
from common.seed import SEED, set_global_seed  # noqa: E402

STAGE = 2
BASELINES = ("persistence", "seasonal_naive", "rolling_mean", "xgb_lag")
SIMPLE_BASELINES = ("persistence", "seasonal_naive", "rolling_mean")
GRAPH_VARIANTS = ("graph_full", "graph_shuffled", "graph_none")
TARGET_COLUMN = {"dwell": "avg_dwell_time_hours", "throughput": "throughput_teu_per_hr", "activity": "activity"}
EPISODE_PCT = 95          # alert threshold = node's 95th percentile of TRAIN dwell (role file, Day 2)
LOOKBACK_H = 168          # an alert counts if it fires up to 7 days before the onset (role file)
ORIGIN_STEP_H = 6         # forecasts are issued every 6 hours (Role 1's file)
P_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)   # candidate probability thresholds for rule B
COLORS = {"persistence": "#7f7f7f", "seasonal_naive": "#bcbd22", "rolling_mean": "#c7c7c7", "xgb_lag": "#ff7f0e",
          "lstm": "#1f77b4", "transformer": "#9467bd", "graph_full": "#2ca02c", "graph_shuffled": "#98df8a",
          "graph_none": "#d62728", "stage1_risk": "#8c564b"}


# ------------------------------------------------------------------ loading and trust checks

def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, default=stage_path(STAGE, "predictions", "dl_forecasts.csv.gz"))
    ap.add_argument("--real-predictions", type=Path, default=stage_path(STAGE, "predictions", "dl_real_forecasts.csv.gz"))
    ap.add_argument("--out-dir", type=Path, default=None, help="custom folder = dry run (no ledger)")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples (1000 for real runs)")
    ap.add_argument("--no-ledger", action="store_true", help="do not touch claims/CLAIMS_LEDGER.md")
    return ap.parse_args()


def load_grid() -> pd.DataFrame:
    """Role 3's complete hourly grid with parsed times."""
    grid = pd.read_csv(stage_path(STAGE, "data", "hourly_node_telemetry.csv.gz"))
    grid["hour_ts"] = pd.to_datetime(grid["hour_ts"], utc=True)
    grid["is_imputed"] = grid["is_imputed"].astype(str).str.lower().eq("true")
    return grid


def load_forecasts(path: Path, validate: bool) -> pd.DataFrame:
    """Load a forecasts file (after the contract check for the hand-over file) with parsed times."""
    if validate:
        report = contracts.validate_csv("dl_forecasts", path)
        if not report.ok:
            for v in report.violations[:10]:
                print(f"  - {v}")
            raise SystemExit("dl_forecasts breaks the contract; ask Role 1 to fix it (scripts/validate_contract.py).")
    df = pd.read_csv(path)
    df["origin_ts"] = pd.to_datetime(df["origin_ts"], utc=True)
    df["target_ts"] = df["origin_ts"] + pd.to_timedelta(df["horizon_h"], unit="h")
    return df


KEYS = ["split", "target", "horizon_h", "node_id", "origin_ts"]


def to_wide(df: pd.DataFrame, errors: list) -> tuple:
    """Align every model on the same forecasts. Returns (keys frame, {model: {pred, lo, hi}}, y_true array)."""
    models = sorted(df["model_name"].unique())
    ref = df[df["model_name"] == ("persistence" if "persistence" in models else models[0])]
    keys = ref.sort_values(KEYS)[KEYS + ["target_ts", "y_true"]].reset_index(drop=True)
    index = pd.MultiIndex.from_frame(keys[KEYS])
    wide = {}
    for m in models:
        part = df[df["model_name"] == m].set_index(KEYS)
        if len(part) != len(index) or not part.index.sort_values().equals(index):
            errors.append(f"model '{m}' did not forecast exactly the same node/origin/horizon/target set as the others.")
            continue
        part = part.reindex(index)
        if not np.allclose(part["y_true"].to_numpy(), keys["y_true"].to_numpy(), atol=1e-3):
            errors.append(f"model '{m}' reports different y_true values for the same forecasts.")
        wide[m] = {"pred": part["y_pred"].to_numpy(float), "lo": part["y_lo"].to_numpy(float),
                   "hi": part["y_hi"].to_numpy(float)}
    return keys, wide, keys["y_true"].to_numpy(float)


def attach_truth_from_grid(keys: pd.DataFrame, grid: pd.DataFrame, time_col: str, errors: list) -> np.ndarray:
    """Re-read each target from Role 3's grid; flag mismatches; return a mask of usable (measured) targets."""
    measured = np.ones(len(keys), dtype=bool)
    for target, col in TARGET_COLUMN.items():
        rows = keys["target"] == target
        if not rows.any():
            continue
        g = grid.set_index(["node_id", time_col])
        idx = pd.MultiIndex.from_arrays([keys.loc[rows, "node_id"], keys.loc[rows, "target_ts"]])
        truth = g[col].reindex(idx).to_numpy(float)
        imputed = g["is_imputed"].reindex(idx).fillna(True).to_numpy(bool)
        missing = np.isnan(truth)
        if missing.any():
            errors.append(f"{int(missing.sum())} {target} targets point at hours that are not in Role 3's grid.")
        diff = np.abs(truth - keys.loc[rows, "y_true"].to_numpy(float))
        bad = int((diff > 1e-3 * np.maximum(1, np.abs(truth))).sum())
        if bad:
            errors.append(f"{bad} {target} y_true values differ from Role 3's grid (wrong target hour or re-scaled data?).")
        # Hours that were filled in by Role 3 are not real measurements: never score a forecast against them.
        measured[rows.to_numpy()] = ~imputed & ~missing
    return measured


def time_checks(keys: pd.DataFrame, bounds: dict, input_steps: int, step_h: int) -> dict:
    """Leakage / time-order checks on the forecast origins (role file Day 2, item 7)."""
    b = {k: pd.Timestamp(v) for k, v in bounds.items() if k.endswith(("_start", "_end"))}
    b = {k: (v.tz_localize("UTC") if v.tzinfo is None else v) for k, v in b.items()}
    first_input = keys["origin_ts"] - pd.to_timedelta((input_steps - 1) * step_h, unit="h")
    out = {}
    test, val = keys["split"] == "test", keys["split"] == "val"
    out["test_inputs_start_inside_test"] = bool((first_input[test] >= b["test_start"]).all())
    out["val_inputs_inside_val"] = bool((first_input[val] >= b["val_start"]).all())
    out["val_targets_before_val_end"] = bool((keys.loc[val, "target_ts"] < b["val_end"]).all())
    out["targets_after_origin"] = bool((keys["target_ts"] > keys["origin_ts"]).all())
    out["no_train_split_rows_in_file"] = bool(not (keys["split"] == "train").any())
    out["passed"] = all(out.values())
    return out


# ------------------------------------------------------------------ 1-2: accuracy, skill, intervals

def accuracy_tables(keys, wide, y, ok, blocks, n_boot: int, horizons) -> tuple:
    """Per split x model x target x horizon: MAE, RMSE, sMAPE, skill vs persistence and vs best baseline, intervals."""
    rows, interval_rows = [], []
    for split in ("val", "test"):
        for target in sorted(keys["target"].unique()):
            for h in horizons:
                sel = ((keys["split"] == split) & (keys["target"] == target) & (keys["horizon_h"] == h)).to_numpy() & ok
                if not sel.any():
                    continue
                yt, blk = y[sel], blocks[sel]
                errs = {m: np.abs(wide[m]["pred"][sel] - yt) for m in wide}
                # The best baseline is chosen on VALIDATION MAE (never on test) for this target and horizon.
                vsel = ((keys["split"] == "val") & (keys["target"] == target) & (keys["horizon_h"] == h)).to_numpy() & ok
                base_mae = {m: np.abs(wide[m]["pred"][vsel] - y[vsel]).mean() for m in BASELINES if m in wide}
                best_base = min(base_mae, key=base_mae.get)
                for m in wide:
                    p = wide[m]["pred"][sel]
                    sk, sk_lo, sk_hi = block_skill_ci(errs[m], errs["persistence"], blk, n=n_boot, seed=SEED)
                    sb, sb_lo, sb_hi = block_skill_ci(errs[m], errs[best_base], blk, n=n_boot, seed=SEED)
                    rows.append({"split": split, "model": m, "target": target, "horizon_h": int(h), "n": int(sel.sum()),
                                 "mae": errs[m].mean(), "rmse": rmse(yt, p), "smape_pct": smape(yt, p),
                                 "skill_vs_persistence": sk, "skill_lo": sk_lo, "skill_hi": sk_hi,
                                 "best_baseline_on_val": best_base, "skill_vs_best_baseline": sb,
                                 "skill_vs_best_baseline_lo": sb_lo, "skill_vs_best_baseline_hi": sb_hi})
                    lo, hi = wide[m]["lo"][sel], wide[m]["hi"][sel]
                    interval_rows.append({"split": split, "model": m, "target": target, "horizon_h": int(h),
                                          "coverage_80": float(((yt >= lo) & (yt <= hi)).mean()),
                                          "mean_width": float((hi - lo).mean()),
                                          "interval_score": interval_score(yt, lo, hi, alpha=0.2),
                                          "pinball_q50": float(np.mean(np.abs(yt - p)) / 2)})
    return pd.DataFrame(rows).round(5), pd.DataFrame(interval_rows).round(5)


def plot_skill(table: pd.DataFrame, out: Path, title: str, split: str = "test") -> None:
    """Skill vs persistence by horizon, one panel per target, 95% interval as a band."""
    t = table[table["split"] == split]
    targets = sorted(t["target"].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(6.2 * len(targets), 4.2), squeeze=False)
    for ax, target in zip(axes[0], targets):
        for m, g in t[t["target"] == target].groupby("model"):
            g = g.sort_values("horizon_h")
            x = np.arange(len(g))
            ax.plot(x, g["skill_vs_persistence"], marker="o", color=COLORS.get(m), label=m, lw=2 if m.startswith("graph_full") else 1.2)
            ax.fill_between(x, g["skill_lo"], g["skill_hi"], color=COLORS.get(m), alpha=0.12)
            ax.set_xticks(x, [f"{h} h" for h in g["horizon_h"]])
        ax.axhline(0, color="black", lw=0.8)
        # A very bad baseline (seasonal naive reaches -1.7) would squash every other line; keep the axis readable.
        values = t[t["target"] == target]["skill_lo"]
        ax.set_ylim(bottom=max(float(values.min()) - 0.02, -0.3))
        if float(values.min()) < -0.3:
            ax.text(0.01, 0.02, "lines below -0.3 are cut off (see the CSV)", transform=ax.transAxes, fontsize=8)
        ax.set_title(f"{target}: skill vs persistence ({split})")
        ax.set_xlabel("forecast horizon")
        ax.set_ylabel("skill = 1 - MAE / MAE(persistence)")
    axes[0][-1].legend(fontsize=8, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ 3: lead time

def node_thresholds(grid: pd.DataFrame, train_end: pd.Timestamp) -> pd.Series:
    """Alert threshold per node = 95th percentile of its measured TRAIN dwell."""
    train = grid[(grid["hour_ts"] < train_end) & ~grid["is_imputed"]]
    return train.groupby("node_id")["avg_dwell_time_hours"].quantile(EPISODE_PCT / 100)


def truth_episodes(grid: pd.DataFrame, thr: pd.Series) -> tuple:
    """Per node: list of episode onsets and the hourly 'inside an episode' mask (whole year of measured dwell)."""
    onsets, inside = {}, {}
    for node, g in grid.sort_values("hour_ts").groupby("node_id"):
        onsets[node] = episodes_from_series(g["hour_ts"], g["avg_dwell_time_hours"], thr[node])
        inside[node] = episode_hours(g["hour_ts"], g["avg_dwell_time_hours"], thr[node])
    return onsets, inside


def alert_frame(keys, wide, model: str, split: str, thr: pd.Series, inside: dict) -> pd.DataFrame:
    """One row per (node, origin) with the max forecast over horizons and P(crossing); in-episode origins dropped."""
    sel = ((keys["split"] == split) & (keys["target"] == "dwell")).to_numpy()
    k = keys.loc[sel, ["node_id", "origin_ts"]].copy()
    t = thr.reindex(k["node_id"]).to_numpy()
    k["crosses"] = wide[model]["pred"][sel] >= t
    k["prob"] = exceedance_probability(wide[model]["lo"][sel], wide[model]["pred"][sel], wide[model]["hi"][sel], t)
    per_origin = k.groupby(["node_id", "origin_ts"]).agg(crosses=("crosses", "max"), prob=("prob", "max")).reset_index()
    # An "alert" raised while the node is already in a bottleneck is not an early warning: drop those origins.
    now_inside = [bool(inside[n].get(ts.floor("h"), False)) for n, ts in zip(per_origin["node_id"], per_origin["origin_ts"])]
    return per_origin[~np.array(now_inside)]


def score_alerts(alerts: pd.DataFrame, onsets: dict, first_origin, last_origin, n_nodes: int) -> dict:
    """Episode recall, lead times and false alarms for one set of alert times (node_id, origin_ts)."""
    lb = pd.Timedelta(hours=LOOKBACK_H)
    leads, false_alarms, n_alerts = [], 0, len(alerts)
    for node, node_onsets in onsets.items():
        a = np.sort(pd.DatetimeIndex(alerts.loc[alerts["node_id"] == node, "origin_ts"]).tz_convert(None).to_numpy())
        all_on = pd.DatetimeIndex(node_onsets).tz_convert(None).to_numpy() if len(node_onsets) else np.array([], dtype="datetime64[ns]")
        # Eligible onsets: the full 7-day look-back is covered by forecast origins of this split.
        for on in node_onsets:
            if not (first_origin + lb <= on <= last_origin + pd.Timedelta(hours=ORIGIN_STEP_H)):
                continue
            on_np = on.tz_convert(None).to_datetime64()
            window = a[(a >= on_np - np.timedelta64(LOOKBACK_H, "h")) & (a <= on_np)]
            leads.append((on - pd.Timestamp(window.min(), tz="UTC")).total_seconds() / 3600 if len(window) else np.nan)
        # False alarm: no onset (eligible or not) at this node within the next 7 days.
        for ts in a:
            if not ((all_on >= ts) & (all_on <= ts + np.timedelta64(LOOKBACK_H, "h"))).any():
                false_alarms += 1
    leads = np.array(leads, dtype=float)
    caught = leads[~np.isnan(leads)]
    weeks = (last_origin - first_origin).total_seconds() / (3600 * 168) or 1
    lo, hi = wilson_interval(len(caught), len(leads))
    return {"n_episodes": int(len(leads)), "n_caught": int(len(caught)),
            "episode_recall": float(len(caught) / len(leads)) if len(leads) else float("nan"),
            "recall_lo": lo, "recall_hi": hi,
            "median_lead_h": float(np.median(caught)) if len(caught) else float("nan"),
            "q25_lead_h": float(np.percentile(caught, 25)) if len(caught) else float("nan"),
            "q75_lead_h": float(np.percentile(caught, 75)) if len(caught) else float("nan"),
            "n_alerts": int(n_alerts), "false_alarms": int(false_alarms),
            "false_alarms_per_node_week": float(false_alarms / (n_nodes * weeks)),
            "false_alarm_share": float(false_alarms / n_alerts) if n_alerts else float("nan")}


def lead_time_tables(keys, wide, thr, onsets, inside, miss_cost: float, fa_cost: float, extra: dict) -> tuple:
    """Rule A (median crosses) and rule B (probability >= p*, p* chosen on VAL by cost) for every model."""
    rows, p_star = [], {}
    span = {sp: (keys.loc[keys["split"] == sp, "origin_ts"].min(), keys.loc[keys["split"] == sp, "origin_ts"].max())
            for sp in ("val", "test")}
    n_nodes = keys["node_id"].nunique()
    for m in wide:
        frames = {sp: alert_frame(keys, wide, m, sp, thr, inside) for sp in ("val", "test")}
        # Choose p* on validation: minimise 10 x missed episodes + 1 x false alarms (contracts/cost_policy.json).
        costs = {}
        for p in P_GRID:
            r = score_alerts(frames["val"][frames["val"]["prob"] >= p], onsets, *span["val"], n_nodes)
            costs[float(p)] = miss_cost * (r["n_episodes"] - r["n_caught"]) + fa_cost * r["false_alarms"]
        p_star[m] = max(p for p, c in costs.items() if c == min(costs.values()))
        for sp in ("val", "test"):
            f = frames[sp]
            for rule, alerts in (("A_median_crosses", f[f["crosses"]]), (f"B_prob_ge_{p_star[m]:.2f}", f[f["prob"] >= p_star[m]])):
                rows.append({"split": sp, "model": m, "rule": rule, **score_alerts(alerts, onsets, *span[sp], n_nodes)})
    # Stage 1 detection baseline (alarm when the node's recent shipments look critical), if available.
    for sp, alerts in extra.items():
        rows.append({"split": sp, "model": "stage1_risk", "rule": "stage1_threshold",
                     **score_alerts(alerts, onsets, *span[sp], n_nodes)})
    table = pd.DataFrame(rows)
    # One cost number per row: 10 per missed bottleneck + 1 per false alarm (contracts/cost_policy.json).
    table["warning_cost"] = miss_cost * (table["n_episodes"] - table["n_caught"]) + fa_cost * table["false_alarms"]
    # Each model's operating rule (A or B) is the one with the lower VALIDATION cost; ties go to fewer alarms.
    table["chosen_on_val"] = False
    for m, g in table[table["split"] == "val"].groupby("model"):
        best = g.sort_values(["warning_cost", "n_alerts"]).iloc[0]["rule"]
        table.loc[(table["model"] == m) & (table["rule"] == best), "chosen_on_val"] = True
    return table.round(4), p_star


def chosen_alerts(keys, wide, model: str, split: str, thr, inside, lead: pd.DataFrame, p_star: dict) -> pd.DataFrame:
    """The alerts a model raises with ITS validation-chosen rule (A: median crosses, B: probability >= p*)."""
    rule = lead[(lead["model"] == model) & lead["chosen_on_val"]]["rule"].iloc[0]
    frame = alert_frame(keys, wide, model, split, thr, inside)
    return frame[frame["crosses"]] if rule.startswith("A") else frame[frame["prob"] >= p_star[model]]


def stage1_alerts(keys: pd.DataFrame) -> dict:
    """Stage 1 champion's P(critical) averaged per node over each 6-hour block -> alert at its frozen threshold."""
    champ_path = stage_path(1, "reports", "eval_champion.json")
    pred_path = stage_path(1, "predictions", "ml_predictions.csv")
    prep_path = stage_path(1, "data", "prepared_dataset.csv.gz")
    if not (champ_path.exists() and pred_path.exists() and prep_path.exists()):
        return {}
    champ = json.loads(champ_path.read_text())
    pred = pd.read_csv(pred_path)
    pred = pred[pred["model_name"] == champ["model_name"]]
    prep = pd.read_csv(prep_path, usecols=["record_id", "timestamp", "current_node_id"])
    df = pred.merge(prep, on="record_id")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out = {}
    for sp in ("val", "test"):
        origins = np.sort(keys.loc[keys["split"] == sp, "origin_ts"].unique())
        part = df[df["split"] == sp].copy()
        # A shipment seen in the 6 hours before an origin counts for that origin (only the past is used).
        part["origin_ts"] = part["timestamp"].dt.ceil(f"{ORIGIN_STEP_H}h")
        agg = part.groupby(["current_node_id", "origin_ts"])["p_critical"].mean().reset_index()
        agg = agg[agg["origin_ts"].isin(origins) & (agg["p_critical"] >= champ["threshold"])]
        out[sp] = agg.rename(columns={"current_node_id": "node_id"})[["node_id", "origin_ts"]]
    return out


def plot_lead_time(table: pd.DataFrame, out: Path) -> None:
    """Recall, lead time and warning cost per model on TEST, each with its validation-chosen alert rule."""
    t = table[(table["split"] == "test") & table["chosen_on_val"]]
    t = t.sort_values("warning_cost", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    y = np.arange(len(t))
    err = np.vstack([t["episode_recall"] - t["recall_lo"], t["recall_hi"] - t["episode_recall"]])
    axes[0].barh(y, t["episode_recall"], xerr=err, color=[COLORS.get(m, "#555") for m in t["model"]])
    axes[0].set_yticks(y, t["model"])
    axes[0].set_xlabel("share of test bottleneck episodes warned about in advance")
    axes[0].set_title("Episode recall (test, 95% Wilson interval)")
    axes[1].barh(y, t["median_lead_h"], color=[COLORS.get(m, "#555") for m in t["model"]])
    axes[1].errorbar(t["median_lead_h"], y, xerr=np.vstack([t["median_lead_h"] - t["q25_lead_h"], t["q75_lead_h"] - t["median_lead_h"]]),
                     fmt="none", ecolor="black", lw=1)
    axes[1].set_yticks(y, t["model"])
    axes[1].set_xlabel("hours of warning before onset (median, IQR)")
    axes[1].set_title("Lead time of caught episodes (test)")
    axes[2].barh(y, t["warning_cost"], color=[COLORS.get(m, "#555") for m in t["model"]])
    axes[2].set_yticks(y, [f"{m} ({r.split('_')[0]})" for m, r in zip(t["model"], t["rule"])])
    axes[2].set_xlabel("10 x missed episodes + 1 x false alarms (lower is better)")
    axes[2].set_title("Cost of the early-warning decisions (test)")
    fig.suptitle("Early warning on test; each model uses the alert rule it chose on validation")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ 4: graph ablation

def ablation_table(keys, wide, y, ok, blocks, acc: pd.DataFrame, lead: pd.DataFrame, params: dict, n_boot: int) -> pd.DataFrame:
    """Does the REAL graph help? Paired skill of graph_full vs graph_none and vs graph_shuffled, per horizon (test)."""
    rows = []
    present = [m for m in GRAPH_VARIANTS + ("lstm",) if m in wide]
    for target in sorted(keys["target"].unique()):
        for h in sorted(keys["horizon_h"].unique()):
            sel = ((keys["split"] == "test") & (keys["target"] == target) & (keys["horizon_h"] == h)).to_numpy() & ok
            if not sel.any():
                continue
            row = {"target": target, "horizon_h": int(h)}
            for m in present:
                r = acc[(acc["split"] == "test") & (acc["model"] == m) & (acc["target"] == target) & (acc["horizon_h"] == h)]
                row[f"skill_{m}"] = float(r["skill_vs_persistence"].iloc[0])
            if "graph_full" in wide:
                e_full = np.abs(wide["graph_full"]["pred"][sel] - y[sel])
                for other in ("graph_none", "graph_shuffled", "lstm"):
                    if other in wide:
                        s, lo, hi = block_skill_ci(e_full, np.abs(wide[other]["pred"][sel] - y[sel]), blocks[sel], n=n_boot, seed=SEED)
                        row[f"full_vs_{other}"], row[f"full_vs_{other}_lo"], row[f"full_vs_{other}_hi"] = s, lo, hi
            rows.append(row)
    table = pd.DataFrame(rows)
    # Append lead time (test, each model's validation-chosen rule) and size, so the whole story is in one table.
    for m in present:
        r = lead[(lead["split"] == "test") & (lead["model"] == m) & lead.get("chosen_on_val", pd.Series(False, index=lead.index))]
        if len(r):
            table[f"recall_{m}"] = float(r["episode_recall"].iloc[0])
            table[f"median_lead_h_{m}"] = float(r["median_lead_h"].iloc[0])
        table[f"params_{m}"] = params.get(m)
    return table.round(4)


def ablation_verdict(table: pd.DataFrame, other: str) -> dict:
    """How many target-horizons graph_full is better than `other` with the paired 95% interval above zero."""
    col = f"full_vs_{other}"
    if col not in table:
        return {"available": False}
    better = int((table[f"{col}_lo"] > 0).sum())
    worse = int((table[f"{col}_hi"] < 0).sum())
    return {"available": True, "cells": int(len(table)), "graph_full_better": better, "graph_full_worse": worse,
            "no_clear_difference": int(len(table) - better - worse)}


# ------------------------------------------------------------------ 5: visual benchmark

def visual_benchmark(grid, keys, wide, thr, onsets, lead_alerts: dict, out_png: Path) -> list:
    """Three test episodes: measured dwell vs 24-hour-ahead forecasts of four models, with the graph model's band."""
    first, last = keys.loc[keys["split"] == "test", "origin_ts"].agg(["min", "max"])
    candidates = []
    for node, ons in onsets.items():
        g = grid[grid["node_id"] == node].set_index("hour_ts")["avg_dwell_time_hours"]
        for on in ons:
            if first + pd.Timedelta(hours=LOOKBACK_H) <= on <= last:
                peak = g[on:on + pd.Timedelta(hours=48)].max() / thr[node]
                candidates.append((peak, node, on))
    picks = sorted(candidates, reverse=True)[:3]
    if not picks:
        return []
    # Lines for the models whose alerts are passed in (the champion included); the Stage 1 alarm has no forecast line.
    show = [m for m in lead_alerts if m in wide]
    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 3.4 * len(picks)), squeeze=False)
    chosen = []
    for ax, (peak, node, on) in zip(axes[:, 0], picks):
        lo_t, hi_t = on - pd.Timedelta(days=7), on + pd.Timedelta(days=3)
        g = grid[(grid["node_id"] == node) & grid["hour_ts"].between(lo_t, hi_t)]
        ax.plot(g["hour_ts"], g["avg_dwell_time_hours"], color="black", lw=1, label="measured dwell")
        ax.axhline(thr[node], color="black", ls="--", lw=0.8, label="alert threshold (train p95)")
        ax.axvline(on, color="red", lw=1, label="episode onset")
        sel = ((keys["split"] == "test") & (keys["target"] == "dwell") & (keys["horizon_h"] == 24) & (keys["node_id"] == node)
               & keys["target_ts"].between(lo_t, hi_t)).to_numpy()
        ts = keys.loc[sel, "target_ts"]
        for m in show:
            ax.plot(ts, wide[m]["pred"][sel], color=COLORS[m], lw=1.6 if m == "graph_full" else 1, label=f"{m} (24 h ahead)")
        if "graph_full" in wide:
            ax.fill_between(ts, wide["graph_full"]["lo"][sel], wide["graph_full"]["hi"][sel], color=COLORS["graph_full"], alpha=0.15,
                            label="graph_full 80% band")
        first_alerts = {}
        for m, alerts in lead_alerts.items():
            a = alerts[(alerts["node_id"] == node) & alerts["origin_ts"].between(on - pd.Timedelta(hours=LOOKBACK_H), on)]
            if len(a):
                first_alerts[m] = a["origin_ts"].min()
                ax.scatter([first_alerts[m]], [thr[node]], marker="v", color=COLORS.get(m, "#555"), zorder=5, s=40)
        ax.set_title(f"{node}: onset {on:%Y-%m-%d %H:%M} UTC, peak {peak:.2f}x threshold (triangles = first alert, rule chosen on val)")
        ax.set_ylabel("dwell (hours)")
        chosen.append({"node_id": node, "onset": on.isoformat(), "peak_vs_threshold": round(float(peak), 3),
                       "first_alert_lead_h": {m: round((on - t).total_seconds() / 3600, 1) for m, t in first_alerts.items()}})
    axes[0, 0].legend(fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return chosen


# ------------------------------------------------------------------ 6: slices

def slices(keys, wide, y, ok, model: str) -> pd.DataFrame:
    """Skill of `model` vs persistence per node type and per node (test, dwell, 24 h and 168 h)."""
    types = pd.read_csv(NODES_CSV).set_index("node_id")["node_type"]
    rows = []
    for h in (24, 168):
        sel = ((keys["split"] == "test") & (keys["target"] == "dwell") & (keys["horizon_h"] == h)).to_numpy() & ok
        k = keys.loc[sel, ["node_id"]].copy()
        k["node_type"] = types.reindex(k["node_id"]).to_numpy()
        k["e_m"] = np.abs(wide[model]["pred"][sel] - y[sel])
        k["e_p"] = np.abs(wide["persistence"]["pred"][sel] - y[sel])
        for level in ("node_type", "node_id"):
            for name, g in k.groupby(level):
                rows.append({"horizon_h": h, "level": level, "slice": name, "n": len(g), "mae_model": g["e_m"].mean(),
                             "mae_persistence": g["e_p"].mean(), "skill": 1 - g["e_m"].sum() / g["e_p"].sum()})
    return pd.DataFrame(rows).round(4)


# ------------------------------------------------------------------ 7: real track

def real_track(path: Path, n_boot: int, out_dir: Path, champion: str) -> dict | None:
    """Skill, intervals and ablation on the REAL PortWatch track (daily ship counts)."""
    if not path.exists():
        return None
    df = load_forecasts(path, validate=False)
    grid = pd.read_csv(stage_path(STAGE, "data", "real_daily_activity.csv.gz"))
    grid["day_ts"] = pd.to_datetime(grid["date"]).dt.tz_localize("UTC")
    grid["is_imputed"] = grid["is_imputed"].astype(str).str.lower().eq("true")
    errors: list = []
    keys, wide, y = to_wide(df, errors)
    ok = attach_truth_from_grid(keys, grid, "day_ts", errors)
    # Real data blocks = ISO weeks of the forecast origin (daily forecasts within a week are correlated).
    blocks = keys["origin_ts"].dt.strftime("%G-%V").to_numpy()
    horizons = sorted(keys["horizon_h"].unique())
    acc, intervals = accuracy_tables(keys, wide, y, ok, blocks, n_boot, horizons)
    acc.to_csv(out_dir / "eval_real_forecast_metrics.csv", index=False)
    intervals.to_csv(out_dir / "eval_real_interval_quality.csv", index=False)
    plot_skill(acc, out_dir / "eval_real_skill_by_horizon.png", "REAL track: IMF PortWatch daily container ship counts (test 2025-26)")
    abl = ablation_table(keys, wide, y, ok, blocks, acc, pd.DataFrame(columns=["split", "model", "rule"]), {}, n_boot)
    abl.to_csv(out_dir / "eval_real_graph_ablation.csv", index=False)
    plot_real_example(keys, wide, out_dir / "eval_real_example.png", champion)
    t = acc[acc["split"] == "test"]
    pos = t.assign(pos=t["skill_lo"] > 0).groupby("model")["pos"].sum().astype(int).to_dict()
    return {"errors": errors, "n_test_forecasts": int(((keys["split"] == "test").to_numpy() & ok).sum()),
            "test_period": [str(keys.loc[keys["split"] == "test", "origin_ts"].min()), str(keys.loc[keys["split"] == "test", "origin_ts"].max())],
            "horizons_with_ci_positive_skill": pos, "n_horizons": len(horizons),
            "skill_test": {f"{r.model}|h{r.horizon_h}": [round(r.skill_vs_persistence, 4), round(r.skill_lo, 4), round(r.skill_hi, 4)]
                           for r in t.itertuples()},
            "skill_vs_best_test": {f"{r.model}|h{r.horizon_h}": [round(r.skill_vs_best_baseline, 4), round(r.skill_vs_best_baseline_lo, 4),
                                                                  round(r.skill_vs_best_baseline_hi, 4)] for r in t.itertuples()},
            "ablation_vs_none": ablation_verdict(abl, "graph_none"), "ablation_vs_shuffled": ablation_verdict(abl, "graph_shuffled")}


def plot_real_example(keys, wide, out: Path, champion: str) -> None:
    """Real 7-day-ahead forecasts of the champion at two nodes over the last 4 months of test, with the 80% band."""
    nodes = [n for n in ("CHOKE_BAB_EL_MANDEB", "PORT_US_LAX") if n in set(keys["node_id"])]
    h = max(keys["horizon_h"])
    model = champion if champion in wide else "graph_full"
    # 18 months of daily points is unreadable; the last 120 days of the test period show the behaviour clearly.
    start = keys.loc[keys["split"] == "test", "target_ts"].max() - pd.Timedelta(days=120)
    fig, axes = plt.subplots(len(nodes), 1, figsize=(12, 3.3 * len(nodes)), squeeze=False)
    for ax, node in zip(axes[:, 0], nodes):
        sel = ((keys["split"] == "test") & (keys["node_id"] == node) & (keys["horizon_h"] == h) & (keys["target_ts"] >= start)).to_numpy()
        ts = keys.loc[sel, "target_ts"]
        ax.plot(ts, keys.loc[sel, "y_true"], color="black", lw=0.8, label="real ships per day (PortWatch)")
        for m in ("seasonal_naive", model):
            if m in wide:
                ax.plot(ts, wide[m]["pred"][sel], color=COLORS[m], lw=1.4 if m == model else 1, label=f"{m} ({h // 24} days ahead)")
        ax.fill_between(ts, wide[model]["lo"][sel], wide[model]["hi"][sel], color=COLORS[model], alpha=0.15, label=f"{model} 80% band")
        ax.set_title(f"{node}: real daily container ship count vs {h // 24}-day-ahead forecast (test)")
        ax.set_ylabel("ships / day")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ 8: champion, ledger, report

def choose_champion(acc: pd.DataFrame, lead: pd.DataFrame) -> dict:
    """Best NEURAL model on VALIDATION: skill at 24 h for dwell, tie-broken by validation episode recall."""
    v = acc[(acc["split"] == "val") & (acc["target"] == "dwell") & (acc["horizon_h"] == 24) & ~acc["model"].isin(BASELINES)]
    if v.empty:
        return {}
    rec = lead[(lead["split"] == "val") & (lead["rule"] == "A_median_crosses")].set_index("model")["episode_recall"]
    v = v.assign(val_recall=v["model"].map(rec).fillna(0)).sort_values(["skill_vs_persistence", "val_recall"], ascending=False)
    best = v.iloc[0]
    base = acc[(acc["split"] == "val") & (acc["target"] == "dwell") & (acc["horizon_h"] == 24) & acc["model"].isin(BASELINES)]
    best_base = base.sort_values("skill_vs_persistence", ascending=False).iloc[0]
    return {"model": best["model"], "chosen_on": "validation: skill vs persistence for dwell at 24 h, then episode recall",
            "val_skill_dwell_24h": round(float(best["skill_vs_persistence"]), 4),
            "best_baseline_on_val": best_base["model"], "best_baseline_val_skill_dwell_24h": round(float(best_base["skill_vs_persistence"]), 4),
            "beats_best_baseline_on_val": bool(best["skill_vs_persistence"] > best_base["skill_vs_persistence"])}


def md_table(df: pd.DataFrame) -> str:
    """A small markdown table without extra libraries."""
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False))


def fmt_ci(v, lo, hi, d: int = 3) -> str:
    """Format 'value [lo, hi]'."""
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def write_ledger(m: dict) -> None:
    """One ledger row per Stage 2 number we may say in the debate (PENDING until a clean-clone re-run)."""
    cmd = "python Stage02_DL/04_evaluation_engineer.py"
    src = "Stage02_DL/reports/eval_metrics.json"
    c = m["champion"]["model"]
    rows = [
        ("C-DL-01", f"{c} skill vs persistence, dwell, 24 h ahead (test)", m["headline"]["skill_dwell_24h"], src),
        ("C-DL-02", f"{c} skill vs persistence, dwell, 168 h (7 days) ahead (test)", m["headline"]["skill_dwell_168h"], src),
        ("C-DL-03", f"Horizons (of 5) where {c} has skill > 0 with the 95% interval above 0, dwell / throughput (test)",
         m["headline"]["positive_skill_horizons"], "Stage02_DL/reports/eval_forecast_metrics.csv"),
        ("C-DL-04", f"{c} skill vs the best baseline (chosen on val), dwell, 24 h (test)", m["headline"]["skill_vs_best_baseline_dwell_24h"], src),
        ("C-DL-05", f"{c} warns about dwell bottleneck episodes in advance: recall [95% CI] (n episodes), test, alert rule chosen on val ({m['headline']['alert_rule_chosen_on_val']})",
         m["headline"]["episode_recall"], "Stage02_DL/reports/eval_lead_time.csv"),
        ("C-DL-06", f"{c} median lead time before onset [IQR], hours (test, same rule)", m["headline"]["median_lead_h"], "Stage02_DL/reports/eval_lead_time.csv"),
        ("C-DL-07", f"{c} false alarms per node-week (test, same rule)", m["headline"]["false_alarms_per_node_week"], "Stage02_DL/reports/eval_lead_time.csv"),
        ("C-DL-13", "Early-warning cost on test (10 per missed bottleneck + 1 per false alarm; each model's rule chosen on val), lower is better",
         m["headline"]["warning_cost_test"], "Stage02_DL/reports/eval_lead_time.csv"),
        ("C-DL-08", f"{c} 80% interval coverage, dwell, all horizons (test; target 0.80)", m["headline"]["coverage_dwell"], "Stage02_DL/reports/eval_interval_quality.csv"),
        ("C-DL-09", "Graph ablation: target-horizons where graph_full beats graph_none (paired 95% CI above 0), test",
         m["headline"]["ablation_vs_none"], "Stage02_DL/reports/eval_graph_ablation.csv"),
        ("C-DL-10", "Graph ablation: target-horizons where graph_full beats graph_shuffled (paired 95% CI above 0), test",
         m["headline"]["ablation_vs_shuffled"], "Stage02_DL/reports/eval_graph_ablation.csv"),
        ("C-DL-11", "Leakage and time-order checks on the forecast file (origins, purge gap, y_true re-read from the grid)",
         "PASS" if m["trust"]["passed"] else "FAIL", "Stage02_DL/reports/eval_leakage_tests.json"),
    ]
    if m.get("real_track"):
        rows.append(("C-DL-12", f"REAL data (IMF PortWatch daily ship counts, test 2025-26): {c} skill vs persistence by horizon",
                     m["headline"]["real_skill"], "Stage02_DL/reports/eval_real_forecast_metrics.csv"))
        rows.append(("C-DL-14", f"REAL data: {c} skill vs the best baseline chosen on val, by horizon (test 2025-26)",
                     m["headline"]["real_skill_vs_best_baseline"], "Stage02_DL/reports/eval_real_forecast_metrics.csv"))
    for cid, claim, value, source in rows:
        ledger.upsert(cid, 2, claim, value, source, cmd, status="PENDING")


def write_report(path: Path, m: dict, acc: pd.DataFrame, lead: pd.DataFrame, abl_v: dict, chosen: list) -> None:
    """Plain-English summary of every Stage 2 result (the file the debate team reads first)."""
    c = m["champion"]["model"]
    h = m["headline"]
    lines = [
        "# Stage 2 evaluation (deep learning forecasts)",
        "",
        "> **What this shows:** how well each forecaster predicts node dwell and throughput 1 h to 7 days ahead, how early",
        "> it warns about dwell bottlenecks, and whether the graph (trade lanes) really helps. **Reproduce:**",
        "> `python Stage02_DL/04_evaluation_engineer.py`. Every number below is in `eval_metrics.json` or a CSV next to it.",
        "",
        f"**Champion (chosen on validation, never on test):** `{c}`. Beats the best baseline on validation: "
        f"{m['champion']['beats_best_baseline_on_val']} (best baseline `{m['champion']['best_baseline_on_val']}`).",
        "",
        "## Headline numbers (test period)",
        f"- Skill vs persistence, dwell 24 h: **{h['skill_dwell_24h']}**; 168 h: **{h['skill_dwell_168h']}**.",
        f"- Horizons with CI-positive skill (dwell / throughput): **{h['positive_skill_horizons']}** (target: >= 4 of 5).",
        f"- Skill vs the best baseline (dwell 24 h): **{h['skill_vs_best_baseline_dwell_24h']}**.",
        f"- Early warning (rule {h['alert_rule_chosen_on_val']}, chosen on validation): recall **{h['episode_recall']}**, "
        f"median lead **{h['median_lead_h']} h**, false alarms per node-week **{h['false_alarms_per_node_week']}**.",
        f"- Early-warning cost on test (10 per miss + 1 per false alarm): **{h['warning_cost_test']}**.",
        f"- 80% interval coverage (dwell): **{h['coverage_dwell']}**.",
        f"- Graph ablation vs graph_none: {h['ablation_vs_none']}; vs graph_shuffled: {h['ablation_vs_shuffled']}.",
    ]
    if m.get("real_track"):
        lines.append(f"- REAL data (PortWatch, test 2025-26), skill vs persistence: **{h['real_skill']}**; "
                     f"vs best baseline: **{h['real_skill_vs_best_baseline']}**.")
    lines += ["", "## Honest reading", *m["honest_notes"], "",
              "## Visual benchmark episodes (test)", *[f"- {e['node_id']} onset {e['onset']}: first-alert lead (h) {e['first_alert_lead_h']}" for e in chosen],
              "", "## Files", "- `eval_forecast_metrics.csv`, `eval_skill_by_horizon.png`: accuracy and skill with 95% CIs",
              "- `eval_interval_quality.csv`: coverage, width, interval score", "- `eval_lead_time.csv`, `eval_lead_time.png`: early warning",
              "- `eval_graph_ablation.csv`: does the real graph help", "- `eval_visual_benchmark.png/.csv/.md`: problem-statement benchmark",
              "- `eval_slices.csv`, `eval_leakage_tests.json`, `eval_champion.json`, `eval_alert_thresholds.csv`",
              "- `eval_real_*`: the same on REAL IMF PortWatch data"]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    """Run the whole Stage 2 evaluation."""
    args = parse_args()
    set_global_seed(SEED)
    out_dir = args.out_dir or stage_path(STAGE, "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    use_ledger = not args.no_ledger and args.out_dir is None

    # Step 0: load and trust-check before computing any number.
    grid = load_grid()
    bounds = json.loads(stage_path(STAGE, "data", "splits.json").read_text())
    df = load_forecasts(args.predictions, validate=True)
    errors: list = []
    keys, wide, y = to_wide(df, errors)
    ok = attach_truth_from_grid(keys, grid, "hour_ts", errors)
    trust = time_checks(keys, bounds, input_steps=72, step_h=1)
    trust["y_true_matches_grid_and_models_aligned"] = not errors
    trust["errors"] = errors
    trust["imputed_targets_excluded"] = int((~ok).sum())
    if "persistence" not in wide:
        raise SystemExit("No persistence rows: every model must ship next to its baseline (Role 1 rule 3).")
    trust["persistence_self_skill_is_zero"] = bool(abs(block_skill_ci(np.abs(wide["persistence"]["pred"] - y)[ok],
                                                                      np.abs(wide["persistence"]["pred"] - y)[ok],
                                                                      keys["origin_ts"].dt.date.to_numpy()[ok], n=10)[0]) < 1e-12)
    trust["passed"] = trust["passed"] and not errors and trust["persistence_self_skill_is_zero"]
    print(f"trust checks: {'PASS' if trust['passed'] else 'FAIL'} ({len(errors)} errors); {len(wide)} models")
    if not trust["passed"]:
        (out_dir / "eval_leakage_tests.json").write_text(json.dumps(trust, indent=2, default=str))
        print("BLOCKED: trust checks failed; no metrics, champion or ledger rows written. See eval_leakage_tests.json.")
        return 1

    # Step 1-2: accuracy, skill and interval quality (blocks = calendar day of the forecast origin).
    blocks = keys["origin_ts"].dt.strftime("%Y-%m-%d").to_numpy()
    horizons = sorted(keys["horizon_h"].unique())
    acc, intervals = accuracy_tables(keys, wide, y, ok, blocks, args.n_boot, horizons)
    acc.to_csv(out_dir / "eval_forecast_metrics.csv", index=False)
    intervals.to_csv(out_dir / "eval_interval_quality.csv", index=False)
    plot_skill(acc, out_dir / "eval_skill_by_horizon.png", "Simulated track: skill vs persistence by horizon (test, 95% block-bootstrap CI)")
    print("accuracy and intervals done")

    # Step 3: lead time. Thresholds come from TRAIN hours only; p* for rule B is chosen on VAL.
    train_end = pd.Timestamp(bounds["train_end"])
    thr = node_thresholds(grid, train_end)
    thr.rename("dwell_alert_threshold_h").round(3).to_csv(out_dir / "eval_alert_thresholds.csv")
    onsets, inside = truth_episodes(grid, thr)
    cm = load_cost_matrix()
    miss_cost, fa_cost = float(cm[2, 0]), float(cm[0, 2])   # missed critical = 10, unnecessary alarm = 1
    s1 = stage1_alerts(keys)
    lead, p_star = lead_time_tables(keys, wide, thr, onsets, inside, miss_cost, fa_cost, s1)
    lead.to_csv(out_dir / "eval_lead_time.csv", index=False)
    plot_lead_time(lead, out_dir / "eval_lead_time.png")
    print("lead time done")

    # Step 8 (early): champion on validation, so every later "headline" uses a model chosen without test data.
    champion = choose_champion(acc, lead)
    c = champion["model"]
    champion["alert_rule"] = {"A": "alert when any horizon's median dwell forecast >= node threshold",
                              "B": f"alert when max P(dwell >= threshold) >= {p_star[c]:.2f} (p* chosen on validation by cost)",
                              "p_star": p_star[c], "thresholds_file": "Stage02_DL/reports/eval_alert_thresholds.csv",
                              "operating_rule": str(lead[(lead["model"] == c) & lead["chosen_on_val"]]["rule"].iloc[0]),
                              "operating_rule_chosen_on": "validation warning cost (10 per missed episode + 1 per false alarm)"}
    champion["model_version"] = str(df.loc[df["model_name"] == c, "model_version"].iloc[0])
    (out_dir / "eval_champion.json").write_text(json.dumps(champion, indent=2))

    # Step 4: graph ablation (parameter counts from Role 1's model card).
    card_path = stage_path(STAGE, "artifacts", "model_card.json")
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    params = {k: v.get("parameters") for k, v in card.get("tracks", {}).get("sim", {}).get("models", {}).items()}
    abl = ablation_table(keys, wide, y, ok, blocks, acc, lead, params, args.n_boot)
    abl.to_csv(out_dir / "eval_graph_ablation.csv", index=False)
    abl_none, abl_shuf = ablation_verdict(abl, "graph_none"), ablation_verdict(abl, "graph_shuffled")

    # Step 5: visual benchmark (each model's validation-chosen alerts, plus the Stage 1 detection baseline).
    shown = [m for m in ("persistence", "xgb_lag", "lstm", "graph_full", c) if m in wide]
    alerts_by_model = {m: chosen_alerts(keys, wide, m, "test", thr, inside, lead, p_star) for m in dict.fromkeys(shown)}
    if "test" in s1:
        alerts_by_model["stage1_risk"] = s1["test"]
    chosen = visual_benchmark(grid, keys, wide, thr, onsets, alerts_by_model, out_dir / "eval_visual_benchmark.png")
    bench = lead[(lead["split"] == "test") & lead["chosen_on_val"]
                 & lead["model"].isin(["persistence", "xgb_lag", "lstm", "transformer", "graph_full", "stage1_risk"])]
    bench.to_csv(out_dir / "eval_visual_benchmark.csv", index=False)
    (out_dir / "eval_visual_benchmark.md").write_text(
        "# Visual prediction benchmark (test)\n\nBaseline ML detection (`xgb_lag`; `stage1_risk` = Stage 1 champion's shipment "
        "P(critical) averaged per node, at its frozen threshold) vs the DL forecasters, same episodes, same alert scoring.\n\n"
        + md_table(bench[["model", "rule", "n_episodes", "n_caught", "episode_recall", "median_lead_h", "false_alarms_per_node_week", "warning_cost"]])
        + "\n\nEpisodes in the figure: " + json.dumps(chosen, default=str) + "\n")

    # Step 6: slices, Diebold-Mariano tests, leakage file.
    slices(keys, wide, y, ok, c).to_csv(out_dir / "eval_slices.csv", index=False)
    dm = {}
    for other in ("persistence", "xgb_lag", "graph_none", "lstm"):
        if other in wide and other != c:
            for h in (24, 168):
                sel = ((keys["split"] == "test") & (keys["target"] == "dwell") & (keys["horizon_h"] == h)).to_numpy() & ok
                k = keys.loc[sel, ["origin_ts"]].assign(a=np.abs(wide[c]["pred"][sel] - y[sel]), b=np.abs(wide[other]["pred"][sel] - y[sel]))
                per_origin = k.groupby("origin_ts")[["a", "b"]].mean()
                dm[f"{c}_vs_{other}_dwell_{h}h"] = diebold_mariano(per_origin["a"], per_origin["b"], max_lag=int(np.ceil(h / ORIGIN_STEP_H)))
    (out_dir / "eval_leakage_tests.json").write_text(json.dumps(trust, indent=2, default=str))

    # Step 7: the real track.
    real = real_track(args.real_predictions, args.n_boot, out_dir, c)

    # Headline numbers (champion, test) and honest notes.
    def acc_row(model, target, h, split="test"):
        return acc[(acc["split"] == split) & (acc["model"] == model) & (acc["target"] == target) & (acc["horizon_h"] == h)].iloc[0]

    t = acc[acc["split"] == "test"]
    pos = t.assign(pos=t["skill_lo"] > 0).groupby(["model", "target"])["pos"].sum().astype(int)
    test_chosen = lead[(lead["split"] == "test") & lead["chosen_on_val"]].set_index("model")
    lead_c = test_chosen.loc[c]
    persist_a = lead[(lead["split"] == "test") & (lead["model"] == "persistence") & (lead["rule"] == "A_median_crosses")].iloc[0]
    cov = intervals[(intervals["split"] == "test") & (intervals["model"] == c) & (intervals["target"] == "dwell")]["coverage_80"].mean()
    r24, r168 = acc_row(c, "dwell", 24), acc_row(c, "dwell", 168)
    headline = {
        "skill_dwell_24h": fmt_ci(r24.skill_vs_persistence, r24.skill_lo, r24.skill_hi),
        "skill_dwell_168h": fmt_ci(r168.skill_vs_persistence, r168.skill_lo, r168.skill_hi),
        "positive_skill_horizons": f"{pos.get((c, 'dwell'), 0)}/5 dwell, {pos.get((c, 'throughput'), 0)}/5 throughput",
        "skill_vs_best_baseline_dwell_24h": fmt_ci(r24.skill_vs_best_baseline, r24.skill_vs_best_baseline_lo, r24.skill_vs_best_baseline_hi)
                                            + f" vs {r24.best_baseline_on_val}",
        "episode_recall": f"{lead_c.episode_recall:.2f} [{lead_c.recall_lo:.2f}, {lead_c.recall_hi:.2f}] (n = {lead_c.n_episodes})",
        "median_lead_h": f"{lead_c.median_lead_h:.0f} [{lead_c.q25_lead_h:.0f}, {lead_c.q75_lead_h:.0f}]",
        "alert_rule_chosen_on_val": str(lead_c.rule),
        "warning_cost_test": f"{c} {lead_c.warning_cost:.0f} vs " + ", ".join(
            f"{m} {test_chosen.loc[m, 'warning_cost']:.0f}" for m in ("lstm", "xgb_lag", "persistence", "stage1_risk") if m in test_chosen.index and m != c),
        "false_alarms_per_node_week": f"{lead_c.false_alarms_per_node_week:.2f}",
        "coverage_dwell": f"{cov:.3f}",
        "ablation_vs_none": f"better in {abl_none.get('graph_full_better')}/{abl_none.get('cells')}, worse in {abl_none.get('graph_full_worse')}",
        "ablation_vs_shuffled": f"better in {abl_shuf.get('graph_full_better')}/{abl_shuf.get('cells')}, worse in {abl_shuf.get('graph_full_worse')}",
    }
    if real:
        rp = real["horizons_with_ci_positive_skill"]
        headline["real_positive_skill"] = f"{c} {rp.get(c, 0)}/{real['n_horizons']} horizons (CI above 0)"
        headline["real_skill"] = "; ".join(f"{k.split('|h')[1]} h: {fmt_ci(*v)}" for k, v in real["skill_test"].items() if k.startswith(c + "|"))
        headline["real_skill_vs_best_baseline"] = "; ".join(f"{k.split('|h')[1]} h: {fmt_ci(*v)}" for k, v in real["skill_vs_best_test"].items() if k.startswith(c + "|"))
    notes = [f"- The champion was picked on validation; its test skill is reported as is, even where it is not significant."]
    if not champion["beats_best_baseline_on_val"]:
        notes.append(f"- **No neural model beats the best baseline ({champion['best_baseline_on_val']}) on validation at 24 h.** Say so first.")
    notes.append(f"- **Recall alone is not a fair score.** With rule A even persistence catches {persist_a.n_caught}/{persist_a.n_episodes} "
                 f"test episodes (noisy readings crossing the line), at {persist_a.false_alarms} false alarms. Compare models on the "
                 "warning cost of their validation-chosen rule: " + headline["warning_cost_test"] + ".")
    if abl_none.get("graph_full_better", 0) == 0:
        notes.append(f"- **The graph does not clearly help** on the simulated track: graph_full vs graph_none {headline['ablation_vs_none']}, "
                     f"vs graph_shuffled {headline['ablation_vs_shuffled']} (target x horizon cells, paired 95% CI). That is a finding, not a failure.")
    if real and real["ablation_vs_none"].get("graph_full_better", 0) == 0:
        notes.append("- The graph does not help on the REAL track either (eval_real_graph_ablation.csv).")
    if "-fast" in champion["model_version"]:
        notes.append("- These numbers come from the --fast training run (hidden 32, <= 10 epochs); claims stay PENDING until the full run.")
    max_skill = float(t["skill_vs_persistence"].max())
    if max_skill > 0.9:
        notes.append(f"- WARNING: a skill of {max_skill:.2f} is suspiciously high; investigate leakage before quoting it.")
    notes.append("- Data are simulated (SYNTHETIC_CALIBRATED); the real-data track checks the same models on IMF PortWatch counts.")
    metrics = {"champion": champion, "headline": headline, "trust": {"passed": trust["passed"]}, "diebold_mariano": dm,
               "p_star_by_model": p_star, "episode_threshold": f"node p{EPISODE_PCT} of TRAIN dwell", "lookback_h": LOOKBACK_H,
               "n_test_episodes_eligible": int(lead_c.n_episodes), "graph_ablation": {"vs_none": abl_none, "vs_shuffled": abl_shuf},
               "positive_skill_horizons_by_model": {f"{m}|{tg}": int(v) for (m, tg), v in pos.items()},
               "real_track": real, "honest_notes": notes, "n_boot": args.n_boot}
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    write_report(out_dir / "eval_report.md", metrics, acc, lead, abl_none, chosen)
    print(json.dumps(headline, indent=2))
    if use_ledger:
        write_ledger(metrics)
        print("ledger: Stage 2 rows written to claims/CLAIMS_LEDGER.md (status PENDING until a clean-clone run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
