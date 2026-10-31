"""ROLE: DL ENGINEER  |  STAGE 02 - DEEP LEARNING
Owner: Role 1 - ML Engineer (model role). Rewritten by Role 4 at the team lead's request; Role 1 to review.
Full instructions: docs/roles/ROLE_1_ML_ENGINEER.md (Day 2 section)

WHAT THIS SCRIPT DOES
    Trains 7-day multi-horizon forecasters with 80% intervals and writes them in the dl_forecasts contract format.
    Simulated track (hourly, 27 nodes, targets dwell + throughput, horizons 1/6/24/72/168 h):
        baselines  persistence, seasonal_naive (same hour last week), rolling_mean (last 24 h), xgb_lag (XGBoost)
        neural     lstm (shared LSTM + node embedding), transformer (encoder, 2 layers x 4 heads),
                   graph_full (LSTM + graph convolution + transit-delayed lane messages),
                   graph_shuffled (same model, lanes relabelled at random), graph_none (same model, no lanes)
    Real track (daily IMF PortWatch ship counts, 23 nodes, horizons 1/3/7 days): the same models, when the data
        engineer has built Stage02_DL/data/real_daily_activity.csv.gz.
    Every model is fitted on TRAIN, early-stopped on VAL, gets the same split-conformal interval correction on VAL,
    and is then asked for VAL and TEST forecasts. Test numbers are never looked at here.

READS : Stage02_DL/data/hourly_node_telemetry.csv.gz, scaler.json, splits.json      (Role 3)
        Stage02_DL/data/real_daily_activity.csv.gz, real_scaler.json, real_splits.json  (Role 3, optional)
        contracts/network_nodes.csv, contracts/network_edges.csv
WRITES: Stage02_DL/predictions/dl_forecasts.csv.gz        (contract dl_forecasts, simulated track)
        Stage02_DL/predictions/dl_real_forecasts.csv.gz   (same columns, real track, target = activity)
        Stage02_DL/artifacts/*.pt (weights + config + interval margins), xgb_lag_*.joblib,
        Stage02_DL/artifacts/model_card.json, training_curve.csv

RUN   : python 03_dl_engineer.py --fast      (from Stage02_DL/; about 10 min on a laptop CPU: hidden 32, <= 10 epochs)
        python 03_dl_engineer.py             (full run: hidden 64, <= 30 epochs; ~45-60 min on CPU)
        Partial runs (--models / --epochs / --track sim|real) write to artifacts/scratch/ and predictions/scratch/ only.
        python 03_dl_engineer.py --noise-channel   (Role 2's sanity test: trains one LSTM with 2 extra noise inputs,
                                                    saves artifacts/noise_check/lstm_noise.pt, writes no forecasts)
"""

import os

# PyTorch and XGBoost each bring their own OpenMP library. On macOS the two deadlock in one process unless OpenMP
# itself is limited to one thread; both libraries still use several threads through their own settings below.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

# Make "import common" work when this file is run from inside the stage folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from common.contracts import write_validated_csv  # noqa: E402
from common.modelkit import save_model_card  # noqa: E402
from common.modelkit.dl_data import (  # noqa: E402
    SIM_TRACK, ForecastData, real_data_from_grid, shuffle_permutation, sim_data_from_grid, split_origins,
    symmetric_adjacency, transit_edges,
)
from common.modelkit.dl_models import build_model, count_parameters, save_model  # noqa: E402
from common.modelkit.dl_train import (  # noqa: E402
    apply_margins, conformal_margins, long_frame, naive_point, predict_scaled, predict_xgb_lag, residual_quantiles,
    train_neural, train_xgb_lag, val_pinball, with_residual_band,
)
from common.paths import EDGES_CSV, NODES_CSV, stage_path  # noqa: E402
from common.seed import SEED, set_global_seed  # noqa: E402

STAGE = 2
MODEL_VERSION = "v2.0"
THREADS = 4                          # CPU threads for PyTorch and XGBoost (a typical 8-core laptop)
SIM_EVAL_HOURS = (0, 6, 12, 18)      # role file: val/test forecast origins every 6 hours
BASELINES = ("persistence", "seasonal_naive", "rolling_mean")
NEURAL = ("lstm", "transformer", "graph_full", "graph_shuffled", "graph_none")
# Where outputs go. A PARTIAL run (--models, --epochs or one --track only) writes to scratch/ sub-folders
# (git-ignored), so a quick experiment can never overwrite the complete, committed hand-off files.
SCRATCH = {"on": False}


def out_path(sub: str, name: str = ""):
    """Output path in artifacts/ or predictions/, or in their scratch/ sub-folder during a partial run."""
    folder = stage_path(STAGE, sub) / ("scratch" if SCRATCH["on"] else "")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name if name else folder


ARTIFACT_NAMES = {"lstm": "lstm_forecaster", "transformer": "transformer_forecaster", "graph_full": "graph_forecaster",
                  "graph_shuffled": "graph_shuffled", "graph_none": "graph_none"}


def _get_git_commit() -> str:
    """Return short commit SHA if in git repo, else 'unknown'."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def log_hparam_row(row: dict) -> None:
    """Append one row to Stage02_DL/artifacts/hparam_log.csv (creating with header if missing).

    This file is kept outside scratch/ so hyper-parameter exploration on validation is preserved.
    """
    log_path = stage_path(STAGE, "artifacts", "hparam_log.csv")
    cols = ["utc_time", "tag", "track", "model", "hidden", "input_len", "epochs_run",
            "best_epoch", "val_pinball_common", "parameters", "train_seconds", "train_stride", "git_commit"]
    df = pd.DataFrame([[row.get(c, "") for c in cols]], columns=cols)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode="a", header=False, index=False)


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="hidden 32, <= 10 epochs, sparser training origins (~10 min)")
    ap.add_argument("--track", choices=["sim", "real", "both"], default="both")
    ap.add_argument("--models", nargs="*", default=None, help="subset of models to train, e.g. lstm graph_full")
    ap.add_argument("--epochs", type=int, default=None, help="override the epoch limit")
    ap.add_argument("--noise-channel", action="store_true", help="Role 2's attribution sanity test (see RUN above)")
    ap.add_argument("--hidden", type=int, default=None, help="override hidden dimension (default 64, or 32 in --fast)")
    ap.add_argument("--input-len", type=int, default=None, help="override input window steps for sim track (default 72)")
    ap.add_argument("--tag", type=str, default=None, help="experiment label for artifacts/hparam_log.csv")
    ap.add_argument("--fold", type=int, choices=[1, 2, 3], default=None, help="rolling-origin backtest fold (1, 2, or 3)")
    return ap.parse_args()


def settings(fast: bool, track: str, epochs: int | None, hidden: int | None = None) -> dict:
    """Hyper-parameters per mode (role file: full = hidden 64, <= 30 epochs, patience 5; fast = hidden 32, 10 epochs)."""
    h_dim = hidden if hidden is not None else (32 if fast else 64)
    s = {"hidden": h_dim, "layers": 2, "epochs": 10 if fast else 30, "patience": 3 if fast else 5,
         "lr": 1e-3, "batch_origins": 16, "xgb_trees": 300 if fast else 600,
         # Hourly track: use every 6th (fast) or 2nd (full) training hour as an origin; neighbouring hours are
         # almost identical windows, so this loses little and makes an epoch 2-6x faster. Daily track: every day.
         "train_stride": (6 if fast else 2) if track == "sim" else 1}
    if epochs:
        s["epochs"] = epochs
    return s


def load_track(track: str, noise: bool = False, input_len: int | None = None,
               custom_scaler: dict | None = None) -> tuple:
    """Return (ForecastData, split boundaries, lane table) for the simulated or the real track (None if missing)."""
    edges = pd.read_csv(EDGES_CSV)
    node_ids = pd.read_csv(NODES_CSV)["node_id"].tolist()
    if track == "sim":
        grid = pd.read_csv(stage_path(STAGE, "data", "hourly_node_telemetry.csv.gz"))
        scaler = custom_scaler if custom_scaler is not None else json.loads(stage_path(STAGE, "data", "scaler.json").read_text())
        bounds = json.loads(stage_path(STAGE, "data", "splits.json").read_text())
        spec = dataclasses.replace(SIM_TRACK, input_len=input_len) if input_len is not None else SIM_TRACK
        return sim_data_from_grid(grid, node_ids, scaler, noise_channels=noise, spec=spec), bounds, edges
    path = stage_path(STAGE, "data", "real_daily_activity.csv.gz")
    if not path.exists():
        return None, None, edges
    scaler = json.loads(stage_path(STAGE, "data", "real_scaler.json").read_text())
    bounds = json.loads(stage_path(STAGE, "data", "real_splits.json").read_text())
    return real_data_from_grid(pd.read_csv(path), scaler), bounds, edges


def model_config(name: str, data: ForecastData, edges: pd.DataFrame, s: dict) -> tuple:
    """Return (kind, config) for a neural model; the three graph variants differ ONLY in their lanes."""
    spec = data.spec
    cfg = {"n_inputs": data.n_inputs, "n_nodes": data.n_nodes, "n_horizons": len(spec.horizons),
           "n_targets": len(data.target_idx), "hidden": s["hidden"], "layers": s["layers"],
           "input_len": spec.input_len, "n_ctx_channels": len(spec.channels), "n_blocks": spec.n_blocks}
    if name == "lstm":
        return "lstm", {**cfg, "dropout": 0.2}
    if name == "transformer":
        return "transformer", {**cfg, "dropout": 0.1, "heads": 4}
    perm = shuffle_permutation(data.n_nodes) if name == "graph_shuffled" else None
    cfg["adjacency"] = symmetric_adjacency(data.node_ids, edges, permutation=perm, identity=(name == "graph_none"))
    all_edges = transit_edges(data.node_ids, edges, spec, permutation=perm)
    if name == "graph_none":
        # No lanes at all: an empty edge list, so no messages are passed (same weights, nothing to read).
        all_edges = {k: v[:0] for k, v in all_edges.items()}
    cfg["edges"] = all_edges
    return "graph", {**cfg, "dropout": 0.2}


def to_reported_units(data: ForecastData, q_scaled: np.ndarray) -> np.ndarray:
    """Scaled quantiles -> original units; physical quantities cannot be negative, so bounds are floored at 0."""
    return np.maximum(data.to_original(q_scaled), 0.0)


def run_track(track: str, args: argparse.Namespace, card: dict, curves: list) -> pd.DataFrame | None:
    """Train every requested model on one track and return its val+test forecasts as one long DataFrame."""
    data, bounds, edges = load_track(track, input_len=args.input_len if track == "sim" else None)
    if data is None:
        print(f"[{track}] skipped: no prepared data (run 01_data_engineer.py after scripts/fetch_real_data.py)")
        card["tracks"][track] = {"status": "SKIPPED: real_daily_activity.csv.gz not found"}
        return None
    s = settings(args.fast, track, args.epochs, hidden=args.hidden)
    spec = data.spec
    wanted = args.models or list(BASELINES) + ["xgb_lag"] + list(NEURAL)
    print(f"\n=== {track} track: {data.n_nodes} nodes, {len(data.times)} steps of {spec.step_hours} h, "
          f"horizons {spec.horizons} steps ===")

    # Step 1: forecast origins. Training uses a strided subset; val/test use fixed clock hours (sim) or every day.
    all_origins = split_origins(data.times, spec, bounds)
    train_o = all_origins["train"][::s["train_stride"]]
    eval_o = split_origins(data.times, spec, bounds, origin_hours=SIM_EVAL_HOURS if track == "sim" else None)
    val_o, test_o = eval_o["val"], eval_o["test"]
    print(f"origins: train {len(train_o)} (stride {s['train_stride']}), val {len(val_o)}, test {len(test_o)}")
    y_val = data.batch(val_o)["y"]
    mask_val = ~data.target_imputed(val_o)

    forecasts, info, trained = {}, {}, {}

    # Step 2: simple baselines. Their 80% band = their own TRAIN errors (10th/90th percentile) per node and horizon.
    train_dense = all_origins["train"][::max(1, s["train_stride"])]
    y_train = data.batch(train_dense)["y"]
    mask_train = ~data.target_imputed(train_dense)
    for name in BASELINES:
        if name not in wanted:
            continue
        band = residual_quantiles(naive_point(data, train_dense, name), y_train, mask_train)
        forecasts[name] = {sp: with_residual_band(naive_point(data, o, name), band) for sp, o in (("val", val_o), ("test", test_o))}
        info[name] = {"kind": "baseline", "parameters": 0}

    # Step 3: XGBoost on lag features (the "baseline ML" the DL models must beat).
    if "xgb_lag" in wanted:
        print("training xgb_lag ...")
        t0 = time.time()
        xgb_bundle = train_xgb_lag(data, train_o, val_o, n_estimators=s["xgb_trees"], n_jobs=THREADS)
        forecasts["xgb_lag"] = {sp: predict_xgb_lag(xgb_bundle, data, o) for sp, o in (("val", val_o), ("test", test_o))}
        joblib.dump(xgb_bundle, out_path("artifacts", f"xgb_lag_{track}.joblib"), compress=3)
        info["xgb_lag"] = {"kind": "xgboost", "train_seconds": round(time.time() - t0, 1),
                           "best_iterations": {f"h{spec.horizons[h]}_{data.target_names[t]}": int(m.best_iteration)
                                               for (h, t), m in xgb_bundle["models"].items()}}

    # Step 4: the neural models, each trained from the same seed so differences come from the design, not luck.
    for name in NEURAL:
        if name not in wanted:
            continue
        set_global_seed(SEED)
        kind, cfg = model_config(name, data, edges, s)
        model = build_model(kind, cfg)
        print(f"training {name} ({count_parameters(model):,} parameters) ...")
        t0 = time.time()
        model, curve = train_neural(model, data, train_o, val_o, epochs=s["epochs"], batch_size=s["batch_origins"],
                                    lr=s["lr"], patience=s["patience"])
        forecasts[name] = {sp: predict_scaled(model, data, o) for sp, o in (("val", val_o), ("test", test_o))}
        curves.extend({"track": track, "model": name, **row} for row in curve)
        info[name] = {"kind": kind, "parameters": count_parameters(model), "train_seconds": round(time.time() - t0, 1),
                      "epochs_run": len(curve), "best_epoch": curve[-1]["best_epoch"] if curve else None,
                      "val_pinball_scaled": round(val_pinball(model, data, val_o), 5)}

        # Fair comparison origin set: evaluate on validation origins valid for 168h window (Runbook §5.1)
        common_spec = dataclasses.replace(SIM_TRACK, input_len=168) if track == "sim" else spec
        common_val_o = split_origins(data.times, common_spec, bounds, origin_hours=SIM_EVAL_HOURS if track == "sim" else None)["val"]
        val_pb_common = round(val_pinball(model, data, common_val_o), 5)
        info[name]["val_pinball_common"] = val_pb_common

        if args.tag or args.hidden or args.input_len:
            # Append hyperparameter evaluation metrics on the common validation origin set to hparam_log.csv.
            log_hparam_row({
                "utc_time": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tag": args.tag or "",
                "track": track,
                "model": name,
                "hidden": s["hidden"],
                "input_len": spec.input_len,
                "epochs_run": len(curve),
                "best_epoch": curve[-1]["best_epoch"] if curve else None,
                "val_pinball_common": val_pb_common,
                "parameters": count_parameters(model),
                "train_seconds": info[name]["train_seconds"],
                "train_stride": s["train_stride"],
                "git_commit": _get_git_commit(),
            })

        if kind == "graph":
            info[name]["lanes_used"] = int(len(cfg["edges"]["src"]))
        prefix = "" if track == "sim" else "real_"
        info[name]["artifact"] = f"Stage02_DL/artifacts/{prefix}{ARTIFACT_NAMES[name]}.pt"
        trained[name] = (model, out_path("artifacts", f"{prefix}{ARTIFACT_NAMES[name]}.pt"), kind, cfg)

    # Step 5: one split-conformal correction per model on VAL, then the long-format rows for val and test.
    frames = []
    for name, by_split in forecasts.items():
        margins = conformal_margins(by_split["val"], y_val, mask_val)
        info[name]["conformal_margin_scaled"] = np.round(margins, 4).tolist()
        raw_cov = _coverage(by_split["val"], y_val, mask_val)
        info[name]["val_coverage_before_conformal"] = round(raw_cov, 3)
        for sp, origins in (("val", val_o), ("test", test_o)):
            q = to_reported_units(data, apply_margins(by_split[sp], margins))
            frames.append(long_frame(data, origins, q, sp, name, MODEL_VERSION + ("-fast" if args.fast else "")))
        # Val-only accuracy for the model card (MAE of the median in reported units, by horizon).
        info[name]["val_mae_by_horizon"] = _val_mae(data, by_split["val"], val_o, mask_val)
        if name in trained:
            # Save weights + config + the interval margins, so the API and Role 2 rebuild exactly this model.
            model, path, kind, cfg = trained[name]
            save_model(model, path, kind, cfg, meta={
                "track": track, "model_name": name, "model_version": MODEL_VERSION + ("-fast" if args.fast else ""),
                "input_channels": data.input_channel_names, "targets": data.target_names,
                "horizons_steps": list(spec.horizons), "step_hours": spec.step_hours, "input_len": spec.input_len,
                "conformal_margin_scaled": margins.tolist(), "node_ids": data.node_ids,
                "scaler_file": "Stage02_DL/data/scaler.json" if track == "sim" else "Stage02_DL/data/real_scaler.json"})
    card["tracks"][track] = {"status": "TRAINED", "settings": s, "nodes": data.n_nodes,
                             "origins": {"train": int(len(train_o)), "val": int(len(val_o)), "test": int(len(test_o))},
                             "input_channels": data.input_channel_names, "models": info}
    return pd.concat(frames, ignore_index=True)


def _coverage(q: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Share of measured val targets inside [q10, q90] before the conformal correction."""
    inside = (y >= q[..., 0]) & (y <= q[..., 2])
    return float(inside[np.repeat(mask[..., None], y.shape[-1], axis=-1)].mean())


def _val_mae(data: ForecastData, q_scaled: np.ndarray, origins: np.ndarray, mask: np.ndarray) -> dict:
    """Validation MAE of the median forecast per target and horizon, in reported units."""
    pred = to_reported_units(data, q_scaled)[..., 1]
    truth = data.truth_original(origins)
    if data.output_transform is not None:
        pred, truth = data.output_transform(pred), data.output_transform(truth)
    out = {}
    for ti, tname in enumerate(data.target_names):
        for hi, h in enumerate(data.spec.horizons):
            m = mask[:, :, hi]
            out[f"{tname}_h{h * data.spec.step_hours}"] = round(float(np.abs(pred[:, :, hi, ti] - truth[:, :, hi, ti])[m].mean()), 4)
    return out


def noise_channel_check(args: argparse.Namespace) -> int:
    """Role 2's sanity test: train one LSTM whose inputs include pure noise + a time-shuffled channel, save it."""
    data, bounds, edges = load_track("sim", noise=True)
    s = settings(True, "sim", args.epochs)
    origins = split_origins(data.times, data.spec, bounds)
    val_o = split_origins(data.times, data.spec, bounds, origin_hours=SIM_EVAL_HOURS)["val"]
    kind, cfg = model_config("lstm", data, edges, s)
    model, curve = train_neural(build_model(kind, cfg), data, origins["train"][::s["train_stride"]], val_o,
                                epochs=s["epochs"], batch_size=s["batch_origins"], patience=s["patience"])
    out = stage_path(STAGE, "artifacts", "noise_check/lstm_noise.pt")
    save_model(model, out, kind, cfg, meta={"input_channels": data.input_channel_names, "noise_channels": True,
                                            "note": "inputs -2 and -1 are noise; attribution to them must be ~0"})
    print(f"saved {out} (input channels: {data.input_channel_names})")
    return 0


def run_fold(fold_k: int, args: argparse.Namespace) -> int:
    """Rolling-origin backtest fold for Role 4 (Section 7 in runbook).

    Trains models on historical train data prior to cutoff C_k, evaluates point forecasts on a 21-day
    purge-isolated validation window, and writes rows to Stage02_DL/predictions/dl_backtest_folds.csv.gz.
    """
    from common.dataio.hourly_grid import fit_robust_scaler

    print(f"\n=== Rolling-origin backtest: Fold {fold_k} (sim track) ===")
    cutoffs = {1: "2025-06-01T00:00:00Z", 2: "2025-07-01T00:00:00Z", 3: "2025-08-01T00:00:00Z"}
    c_str = cutoffs[fold_k]
    c_ts = pd.Timestamp(c_str)

    # 1. Fold boundaries: purge gap of 168 h after train_end, followed by 21 days of validation.
    val_start_ts = c_ts + pd.Timedelta(hours=168)
    val_end_ts = val_start_ts + pd.Timedelta(days=21)
    fold_bounds = {
        "train_end": c_str,
        "val_start": val_start_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "val_end": val_end_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_start": "2099-01-01T00:00:00Z",
    }

    # 2. Refit scaler strictly on data prior to cutoff C_k so no statistics leak from later dates.
    grid = pd.read_csv(stage_path(STAGE, "data", "hourly_node_telemetry.csv.gz"))
    fold_scaler = fit_robust_scaler(grid, c_ts)

    node_ids = pd.read_csv(NODES_CSV)["node_id"].tolist()
    edges = pd.read_csv(EDGES_CSV)
    data = sim_data_from_grid(grid, node_ids, fold_scaler)
    spec = data.spec

    # 3. Fast settings per runbook: patience 3, stride 6, trees 300
    s = settings(True, "sim", args.epochs, hidden=args.hidden)
    all_origins = split_origins(data.times, spec, fold_bounds)
    train_o = all_origins["train"][::s["train_stride"]]
    val_o = split_origins(data.times, spec, fold_bounds, origin_hours=SIM_EVAL_HOURS)["val"]
    print(f"Fold {fold_k}: train origins {len(train_o)} (stride {s['train_stride']}), val origins {len(val_o)}")

    frames = []

    # Baselines: residual band from train, predict on fold-val origins
    train_dense = all_origins["train"][::max(1, s["train_stride"])]
    y_train = data.batch(train_dense)["y"]
    mask_train = ~data.target_imputed(train_dense)
    for name in ("persistence", "seasonal_naive"):
        band = residual_quantiles(naive_point(data, train_dense, name), y_train, mask_train)
        q_raw = with_residual_band(naive_point(data, val_o, name), band)
        q = to_reported_units(data, q_raw)
        df = long_frame(data, val_o, q, "val", name, "v2.0-fold-fast")
        df["fold"] = fold_k
        frames.append(df)

    # XGBoost on lag features
    print(f"training xgb_lag for fold {fold_k} ...")
    xgb_bundle = train_xgb_lag(data, train_o, val_o, n_estimators=s["xgb_trees"], n_jobs=THREADS)
    q_raw = predict_xgb_lag(xgb_bundle, data, val_o)
    q = to_reported_units(data, q_raw)
    df = long_frame(data, val_o, q, "val", "xgb_lag", "v2.0-fold-fast")
    df["fold"] = fold_k
    frames.append(df)

    # Neural models: early stop on fold-val origins, no conformal widening (eval uses point forecast)
    neural_models = ("lstm", "transformer", "graph_full", "graph_none")
    for name in neural_models:
        set_global_seed(SEED)
        kind, cfg = model_config(name, data, edges, s)
        model = build_model(kind, cfg)
        print(f"training {name} for fold {fold_k} ({count_parameters(model):,} parameters) ...")
        model, _ = train_neural(model, data, train_o, val_o, epochs=s["epochs"],
                                batch_size=s["batch_origins"], lr=s["lr"], patience=s["patience"])
        q_raw = predict_scaled(model, data, val_o)
        q = to_reported_units(data, q_raw)
        df = long_frame(data, val_o, q, "val", name, "v2.0-fold-fast")
        df["fold"] = fold_k
        frames.append(df)

    # Combine fold forecasts and append/merge into dl_backtest_folds.csv.gz
    fold_df = pd.concat(frames, ignore_index=True)
    out_file = stage_path(STAGE, "predictions", "dl_backtest_folds.csv.gz")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        existing = pd.read_csv(out_file)
        combined = pd.concat([existing[existing["fold"] != fold_k], fold_df], ignore_index=True)
    else:
        combined = fold_df
    combined.to_csv(out_file, index=False, compression="gzip")
    print(f"Wrote {len(fold_df):,} rows for fold {fold_k} (total: {len(combined):,}) -> {out_file}")
    return 0


def main() -> int:
    """Train, forecast, validate and save everything for Stage 2."""
    args = parse_args()
    set_global_seed(SEED)
    torch.set_num_threads(THREADS)
    stage_path(STAGE, "artifacts").mkdir(parents=True, exist_ok=True)
    if args.noise_channel:
        return noise_channel_check(args)
    if args.fold is not None:
        return run_fold(args.fold, args)
    SCRATCH["on"] = bool(args.models or args.epochs or args.track != "both" or args.hidden or args.input_len or args.tag)
    if SCRATCH["on"]:
        print("PARTIAL RUN: outputs go to Stage02_DL/artifacts/scratch/ and predictions/scratch/ (not the hand-off files).")

    start = time.time()
    card = {"tracks": {}}
    curves: list = []
    tracks = ["sim", "real"] if args.track == "both" else [args.track]
    outputs = {t: run_track(t, args, card, curves) for t in tracks}

    # Step 6: write forecasts. The simulated track goes through the contract checker (it is a hand-over file).
    if outputs.get("sim") is not None:
        path = out_path("predictions", "dl_forecasts.csv.gz")
        write_validated_csv(outputs["sim"], "dl_forecasts", path)
        old = stage_path(STAGE, "predictions", "dl_forecasts.csv")
        if old.exists() and not SCRATCH["on"]:
            old.unlink()  # the v1 file is replaced; a stale copy next to the new one would confuse readers
        print(f"wrote {len(outputs['sim']):,} rows -> {path}")
    if outputs.get("real") is not None:
        path = out_path("predictions", "dl_real_forecasts.csv.gz")
        outputs["real"].to_csv(path, index=False)
        print(f"wrote {len(outputs['real']):,} rows -> {path}")

    # Step 7: training curves and the model card (val numbers only; Role 4 owns every test number).
    if curves:
        pd.DataFrame(curves).to_csv(out_path("artifacts", "training_curve.csv"), index=False)
    sim_info = card["tracks"].get("sim", {}).get("models", {})
    proposal = None
    if sim_info:
        # Proposal only: lowest val pinball among the neural models. Role 4 decides in eval_champion.json.
        neural = {k: v["val_pinball_scaled"] for k, v in sim_info.items() if "val_pinball_scaled" in v}
        proposal = min(neural, key=neural.get) if neural else None
    save_model_card(
        output_dir=out_path("artifacts"), name="stage02_forecasters", version=MODEL_VERSION + ("-fast" if args.fast else ""),
        features=list(card["tracks"].get("sim", {}).get("input_channels", [])),
        train_window={"sim": "origins whose last target hour is before splits.json train_end",
                      "real": "origins whose last target day is before 2024-01-01"},
        metrics_on_val={t: {m: {k: v[k] for k in ("val_pinball_scaled", "val_mae_by_horizon", "val_coverage_before_conformal")
                                if k in v} for m, v in d.get("models", {}).items()} for t, d in card["tracks"].items()},
        hyperparameters={t: d.get("settings") for t, d in card["tracks"].items()},
        notes=("Quantile forecasts (q10/q50/q90) for every model; intervals corrected by split-conformal on VAL. "
               "Graph variants share one architecture and parameter count; only the lanes differ. "
               "Run mode: " + ("--fast" if args.fast else "full")),
        extra_fields={"tracks": card["tracks"], "champion_proposal": proposal,
                      "run_seconds": round(time.time() - start, 1), "command": " ".join(sys.argv)},
    )
    print(f"\nStage 02 models done in {(time.time() - start) / 60:.1f} min. Champion proposal (val): {proposal}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
