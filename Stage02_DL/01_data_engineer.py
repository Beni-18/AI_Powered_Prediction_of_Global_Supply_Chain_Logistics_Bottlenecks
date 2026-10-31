"""ROLE: DATA ENGINEER  |  STAGE 02 - DEEP LEARNING
Owner: Role 3 - Data & Integration Engineer. Written by Role 4 at the team lead's request; Role 3 to review and run.
Full instructions: docs/roles/ROLE_3_DATA_INTEGRATION_ENGINEER.md (Day 2 section)

WHAT THIS SCRIPT DOES
    Simulated track (the main Stage 2 data):
      1. Rebuilds the complete hourly grid (27 nodes x every hour of 2025): missing hours are put back,
         flagged is_imputed=True, and filled from the past (<= 6 h) or the node's TRAIN hour-of-week median.
      2. Writes the split dates (the SAME dates as Stage 1, 168 h purge gap) and a robust scaler fitted on TRAIN only.
      3. Writes the network as matrices (symmetric 0/1 and directed transit days) in registry node order.
    Real track (IMF PortWatch, if downloaded):
      4. Daily container port calls / chokepoint transits since 2019 for 23 nodes, complete daily grid,
         calendar-year split (train <= 2023, val 2024, test >= 2025) and a train-only scaler.

INPUT : data/raw/sim/node_hourly_raw.csv.gz, Stage01_ML/data/split_boundaries.json, contracts/network_*.csv
        data/raw/real/portwatch_*.csv  (optional: python scripts/fetch_real_data.py downloads them, ~10 min)
OUTPUT: Stage02_DL/data/hourly_node_telemetry.csv.gz   (contract: hourly_node_telemetry)
        Stage02_DL/data/splits.json, scaler.json, adjacency_matrix.csv, adjacency_directed_transit_days.csv
        Stage02_DL/data/real_daily_activity.csv.gz, real_splits.json, real_scaler.json   (real track)
        Stage02_DL/data/data_quality_report.json      (every change made, with counts)

RUN   : python 01_data_engineer.py        (from inside Stage02_DL/, about 20 seconds)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make "import common" work when this file is run from inside the stage folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.contracts import write_validated_csv  # noqa: E402
from common.dataio import real_activity  # noqa: E402
from common.dataio.hourly_grid import CHANNELS, complete_grid, fit_robust_scaler  # noqa: E402
from common.paths import EDGES_CSV, NODES_CSV, REPO_ROOT, stage_path  # noqa: E402
from common.seed import set_global_seed  # noqa: E402

STAGE = 2
RAW_HOURLY = REPO_ROOT / "data" / "raw" / "sim" / "node_hourly_raw.csv.gz"
STAGE1_BOUNDARIES = stage_path(1, "data", "split_boundaries.json")
CONTRACT_COLUMNS = ["hour_ts", "node_id", *CHANNELS, "vessels_waiting", "n_records", "is_imputed", "data_origin", "scenario_id"]


def split_period(ts: pd.Series, b: dict) -> pd.Series:
    """Label each timestamp train / gap / val / test with the Stage 1 dates (for counting only, not a column)."""
    t = pd.to_datetime(ts, utc=True)
    out = pd.Series("gap", index=ts.index)
    out[t < pd.Timestamp(b["train_end"])] = "train"
    out[(t >= pd.Timestamp(b["val_start"])) & (t < pd.Timestamp(b["val_end"]))] = "val"
    out[t >= pd.Timestamp(b["test_start"])] = "test"
    return out


def build_sim_track(node_ids: list, report: dict) -> None:
    """Steps 1-3: the complete hourly grid, the split file and the train-only scaler."""
    # Step 1: rebuild every missing node-hour without looking into the future.
    boundaries = json.loads(STAGE1_BOUNDARIES.read_text())
    train_end = pd.Timestamp(boundaries["train_end"])
    raw = pd.read_csv(RAW_HOURLY)
    grid, log = complete_grid(raw, node_ids, train_end)
    periods = split_period(grid["hour_ts"], boundaries)
    log["imputed_rows_by_split_period"] = grid.groupby(periods)["is_imputed"].sum().astype(int).to_dict()
    log["rows_by_split_period"] = periods.value_counts().to_dict()
    # Values are never clipped here: overloads above 100% utilisation are real congestion (SIM_CARD).
    log["values_clipped"] = 0
    report["sim_track"] = log

    # Step 2: write the grid through the contract checker; > 10 MB of CSV goes out gzipped (architecture section 5).
    grid["hour_ts"] = pd.to_datetime(grid["hour_ts"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = stage_path(STAGE, "data", "hourly_node_telemetry.csv.gz")
    write_validated_csv(grid[CONTRACT_COLUMNS], "hourly_node_telemetry", out)
    print(f"hourly grid: {len(grid):,} rows ({log['imputed_rows']} imputed) -> {out.relative_to(REPO_ROOT)}")

    # Step 3: split dates = Stage 1's dates (one split for the whole project), plus how Stage 2 applies them.
    splits = dict(boundaries)
    splits["stage2_rule"] = ("a forecast window is train only if its last TARGET hour is before train_end; "
                             "val/test only if its FIRST INPUT hour is on/after val_start/test_start "
                             "(common/dataio/windows.py). Long-context summaries may look further back: "
                             "that is history a forecaster would really have, never a training target.")
    stage_path(STAGE, "data", "splits.json").write_text(json.dumps(splits, indent=2))

    # Step 4: robust scaler per node and channel, fitted on train hours only.
    scaler = fit_robust_scaler(grid, train_end)
    scaler["inverse"] = "x = scaled * iqr + median  (common.dataio.hourly_grid.scale_array(..., inverse=True))"
    stage_path(STAGE, "data", "scaler.json").write_text(json.dumps(scaler, indent=2))


def build_network_files(node_ids: list, report: dict) -> None:
    """Write the lanes as matrices in registry order: symmetric 0/1 (for the GCN) and directed transit days."""
    edges = pd.read_csv(EDGES_CSV)
    index = {n: i for i, n in enumerate(node_ids)}
    sym = np.zeros((len(node_ids), len(node_ids)), dtype=int)
    directed = np.zeros((len(node_ids), len(node_ids)))
    for e in edges.itertuples():
        i, j = index[e.src], index[e.dst]
        sym[i, j] = sym[j, i] = 1
        # Directed: row = where congestion starts, column = where it arrives, value = days it takes.
        directed[i, j] = e.typical_transit_days
        if str(e.bidirectional).lower() == "true":
            directed[j, i] = e.typical_transit_days
    pd.DataFrame(sym, index=node_ids, columns=node_ids).to_csv(stage_path(STAGE, "data", "adjacency_matrix.csv"))
    pd.DataFrame(directed, index=node_ids, columns=node_ids).to_csv(
        stage_path(STAGE, "data", "adjacency_directed_transit_days.csv"))
    report["network"] = {"nodes": len(node_ids), "lanes": int(len(edges)), "undirected_links": int(sym.sum() // 2),
                         "directed_links": int((directed > 0).sum()),
                         "isolated_nodes": [n for n in node_ids if sym[index[n]].sum() == 0]}


def build_real_track(node_ids: list, report: dict) -> None:
    """Step 4: the PortWatch daily grid, its calendar-year split and a train-only scaler (skipped if not downloaded)."""
    if not real_activity.raw_files_present():
        report["real_track"] = {"status": "SKIPPED", "why": "data/raw/real/portwatch_*.csv not found; "
                                "run python scripts/fetch_real_data.py first (downloads ~650 MB once, keeps ~20 MB)"}
        print("real track: SKIPPED (PortWatch files not downloaded)")
        return
    grid, log = real_activity.load_real_daily(node_ids)
    days = pd.DatetimeIndex(sorted(grid["date"].unique()))
    splits = real_activity.real_split_boundaries(days)

    # Train-only robust scaler on log(1 + count), per node.
    train = grid[(grid["date"] < pd.Timestamp(splits["train_end"]).tz_localize(None)) & ~grid["is_imputed"]]
    log_act = real_activity.activity_log_transform(train["activity"].to_numpy())
    train = train.assign(log_activity=log_act)
    scaler = {"method": "robust on log(1 + activity), per node, TRAIN days only", "nodes": {}}
    for node, g in train.groupby("node_id"):
        med = float(g["log_activity"].median())
        iqr = float(g["log_activity"].quantile(0.75) - g["log_activity"].quantile(0.25)) or float(g["log_activity"].std()) or 1.0
        scaler["nodes"][node] = {"log_activity": {"median": round(med, 6), "iqr": round(iqr, 6)}}

    out = grid.assign(date=grid["date"].dt.strftime("%Y-%m-%d"))
    out.to_csv(stage_path(STAGE, "data", "real_daily_activity.csv.gz"), index=False)
    stage_path(STAGE, "data", "real_splits.json").write_text(json.dumps(splits, indent=2))
    stage_path(STAGE, "data", "real_scaler.json").write_text(json.dumps(scaler, indent=2))
    report["real_track"] = {"status": "BUILT", **log, "split": splits}
    print(f"real track: {log['n_nodes']} nodes x {log['n_days']} days ({log['first_day']} .. {log['last_day']})")


def main() -> int:
    """Run the whole Stage 02 data preparation."""
    set_global_seed()  # same random numbers on every laptop
    node_ids = pd.read_csv(NODES_CSV)["node_id"].tolist()
    report: dict = {"input_files": [str(RAW_HOURLY.relative_to(REPO_ROOT)), str(STAGE1_BOUNDARIES.relative_to(REPO_ROOT))],
                    "data_origin_note": "sim track = SYNTHETIC_CALIBRATED (simulator v0.3, real drivers); real track = REAL (PortWatch)"}
    stage_path(STAGE, "data").mkdir(parents=True, exist_ok=True)

    build_sim_track(node_ids, report)
    build_network_files(node_ids, report)
    build_real_track(node_ids, report)

    # Every change made to the data, with counts, in one file (Role 3 convention).
    stage_path(STAGE, "data", "data_quality_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("wrote Stage02_DL/data/data_quality_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
