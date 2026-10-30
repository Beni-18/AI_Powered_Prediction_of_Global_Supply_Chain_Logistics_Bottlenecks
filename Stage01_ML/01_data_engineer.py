"""ROLE: DATA ENGINEER  |  STAGE 01 - MACHINE LEARNING
Owner: Role 3 - Data & Integration Engineer
Full instructions: docs/roles/ROLE_3_DATA_INTEGRATION_ENGINEER.md  (Day 1 section)

WHAT THIS SCRIPT DOES
    Cleans the raw simulated shipment telemetry, derives the label (delay_status) from the realised
    delay, and assigns the ONE chronological train/val/test split that every later stage reuses.

INPUT : data/raw/sim/shipment_telemetry_raw.csv.gz
OUTPUT: Stage01_ML/data/prepared_dataset.csv.gz   (contract: telemetry, plus *_flag helper columns)
        Stage01_ML/data/split_boundaries.json      (the split dates; Stage 2 uses the same ones)
        Stage01_ML/data/data_quality_report.json   (every change made to the data, with counts)
READS : data/raw/sim/node_hourly_raw.csv.gz (node averages, to spot dwell sensor spikes; via common/dataio/node_context.py)

RUN   : python 01_data_engineer.py     (from inside Stage01_ML/)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make "import common" work when this file is run from inside the stage folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.contracts import load_schema, write_validated_csv  # noqa: E402
from common.dataio.node_context import node_lag  # noqa: E402
from common.paths import CONTRACTS_DIR, REPO_ROOT, stage_path  # noqa: E402
from common.seed import set_global_seed  # noqa: E402

RAW_FILE = REPO_ROOT / "data" / "raw" / "sim" / "shipment_telemetry_raw.csv.gz"
LABEL_POLICY = json.loads((CONTRACTS_DIR / "label_policy.json").read_text())

# Numeric feature columns that can be missing in the raw feed (the label column never is).
FEATURE_NUMERICS = ["dwell_time_hours", "throughput_teu_per_hr", "capacity_utilization_pct",
                    "customs_hold_hours", "weather_severity_index", "labor_deficit_ratio",
                    "equipment_availability_pct", "historical_delay_rate"]
ALL_NUMERICS = FEATURE_NUMERICS + ["realized_delay_hours"]

# Reporting problems Role 2's EDA found (eda_drift_report.json: carrier ZIM's dwell "SHIFTED" mid-year).
OFFSET_CARRIER, OFFSET_FROM = "ZIM", pd.Timestamp("2025-07-01", tz="UTC")
SPIKE_RATIO = 2.3   # a dwell reading above 2.3x the node's previous-hour average is a sensor spike (normal 99th pct is 1.4x)


def derive_delay_status(realized_delay_hours: pd.Series) -> pd.Series:
    """Turn realised delay (hours) into ON_TIME / AT_RISK / CRITICAL_DELAY using label_policy.json."""
    t = LABEL_POLICY["thresholds_hours"]
    # Below 6 h = on time; 6 up to 24 h = at risk; 24 h or more = critical (numbers live in the policy file).
    return pd.Series(
        np.select(
            [realized_delay_hours < t["ON_TIME_below"], realized_delay_hours < t["AT_RISK_below"]],
            ["ON_TIME", "AT_RISK"],
            default="CRITICAL_DELAY",
        ),
        index=realized_delay_hours.index,
    )


def assign_split(timestamps: pd.Series) -> tuple[pd.Series, pd.Series, dict]:
    """Chronological 70/15/15 split with a 168 h purge gap. Returns (split, in_purge_gap, boundaries).

    Rule we implement:
      - train_end and val_end are the 70% and 85% points in time.
      - Rows in the 168 h right AFTER a boundary get in_purge_gap=True. main() REMOVES them (and logs
        the count), so there is a real 7-day hole between train and val, and between val and test.
        Without the hole, a train shipment's outcome (which happens days later) could overlap val time.
      - Each row is its own shipment_id here, so a shipment can never sit in two splits.
    """
    policy = LABEL_POLICY["split_policy"]
    gap = pd.Timedelta(hours=policy["purge_gap_hours"])
    train_end = timestamps.quantile(policy["train_fraction"])
    val_end = timestamps.quantile(policy["train_fraction"] + policy["val_fraction"])

    # Start with the plain three-way cut by time.
    split = pd.Series("test", index=timestamps.index)
    split[timestamps < val_end] = "val"
    split[timestamps < train_end] = "train"

    # Rows inside the gap after each boundary are pushed back to the earlier split and flagged.
    in_train_gap = (timestamps >= train_end) & (timestamps < train_end + gap)
    in_val_gap = (timestamps >= val_end) & (timestamps < val_end + gap)
    split[in_train_gap] = "train"
    split[in_val_gap] = "val"
    in_gap = in_train_gap | in_val_gap

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    boundaries = {
        "train_end": train_end.strftime(fmt), "val_start": (train_end + gap).strftime(fmt),
        "val_end": val_end.strftime(fmt), "test_start": (val_end + gap).strftime(fmt),
        "purge_gap_hours": policy["purge_gap_hours"],
        "rule": "rows in [boundary, boundary+gap) are removed by the purge; count in data_quality_report.json",
    }
    return split, in_gap, boundaries


def correct_reporting_issues(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Undo two reporting faults in dwell_time_hours, estimating each correction FROM THE DATA, and flag every change.

    1. ZIM's dwell reports carry a constant offset from 1 July. Estimate: how much ZIM's gap to the node's average
       dwell grew after that date, minus how much other carriers' gap grew (removes the season). Subtract it.
    2. Sensor spikes: dwell far above the node's previous-hour average. Divide by the typical spike factor
       (median ratio among the flagged rows).
    Values are corrected, never dropped; dwell_offset_corrected / dwell_spike_corrected mark the rows.
    """
    node_avg = node_lag(df, ["avg_dwell_time_hours"])["lag_avg_dwell_time_hours"]
    gap = df["dwell_time_hours"] - node_avg
    after = df["timestamp"] >= OFFSET_FROM
    carrier = df["carrier_code"] == OFFSET_CARRIER
    offset = ((gap[carrier & after].median() - gap[carrier & ~after].median())
              - (gap[~carrier & after].median() - gap[~carrier & ~after].median()))
    df["dwell_offset_corrected"] = carrier & after & df["dwell_time_hours"].notna()
    df.loc[df["dwell_offset_corrected"], "dwell_time_hours"] = (df.loc[df["dwell_offset_corrected"], "dwell_time_hours"] - offset).clip(lower=0)

    ratio = df["dwell_time_hours"] / node_avg
    df["dwell_spike_corrected"] = (ratio > SPIKE_RATIO).fillna(False)
    factor = float(ratio[df["dwell_spike_corrected"]].median()) if df["dwell_spike_corrected"].any() else 1.0
    df.loc[df["dwell_spike_corrected"], "dwell_time_hours"] = df.loc[df["dwell_spike_corrected"], "dwell_time_hours"] / factor

    report["reporting_corrections"] = {
        "dwell_offset": {"carrier": OFFSET_CARRIER, "from": OFFSET_FROM.strftime("%Y-%m-%d"),
                         "estimated_offset_hours": round(float(offset), 3), "rows": int(df["dwell_offset_corrected"].sum())},
        "dwell_spikes": {"rule": f"dwell > {SPIKE_RATIO}x node previous-hour average", "estimated_factor": round(factor, 3),
                         "rows": int(df["dwell_spike_corrected"].sum())},
    }
    return df


def main() -> int:
    """Run the whole Stage 01 data preparation."""
    set_global_seed()  # same random numbers on every laptop
    report: dict = {"input_file": str(RAW_FILE.relative_to(REPO_ROOT))}

    # Step 1: load everything as text so nothing is silently converted, then check the columns.
    df = pd.read_csv(RAW_FILE, dtype=str, keep_default_na=False, na_values=[""])
    schema = load_schema("telemetry")
    required = [c["name"] for c in schema["columns"]]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"ERROR: raw file is missing columns: {missing_cols}")
        return 1
    report["rows_in"] = len(df)

    # Step 2: parse timestamps to UTC and numbers to floats; COUNT every value that fails to convert.
    report["coercion_failures"] = {}
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="ISO8601")
    report["coercion_failures"]["timestamp"] = int((ts.isna() & df["timestamp"].notna()).sum())
    for col in ALL_NUMERICS:
        converted = pd.to_numeric(df[col], errors="coerce")
        report["coercion_failures"][col] = int((converted.isna() & df[col].notna()).sum())
        df[col] = converted
    df["timestamp"] = ts
    if df["timestamp"].isna().any() or df["realized_delay_hours"].isna().any():
        # Without a time or an outcome a row cannot be split or labelled; stop loudly instead of guessing.
        print("ERROR: some rows have no valid timestamp or realized_delay_hours; fix the raw data first.")
        return 1

    # Step 3: remove EXACT duplicate records (every column equal), e.g. a feed that was sent twice.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["exact_duplicates_removed"] = before - len(df)
    # Near-duplicates (same shipment and time, other values differ) are only flagged, never removed.
    df["dup_flag"] = df.duplicated(subset=["shipment_id", "timestamp"], keep=False)
    report["near_duplicates_flagged"] = int(df["dup_flag"].sum())

    # Step 4: derive the label from the OUTCOME only, and assign the one split.
    df["delay_status"] = derive_delay_status(df["realized_delay_hours"])
    df["split"], in_gap, boundaries = assign_split(df["timestamp"])
    # Purge: remove the rows inside the 7-day gaps so the splits are truly separated in time.
    # This is the ONE place rows are removed on purpose; the count goes into the report.
    report["purge_gap_rows_removed"] = int(in_gap.sum())
    df = df[~in_gap].reset_index(drop=True)

    # Step 5: range rules from the schema. Out-of-range values are FLAGGED and KEPT, never clipped.
    report["range_flags"] = {}
    ranges = {c["name"]: c for c in schema["columns"] if c["name"] in FEATURE_NUMERICS}
    for col, spec in ranges.items():
        out = (df[col] < spec.get("min", -np.inf)) | (df[col] > spec.get("max", np.inf))
        if out.any():
            df[f"range_flag_{col}"] = out
        report["range_flags"][col] = int(out.sum())
    # Utilisation above 100% is real overload, not an error: flag it so models and planners can see it.
    df["overload_flag"] = df["capacity_utilization_pct"] > 100
    report["overload_rows"] = int(df["overload_flag"].sum())

    # Step 5b: correct the dwell reporting faults Role 2's EDA found (offset and sensor spikes), with flags.
    df = correct_reporting_issues(df, report)

    # Step 6: missing feature values.
    # CONFLICT NOTE: the role file says "keep missing, the model imputes", but the telemetry contract
    # marks these columns NOT nullable, and contracts win (AGENTS.md R3). So we fill each gap with the
    # TRAIN-split median of that column (never val/test, so no information leaks from the future) and
    # add was_missing_<col>=True so the model still knows the value was unknown. Raised in the HANDOFF.
    train_rows = df["split"] == "train"
    report["missing_filled_with_train_median"] = {}
    for col in FEATURE_NUMERICS:
        was_missing = df[col].isna()
        df[f"was_missing_{col}"] = was_missing
        median = float(df.loc[train_rows, col].median())
        df[col] = df[col].fillna(median)
        report["missing_filled_with_train_median"][col] = {"rows": int(was_missing.sum()), "median": round(median, 4)}

    # Step 7: write timestamps back as ISO text and save through the contract checker.
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out_file = stage_path(1, "data", "prepared_dataset.csv.gz")
    contract = write_validated_csv(df, "telemetry", out_file)
    stage_path(1, "data", "split_boundaries.json").write_text(json.dumps(boundaries, indent=2))

    # Step 8: data quality report - every change above, plus class mix and dates per split.
    report["rows_out"] = len(df)
    report["extra_columns"] = contract.extra_columns
    report["split_boundaries"] = boundaries
    report["per_split"] = {}
    for name, part in df.groupby("split"):
        report["per_split"][name] = {
            "rows": len(part),
            "date_min": part["timestamp"].min(), "date_max": part["timestamp"].max(),
            "class_mix": part["delay_status"].value_counts(normalize=True).round(4).to_dict(),
        }
    stage_path(1, "data", "data_quality_report.json").write_text(json.dumps(report, indent=2))

    print(f"OK: wrote {len(df)} rows to {out_file.relative_to(REPO_ROOT)}")
    for name, info in report["per_split"].items():
        print(f"  {name:5s} {info['rows']:6d} rows  {info['class_mix']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
