"""ROLE: DATA ENGINEER  |  DAY 0 - FOUNDATIONS
Builds the raw simulated datasets that every stage starts from. One command, same output on every laptop:

    python scripts/build_sim_data.py --seed 42

READS : contracts/network_nodes.csv, contracts/network_edges.csv (through common/dataio/simulator.py)
WRITES: data/raw/sim/shipment_telemetry_raw.csv.gz  (shipment snapshots, split + delay_status still empty)
        data/raw/sim/node_hourly_raw.csv.gz         (one row per node per hour, some hours missing on purpose)
        data/raw/sim/events_ground_truth.csv        (every disruption event the simulator sampled)
        data/raw/sim/INJECTED_ISSUES.json           (PRIVATE until Role 2 reports Day 1 findings - do not commit early)
OWNER : Role 3 - Data & Integration Engineer
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make "import common" work when this file is run from the repo root or from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dataio import simulator as sim  # noqa: E402
from common.dataio.calibration import real_drivers  # noqa: E402
from common.paths import REPO_ROOT  # noqa: E402
from common.seed import set_global_seed  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "raw" / "sim"

# Hidden-factor scale, tuned ONCE so the class mix lands near the target 65/22/13 (role file 6.3).
# Never re-tune this to make a model look better.
HIDDEN_SCALE = 1.0

# Thresholds from contracts/label_policy.json, used here only to REPORT the class mix.
ON_TIME_BELOW, AT_RISK_BELOW = 6, 24

# mtime=0 keeps the .gz byte-identical between runs, so git only sees a change when the data changes.
GZIP = {"method": "gzip", "mtime": 0}


def measurement_noise(values: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Multiply by lognormal noise (sigma 0.05-0.15): real sensors and reports are never exact."""
    return values * rng.lognormal(0.0, sigma, size=len(values))


def plant_issues(ship: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, list]:
    """Inject realistic data problems for the EDA role to discover (role file 6.5). Returns (data, log)."""
    log = []
    numeric = ["dwell_time_hours", "throughput_teu_per_hr", "capacity_utilization_pct",
               "customs_hold_hours", "weather_severity_index", "labor_deficit_ratio",
               "equipment_availability_pct", "historical_delay_rate"]

    # 1. One carrier's dwell reports gain a constant +5 h offset after 1 July (a reporting change).
    mask = (ship["carrier_code"] == "ZIM") & (ship["timestamp"] >= pd.Timestamp("2025-07-01", tz="UTC"))
    ship.loc[mask, "dwell_time_hours"] = ship.loc[mask, "dwell_time_hours"] + 5.0
    log.append({"issue": "unit_offset", "carrier": "ZIM", "after": "2025-07-01", "column": "dwell_time_hours",
                "change": "+5 h offset", "rows": int(mask.sum())})

    # 2. Weekly reporting issue: Shanghai's Monday customs values arrive rounded to whole hours (stale feed).
    mask = (ship["current_node_id"] == "PORT_CN_SHA") & (ship["timestamp"].dt.day_of_week == 0)
    ship.loc[mask, "customs_hold_hours"] = ship.loc[mask, "customs_hold_hours"].round(0)
    log.append({"issue": "weekly_reporting_lag", "node": "PORT_CN_SHA", "weekday": "Monday",
                "column": "customs_hold_hours", "change": "rounded/stale values", "rows": int(mask.sum())})

    # 3. 0.5% sensor outliers: a value multiplied by 3 (still inside the contract range).
    out_rows = rng.choice(len(ship), size=int(0.005 * len(ship)), replace=False)
    ship.loc[ship.index[out_rows], "dwell_time_hours"] = (ship["dwell_time_hours"].iloc[out_rows] * 3).clip(0, 400)
    log.append({"issue": "sensor_outliers", "column": "dwell_time_hours", "change": "x3", "rows": len(out_rows)})

    # 4. About 1.5% missing values spread over the numeric columns.
    total_missing = 0
    for col in numeric:
        miss = rng.random(len(ship)) < 0.015 / len(numeric) * 4
        ship.loc[miss, col] = np.nan
        total_missing += int(miss.sum())
    log.append({"issue": "missing_values", "columns": numeric, "cells": total_missing})

    # 5. Two duplicated blocks (a feed re-sent the same 50 records twice).
    blocks = [ship.iloc[s:s + 50] for s in rng.integers(0, len(ship) - 50, 2)]
    ship = pd.concat([ship, *blocks], ignore_index=True)
    log.append({"issue": "duplicated_blocks", "blocks": 2, "rows_per_block": 50})
    return ship, log


def main() -> int:
    """Run the simulator and save all raw files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-shipments", type=int, default=60000)
    parser.add_argument("--synthetic-drivers", action="store_true",
                        help="use the old made-up weather/season/stress instead of the real calibration data")
    args = parser.parse_args()
    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: load the trade network and sample disruption events for the whole simulated period.
    nodes, edges = sim.load_network()
    # Real drivers (weather, monthly traffic, port size, GSCPI) unless the old synthetic ones are asked for.
    hours = sim.SIM_START + pd.to_timedelta(np.arange(sim.SIM_HOURS + sim.EXTRA_HOURS), unit="h")
    real = None if args.synthetic_drivers else real_drivers(hours, list(nodes["node_id"]))
    print("drivers:", "synthetic" if real is None else "REAL (Open-Meteo weather, PortWatch traffic and size, NY Fed GSCPI)")
    events = sim.sample_events(nodes, sim.SIM_HOURS + sim.EXTRA_HOURS, rng, skip_types=() if real is None else ("STORM",))
    print(f"events sampled: {len(events)}")

    # Step 2: simulate every node hour by hour (this is the slow part, about a minute).
    state = sim.simulate_nodes(nodes, edges, events, rng, real=real)
    print("node simulation done")

    # Step 3: sample shipments and read their realised delay from the future node states.
    ship = sim.simulate_shipments(state, nodes, edges, args.n_shipments, rng, HIDDEN_SCALE)
    ship = ship.sort_values("timestamp").reset_index(drop=True)
    ship["historical_delay_rate"] = sim.add_historical_delay_rate(ship)

    # Step 4: report the class mix so we can check it against the 65/22/13 target.
    d = ship["realized_delay_hours"]
    mix = {"ON_TIME": float((d < ON_TIME_BELOW).mean()),
           "AT_RISK": float(((d >= ON_TIME_BELOW) & (d < AT_RISK_BELOW)).mean()),
           "CRITICAL_DELAY": float((d >= AT_RISK_BELOW).mean())}
    print("class mix:", {k: round(v, 3) for k, v in mix.items()})

    # Step 5: what we OBSERVE is the true value plus measurement noise (the label is not noised).
    for col, sigma in [("dwell_time_hours", 0.05), ("throughput_teu_per_hr", 0.08),
                       ("capacity_utilization_pct", 0.05), ("customs_hold_hours", 0.15)]:
        ship[col] = measurement_noise(ship[col].to_numpy(), rng, sigma)
    for col, hi in [("dwell_time_hours", 400), ("throughput_teu_per_hr", 2500),
                    ("capacity_utilization_pct", 150), ("customs_hold_hours", 240)]:
        ship[col] = ship[col].clip(0, hi)  # physical limits from the telemetry contract

    # Step 6: add ids and the honesty columns the contract requires.
    ship.insert(0, "record_id", [f"R{i + 1:06d}" for i in range(len(ship))])
    ship.insert(2, "shipment_id", [f"SHP-{i + 100001}" for i in range(len(ship))])
    ship.insert(3, "container_id", [f"{c[:3].upper()}U{rng.integers(1000000, 9999999)}" for c in ship["carrier_code"]])
    ship["delay_status"] = ""   # set by Stage01_ML/01_data_engineer.py from label_policy.json
    ship["split"] = ""          # set by Stage01_ML/01_data_engineer.py - ONE place only
    ship["data_origin"] = "SYNTHETIC_CALIBRATED"
    ship["is_synthetic"] = True
    ship["scenario_id"] = "NOMINAL"
    ship = ship.round(3)

    # Step 7: plant the issues for EDA, then fix the duplicated rows' record ids so the
    # duplicates are "same content" but the planted log stays truthful.
    ship, issues = plant_issues(ship, rng)
    cols = ["record_id", "timestamp", "shipment_id", "container_id", "carrier_code", "origin_port",
            "destination_port", "current_node_id", "dwell_time_hours", "throughput_teu_per_hr",
            "capacity_utilization_pct", "customs_hold_hours", "weather_severity_index",
            "labor_deficit_ratio", "equipment_availability_pct", "historical_delay_rate",
            "priority_tier", "realized_delay_hours", "delay_status", "split", "data_origin",
            "is_synthetic", "scenario_id"]
    ship["timestamp"] = ship["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    ship[cols].to_csv(OUT_DIR / "shipment_telemetry_raw.csv.gz", index=False, compression=GZIP)

    # Step 8: hourly node table with noise and ~1% of hours missing (Stage 2 must rebuild the grid).
    hourly = sim.node_hourly_table(state)
    for col in ["avg_dwell_time_hours", "throughput_teu_per_hr", "customs_hold_hours"]:
        hourly[col] = measurement_noise(hourly[col].to_numpy(), rng, 0.07)
    hourly["n_records"] = rng.poisson(8, len(hourly))
    hourly["data_origin"] = "SYNTHETIC_CALIBRATED"
    hourly["scenario_id"] = "NOMINAL"
    dropped = rng.random(len(hourly)) < 0.01
    hourly = hourly[~dropped].round(3)
    hourly["hour_ts"] = hourly["hour_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    hourly.to_csv(OUT_DIR / "node_hourly_raw.csv.gz", index=False, compression=GZIP)
    issues.append({"issue": "missing_hours", "table": "node_hourly_raw", "rows_dropped": int(dropped.sum())})

    # Step 9: ground-truth events with real timestamps (Stage 3 text and Stage 5 scenarios use them).
    events["start_ts"] = (sim.SIM_START + pd.to_timedelta(events["start_h"], unit="h")).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    events["end_ts"] = (sim.SIM_START + pd.to_timedelta(events["end_h"], unit="h")).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    events[["event_id", "type", "node_id", "start_ts", "end_ts", "severity", "cause"]].to_csv(
        OUT_DIR / "events_ground_truth.csv", index=False)

    # Step 10: private list of planted issues + the numbers SIM_CARD.md quotes.
    (OUT_DIR / "INJECTED_ISSUES.json").write_text(json.dumps(issues, indent=2))
    (OUT_DIR / "sim_summary.json").write_text(json.dumps({
        "seed": args.seed, "n_shipments_raw": len(ship), "n_events": len(events),
        "n_hourly_rows": len(hourly), "hidden_scale": HIDDEN_SCALE,
        "drivers": "synthetic" if real is None else "real: weather_hourly, activity_profile, gscpi_monthly",
        "class_mix_before_planting": mix}, indent=2))
    print(f"wrote {len(ship)} shipments, {len(hourly)} node-hours to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
