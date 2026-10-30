"""Shared helpers for the CASCADE control tower (Streamlit pages).
Owner: Role 3 (Data & Integration Engineer). Built by Role 4 at the team lead's request; Role 3 to review.

What lives here: plain-English names for nodes and model features, the status colours, and cached loaders
for the files the pages read. Every number on screen comes from a file written by a role script.

READS : contracts/network_nodes.csv, contracts/network_edges.csv, Stage01_ML/data/prepared_dataset.csv.gz,
        Stage01_ML/reports/*.json, Stage01_ML/05_integration_engineer.py (the scoring code the API uses)
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from common.paths import CONTRACTS_DIR, REPO_ROOT, stage_path

# Status colours from the team's data-viz palette. They always appear next to the status word, never alone.
STATUS = {
    "ON_TIME": {"label": "On time", "color": "#0ca30c"},
    "AT_RISK": {"label": "At risk", "color": "#fab219"},
    "CRITICAL_DELAY": {"label": "Critical", "color": "#d03b3b"},
}
NODE_TYPE_LABEL = {"PORT": "Port", "CHOKEPOINT": "Chokepoint", "RAIL_HUB": "Rail hub", "DC": "Distribution hub"}

# What each model input means, in words a planner uses.
FEATURE_LABEL = {
    "dwell_time_hours": "Dwell time at node (h)",
    "throughput_teu_per_hr": "Throughput (TEU/h)",
    "capacity_utilization_pct": "Capacity used (%)",
    "customs_hold_hours": "Customs hold (h)",
    "weather_severity_index": "Weather severity (0-10)",
    "labor_deficit_ratio": "Labour shortage (0-1)",
    "equipment_availability_pct": "Equipment available (%)",
    "historical_delay_rate": "Historical delay rate",
    "cal_hour": "Hour of day",
    "cal_dow": "Day of week",
    "cal_month": "Month",
    "eng_util_x_customs": "Full yard x customs hold",
    "eng_log_dwell": "Dwell time (log)",
    "eng_throughput_per_util": "Throughput per % capacity",
    "eng_labor_x_equipment_gap": "Labour shortage x missing equipment",
    "roll_node_dwell_mean_24h": "Recent dwell at this node",
}
ONE_HOT_PREFIX = {"current_node_id_": "At", "origin_port_": "From", "destination_port_": "To",
                  "carrier_code_": "Carrier", "priority_tier_": "Priority"}


@st.cache_data
def nodes() -> pd.DataFrame:
    """The 27 official network nodes with names, types and coordinates."""
    return pd.read_csv(CONTRACTS_DIR / "network_nodes.csv")


@st.cache_data
def edges() -> pd.DataFrame:
    """The shipping lanes between nodes."""
    return pd.read_csv(CONTRACTS_DIR / "network_edges.csv")


def node_name(node_id: str) -> str:
    """'CHOKE_SUEZ' -> 'Suez Canal'."""
    match = nodes().set_index("node_id")["display_name"]
    return match.get(node_id, node_id)


def feature_label(feature: str) -> str:
    """Model column name -> plain words, e.g. 'current_node_id_CHOKE_SUEZ' -> 'At: Suez Canal'."""
    if feature in FEATURE_LABEL:
        return FEATURE_LABEL[feature]
    for prefix, word in ONE_HOT_PREFIX.items():
        if feature.startswith(prefix):
            value = feature[len(prefix):]
            pretty = node_name(value) if word in ("At", "From", "To") else value.replace("_", " ").title()
            return f"{word}: {pretty}"
    return feature


def status_chip(status: str) -> str:
    """Small coloured label (HTML) for a delay status; the word is always shown, colour is only a helper."""
    s = STATUS.get(status, {"label": status, "color": "#8b8b88"})
    return (f"<span style='display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.85rem;font-weight:600;"
            f"border:1px solid {s['color']};color:{s['color']};background:{s['color']}1a'>{s['label']}</span>")


def status_label(status: str) -> str:
    """Delay status as a short word for tables."""
    return STATUS.get(status, {"label": status})["label"]


def read_json(path: Path) -> dict:
    """Read a JSON report, or an empty dict if it has not been produced yet."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_resource
def api():
    """Load the Stage 1 API module once, so the pages score with exactly the same code as the API."""
    spec = importlib.util.spec_from_file_location("stage01_api", REPO_ROOT / "Stage01_ML" / "05_integration_engineer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.service()  # load the model now, on the main thread (see the API's start-up note)
    return module


@st.cache_data
def shipments() -> pd.DataFrame:
    """All labelled shipments from the prepared dataset."""
    return pd.read_csv(stage_path(1, "data", "prepared_dataset.csv.gz"))


@st.cache_data
def node_typical() -> pd.DataFrame:
    """Median of each input per node over the TRAIN period: what 'normal' looks like at that node."""
    data = shipments()
    return data[data["split"] == "train"].groupby("current_node_id")[list(FEATURE_LABEL)[:8]].median()


def validate_and_score(records: list) -> tuple:
    """Check rows against the API's contract model, then score them. Bad rows are returned, never dropped silently."""
    module = api()
    fields = module.NUMERIC_FIELDS + ["priority_tier", "carrier_code", "origin_port", "destination_port",
                                      "current_node_id", "timestamp", "record_id"]
    good, problems = [], []
    for i, record in enumerate(records):
        try:
            good.append(module.Shipment(**{k: record.get(k) for k in fields if k in record}))
        except Exception as err:  # pydantic ValidationError: shown to the planner as-is
            problems.append({"row": i, "problem": str(err).splitlines()[1] if "\n" in str(err) else str(err)})
    answers = module.service().score(good) if good else []
    for answer, shipment in zip(answers, good):
        answer["node"] = shipment.current_node_id
    return answers, problems


@st.cache_data
def scored_period(split: str) -> pd.DataFrame:
    """Score every shipment of one period once, with the real outcome next to the model's answer."""
    rows = shipments()
    rows = rows[rows["split"] == split]
    answers, _ = validate_and_score(rows.to_dict(orient="records"))
    table = pd.DataFrame(answers)
    table["p_critical"] = table["probabilities"].map(lambda p: p["CRITICAL_DELAY"])
    table["reasons"] = table["top_drivers"].map(lambda d: " · ".join(feature_label(x) for x in d))
    real = rows.set_index("record_id")[["timestamp", "delay_status", "realized_delay_hours", "carrier_code",
                                        "priority_tier", "origin_port", "destination_port"]]
    return table.drop(columns=["probabilities", "top_drivers"]).join(real, on="record_id")


def node_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Per node: shipments, alert rate, share that really went critical, and how many of those the alert caught."""
    crit = scored["delay_status"] == "CRITICAL_DELAY"
    grouped = scored.assign(crit=crit, caught=crit & scored["alert"]).groupby("node")
    out = pd.DataFrame({
        "shipments": grouped.size(),
        "alerts": grouped["alert"].sum(),
        "critical": grouped["crit"].sum(),
        "caught": grouped["caught"].sum(),
    })
    out["alert_rate"] = out["alerts"] / out["shipments"]
    out["critical_share"] = out["critical"] / out["shipments"]
    out["catch_rate"] = (out["caught"] / out["critical"]).where(out["critical"] > 0)
    info = nodes().set_index("node_id")
    out = out.join(info[["display_name", "node_type", "region", "lat", "lon"]])
    out["type"] = out["node_type"].map(NODE_TYPE_LABEL)
    return out.reset_index().rename(columns={"node": "node_id"})


def sources(*paths: str) -> None:
    """One quiet line naming the files the numbers on screen came from."""
    st.caption("Source: " + " · ".join(f"`{p}`" for p in paths))
