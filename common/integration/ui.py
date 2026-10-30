"""Shared helpers for the CASCADE control tower (Streamlit pages).
Owner: Role 3 (Data & Integration Engineer). Built by Role 4 at the team lead's request; Role 3 to review.

What lives here: plain-English names for nodes and model features, the status colours, the sidebar model
picker, and cached loaders for the files the pages read. Every number on screen comes from a file written
by a role script.

READS : contracts/network_nodes.csv, contracts/network_edges.csv, Stage01_ML/data/prepared_dataset.csv.gz,
        Stage01_ML/reports/*.json, Stage01_ML/05_integration_engineer.py (the scoring code the API uses)
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError

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
# Model names as a reader would say them.
MODEL_LABEL = {
    "logreg": "Logistic regression",
    "xgboost": "XGBoost",
    "xgboost_cal": "XGBoost (calibrated)",
    "xgboost_monotone": "XGBoost (monotone)",
    "lightgbm": "LightGBM",
    "random_forest": "Random forest",
    "rule_dwell": "Dwell-time rule",
    "majority": "Always 'on time'",
}
ONE_HOT_PREFIX = {"current_node_id_": "At", "origin_port_": "From", "destination_port_": "To",
                  "carrier_code_": "Carrier", "priority_tier_": "Priority"}

# Session-state key for the sidebar model picker; every page reads the same value.
_MODEL_KEY = "_cascade_selected_model"


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


def model_label(name: str) -> str:
    """'logreg' -> 'Logistic regression'."""
    return MODEL_LABEL.get(name, name)


def available_models() -> list[dict]:
    """Models the API can serve, each with its own threshold and a champion flag.

    Returns a list of dicts with keys 'model_name', 'alert_threshold', 'is_champion' (same shape as GET /models).
    Falls back to the champion alone when the model artefact is missing.
    """
    module = api()
    svc = module.service()
    if svc.stub:
        return [{"model_name": "STUB", "alert_threshold": 0.5, "is_champion": True}]
    return [{"model_name": m, "alert_threshold": svc.thresholds[m], "is_champion": m == svc.model_name}
            for m in svc.available]


def model_picker() -> str:
    """Sidebar selectbox that lets the user switch the scoring model on every page.

    Persists the choice in st.session_state so the overview, scoring, and evidence pages all score
    with the same model without re-selecting. Returns the chosen model's internal name (e.g. 'logreg').
    """
    models = available_models()
    names = [m["model_name"] for m in models]
    champion = next((m["model_name"] for m in models if m["is_champion"]), names[0])
    thresholds = {m["model_name"]: m["alert_threshold"] for m in models}

    # Default to the champion on first visit.
    if _MODEL_KEY not in st.session_state or st.session_state[_MODEL_KEY] not in names:
        st.session_state[_MODEL_KEY] = champion

    def _format(name: str) -> str:
        """Show the human name plus '(champion)' for the current best."""
        label = model_label(name)
        return f"{label} (champion)" if name == champion else label

    selected = st.sidebar.selectbox(
        "Scoring model",
        names,
        index=names.index(st.session_state[_MODEL_KEY]),
        format_func=_format,
        key=_MODEL_KEY,
        help="Switch the model used to score shipments on every page. The champion was chosen on the validation period.",
    )
    # Show the alert threshold for the selected model so the user knows the decision rule.
    st.sidebar.caption(f"Alert threshold: {thresholds[selected]:.0%}")
    return selected


def selected_model() -> str:
    """The model currently chosen in the sidebar picker, or the champion if the picker has not been shown yet."""
    return st.session_state.get(_MODEL_KEY, None)


def join_words(items: list) -> str:
    """['a', 'b', 'c'] -> 'a, b and c'."""
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


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


def validate_and_score(records: list, model_name: str | None = None) -> tuple:
    """Check rows against the API's contract model, then score them. Bad rows are returned, never dropped silently.

    When model_name is None, the sidebar selection (session state) is used; when that is also unset the champion
    is used. Pass a model_name explicitly to override.
    """
    module = api()
    # Resolve which model to score with: explicit argument > sidebar picker > champion default.
    name = model_name if model_name is not None else selected_model()
    fields = module.NUMERIC_FIELDS + ["priority_tier", "carrier_code", "origin_port", "destination_port",
                                      "current_node_id", "timestamp", "record_id"]
    good, problems = [], []
    for i, record in enumerate(records):
        # Blank CSV cells arrive as NaN; treat them as "missing" so the contract check names the real problem (Role 3).
        values = {k: (None if pd.isna(record[k]) else record[k]) for k in fields if k in record}
        if values.get("record_id") is not None:
            values["record_id"] = str(values["record_id"])
        try:
            good.append(module.Shipment(**values))
        except ValidationError as err:
            # One readable line per broken field, e.g. "dwell_time_hours: Input should be greater than or equal to 0".
            reasons = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors()]
            problems.append({"row": i, "problem": "; ".join(reasons)})
    answers = module.service().score(good, model_name=name) if good else []
    for answer, shipment in zip(answers, good):
        answer["node"] = shipment.current_node_id
    return answers, problems


@st.cache_data
def official_predictions(split: str) -> pd.DataFrame:
    """The champion's evaluated predictions for one period (Role 1's file, the numbers Role 4 reports)."""
    champion = read_json(stage_path(1, "reports", "eval_champion.json"))
    path = stage_path(1, "predictions", "ml_predictions.csv")
    if not champion or not path.exists():
        return pd.DataFrame()
    pred = pd.read_csv(path, usecols=["record_id", "split", "model_name", "y_pred", "p_critical"])
    return pred[(pred["model_name"] == champion["model_name"]) & (pred["split"] == split)].set_index("record_id")


def scored_period(split: str, model_name: str | None = None) -> pd.DataFrame:
    """Every shipment of one period with the model's answer and the real outcome.

    When model_name is None the sidebar selection is used (champion is the fallback).

    For the *champion* model, risk, label and alert come from the official predictions file so the dashboard
    shows exactly the numbers in the evaluation report. For any other model, all numbers come from live scoring.
    """
    name = model_name if model_name is not None else selected_model()
    rows = shipments()
    rows = rows[rows["split"] == split]
    answers, _ = validate_and_score(rows.to_dict(orient="records"), model_name=name)
    table = pd.DataFrame(answers)
    table["p_critical"] = table["probabilities"].map(lambda p: p["CRITICAL_DELAY"])

    # For the champion, overlay the official evaluated predictions so the numbers match the evaluation report.
    svc = api().service()
    is_champion = name is None or name == svc.model_name
    if is_champion:
        official = official_predictions(split)
        if len(official):
            threshold = float(read_json(stage_path(1, "reports", "eval_champion.json"))["threshold"])
            table["p_critical"] = table["record_id"].map(official["p_critical"]).fillna(table["p_critical"])
            table["label"] = table["record_id"].map(official["y_pred"]).fillna(table["label"])
            table["alert"] = table["p_critical"] >= threshold

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
