"""CASCADE control tower - Stage 1 page: shipment bottleneck scoring.
Owner: Role 3 (Data & Integration Engineer). Built by Role 4 at the team lead's request; Role 3 to review.

READS : Stage01_ML/reports/eval_champion.json, eval_model_comparison.csv   (Role 4)
        Stage01_ML/05_integration_engineer.py (the same scoring code the API uses)
        Stage01_ML/data/prepared_dataset.csv.gz (shipments to look up and browse)
WRITES: nothing. Decision support only: every row needs a planner's review.

Start with:   streamlit run app.py   (then open "Bottleneck Scoring" in the sidebar)
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.paths import stage_path  # noqa: E402

st.set_page_config(page_title="Bottleneck scoring", layout="wide")
TEXT_FIELDS = ["priority_tier", "carrier_code", "origin_port", "destination_port", "current_node_id"]


@st.cache_resource
def api_module():
    """Load the Stage 1 API module once, so the page scores with exactly the same code as the API."""
    spec = importlib.util.spec_from_file_location("stage01_api", REPO_ROOT / "Stage01_ML" / "05_integration_engineer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.service()  # load the model now, on the main thread
    return module


@st.cache_data
def shipments() -> pd.DataFrame:
    """All labelled shipments from the prepared dataset (the val and test periods are the ones no model trained on)."""
    return pd.read_csv(stage_path(1, "data", "prepared_dataset.csv.gz"))


@st.cache_data
def scored_period(split: str) -> pd.DataFrame:
    """Score every shipment of one period once, and keep the real outcome next to the model's answer."""
    api = api_module()
    rows = shipments()
    rows = rows[rows["split"] == split]
    answers = to_table(api, score_records(api, rows.to_dict(orient="records"))[0])
    real = rows.set_index("record_id")[["timestamp", "delay_status", "realized_delay_hours"]]
    return answers.join(real, on="record_id")


def read_json(path: Path) -> dict:
    """Read a JSON report, or an empty dict if it has not been produced yet."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def score_records(api, records: list) -> tuple:
    """Validate rows with the API's own contract model, then score them. Bad rows are listed, never dropped silently."""
    fields = api.NUMERIC_FIELDS + TEXT_FIELDS + ["timestamp", "record_id"]
    good, problems = [], []
    for i, record in enumerate(records):
        try:
            good.append(api.Shipment(**{k: record.get(k) for k in fields if k in record}))
        except Exception as err:  # pydantic ValidationError: show it to the planner
            problems.append({"row": i, "problem": str(err).splitlines()[1] if "\n" in str(err) else str(err)})
    answers = api.service().score(good) if good else []
    for answer, shipment in zip(answers, good):
        answer["node"] = shipment.current_node_id
    return answers, problems


def to_table(api, answers: list) -> pd.DataFrame:
    """Flatten API answers into one row per shipment for display."""
    if not answers:
        return pd.DataFrame()
    table = pd.DataFrame(answers)
    table["p_critical"] = table["probabilities"].map(lambda p: p["CRITICAL_DELAY"])
    table["drivers"] = table["top_drivers"].map(", ".join)
    return table[["record_id", "node", "label", "p_critical", "risk_score", "alert", "anomaly_flag", "drivers"]]


# Step 1: the decision rule in force, read from Role 4's champion file (never typed by hand).
champion = read_json(stage_path(1, "reports", "eval_champion.json"))
st.title("Shipment bottleneck scoring")
if not champion:
    st.error("No champion file yet (Stage01_ML/reports/eval_champion.json). Scores below come from the STUB model.")
cols = st.columns(4)
cols[0].metric("Model served", champion.get("model_name", "STUB"))
cols[1].metric("Alert when P(critical) is at least", f"{champion.get('threshold', 0.5):.2f}")
cols[2].metric("Chosen on", champion.get("selected_on", "-"))
cols[3].metric("Beats best baseline", str(champion.get("beats_baseline", "-")))
st.caption("Decision support only: every alert requires operator review. Nothing here reroutes or books a shipment.")

# Step 2: model-vs-baseline mini table from Role 4's evaluation.
comparison_file = stage_path(1, "reports", "eval_model_comparison.csv")
if comparison_file.exists():
    table = pd.read_csv(comparison_file)
    show = table[["model_name", "is_baseline", "macro_f1", "pr_auc_critical", "expected_cost_at_threshold", "threshold",
                  "at_threshold_critical_recall", "at_threshold_false_alarms_per_100_alerts"]]
    show = show.rename(columns={"expected_cost_at_threshold": "cost_per_shipment", "at_threshold_critical_recall": "critical_caught",
                                "at_threshold_false_alarms_per_100_alerts": "false_alarms_per_100"})
    with st.expander("Models on the held-out test period", expanded=False):
        st.dataframe(show.sort_values("cost_per_shipment").round(3), hide_index=True, use_container_width=True)

api = api_module()
lookup_tab, browse_tab, new_tab = st.tabs(["Look up a shipment", "Browse the test period", "Score a new shipment or file"])

# Step 3: look up one shipment by its record id and compare the model's answer with what really happened.
with lookup_tab:
    record_id = st.text_input("Record id (for example R059864)", value="R059864").strip().upper()
    data = shipments()
    match = data[data["record_id"] == record_id]
    if match.empty:
        st.info("No shipment with that record id in the prepared dataset.")
    else:
        row = match.iloc[0]
        answers, problems = score_records(api, [row.to_dict()])
        if problems:
            st.warning(problems[0]["problem"])
        else:
            answer = answers[0]
            c = st.columns(4)
            c[0].metric("Model says", answer["label"])
            c[1].metric("P(critical)", f"{answer['probabilities']['CRITICAL_DELAY']:.0%}")
            c[2].metric("Alert", "YES" if answer["alert"] else "no")
            c[3].metric("What really happened", f"{row['delay_status']} ({row['realized_delay_hours']:.1f} h late)")
            st.write(f"**Top reasons:** {', '.join(answer['top_drivers'])}  ·  **Unusual shipment:** {'yes' if answer['anomaly_flag'] else 'no'}"
                     f"  ·  **Period:** {row['split']} (the model never trained on val or test)")
            st.dataframe(match[["timestamp", "shipment_id", "carrier_code", "priority_tier", "origin_port", "destination_port",
                                "current_node_id"] + api.NUMERIC_FIELDS].T.rename(columns={match.index[0]: "value"}).astype(str), use_container_width=True)

# Step 4: browse every test-period shipment with filters, real outcome next to the model's answer.
with browse_tab:
    scored = scored_period("test")
    f = st.columns(3)
    node = f[0].selectbox("Node", ["All"] + sorted(scored["node"].unique()))
    only_alerts = f[1].checkbox("Only shipments with an alert")
    outcome = f[2].selectbox("What really happened", ["All", "ON_TIME", "AT_RISK", "CRITICAL_DELAY"])
    view = scored
    if node != "All":
        view = view[view["node"] == node]
    if only_alerts:
        view = view[view["alert"]]
    if outcome != "All":
        view = view[view["delay_status"] == outcome]
    caught = int((view["alert"] & (view["delay_status"] == "CRITICAL_DELAY")).sum())
    crit = int((view["delay_status"] == "CRITICAL_DELAY").sum())
    st.caption(f"{len(view):,} shipments · {int(view['alert'].sum()):,} alerts · {caught:,} of {crit:,} real critical delays caught")
    st.dataframe(view.sort_values("risk_score", ascending=False).rename(columns={"delay_status": "really_happened"}).round(3),
                 hide_index=True, use_container_width=True)

# Step 5: score a brand-new shipment typed in by hand, or a whole CSV file.
with new_tab:
    nodes = sorted(api._node_ids())
    with st.form("new_shipment"):
        g = st.columns(3)
        values = {
            "dwell_time_hours": g[0].number_input("Dwell time (hours)", 0.0, 400.0, 30.0),
            "capacity_utilization_pct": g[0].number_input("Capacity utilisation (%)", 0.0, 150.0, 75.0),
            "throughput_teu_per_hr": g[0].number_input("Throughput (TEU per hour)", 0.0, 2500.0, 400.0),
            "customs_hold_hours": g[1].number_input("Customs hold (hours)", 0.0, 240.0, 2.0),
            "weather_severity_index": g[1].number_input("Weather severity (0-10)", 0.0, 10.0, 2.0),
            "labor_deficit_ratio": g[1].number_input("Labour shortage (0-1)", 0.0, 1.0, 0.05),
            "equipment_availability_pct": g[2].number_input("Equipment available (%)", 0.0, 100.0, 92.0),
            "historical_delay_rate": g[2].number_input("Historical delay rate (0-1)", 0.0, 1.0, 0.3),
            "priority_tier": g[2].selectbox("Priority", ["STANDARD", "EXPEDITED", "CRITICAL_MEDICAL", "PERISHABLE_COLD_CHAIN"]),
            "carrier_code": g[0].selectbox("Carrier", ["MSC", "MSK", "CMA", "COSCO", "HAPAG", "ONE", "EVERGREEN", "HMM", "YANGMING", "ZIM"]),
            "origin_port": g[1].selectbox("Origin", nodes, index=nodes.index("PORT_CN_SHA") if "PORT_CN_SHA" in nodes else 0),
            "destination_port": g[2].selectbox("Destination", nodes, index=nodes.index("PORT_US_LAX") if "PORT_US_LAX" in nodes else 0),
            "current_node_id": g[0].selectbox("Current location", nodes, index=nodes.index("CHOKE_SUEZ") if "CHOKE_SUEZ" in nodes else 0),
            "record_id": "MANUAL-1",
        }
        submitted = st.form_submit_button("Score this shipment")
    if submitted:
        answers, problems = score_records(api, [values])
        if problems:
            st.warning(problems[0]["problem"])
        else:
            a = answers[0]
            st.success(f"{a['label']} · P(critical) {a['probabilities']['CRITICAL_DELAY']:.0%} · alert: {'YES' if a['alert'] else 'no'} "
                       f"· top reasons: {', '.join(a['top_drivers'])}")

    uploaded = st.file_uploader("Or upload a CSV with the telemetry columns (up to 1,000 rows)", type="csv")
    if uploaded is not None:
        answers, problems = score_records(api, pd.read_csv(uploaded).head(api.MAX_BATCH).to_dict(orient="records"))
        if problems:
            st.warning(f"{len(problems)} row(s) break the data contract and were not scored.")
            st.dataframe(pd.DataFrame(problems), hide_index=True)
        if answers:
            st.dataframe(to_table(api, answers).sort_values("risk_score", ascending=False), hide_index=True, use_container_width=True)
