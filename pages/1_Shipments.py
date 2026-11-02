"""CASCADE control tower - Shipments page (Stage 1, machine learning): On time, At risk or Critical?
Owner: Role 3 (Data & Integration Engineer). Simplified by Role 4 at the team lead's request; Role 3 to review.

What the planner sees, top to bottom:
  1. "Check a shipment": a form (open, every input explained) that scores one shipment with the Stage 1 model
  2. how many test-period shipments the model puts in each of the three statuses
  3. a list filtered by status and location; clicking a row shows the chance of each status and the main reasons
Only the champion model is used.

READS : Stage01_ML/data/prepared_dataset.csv.gz (test-period shipments), Stage01_ML/reports/eval_champion.json,
        the Stage 1 scoring code through common/integration/ui.py (the same code as the API)
WRITES: nothing. Decision support only: every flagged shipment needs a planner's review.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.contracts import load_schema  # noqa: E402
from common.integration import ui  # noqa: E402
from common.paths import stage_path  # noqa: E402

champion = ui.read_json(stage_path(1, "reports", "eval_champion.json"))
threshold = float(champion.get("threshold", 0.5))
SHORT = {"CRITICAL_DELAY": "Critical", "AT_RISK": "At risk", "ON_TIME": "On time"}
PRIORITY_LABEL = {"STANDARD": "Standard", "EXPEDITED": "Expedited (faster service)",
                  "CRITICAL_MEDICAL": "Critical medical supplies", "PERISHABLE_COLD_CHAIN": "Perishable (cold chain)"}


@st.cache_data
def table() -> pd.DataFrame:
    """Every test-period shipment with its predicted status, in readable columns, worst first (built once)."""
    scored = ui.test_period_scores()
    # Look node names up once as a plain dict (calling ui.node_name per row is slow on 8,000 rows).
    name = ui.nodes().set_index("node_id")["display_name"].to_dict()
    out = pd.DataFrame({
        "Record": scored["record_id"],
        "Time (UTC)": pd.to_datetime(scored["timestamp"]).dt.strftime("%d %b %H:%M"),
        "Location": scored["node"].map(name),
        "Route": scored["origin_port"].map(name) + " → " + scored["destination_port"].map(name),
        "Predicted status": scored["label"].map(SHORT),
        "Chance of Critical": 100 * scored["p_critical"],
        "Planner check": scored["alert"],
        "Main reasons": scored["reasons"],
    })
    return out.sort_values("Chance of Critical", ascending=False).reset_index(drop=True)


def colour(status: str) -> str:
    """Text colour for a status cell (the word is always shown; colour only helps)."""
    key = {v: k for k, v in SHORT.items()}.get(status)
    return f"color: {ui.STATUS[key]['color']}; font-weight: 600" if key else ""


def explain(answer: dict) -> None:
    """One shipment in plain words: predicted status, the chance of each status, planner check, main reasons."""
    p = answer["probabilities"]
    st.markdown(f"Predicted status: {ui.status_chip(answer['label'])} &nbsp; ({ui.STATUS_MEANING[answer['label']]})",
                unsafe_allow_html=True)
    cols = st.columns(3)
    for col, status in zip(cols, reversed(ui.STATUS_ORDER)):
        col.markdown(f"Chance of **{SHORT[status]}**: {p[status]:.0%}")
        col.progress(min(1.0, float(p[status])))
    if answer["alert"]:
        st.warning(f"Planner check: yes. The chance of Critical is {p['CRITICAL_DELAY']:.0%}, at or above {threshold:.0%}.")
    else:
        st.success(f"Planner check: no. The chance of Critical is {p['CRITICAL_DELAY']:.0%}, below {threshold:.0%}.")
    st.markdown("**Main reasons:** " + " · ".join(map(ui.feature_label, answer["top_drivers"])))


st.title("Shipments")
st.markdown(f"**Machine learning (Stage 1)** · model: {ui.model_label(champion.get('model_name', 'STUB'))}, "
            "trained on past shipment records")
st.markdown("For every shipment, the model predicts one of three outcomes: "
            + " · ".join(f"{ui.status_chip(s)} {ui.STATUS_MEANING[s]}" for s in reversed(ui.STATUS_ORDER)),
            unsafe_allow_html=True)
st.caption(f"A shipment is sent to a planner for a check when its chance of Critical is {threshold:.0%} or more. "
           "That covers almost every Critical shipment and a few At risk ones. The tool never acts by itself.")

# Step 1: check one shipment. The form is open at the top; every input says what it means and its unit.
with st.container(border=True):
    st.subheader("Check a shipment")
    st.caption("Enter the shipment's route and the conditions where it is now, then press **Check**. "
               "Hover over the (?) next to a field for an explanation.")
    node_ids = list(ui.nodes()["node_id"])
    # Carrier and priority choices come from the data contract, so the form never offers a value the API rejects.
    allowed = {c["name"]: c.get("allowed") for c in load_schema("telemetry")["columns"]}

    def pick(col, label, default, help_text):
        """Node picker that shows names but returns the official node id."""
        return col.selectbox(label, node_ids, index=node_ids.index(default), format_func=ui.node_name, help=help_text)

    with st.form("new_shipment"):
        st.markdown("**Route**")
        r = st.columns(5)
        values = {
            "origin_port": pick(r[0], "Origin port", "PORT_CN_SHA", "Port where the shipment was loaded."),
            "destination_port": pick(r[1], "Destination", "PORT_US_LAX", "Final port or hub the shipment is going to."),
            "current_node_id": pick(r[2], "Current location", "CHOKE_SUEZ",
                                    "Port, canal or hub where the shipment is at the moment of this check."),
            "carrier_code": r[3].selectbox("Shipping line", allowed["carrier_code"], help="Company carrying the container."),
            "priority_tier": r[4].selectbox("Priority", allowed["priority_tier"], format_func=PRIORITY_LABEL.get,
                                            help="Service level booked by the customer."),
        }
        st.markdown("**Conditions at the current location**")
        c = st.columns(4)
        values |= {
            "dwell_time_hours": c[0].number_input(
                "Waiting time so far (hours)", 0.0, 400.0, 30.0,
                help="How long the shipment has already waited at its current location (dwell time), 0 to 400 hours."),
            "capacity_utilization_pct": c[1].number_input(
                "Terminal capacity used (%)", 0.0, 150.0, 75.0,
                help="How full the terminal or canal is. 100% is full; values up to 150% mean it is overloaded."),
            "throughput_teu_per_hr": c[2].number_input(
                "Containers handled per hour (TEU)", 0.0, 2500.0, 400.0,
                help="Throughput of the location. TEU = twenty-foot equivalent unit, one standard container."),
            "customs_hold_hours": c[3].number_input(
                "Customs hold (hours)", 0.0, 240.0, 2.0,
                help="Hours the shipment is held for customs inspection, 0 to 240."),
            "weather_severity_index": c[0].number_input(
                "Weather severity (0 = calm, 10 = extreme)", 0.0, 10.0, 2.0,
                help="Weather at the location on a 0 to 10 scale."),
            "labor_deficit_ratio": c[1].number_input(
                "Staff shortage (0 = none, 1 = no staff)", 0.0, 1.0, 0.05,
                help="Share of the needed workers who are missing. 0.10 means 10% of the workforce is missing."),
            "equipment_availability_pct": c[2].number_input(
                "Equipment working (%)", 0.0, 100.0, 92.0,
                help="Share of cranes and other handling equipment that is working, 0 to 100%."),
            "historical_delay_rate": c[3].number_input(
                "Past late-shipment rate on this route (0 to 1)", 0.0, 1.0, 0.3,
                help="Share of earlier shipments on this route that arrived late. 0.30 means 30%."),
            "record_id": "MANUAL-1",
        }
        submitted = st.form_submit_button("Check", type="primary")
    if submitted:
        answers, problems = ui.validate_and_score([values])
        if problems:
            st.warning(problems[0]["problem"])
        else:
            explain(answers[0])

# Step 2: how many test-period shipments are in each status.
st.subheader("All shipments in the test period")
st.caption("Shipments the model has never seen, replayed so you can browse its predictions.")
data = table()
counts = data["Predicted status"].value_counts()
tiles = st.columns(3)
for tile, status in zip(tiles, ui.STATUS_ORDER):
    name = SHORT[status]
    tile.metric(f"{name} ({ui.STATUS_MEANING[status]})", f"{int(counts.get(name, 0)):,}")

# Step 3: filter by status and location, then the list (highest chance of Critical on top).
f1, f2 = st.columns([2, 3])
status = f1.radio("Show shipments predicted as", ["Critical", "At risk", "On time", "All"], horizontal=True)
location = f2.selectbox("Current location", ["All locations"] + sorted(data["Location"].unique()),
                        help="Port, canal or hub where the shipment was at the time of the record.")
view = data
if status != "All":
    view = view[view["Predicted status"] == status]
if location != "All locations":
    view = view[view["Location"] == location]
view = view.reset_index(drop=True)
st.markdown(f"**{len(view):,}** shipments shown. Click a row to see why.")
event = st.dataframe(
    view.style.map(colour, subset=["Predicted status"]), hide_index=True, width="stretch", height=380,
    on_select="rerun", selection_mode="single-row", key="shipments",
    column_config={
        "Time (UTC)": st.column_config.TextColumn(help="When the shipment record was taken"),
        "Location": st.column_config.TextColumn(help="Where the shipment was at that time"),
        "Predicted status": st.column_config.TextColumn(help="The model's most likely outcome"),
        "Chance of Critical": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100,
                                                              help="Chance the shipment ends 24 hours or more late"),
        "Planner check": st.column_config.CheckboxColumn(help=f"Ticked when the chance of Critical is {threshold:.0%} or more"),
        "Main reasons": st.column_config.TextColumn(width="large", help="Inputs that pushed the risk up the most"),
    })

# Step 4: the selected shipment, re-scored with the same code as the API so all three chances are shown.
rows = event.selection.rows if event is not None else []
if rows:
    row = view.iloc[rows[0]]
    record = ui.shipments()
    record = record[record["record_id"] == row["Record"]]
    answers, _ = ui.validate_and_score(record.to_dict(orient="records"))
    if answers:
        st.subheader(f"Shipment {row['Record']}")
        st.caption(f"{row['Route']} · now at {row['Location']} · {row['Time (UTC)']} UTC")
        explain(answers[0])
