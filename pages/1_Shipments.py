"""CASCADE control tower - Shipments page (Stage 1, machine learning): which shipments will be late, and why?
Owner: Role 3 (Data & Integration Engineer).

What the planner sees, top to bottom (most urgent first, nothing to search for):
  1. one line: what the three statuses mean and when a shipment needs a planner check
  2. a status filter that also shows how many shipments are in each status (opens on Critical), and a location filter
  3. the list, highest chance of Critical first; clicking a row shows that shipment's chances and reasons right below
  4. "Check a new shipment": a closed section with the form that scores one shipment with the same code as the API
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
PRIORITY_LABEL = {"STANDARD": "Standard", "EXPEDITED": "Expedited",
                  "CRITICAL_MEDICAL": "Critical medical", "PERISHABLE_COLD_CHAIN": "Perishable (cold chain)"}


@st.cache_data
def table() -> pd.DataFrame:
    """Every test-period shipment with its predicted status, in readable columns, worst first (built once)."""
    scored = ui.test_period_scores()
    # Look node names up once as a plain dict (calling ui.node_name per row is slow on 8,000 rows).
    name = ui.nodes().set_index("node_id")["display_name"].to_dict()
    out = pd.DataFrame({
        "Shipment": scored["record_id"],
        "Time (UTC)": pd.to_datetime(scored["timestamp"]).dt.strftime("%d %b %H:%M"),
        "Now at": scored["node"].map(name),
        "Route": scored["origin_port"].map(name) + " → " + scored["destination_port"].map(name),
        "Status": scored["label"].map(SHORT),
        "Chance of Critical": 100 * scored["p_critical"],
        "Planner check": scored["alert"],
        "Why": scored["reasons"],
    })
    return out.sort_values("Chance of Critical", ascending=False).reset_index(drop=True)


def colour(status: str) -> str:
    """Text colour for a status cell (the word is always shown; colour only helps)."""
    key = {v: k for k, v in SHORT.items()}.get(status)
    return f"color: {ui.STATUS[key]['color']}; font-weight: 600" if key else ""


def explain(answer: dict) -> None:
    """One shipment in two lines: predicted status with the chance of each status and the check decision, then why."""
    p = answer["probabilities"]
    chances = " · ".join(f"{SHORT[s]} {p[s]:.0%}" for s in reversed(ui.STATUS_ORDER))
    check = "**needs a planner check**" if answer["alert"] else "no check needed"
    st.markdown(f"{ui.status_chip(answer['label'])} &nbsp; {chances} &nbsp;→&nbsp; {check}", unsafe_allow_html=True)
    st.markdown("**Why:** " + " · ".join(map(ui.feature_label, answer["top_drivers"])))


st.title("Shipments")
st.caption(f"{ui.model_label(champion.get('model_name', 'STUB'))} (machine learning) predicts each shipment as "
           + ", ".join(f"**{SHORT[s]}** ({ui.STATUS_MEANING[s]})" for s in reversed(ui.STATUS_ORDER))
           + f". A shipment needs a planner check when its chance of Critical is {threshold:.0%} or more; "
           "the tool never changes a booking by itself. Shown: the test period (Nov-Dec 2025), which the model never saw.")

# Step 1: filters. The status buttons carry the counts, so no separate count tiles are needed.
data = table()
counts = data["Status"].value_counts()
options = ["Critical", "At risk", "On time", "All"]
f1, f2 = st.columns([3, 2])
status = f1.segmented_control(
    "Status", options, default="Critical", key="status",
    format_func=lambda s: f"{s} ({len(data) if s == 'All' else int(counts.get(s, 0)):,})")
location = f2.selectbox("Now at", ["All locations"] + sorted(data["Now at"].unique()))
view = data
if status and status != "All":
    view = view[view["Status"] == status]
if location != "All locations":
    view = view[view["Now at"] == location]
view = view.reset_index(drop=True)

# Step 2: the list, highest chance of Critical first. Clicking a row shows its details right below.
event = st.dataframe(
    view.style.map(colour, subset=["Status"]), hide_index=True, width="stretch", height=360,
    on_select="rerun", selection_mode="single-row", key="shipments",
    column_config={
        "Chance of Critical": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "Planner check": st.column_config.CheckboxColumn(help=f"Chance of Critical {threshold:.0%} or more"),
        "Why": st.column_config.TextColumn(width="large", help="Inputs that pushed the risk up the most"),
    })
rows = event.selection.rows if event is not None else []
if rows:
    # Re-score the chosen shipment with the same code as the API, so all three chances are shown.
    row = view.iloc[rows[0]]
    record = ui.shipments()
    answers, _ = ui.validate_and_score(record[record["record_id"] == row["Shipment"]].to_dict(orient="records"))
    if answers:
        with st.container(border=True):
            st.markdown(f"**{row['Shipment']}** · {row['Route']} · now at {row['Now at']} · {row['Time (UTC)']} UTC")
            explain(answers[0])
else:
    st.caption(f"{len(view):,} shipments. Click a row to see its chances and why.")

# Step 3: check a new shipment (closed until needed, so it does not push the list down).
with st.expander("Check a new shipment"):
    node_ids = list(ui.nodes()["node_id"])
    # Carrier and priority choices come from the data contract, so the form never offers a value the API rejects.
    allowed = {c["name"]: c.get("allowed") for c in load_schema("telemetry")["columns"]}

    def pick(col, label, default):
        """Node picker that shows names but returns the official node id."""
        return col.selectbox(label, node_ids, index=node_ids.index(default), format_func=ui.node_name)

    with st.form("new_shipment", border=False):
        r = st.columns(5)
        values = {
            "origin_port": pick(r[0], "From", "PORT_CN_SHA"),
            "destination_port": pick(r[1], "To", "PORT_US_LAX"),
            "current_node_id": pick(r[2], "Now at", "CHOKE_SUEZ"),
            "carrier_code": r[3].selectbox("Shipping line", allowed["carrier_code"]),
            "priority_tier": r[4].selectbox("Priority", allowed["priority_tier"], format_func=PRIORITY_LABEL.get),
        }
        c = st.columns(4)
        values |= {
            "dwell_time_hours": c[0].number_input("Waiting so far (h)", 0.0, 400.0, 30.0,
                                                  help="Hours already waited at the current location"),
            "capacity_utilization_pct": c[1].number_input("Terminal full (%)", 0.0, 150.0, 75.0,
                                                          help="100% = full; up to 150% = overloaded"),
            "throughput_teu_per_hr": c[2].number_input("Containers per hour (TEU)", 0.0, 2500.0, 400.0),
            "customs_hold_hours": c[3].number_input("Customs hold (h)", 0.0, 240.0, 2.0),
            "weather_severity_index": c[0].number_input("Weather (0 calm - 10 extreme)", 0.0, 10.0, 2.0),
            "labor_deficit_ratio": c[1].number_input("Staff missing (0-1)", 0.0, 1.0, 0.05,
                                                     help="0.10 = 10% of the workforce is missing"),
            "equipment_availability_pct": c[2].number_input("Equipment working (%)", 0.0, 100.0, 92.0),
            "historical_delay_rate": c[3].number_input("Route late rate (0-1)", 0.0, 1.0, 0.3,
                                                       help="Share of earlier shipments on this route that were late"),
            "record_id": "MANUAL-1",
        }
        submitted = st.form_submit_button("Check", type="primary")
    if submitted:
        answers, problems = ui.validate_and_score([values])
        if problems:
            st.warning(problems[0]["problem"])
        else:
            explain(answers[0])
