"""CASCADE control tower - Shipments page (Stage 1): which shipments are likely to end critically late.
Owner: Role 3 (Data & Integration Engineer). Simplified by Role 4 at the team lead's request; Role 3 to review.

What the planner sees: a list of shipments with the chance of a delay of 24 hours or more, alerts first; click a
row for a short explanation; or type in a new shipment and check it. Only the champion model is used.

READS : Stage01_ML/data/prepared_dataset.csv.gz (test-period shipments), Stage01_ML/reports/eval_champion.json,
        the Stage 1 scoring code through common/integration/ui.py (the same code as the API)
WRITES: nothing. Decision support only: every alert needs a planner's review.
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


@st.cache_data
def shipments_table() -> pd.DataFrame:
    """Every test-period shipment scored by the champion model, with readable names."""
    scored = ui.scored_period("test")
    return pd.DataFrame({
        "Record": scored["record_id"],
        "Time (UTC)": pd.to_datetime(scored["timestamp"]).dt.strftime("%d %b %H:%M"),
        "Location": scored["node"].map(ui.node_name),
        "Route": scored["origin_port"].map(ui.node_name) + " → " + scored["destination_port"].map(ui.node_name),
        "Chance of 24 h+ delay": 100 * scored["p_critical"],
        "Alert": scored["alert"],
        "Main reasons": scored["reasons"],
    }).sort_values("Chance of 24 h+ delay", ascending=False).reset_index(drop=True)


def explain(p_critical: float, alert: bool, reasons: str) -> None:
    """Three short lines: the chance, the alert decision, and the main reasons."""
    c1, c2 = st.columns(2)
    c1.metric("Chance of a 24 h+ delay", f"{p_critical:.0%}")
    c2.metric("Alert", "Yes - check this shipment" if alert else "No")
    st.markdown(f"**Main reasons:** {reasons}")


st.title("Shipments")
st.caption(f"Chance that a shipment ends 24 hours or more late. An alert is raised at {threshold:.0%} or higher. "
           "Alerts are advice for a planner, never an automatic action.")

# Step 1: two filters, then the list (alerts first, highest chance on top).
table = shipments_table()
f1, f2 = st.columns([3, 1])
location = f1.selectbox("Location", ["All locations"] + sorted(table["Location"].unique()))
alerts_only = f2.toggle("Alerts only", value=True)
view = table
if location != "All locations":
    view = view[view["Location"] == location]
if alerts_only:
    view = view[view["Alert"]]
view = view.reset_index(drop=True)
st.markdown(f"**{len(view):,}** shipments shown · **{int(view['Alert'].sum()):,}** with an alert. Click a row for details.")
event = st.dataframe(
    view, hide_index=True, width="stretch", height=420, on_select="rerun", selection_mode="single-row", key="shipments",
    column_config={
        "Chance of 24 h+ delay": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "Alert": st.column_config.CheckboxColumn(),
        "Main reasons": st.column_config.TextColumn(width="large"),
    })

# Step 2: the selected shipment in plain words.
rows = event.selection.rows if event is not None else []
if rows:
    row = view.iloc[rows[0]]
    st.subheader(f"Shipment {row['Record']}")
    st.caption(f"{row['Route']} · now at {row['Location']} · {row['Time (UTC)']} UTC")
    explain(float(row["Chance of 24 h+ delay"]) / 100, bool(row["Alert"]), row["Main reasons"])

# Step 3: check a new shipment typed in by hand (same checks and scoring as the API).
st.divider()
with st.expander("Check a new shipment"):
    node_ids = list(ui.nodes()["node_id"])
    # Carrier and priority choices come from the data contract, so the form never offers a value the API rejects.
    allowed = {c["name"]: c.get("allowed") for c in load_schema("telemetry")["columns"]}

    def pick(col, label, default):
        """Node picker that shows names but returns the official node id."""
        return col.selectbox(label, node_ids, index=node_ids.index(default), format_func=ui.node_name)

    with st.form("new_shipment"):
        r = st.columns(5)
        values = {
            "origin_port": pick(r[0], "From", "PORT_CN_SHA"),
            "destination_port": pick(r[1], "To", "PORT_US_LAX"),
            "current_node_id": pick(r[2], "Now at", "CHOKE_SUEZ"),
            "carrier_code": r[3].selectbox("Carrier", allowed["carrier_code"]),
            "priority_tier": r[4].selectbox("Priority", allowed["priority_tier"], format_func=lambda v: v.replace("_", " ").title()),
        }
        c = st.columns(4)
        values |= {
            "dwell_time_hours": c[0].number_input("Dwell time at node (h)", 0.0, 400.0, 30.0),
            "capacity_utilization_pct": c[1].number_input("Capacity used (%)", 0.0, 150.0, 75.0),
            "throughput_teu_per_hr": c[2].number_input("Throughput (TEU/h)", 0.0, 2500.0, 400.0),
            "customs_hold_hours": c[3].number_input("Customs hold (h)", 0.0, 240.0, 2.0),
            "weather_severity_index": c[0].number_input("Weather severity (0-10)", 0.0, 10.0, 2.0),
            "labor_deficit_ratio": c[1].number_input("Labour shortage (0-1)", 0.0, 1.0, 0.05),
            "equipment_availability_pct": c[2].number_input("Equipment available (%)", 0.0, 100.0, 92.0),
            "historical_delay_rate": c[3].number_input("Historical delay rate (0-1)", 0.0, 1.0, 0.3),
            "record_id": "MANUAL-1",
        }
        submitted = st.form_submit_button("Check", type="primary")
    if submitted:
        answers, problems = ui.validate_and_score([values])
        if problems:
            st.warning(problems[0]["problem"])
        else:
            a = answers[0]
            explain(a["probabilities"]["CRITICAL_DELAY"], a["alert"], " · ".join(map(ui.feature_label, a["top_drivers"])))
