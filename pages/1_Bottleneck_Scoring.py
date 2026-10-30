"""CASCADE control tower - Stage 1 page: shipment bottleneck scoring.
Owner: Role 3 (Data & Integration Engineer). Built by Role 4 at the team lead's request; Role 3 to review.

READS : Stage01_ML/reports/eval_champion.json (model + alert threshold, Role 4)
        Stage01_ML/data/prepared_dataset.csv.gz (shipments to look up and browse, Role 3)
        Stage01_ML/05_integration_engineer.py via common/integration/ui.py (the same scoring code as the API)
WRITES: nothing. Decision support only: every row needs a planner's review.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.integration import ui  # noqa: E402
from common.paths import stage_path  # noqa: E402

champion = ui.read_json(stage_path(1, "reports", "eval_champion.json"))
threshold = float(champion.get("threshold", 0.5))
NUMERIC = list(ui.FEATURE_LABEL)[:8]

st.title("Bottleneck scoring")
st.caption(f"Model **{champion.get('model_name', 'STUB')}** · alert when P(critical) ≥ **{threshold:.2f}** "
           f"(chosen on validation, team decision 0004) · every alert needs a planner's review")
if not champion:
    st.error("No champion file yet (Stage01_ML/reports/eval_champion.json): scores come from the STUB model.")


def answer_card(answer: dict, actual: tuple | None = None) -> None:
    """Prediction tiles: the model's call, risk, alert, and (for past shipments) what really happened."""
    p_crit = answer["probabilities"]["CRITICAL_DELAY"]
    cols = st.columns(4 if actual else 3)
    cols[0].markdown("**Model says**<br>" + ui.status_chip(answer["label"]), unsafe_allow_html=True)
    cols[1].markdown(f"**P(critical)** &nbsp; {p_crit:.0%}", unsafe_allow_html=True)
    cols[1].progress(min(1.0, p_crit))
    alert_text = "**Alert** — raise to a planner" if answer["alert"] else "**No alert**"
    cols[2].markdown(f"{alert_text}<br><span style='opacity:.7'>threshold {threshold:.2f}"
                     f"{' · unusual shipment' if answer['anomaly_flag'] else ''}</span>", unsafe_allow_html=True)
    if actual:
        status, hours = actual
        cols[3].markdown(f"**What really happened**<br>{ui.status_chip(status)} &nbsp; {hours:.1f} h late", unsafe_allow_html=True)
    st.markdown("**Top reasons:** " + " · ".join(ui.feature_label(d) for d in answer["top_drivers"]))


def conditions_table(row: pd.Series) -> None:
    """Each input next to what is normal at that node (train-period median), so outliers stand out."""
    typical = ui.node_typical()
    normal = typical.loc[row["current_node_id"]] if row["current_node_id"] in typical.index else pd.Series(dtype=float)
    table = pd.DataFrame({
        "Input": [ui.FEATURE_LABEL[c] for c in NUMERIC],
        "This shipment": [row[c] for c in NUMERIC],
        "Normal at this node": [normal.get(c) for c in NUMERIC],
    })
    table["Difference"] = table["This shipment"] - table["Normal at this node"]
    st.dataframe(table.round(2), hide_index=True, use_container_width=True)


lookup_tab, browse_tab, new_tab = st.tabs(["Look up a shipment", "Browse the test period", "Score a new shipment"])

# Step 1: one shipment by record id - header card, the model's answer, the real outcome, and its conditions.
with lookup_tab:
    # Default example: the riskiest test-period shipment that really did end critically late (a caught alert).
    test_scored = ui.scored_period("test")
    caught = test_scored[(test_scored["delay_status"] == "CRITICAL_DELAY") & test_scored["alert"]]
    example = caught.sort_values("p_critical", ascending=False)["record_id"].iloc[0] if len(caught) else "R059864"
    record_id = st.text_input("Record id", value=example, help="Any record id from the prepared dataset, e.g. from the Browse tab").strip().upper()
    data = ui.shipments()
    match = data[data["record_id"] == record_id]
    if match.empty:
        st.info("No shipment with that record id in the prepared dataset.")
    else:
        row = match.iloc[0]
        st.markdown(f"#### {row['shipment_id']} &nbsp;<span style='opacity:.6;font-weight:400'>{row['record_id']}</span>",
                    unsafe_allow_html=True)
        meta = st.columns(5)
        meta[0].markdown(f"**Route**<br>{ui.node_name(row['origin_port'])} → {ui.node_name(row['destination_port'])}", unsafe_allow_html=True)
        meta[1].markdown(f"**Now at**<br>{ui.node_name(row['current_node_id'])}", unsafe_allow_html=True)
        meta[2].markdown(f"**Carrier · priority**<br>{row['carrier_code']} · {row['priority_tier'].replace('_', ' ').title()}", unsafe_allow_html=True)
        meta[3].markdown(f"**Snapshot**<br>{pd.to_datetime(row['timestamp']):%d %b %Y %H:%M} UTC", unsafe_allow_html=True)
        period = {"train": "Training (model learned from it)", "val": "Validation (never trained on)", "test": "Test (never trained on)"}
        meta[4].markdown(f"**Period**<br>{period.get(row['split'], row['split'])}", unsafe_allow_html=True)
        st.divider()
        answers, problems = ui.validate_and_score([row.to_dict()])
        if problems:
            st.warning(problems[0]["problem"])
        else:
            answer_card(answers[0], (row["delay_status"], row["realized_delay_hours"]))
            st.markdown("##### Conditions at the node")
            conditions_table(row)

# Step 2: every test-period shipment, filterable, with the real outcome next to the model's call.
with browse_tab:
    scored = ui.scored_period("test")
    scored = scored.assign(location=scored["node"].map(ui.node_name))
    f = st.columns([2, 2, 1.3, 1.2])
    location = f[0].selectbox("Location", ["All locations"] + sorted(scored["location"].unique()))
    outcome = f[1].selectbox("What really happened", ["Any outcome", "On time", "At risk", "Critical"])
    only_alerts = f[2].toggle("Alerts only")
    missed = f[3].toggle("Missed criticals", help="Really critical, but no alert was raised")
    view = scored
    if location != "All locations":
        view = view[view["location"] == location]
    if outcome != "Any outcome":
        view = view[view["delay_status"].map(ui.status_label) == outcome]
    if only_alerts:
        view = view[view["alert"]]
    if missed:
        view = view[(view["delay_status"] == "CRITICAL_DELAY") & ~view["alert"]]
    crit = view["delay_status"] == "CRITICAL_DELAY"
    st.caption(f"{len(view):,} shipments · {int(view['alert'].sum()):,} alerts · "
               f"{int((crit & view['alert']).sum()):,} of {int(crit.sum()):,} real critical delays caught")
    show = view.sort_values("p_critical", ascending=False).assign(
        model_says=lambda d: d["label"].map(ui.status_label), really=lambda d: d["delay_status"].map(ui.status_label),
        p_critical=lambda d: 100 * d["p_critical"], when=lambda d: pd.to_datetime(d["timestamp"]).dt.strftime("%d %b %H:%M"))
    st.dataframe(
        show[["record_id", "when", "location", "model_says", "p_critical", "alert", "really", "realized_delay_hours", "reasons"]],
        hide_index=True, use_container_width=True, height=520,
        column_config={
            "record_id": st.column_config.TextColumn("Record"),
            "when": st.column_config.TextColumn("Snapshot (UTC)"),
            "location": st.column_config.TextColumn("Location"),
            "model_says": st.column_config.TextColumn("Model says"),
            "p_critical": st.column_config.ProgressColumn("P(critical)", min_value=0, max_value=100, format="%.0f%%"),
            "alert": st.column_config.CheckboxColumn("Alert"),
            "really": st.column_config.TextColumn("Really"),
            "realized_delay_hours": st.column_config.NumberColumn("Delay (h)", format="%.1f"),
            "reasons": st.column_config.TextColumn("Top reasons", width="large"),
        })

# Step 3: a new shipment typed in by hand, or a CSV of up to 1,000 rows.
with new_tab:
    node_ids = list(ui.nodes()["node_id"])

    def pick(col, label, default):
        """Node picker that shows names but returns the official node id."""
        return col.selectbox(label, node_ids, index=node_ids.index(default), format_func=ui.node_name)

    with st.form("new_shipment"):
        st.markdown("##### Route")
        r = st.columns(5)
        values = {
            "origin_port": pick(r[0], "From", "PORT_CN_SHA"),
            "destination_port": pick(r[1], "To", "PORT_US_LAX"),
            "current_node_id": pick(r[2], "Now at", "CHOKE_SUEZ"),
            "carrier_code": r[3].selectbox("Carrier", ["MSC", "MSK", "CMA", "COSCO", "HAPAG", "ONE", "EVERGREEN", "HMM", "YANGMING", "ZIM"]),
            "priority_tier": r[4].selectbox("Priority", ["STANDARD", "EXPEDITED", "CRITICAL_MEDICAL", "PERISHABLE_COLD_CHAIN"],
                                            format_func=lambda v: v.replace("_", " ").title()),
        }
        st.markdown("##### Conditions at the node")
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
        submitted = st.form_submit_button("Score this shipment", type="primary")
    if submitted:
        answers, problems = ui.validate_and_score([values])
        if problems:
            st.warning(problems[0]["problem"])
        else:
            answer_card(answers[0])

    st.divider()
    uploaded = st.file_uploader("Or score a CSV file (telemetry columns, up to 1,000 rows)", type="csv")
    if uploaded is not None:
        answers, problems = ui.validate_and_score(pd.read_csv(uploaded).head(1000).to_dict(orient="records"))
        if problems:
            st.warning(f"{len(problems)} row(s) break the data contract and were not scored.")
            st.dataframe(pd.DataFrame(problems), hide_index=True)
        if answers:
            out = pd.DataFrame(answers)
            out = pd.DataFrame({"Record": out["record_id"], "Location": out["node"].map(ui.node_name),
                                "Model says": out["label"].map(ui.status_label),
                                "P(critical)": out["probabilities"].map(lambda p: p["CRITICAL_DELAY"]),
                                "Alert": out["alert"], "Top reasons": out["top_drivers"].map(lambda d: " · ".join(map(ui.feature_label, d)))})
            st.dataframe(out.sort_values("P(critical)", ascending=False), hide_index=True, use_container_width=True,
                         column_config={"P(critical)": st.column_config.NumberColumn(format="%.2f")})

ui.sources("Stage01_ML/reports/eval_champion.json", "Stage01_ML/data/prepared_dataset.csv.gz", "Stage01_ML/05_integration_engineer.py")
