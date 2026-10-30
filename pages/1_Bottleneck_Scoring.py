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

from common.contracts import load_schema  # noqa: E402
from common.integration import ui  # noqa: E402
from common.paths import stage_path  # noqa: E402

champion = ui.read_json(stage_path(1, "reports", "eval_champion.json"))
threshold = float(champion.get("threshold", 0.5))
NUMERIC = list(ui.FEATURE_LABEL)[:8]

st.title("Bottleneck scoring")
model_name = ui.model_label(champion.get("model_name", "STUB"))
st.caption(f"Shipments are scored by **{model_name}**. An alert is raised when the chance of a critical delay "
           f"(24 hours or more) is **{threshold:.0%} or higher**; this threshold was chosen on the validation period and "
           "agreed by the team (decision 0004). Every alert is advice for a planner, never an automatic action.")
if not champion:
    st.error("No champion file yet (Stage01_ML/reports/eval_champion.json): scores come from the STUB model.")


def verdict(answer: dict, actual: str | None) -> str:
    """One plain-English sentence on whether the alert decision was right, for shipments whose outcome is known."""
    critical = actual == "CRITICAL_DELAY"
    if answer["alert"] and critical:
        return "The alert was **correct**: this shipment really did end up critically late."
    if answer["alert"]:
        return f"This was a **false alarm**: the shipment ended up {ui.status_label(actual).lower()}, not critically late."
    if critical:
        return "This critical delay was **missed**: the risk stayed below the alert threshold, so no planner was warned."
    return "Leaving this shipment alone was **correct**: it did not become critically late."


def answer_card(answer: dict, actual: tuple | None = None) -> None:
    """The model's assessment as tiles, then the verdict and the main risk factors as full sentences."""
    p_crit = answer["probabilities"]["CRITICAL_DELAY"]
    cols = st.columns(4 if actual else 3)
    cols[0].markdown("**Model's assessment**<br>" + ui.status_chip(answer["label"]), unsafe_allow_html=True)
    cols[1].markdown(f"**Chance of a critical delay** &nbsp; {p_crit:.0%}", unsafe_allow_html=True)
    cols[1].progress(min(1.0, p_crit))
    decision = "Raised to a planner" if answer["alert"] else "Not raised"
    cols[2].markdown(f"**Alert decision**<br>{decision}<br><span style='opacity:.7'>alert line {threshold:.0%}</span>",
                     unsafe_allow_html=True)
    if actual:
        status, hours = actual
        cols[3].markdown(f"**Actual outcome**<br>{ui.status_chip(status)} &nbsp; arrived {hours:.1f} h late", unsafe_allow_html=True)

    # The verdict, written as sentences rather than labels.
    side = "above" if answer["alert"] else "below"
    sentences = [f"The model rates this shipment **{ui.status_label(answer['label']).lower()}**, with a **{p_crit:.0%}** chance "
                 f"of a delay of 24 hours or more. That is {side} the {threshold:.0%} alert line, so it is "
                 f"{'raised to a planner' if answer['alert'] else 'not raised'}."]
    if actual:
        sentences.append(verdict(answer, actual[0]))
    if answer["anomaly_flag"]:
        sentences.append("Its conditions are also **unusual** compared with normal traffic, so it deserves a closer look.")
    st.markdown(" ".join(sentences))
    reasons = [ui.feature_label(d).lower() for d in answer["top_drivers"]]
    st.markdown(f"The factors pushing the risk up most were **{ui.join_words(reasons)}**.")


def conditions_table(row: pd.Series) -> None:
    """Each measure next to what is typical at that node (training-period median), with a one-line summary."""
    typical = ui.node_typical()
    normal = typical.loc[row["current_node_id"]] if row["current_node_id"] in typical.index else pd.Series(dtype=float)
    table = pd.DataFrame({
        "Measure": [ui.FEATURE_LABEL[c] for c in NUMERIC],
        "This shipment": [row[c] for c in NUMERIC],
        "Typical at this node": [normal.get(c) for c in NUMERIC],
    })
    table["Difference"] = table["This shipment"] - table["Typical at this node"]
    # Summarise the two measures that are furthest above normal, in relative terms.
    relative = (table["Difference"] / table["Typical at this node"].where(table["Typical at this node"] > 0)).fillna(0)
    top = table.assign(relative=relative).sort_values("relative", ascending=False).head(2)
    top = top[top["relative"] > 0.2]
    if len(top):
        parts = [f"{r.Measure.lower()} is {r.relative + 1:.1f} times the usual level ({r._2:.1f} against {r._3:.1f})"
                 for r in top.itertuples()]
        st.markdown(f"Compared with a typical shipment at {ui.node_name(row['current_node_id'])}, {ui.join_words(parts)}.")
    else:
        st.markdown(f"Conditions are close to normal for {ui.node_name(row['current_node_id'])}.")
    st.dataframe(table.round(2), hide_index=True, use_container_width=True,
                 column_config={"Difference": st.column_config.NumberColumn(format="%+.2f")})


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
        meta[1].markdown(f"**Current location**<br>{ui.node_name(row['current_node_id'])}", unsafe_allow_html=True)
        meta[2].markdown(f"**Carrier and priority**<br>{row['carrier_code']} · {row['priority_tier'].replace('_', ' ').title()}", unsafe_allow_html=True)
        meta[3].markdown(f"**Snapshot taken**<br>{pd.to_datetime(row['timestamp']):%d %b %Y %H:%M} UTC", unsafe_allow_html=True)
        period = {"train": "Training period (the model learned from it)", "val": "Validation period (not used for training)",
                  "test": "Test period (not used for training)"}
        meta[4].markdown(f"**Data period**<br>{period.get(row['split'], row['split'])}", unsafe_allow_html=True)
        st.divider()
        answers, problems = ui.validate_and_score([row.to_dict()])
        if problems:
            st.warning(problems[0]["problem"])
        else:
            answer_card(answers[0], (row["delay_status"], row["realized_delay_hours"]))
            st.markdown(f"##### Conditions at {ui.node_name(row['current_node_id'])} compared with normal")
            conditions_table(row)

# Step 2: every test-period shipment, filterable, with the real outcome next to the model's call.
with browse_tab:
    scored = ui.scored_period("test")
    scored = scored.assign(location=scored["node"].map(ui.node_name))
    f = st.columns([2, 2, 1.3, 1.2])
    location = f[0].selectbox("Location", ["All locations"] + sorted(scored["location"].unique()))
    outcome = f[1].selectbox("Actual outcome", ["Any outcome", "On time", "At risk", "Critical"])
    only_alerts = f[2].toggle("Alerts only")
    missed = f[3].toggle("Missed critical delays", help="Shipments that ended critically late without an alert")
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
    st.markdown(f"Showing **{len(view):,}** shipments. The model raised **{int(view['alert'].sum()):,}** alerts and caught "
                f"**{int((crit & view['alert']).sum()):,}** of the **{int(crit.sum()):,}** shipments that actually ended critically late.")
    show = view.sort_values("p_critical", ascending=False).assign(
        model_says=lambda d: d["label"].map(ui.status_label), really=lambda d: d["delay_status"].map(ui.status_label),
        p_critical=lambda d: 100 * d["p_critical"], when=lambda d: pd.to_datetime(d["timestamp"]).dt.strftime("%d %b %H:%M"))
    st.dataframe(
        show[["record_id", "when", "location", "model_says", "p_critical", "alert", "really", "realized_delay_hours", "reasons"]],
        hide_index=True, use_container_width=True, height=520,
        column_config={
            "record_id": st.column_config.TextColumn("Record"),
            "when": st.column_config.TextColumn("Snapshot taken (UTC)"),
            "location": st.column_config.TextColumn("Location"),
            "model_says": st.column_config.TextColumn("Model's assessment"),
            "p_critical": st.column_config.ProgressColumn("Chance of critical delay", min_value=0, max_value=100, format="%.0f%%"),
            "alert": st.column_config.CheckboxColumn("Alert raised"),
            "really": st.column_config.TextColumn("Actual outcome"),
            "realized_delay_hours": st.column_config.NumberColumn("Actual delay (h)", format="%.1f"),
            "reasons": st.column_config.TextColumn("Main risk factors", width="large"),
        })

# Step 3: a new shipment typed in by hand, or a CSV of up to 1,000 rows.
with new_tab:
    node_ids = list(ui.nodes()["node_id"])
    # Carrier and priority choices come from the telemetry contract, so the form never offers a value the API rejects (Role 3).
    allowed = {c["name"]: c.get("allowed") for c in load_schema("telemetry")["columns"]}

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
            "carrier_code": r[3].selectbox("Carrier", allowed["carrier_code"]),
            "priority_tier": r[4].selectbox("Priority", allowed["priority_tier"], format_func=lambda v: v.replace("_", " ").title()),
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
        upload = pd.read_csv(uploaded)
        limit = ui.api().MAX_BATCH
        if len(upload) > limit:
            # Same limit as the batch API; say so instead of cutting rows off silently (Role 3).
            st.warning(f"The file has {len(upload):,} rows; only the first {limit:,} are scored.")
        answers, problems = ui.validate_and_score(upload.head(limit).to_dict(orient="records"))
        if problems:
            st.warning(f"{len(problems)} row(s) break the data contract and were not scored.")
            st.dataframe(pd.DataFrame(problems), hide_index=True)
        if answers:
            out = pd.DataFrame(answers)
            out = pd.DataFrame({"Record": out["record_id"], "Location": out["node"].map(ui.node_name),
                                "Model's assessment": out["label"].map(ui.status_label),
                                "Chance of critical delay": out["probabilities"].map(lambda p: 100 * p["CRITICAL_DELAY"]),
                                "Alert raised": out["alert"],
                                "Main risk factors": out["top_drivers"].map(lambda d: " · ".join(map(ui.feature_label, d)))})
            st.markdown(f"Scored **{len(out):,}** shipments; **{int(out['Alert raised'].sum()):,}** would be raised to a planner.")
            st.dataframe(out.sort_values("Chance of critical delay", ascending=False), hide_index=True, use_container_width=True,
                         column_config={"Chance of critical delay": st.column_config.NumberColumn(format="%.0f%%")})

ui.sources("Stage01_ML/reports/eval_champion.json", "Stage01_ML/data/prepared_dataset.csv.gz", "Stage01_ML/05_integration_engineer.py")
