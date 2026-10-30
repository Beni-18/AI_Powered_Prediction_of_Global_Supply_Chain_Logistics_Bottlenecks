"""CASCADE control tower - Overview page (network map, headline numbers, weak spots).
Owner: Role 3 (Data & Integration Engineer). Built by Role 4 at the team lead's request; Role 3 to review.
Opened through app.py (streamlit run app.py).

READS : Stage01_ML/reports/eval_champion.json (model + alert threshold), the test-period shipments scored with
        the API's own code (common/integration/ui.py), contracts/network_nodes.csv and network_edges.csv.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.integration import ui  # noqa: E402
from common.paths import stage_path  # noqa: E402

# Sequential blue ramp (palette steps 150 -> 650): darker = a larger share of shipments raise an alert.
RISK_SCALE = [[0.0, "#b7d3f6"], [0.5, "#3987e5"], [1.0, "#104281"]]
NODE_SYMBOL = {"Port": "circle", "Chokepoint": "diamond", "Rail hub": "square", "Distribution hub": "square"}
CATCH_LIMIT = 0.5   # a node where the alert catches fewer than half of the real critical delays needs a human check
MIN_CRITICAL = 10   # ...but only judge nodes with at least 10 real critical delays in the test period

# Step 1: the decision rule in force and the scored test period.
champion = ui.read_json(stage_path(1, "reports", "eval_champion.json"))
scored = ui.scored_period("test")
summary = ui.node_summary(scored)
start, end = pd.to_datetime(scored["timestamp"]).agg(["min", "max"])

st.title("Network overview")
st.caption(f"Stage 1 · shipment delay risk · test period {start:%d %b} – {end:%d %b %Y} (never seen in training) · "
           f"model **{champion.get('model_name', 'STUB')}** · simulated data v1 · every alert needs a planner's review")

# Step 2: headline numbers for the period.
crit = scored["delay_status"] == "CRITICAL_DELAY"
alerts = int(scored["alert"].sum())
caught = int((crit & scored["alert"]).sum())
false_alarms = alerts - caught
tiles = st.columns(5)
tiles[0].metric("Shipments", f"{len(scored):,}", help="Shipment snapshots scored in the test period")
tiles[1].metric("Alerts", f"{alerts:,}", help="Alert = P(critical) at or above the alert threshold")
tiles[2].metric("Critical caught", f"{caught / crit.sum():.0%}", help=f"{caught:,} of {int(crit.sum()):,} shipments that really ended 24 h+ late")
tiles[3].metric("False alarms /100", f"{100 * false_alarms / alerts:.0f}", help="Per 100 alerts. Team budget: 40 (docs/decisions/0004)")
tiles[4].metric("Alert threshold", f"{champion.get('threshold', 0.5):.2f}",
                help=f"P(critical) for model {champion.get('model_name', 'STUB')}, chosen on validation by lowest expected cost")

# Step 3: the network map across the full width, then the node table and the weak spots.
with st.container():
    st.subheader("Where alerts are raised")
    lookup = ui.nodes().set_index("node_id")
    lane_lat, lane_lon = [], []
    for src, dst in ui.edges()[["src", "dst"]].itertuples(index=False):
        if src in lookup.index and dst in lookup.index:
            lane_lat += [lookup.at[src, "lat"], lookup.at[dst, "lat"], None]
            lane_lon += [lookup.at[src, "lon"], lookup.at[dst, "lon"], None]
    fig = go.Figure(go.Scattergeo(lat=lane_lat, lon=lane_lon, mode="lines", hoverinfo="skip", showlegend=False,
                                  line={"width": 1, "color": "rgba(128,128,128,0.35)"}))
    # Nodes with no shipment snapshots in the period (lane ends, Cape route) are drawn grey so the network is complete.
    quiet = ui.nodes()[~ui.nodes()["node_id"].isin(summary["node_id"])]
    fig.add_trace(go.Scattergeo(lat=quiet["lat"], lon=quiet["lon"], mode="markers", name="No shipments in period",
                                marker={"size": 7, "color": "rgba(128,128,128,0.55)", "line": {"width": 1, "color": "white"}},
                                customdata=quiet[["display_name"]].values,
                                hovertemplate="<b>%{customdata[0]}</b><br>No shipment snapshots in this period<extra></extra>"))
    cmax = float(summary["alert_rate"].max())
    for i, (node_type, part) in enumerate(summary.groupby("type")):
        fig.add_trace(go.Scattergeo(
            lat=part["lat"], lon=part["lon"], mode="markers", name=node_type,
            marker={"symbol": NODE_SYMBOL.get(node_type, "circle"), "size": 9 + 14 * (part["shipments"] / summary["shipments"].max()),
                    "color": part["alert_rate"], "colorscale": RISK_SCALE, "cmin": 0, "cmax": cmax,
                    "line": {"width": 1.5, "color": "white"}, "showscale": i == 0,
                    "colorbar": {"title": "Alert rate", "tickformat": ".0%", "thickness": 10, "len": 0.6}},
            customdata=part[["display_name", "type", "shipments", "alert_rate", "critical_share", "catch_rate"]].values,
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]:,} shipments"
                          "<br>Alert rate %{customdata[3]:.0%}<br>Really critical %{customdata[4]:.0%}"
                          "<br>Critical caught %{customdata[5]:.0%}<extra></extra>"))
    # Fit the view to the 27 nodes, so the map fills its box instead of showing empty oceans and poles.
    fig.update_geos(projection_type="natural earth", fitbounds="locations", showcountries=False, showland=True,
                    landcolor="rgba(128,128,128,0.12)", coastlinecolor="rgba(128,128,128,0.35)", showocean=False,
                    bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=420, margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend={"orientation": "h", "y": -0.02, "title": None})
    # Page scrolling must scroll the page, not zoom the map.
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
    st.caption("Marker size = shipments in the period · shape = node type · colour = share of shipments that raised an alert")

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Nodes by alerts raised")
    table = summary.sort_values("alerts", ascending=False)
    st.dataframe(
        table[["display_name", "shipments", "alerts", "critical_share", "catch_rate"]].assign(
            critical_share=lambda d: 100 * d["critical_share"], catch_rate=lambda d: 100 * d["catch_rate"]),
        hide_index=True, use_container_width=True, height=400,
        column_config={
            "display_name": st.column_config.TextColumn("Node"),
            "shipments": st.column_config.NumberColumn("Shipments", format="%d"),
            "alerts": st.column_config.NumberColumn("Alerts", format="%d"),
            "critical_share": st.column_config.ProgressColumn("Really critical", min_value=0, max_value=100, format="%.0f%%"),
            "catch_rate": st.column_config.ProgressColumn("Caught by alert", min_value=0, max_value=100, format="%.0f%%"),
        })

# Step 4: where the model is weak, straight from the numbers (the field guide tells planners the same).
with right:
    st.subheader("Check by hand")
    weak = summary[(summary["critical"] >= MIN_CRITICAL) & (summary["catch_rate"] < CATCH_LIMIT)].sort_values("catch_rate")
    st.caption(f"Nodes where the alert caught fewer than half of the real critical delays (at least {MIN_CRITICAL} in the period)")
    if len(weak):
        st.dataframe(pd.DataFrame({"Node": weak["display_name"], "Type": weak["type"],
                                   "Caught": weak.apply(lambda r: f"{int(r.caught)} of {int(r.critical)}", axis=1)}),
                     hide_index=True, use_container_width=True)
        chokepoints = int((weak["node_type"] == "CHOKEPOINT").sum())
        st.warning(f"{chokepoints} of these {len(weak)} are maritime chokepoints, where the model is weakest. "
                   "Planners should review shipments there manually.")
    else:
        st.success("No node below the limit.")

ui.sources("Stage01_ML/reports/eval_champion.json", "Stage01_ML/predictions (model)", "Stage01_ML/data/prepared_dataset.csv.gz",
           "contracts/network_nodes.csv")
