"""CASCADE control tower - Forecasts page (Stage 2, deep learning): expected dwell time and throughput for 7 days.
Owner: Role 3 (Data & Integration Engineer). Simplified by Role 4 at the team lead's request; Role 3 to review.

What the planner sees: pick a forecast time; a short list of the nodes expected to go above their dwell alert line in
the next 24 hours; then one node's last week and its forecast for the next 7 days, with a one-sentence summary.
Only the champion model is used. This page reads files only (no PyTorch), so the deployed app stays light.

READS : Stage02_DL/predictions/dl_forecasts.csv.gz (Role 1), Stage02_DL/data/hourly_node_telemetry.csv.gz (Role 3),
        Stage02_DL/reports/eval_champion.json and eval_alert_thresholds.csv (Role 4), contracts/network_nodes.csv
WRITES: nothing.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evalkit.forecast import exceedance_probability  # noqa: E402
from common.integration import ui  # noqa: E402
from common.paths import stage_path  # noqa: E402

MEASURES = {"dwell": ("Dwell time (hours cargo waits)", "avg_dwell_time_hours", "h"),
            "throughput": ("Throughput (containers per hour, TEU)", "throughput_teu_per_hr", "TEU/h")}
AHEAD = {1: "in 1 hour", 6: "in 6 hours", 24: "in 1 day", 72: "in 3 days", 168: "in 7 days"}
ATTENTION = 0.5   # list a node when the chance of dwell above its alert line is 50% or more
# The Stage 2 models in words (the champion is read from Role 4's eval_champion.json).
MODEL_WORDS = {"transformer": "Transformer neural network", "lstm": "LSTM neural network",
               "graph_full": "graph neural network", "ensemble": "average of a Transformer neural network and XGBoost",
               "xgb_lag": "XGBoost"}


@st.cache_data
def load() -> tuple:
    """Champion forecasts for the test period with the chance of crossing the alert line, the alert lines, the champion."""
    path = stage_path(2, "predictions", "dl_forecasts.csv.gz")
    champion = ui.read_json(stage_path(2, "reports", "eval_champion.json")).get("model")
    thr_path = stage_path(2, "reports", "eval_alert_thresholds.csv")
    if not path.exists() or champion is None or not thr_path.exists():
        return pd.DataFrame(), pd.Series(dtype=float), champion
    fc = pd.read_csv(path)
    fc = fc[(fc["split"] == "test") & (fc["model_name"] == champion)].copy()
    fc["origin_ts"] = pd.to_datetime(fc["origin_ts"], utc=True)
    fc["target_ts"] = fc["origin_ts"] + pd.to_timedelta(fc["horizon_h"], unit="h")
    thr = pd.read_csv(thr_path, index_col=0).iloc[:, 0]
    # Chance that dwell ends above the node's alert line, read from the forecast's 80% band.
    dwell = fc["target"] == "dwell"
    fc.loc[dwell, "p_above"] = exceedance_probability(fc.loc[dwell, "y_lo"], fc.loc[dwell, "y_pred"], fc.loc[dwell, "y_hi"],
                                                      fc.loc[dwell, "node_id"].map(thr))
    return fc, thr, champion


@st.cache_data
def history() -> pd.DataFrame:
    """Measured hourly dwell and throughput per node."""
    g = pd.read_csv(stage_path(2, "data", "hourly_node_telemetry.csv.gz"),
                    usecols=["hour_ts", "node_id", "avg_dwell_time_hours", "throughput_teu_per_hr"])
    g["hour_ts"] = pd.to_datetime(g["hour_ts"], utc=True)
    return g


def node_label(node_id: str) -> str:
    """'PORT_KR_PUS' -> 'Busan (port)'."""
    kind = ui.nodes().set_index("node_id")["node_type"].map(ui.NODE_TYPE_LABEL).get(node_id, "")
    return f"{ui.node_name(node_id)} ({kind.lower()})" if kind else ui.node_name(node_id)


fc, thr, champion = load()
st.title("Forecasts")
# What history the champion reads (the ensemble's XGBoost part also reads up to 4 weeks of daily averages).
READS = {"ensemble": "the last 72 hours (neural network) and up to 4 weeks (XGBoost) of each node"}
st.markdown(f"**Deep learning (Stage 2)** · model: {MODEL_WORDS.get(champion, champion)}, reading "
            f"{READS.get(champion, 'the last 72 hours of each node')}")
st.markdown("The model forecasts the **dwell time** (how long cargo waits at a node) and the **throughput** (containers "
            "handled per hour) from 1 hour to 7 days ahead. A node **needs attention** when its dwell time is likely to "
            "go above its **alert line**: the level it stays below 95% of the time in normal operation.")
if fc.empty:
    st.error("No forecasts yet: run Stage02_DL/03_dl_engineer.py and 04_evaluation_engineer.py.")
    st.stop()
st.info("**How to use this page.** 1. Choose the moment the forecast is made. "
        "2. Read the list of nodes that need attention in the next 24 hours. "
        "3. Pick a node to see its last 7 days and its forecast for the next 7 days.")

# Step 1: the forecast time. It opens on the time with the most nodes needing attention, so the list is not empty.
day_ahead = fc[(fc["target"] == "dwell") & (fc["horizon_h"] == 24)]
origins = sorted(fc["origin_ts"].unique())
busiest = day_ahead.groupby("origin_ts")["p_above"].apply(lambda p: (p >= ATTENTION).sum()).idxmax()
origin = st.select_slider(
    "1. Forecast made at (UTC)", options=origins, value=busiest, format_func=lambda t: pd.Timestamp(t).strftime("%d %b %Y %H:%M"),
    help="The model only uses data measured up to this moment. Slide to replay the test period (Nov-Dec 2025), "
         "which the model never saw during training.")

# Step 2: nodes likely to be above their dwell alert line in 24 hours.
now = day_ahead[day_ahead["origin_ts"] == origin].set_index("node_id")
attention = now[now["p_above"] >= ATTENTION].sort_values("p_above", ascending=False)
st.subheader("2. Needs attention in the next 24 hours")
if attention.empty:
    st.success("No node is expected to go above its dwell alert line in the next 24 hours.")
else:
    st.caption(f"Nodes with a {ATTENTION:.0%} or higher chance that dwell time will be above their alert line one day "
               "after the forecast time.")
    st.dataframe(pd.DataFrame({
        "Node": attention.index.map(node_label),
        "Expected dwell in 1 day": attention["y_pred"].round(1).astype(str) + " h",
        "Alert line": attention.index.map(thr).round(1).astype(str) + " h",
        "Chance above the line": 100 * attention["p_above"],
    }), hide_index=True, width="stretch",
        column_config={
            "Expected dwell in 1 day": st.column_config.TextColumn(help="The model's most likely dwell time 24 hours ahead"),
            "Alert line": st.column_config.TextColumn(help="The node's normal upper limit: 95% of training hours were below it"),
            "Chance above the line": st.column_config.ProgressColumn(
                format="%.0f%%", min_value=0, max_value=100, help="Chance that dwell time is above the alert line in 24 hours"),
        })

# Step 3: one node's last 7 days and its forecast for the next 7 days.
st.subheader("3. Forecast for one node")
node_ids = list(attention.index) + [n for n in ui.nodes()["node_id"] if n not in attention.index]
c1, c2 = st.columns([3, 2])
node = c1.selectbox("Node (port, canal or hub)", node_ids, format_func=node_label,
                    help="Nodes that need attention are listed first.")
measure = c2.radio("What to forecast", list(MEASURES), format_func=lambda m: MEASURES[m][0], horizontal=True,
                   help="Dwell time: hours cargo waits at the node. Throughput: containers (TEU) handled per hour.")
label, column, unit = MEASURES[measure]
h = history()
past = h[(h["node_id"] == node) & (h["hour_ts"] > origin - pd.Timedelta(days=7)) & (h["hour_ts"] <= origin)]
path = fc[(fc["origin_ts"] == origin) & (fc["node_id"] == node) & (fc["target"] == measure)].sort_values("horizon_h")

# A one-sentence summary of the 1-day forecast, so the chart does not have to be decoded.
day = path[path["horizon_h"] == 24]
if len(day):
    d = day.iloc[0]
    sentence = (f"At **{ui.node_name(node)}**, {label.split(' (')[0].lower()} is expected to be **{d['y_pred']:.1f} {unit}** "
                f"in 1 day (likely between {d['y_lo']:.1f} and {d['y_hi']:.1f} {unit}).")
    if measure == "dwell" and node in thr.index:
        line = float(thr[node])
        sentence += (f" Its alert line is {line:.1f} h, so the chance of being above it is **{d['p_above']:.0%}**.")
    st.markdown(sentence)

start = [origin] if len(past) else []
first = [float(past[column].iloc[-1])] if len(past) else []
fig = go.Figure()
fig.add_trace(go.Scatter(x=past["hour_ts"], y=past[column], name="Measured (last 7 days)", line={"color": "#555", "width": 1}))
fig.add_trace(go.Scatter(x=start + list(path["target_ts"]), y=first + list(path["y_hi"]), mode="lines", line={"width": 0},
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=start + list(path["target_ts"]), y=first + list(path["y_lo"]), mode="lines", fill="tonexty",
                         line={"width": 0}, fillcolor="rgba(31,119,180,0.18)", name="Likely range (8 in 10 outcomes fall inside)"))
fig.add_trace(go.Scatter(x=start + list(path["target_ts"]), y=first + list(path["y_pred"]), name="Forecast (most likely value)",
                         mode="lines+markers", line={"color": "#1f77b4", "width": 2}))
if measure == "dwell" and node in thr.index:
    fig.add_hline(y=float(thr[node]), line_dash="dash", line_color="#d03b3b", annotation_text="alert line")
fig.add_vline(x=origin, line_color="#999", line_width=1)
fig.update_layout(height=380, margin={"l": 10, "r": 10, "t": 10, "b": 10}, yaxis_title=label, xaxis_title="Date (UTC)",
                  legend={"orientation": "h", "y": -0.25})
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
st.caption("Left of the grey line: what was measured. Right of it: the forecast. The red dashed line is the alert line.")

# Step 4: the same forecast as a small table.
st.dataframe(pd.DataFrame({
    "When": path["horizon_h"].map(AHEAD),
    f"Most likely ({unit})": path["y_pred"].round(1),
    f"Likely range ({unit})": [f"{lo:.1f} to {hi:.1f}" for lo, hi in zip(path["y_lo"], path["y_hi"])],
}), hide_index=True, width="stretch")
