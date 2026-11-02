"""CASCADE control tower - Port forecasts page (Stage 2, deep learning): which nodes will get congested in 7 days?
Owner: Role 3 (Data & Integration Engineer).

What the planner sees, top to bottom (most urgent first, nothing to search for):
  1. one line: the model and when a node gets an early warning
  2. the forecast time (a slider over the test period, opening on the time with the most warnings)
  3. the early warnings for the next 7 days, raised with Role 4's validated alert rule (eval_champion.json: the highest
     chance of dwell above the alert line over all horizons >= p*); clicking a row opens that node below
  4. one node: a one-sentence 1-day summary and a chart of its last 7 days, the forecast, the 80% range and the alert line
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

MEASURES = {"dwell": ("Dwell time", "avg_dwell_time_hours", "h"),
            "throughput": ("Throughput", "throughput_teu_per_hr", "TEU/h")}
AHEAD = {1: "in 1 hour", 6: "in 6 hours", 24: "in 1 day", 72: "in 3 days", 168: "in 7 days"}
# The Stage 2 models in words (the champion is read from Role 4's eval_champion.json).
MODEL_WORDS = {"transformer": "Transformer", "lstm": "LSTM", "graph_full": "Graph neural network",
               "ensemble": "Transformer + XGBoost ensemble", "xgb_lag": "XGBoost"}


@st.cache_data
def load() -> tuple:
    """Champion forecasts for the test period with the chance of crossing the alert line, the alert lines, the champion,
    and p* (the validated probability threshold of Role 4's alert rule)."""
    path = stage_path(2, "predictions", "dl_forecasts.csv.gz")
    champ = ui.read_json(stage_path(2, "reports", "eval_champion.json"))
    champion = champ.get("model")
    # p* is chosen by Role 4 on validation; read it from the file so the page always follows the evaluated rule.
    p_star = champ.get("alert_rule", {}).get("p_star")
    thr_path = stage_path(2, "reports", "eval_alert_thresholds.csv")
    if not path.exists() or champion is None or not thr_path.exists():
        return pd.DataFrame(), pd.Series(dtype=float), champion, p_star
    fc = pd.read_csv(path)
    fc = fc[(fc["split"] == "test") & (fc["model_name"] == champion)].copy()
    fc["origin_ts"] = pd.to_datetime(fc["origin_ts"], utc=True)
    fc["target_ts"] = fc["origin_ts"] + pd.to_timedelta(fc["horizon_h"], unit="h")
    thr = pd.read_csv(thr_path, index_col=0).iloc[:, 0]
    # Chance that dwell ends above the node's alert line, read from the forecast's 80% band.
    dwell = fc["target"] == "dwell"
    fc.loc[dwell, "p_above"] = exceedance_probability(fc.loc[dwell, "y_lo"], fc.loc[dwell, "y_pred"], fc.loc[dwell, "y_hi"],
                                                      fc.loc[dwell, "node_id"].map(thr))
    return fc, thr, champion, p_star


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


fc, thr, champion, p_star = load()
st.title("Port forecasts")
if fc.empty or p_star is None:
    st.error("No forecasts or no validated alert rule yet: run Stage02_DL/03_dl_engineer.py and 04_evaluation_engineer.py.")
    st.stop()
st.caption(f"{MODEL_WORDS.get(champion, champion)} (deep learning) forecasts dwell time and throughput at 27 ports, canals "
           f"and hubs, 1 hour to 7 days ahead. A node gets an **early warning** when its chance of dwell time above its "
           f"alert line (its normal upper level) reaches {p_star:.0%} at any point in the next 7 days; {p_star:.0%} was "
           "chosen on the validation period. Shown: the test period (Nov-Dec 2025), which the model never saw.")

# Step 1: the forecast time. It opens on the time with the most early warnings, so the list is not empty.
dwell_fc = fc[fc["target"] == "dwell"]
# Role 4's rule B: per node and forecast time, the HIGHEST chance of dwell above the alert line over all horizons.
peak = dwell_fc.loc[dwell_fc.groupby(["origin_ts", "node_id"])["p_above"].idxmax()]
peak = peak[["origin_ts", "node_id", "horizon_h", "y_pred", "p_above"]]
origins = sorted(fc["origin_ts"].unique())
busiest = peak.groupby("origin_ts")["p_above"].apply(lambda p: (p >= p_star).sum()).idxmax()
origin = st.select_slider("Forecast made at (UTC)", options=origins, value=busiest,
                          format_func=lambda t: pd.Timestamp(t).strftime("%d %b %Y %H:%M"))

# Step 2: early warnings = nodes whose highest chance over the next 7 days reaches p*. Click a row to open it below.
now = peak[peak["origin_ts"] == origin].set_index("node_id")
warnings = now[now["p_above"] >= p_star].sort_values("p_above", ascending=False)
st.subheader(f"Early warnings (next 7 days): {len(warnings)}")
event = None
if warnings.empty:
    st.success("No node is likely to go above its dwell alert line in the next 7 days.")
else:
    event = st.dataframe(pd.DataFrame({
        "Node": warnings.index.map(node_label),
        "Most likely when": warnings["horizon_h"].map(AHEAD),
        "Expected dwell then": warnings["y_pred"].round(1).astype(str) + " h",
        "Alert line": warnings.index.map(thr).round(1).astype(str) + " h",
        "Chance above the line": 100 * warnings["p_above"],
    }), hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="warnings",
        column_config={"Chance above the line": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})

# Step 3: one node. A clicked warning opens that node; otherwise the most urgent warning (or the first node) is shown.
st.subheader("Node forecast")
node_ids = list(warnings.index) + [n for n in ui.nodes()["node_id"] if n not in warnings.index]
rows = event.selection.rows if event is not None else []
c1, c2 = st.columns([3, 2])
node = c1.selectbox("Node", node_ids, index=rows[0] if rows else 0, format_func=node_label)
measure = c2.segmented_control("Measure", list(MEASURES), default="dwell", format_func=lambda m: MEASURES[m][0]) or "dwell"
label, column, unit = MEASURES[measure]
h = history()
past = h[(h["node_id"] == node) & (h["hour_ts"] > origin - pd.Timedelta(days=7)) & (h["hour_ts"] <= origin)]
path = fc[(fc["origin_ts"] == origin) & (fc["node_id"] == node) & (fc["target"] == measure)].sort_values("horizon_h")

# A one-sentence summary of the 1-day forecast, so the chart does not have to be decoded.
day = path[path["horizon_h"] == 24]
if len(day):
    d = day.iloc[0]
    sentence = (f"In 1 day: **{d['y_pred']:.1f} {unit}** (likely {d['y_lo']:.1f} to {d['y_hi']:.1f} {unit})")
    if measure == "dwell" and node in thr.index:
        sentence += f"; alert line {float(thr[node]):.1f} h, chance above it **{d['p_above']:.0%}**"
    st.markdown(sentence + ".")

start = [origin] if len(past) else []
first = [float(past[column].iloc[-1])] if len(past) else []
fig = go.Figure()
fig.add_trace(go.Scatter(x=past["hour_ts"], y=past[column], name="Last 7 days", line={"color": "#555", "width": 1}))
fig.add_trace(go.Scatter(x=start + list(path["target_ts"]), y=first + list(path["y_hi"]), mode="lines", line={"width": 0},
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=start + list(path["target_ts"]), y=first + list(path["y_lo"]), mode="lines", fill="tonexty",
                         line={"width": 0}, fillcolor="rgba(31,119,180,0.18)", name="Likely range (80%)"))
fig.add_trace(go.Scatter(x=start + list(path["target_ts"]), y=first + list(path["y_pred"]), name="Forecast",
                         mode="lines+markers", line={"color": "#1f77b4", "width": 2}))
if measure == "dwell" and node in thr.index:
    fig.add_hline(y=float(thr[node]), line_dash="dash", line_color="#d03b3b", annotation_text="alert line")
fig.add_vline(x=origin, line_color="#999", line_width=1)
fig.update_layout(height=340, margin={"l": 10, "r": 10, "t": 10, "b": 10}, yaxis_title=f"{label} ({unit})",
                  legend={"orientation": "h", "y": -0.2})
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
