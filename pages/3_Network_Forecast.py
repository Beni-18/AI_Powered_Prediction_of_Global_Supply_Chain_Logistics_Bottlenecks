"""CASCADE control tower - Stage 2 page: 7-day node forecasts on the network map.
Owner: Role 3 (Data & Integration Engineer). Built by Role 4 at the team lead's request; Role 3 to review.

What the operator sees: pick a forecast time in the held-out test period and a model; the map colours every node by
the chance its dwell time will be above its alert line at the chosen horizon; click through to one node's fan chart
(history, forecast median and 80% band, and what really happened afterwards).
This page reads files only (no PyTorch), so the deployed app stays light. Live forecasts come from the Stage 2 API.

READS : Stage02_DL/predictions/dl_forecasts.csv.gz (Role 1), Stage02_DL/data/hourly_node_telemetry.csv.gz (Role 3),
        Stage02_DL/reports/eval_champion.json, eval_alert_thresholds.csv (Role 4), contracts/network_*.csv
WRITES: nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.evalkit.forecast import exceedance_probability  # noqa: E402
from common.integration import ui  # noqa: E402
from common.paths import stage_path  # noqa: E402

MODEL_LABEL = {"persistence": "Persistence (last value)", "seasonal_naive": "Same hour last week", "rolling_mean": "Last-24 h average",
               "xgb_lag": "XGBoost on lags (baseline ML)", "lstm": "LSTM", "transformer": "Transformer",
               "graph_full": "Graph model (real lanes)", "graph_shuffled": "Graph model (shuffled lanes)", "graph_none": "Graph model (no lanes)"}
TARGET_LABEL = {"dwell": "Dwell time (h)", "throughput": "Throughput (TEU/h)"}


@st.cache_data
def forecasts() -> pd.DataFrame:
    """Test-period forecasts of every model (val rows are not shown to operators)."""
    path = stage_path(2, "predictions", "dl_forecasts.csv.gz")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["split"] == "test"]
    df["origin_ts"] = pd.to_datetime(df["origin_ts"], utc=True)
    df["target_ts"] = df["origin_ts"] + pd.to_timedelta(df["horizon_h"], unit="h")
    return df


@st.cache_data
def hourly() -> pd.DataFrame:
    """Measured hourly node telemetry (for history and for 'what really happened')."""
    g = pd.read_csv(stage_path(2, "data", "hourly_node_telemetry.csv.gz"),
                    usecols=["hour_ts", "node_id", "avg_dwell_time_hours", "throughput_teu_per_hr", "is_imputed"])
    g["hour_ts"] = pd.to_datetime(g["hour_ts"], utc=True)
    return g


@st.cache_data
def thresholds() -> pd.Series:
    """Per-node dwell alert line (95th percentile of training dwell, set by Role 4)."""
    path = stage_path(2, "reports", "eval_alert_thresholds.csv")
    return pd.read_csv(path, index_col=0).iloc[:, 0] if path.exists() else pd.Series(dtype=float)


fc = forecasts()
if fc.empty:
    st.error("No Stage 2 forecasts yet: run python Stage02_DL/03_dl_engineer.py, then 04_evaluation_engineer.py.")
    st.stop()
thr = thresholds()
champion = ui.read_json(stage_path(2, "reports", "eval_champion.json")).get("model")

st.title("Network forecast")
st.caption("Forecasts of dwell time and throughput 1 hour to 7 days ahead for all 27 nodes, replayed over the held-out test "
           "period (Nov-Dec 2025). Simulated data calibrated on real drivers (SYNTHETIC_CALIBRATED). Decision support only: "
           "a planner reviews every alert.")

# Step 1: the operator picks a model, a forecast time and the horizon used to colour the map.
models = [m for m in MODEL_LABEL if m in set(fc["model_name"])]
default = models.index(champion) if champion in models else 0
c1, c2, c3 = st.columns([2, 3, 1])
model = c1.selectbox("Model", models, index=default, format_func=lambda m: MODEL_LABEL[m] + (" · champion" if m == champion else ""))
origins = sorted(fc["origin_ts"].unique())
origin = c2.select_slider("Forecast made at (UTC)", options=origins, value=origins[len(origins) // 2],
                          format_func=lambda t: pd.Timestamp(t).strftime("%d %b %H:%M"))
horizon = c3.selectbox("Horizon", [24, 72, 168, 6, 1], format_func=lambda h: f"{h} h")

# Step 2: chance that dwell crosses each node's alert line at that horizon (from the forecast's 80% band).
now = fc[(fc["model_name"] == model) & (fc["origin_ts"] == origin) & (fc["target"] == "dwell")]
at_h = now[now["horizon_h"] == horizon].set_index("node_id")
nodes = ui.nodes().set_index("node_id")
risk = pd.DataFrame({"forecast_dwell_h": at_h["y_pred"], "band_low": at_h["y_lo"], "band_high": at_h["y_hi"]})
risk["alert_line_h"] = thr.reindex(risk.index)
risk["p_above_line"] = exceedance_probability(risk["band_low"], risk["forecast_dwell_h"], risk["band_high"], risk["alert_line_h"])
risk = risk.join(nodes[["display_name", "node_type", "lat", "lon"]])
if risk["alert_line_h"].isna().all():
    st.warning("Alert lines are not available yet (run python Stage02_DL/04_evaluation_engineer.py); the map shows forecasts only.")
risk["p_above_line"] = risk["p_above_line"].fillna(0.0)

# Step 3: the map - lanes as thin lines, nodes coloured by the chance of crossing the alert line.
fig = go.Figure()
for e in ui.edges().itertuples():
    if e.src in nodes.index and e.dst in nodes.index:
        fig.add_trace(go.Scattergeo(lon=[nodes.at[e.src, "lon"], nodes.at[e.dst, "lon"]], lat=[nodes.at[e.src, "lat"], nodes.at[e.dst, "lat"]],
                                    mode="lines", line={"width": 0.6, "color": "#9aa5b1"}, hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scattergeo(
    lon=risk["lon"], lat=risk["lat"], mode="markers", showlegend=False,
    marker={"size": 9 + 14 * risk["p_above_line"], "color": risk["p_above_line"], "cmin": 0, "cmax": 1,
            "colorscale": [[0, "#0ca30c"], [0.5, "#fab219"], [1, "#d03b3b"]], "colorbar": {"title": "P(above line)"},
            "line": {"width": 0.5, "color": "#333"}},
    text=[f"{r.display_name}<br>forecast {r.forecast_dwell_h:.1f} h (80%: {r.band_low:.1f}-{r.band_high:.1f})<br>"
          f"alert line {r.alert_line_h:.1f} h · P(above) {r.p_above_line:.0%}" for r in risk.itertuples()],
    hoverinfo="text"))
fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="#d0d0d0", landcolor="#f4f4f4")
fig.update_layout(height=430, margin={"l": 0, "r": 0, "t": 10, "b": 0})
st.plotly_chart(fig, width="stretch")

# Step 4: the ten nodes most likely to be above their alert line.
st.subheader(f"Top 10 nodes by chance of dwell above the alert line, {horizon} h ahead")
top = risk.sort_values("p_above_line", ascending=False).head(10)
st.dataframe(pd.DataFrame({"Node": top["display_name"], "Type": top["node_type"].map(ui.NODE_TYPE_LABEL),
                           "P(above line)": top["p_above_line"], "Forecast dwell (h)": top["forecast_dwell_h"].round(1),
                           "80% band (h)": [f"{a:.1f} - {b:.1f}" for a, b in zip(top["band_low"], top["band_high"])],
                           "Alert line (h)": top["alert_line_h"].round(1)}),
             hide_index=True, width="stretch",
             column_config={"P(above line)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=1)})

# Step 5: one node's fan chart - history, forecast with band, and what really happened.
st.subheader("Node detail")
d1, d2 = st.columns([3, 1])
node = d1.selectbox("Node", list(top.index) + [n for n in nodes.index if n not in top.index],
                    format_func=lambda n: f"{nodes.at[n, 'display_name']} ({n})")
target = d2.radio("Measure", ["dwell", "throughput"], format_func=TARGET_LABEL.get, horizontal=True)
col = "avg_dwell_time_hours" if target == "dwell" else "throughput_teu_per_hr"
h = hourly()
hist = h[(h["node_id"] == node) & h["hour_ts"].between(origin - pd.Timedelta(days=7), origin + pd.Timedelta(days=7))]
path = fc[(fc["model_name"] == model) & (fc["origin_ts"] == origin) & (fc["node_id"] == node) & (fc["target"] == target)].sort_values("horizon_h")
fan = go.Figure()
past, after = hist[hist["hour_ts"] <= origin], hist[hist["hour_ts"] > origin]
fan.add_trace(go.Scatter(x=past["hour_ts"], y=past[col], name="measured (before forecast)", line={"color": "#333", "width": 1}))
fan.add_trace(go.Scatter(x=after["hour_ts"], y=after[col], name="what really happened", line={"color": "#999", "width": 1, "dash": "dot"}))
anchor = past.iloc[-1] if len(past) else None
xs = ([origin] if anchor is not None else []) + list(path["target_ts"])
first = [anchor[col]] if anchor is not None else []
fan.add_trace(go.Scatter(x=xs, y=first + list(path["y_hi"]), line={"width": 0}, showlegend=False, hoverinfo="skip"))
fan.add_trace(go.Scatter(x=xs, y=first + list(path["y_lo"]), fill="tonexty", fillcolor="rgba(44,160,44,0.18)", line={"width": 0},
                         name="80% band"))
fan.add_trace(go.Scatter(x=xs, y=first + list(path["y_pred"]), name=f"{MODEL_LABEL[model]} forecast", mode="lines+markers",
                         line={"color": "#2ca02c", "width": 2}))
if target == "dwell" and node in thr.index:
    fan.add_hline(y=float(thr[node]), line_dash="dash", line_color="#d03b3b", annotation_text="alert line")
fan.add_vline(x=origin, line_color="#1f77b4", line_width=1)
fan.update_layout(height=380, margin={"l": 10, "r": 10, "t": 10, "b": 10}, yaxis_title=TARGET_LABEL[target],
                  legend={"orientation": "h", "y": -0.2})
st.plotly_chart(fan, width="stretch")
st.caption("Forecast points are at 1, 6, 24, 72 and 168 hours after the forecast time. The dotted line was NOT available to the "
           "model; it is shown so you can judge the forecast.")
ui.sources("Stage02_DL/predictions/dl_forecasts.csv.gz", "Stage02_DL/data/hourly_node_telemetry.csv.gz",
           "Stage02_DL/reports/eval_alert_thresholds.csv")
