"""CASCADE control tower - Port forecasts page (Stage 2, deep learning): will a port, canal or hub get congested?
Owner: Role 3 (Data & Integration Engineer). Simplified by Role 4 at the team lead's request; Role 3 to review.

What the planner sees, top to bottom (the same shape as the Shipments page):
  1. "Check a port": pick a port and a date; the answer is one status (Congested now / Congestion warning / Normal),
     one sentence with
     the expected waiting time against the port's normal upper level, a small chart and a 5-row table
  2. "All ports": every port, canal and hub at that date with the same status, warnings first
Only waiting time (dwell) is shown, because it is what the congestion warning is about.
Congested now = the last measured waiting time is already above the port's alert line (its normal upper level).
Congestion warning = not congested now, and Role 4's validated alert rule fires (eval_champion.json): the highest chance
over the next 7 days that the waiting time is above the alert line reaches p*. (The evaluation drops warnings raised while
a port is already congested, because those are not early warnings; this page shows them as "Congested now" instead.) Only the champion model is used. This page
reads files only (no PyTorch), so the deployed app stays light.

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

AHEAD = {1: "In 1 hour", 6: "In 6 hours", 24: "In 1 day", 72: "In 3 days", 168: "In 7 days"}
CONGESTED, WARNING, NORMAL = "Congested now", "Congestion warning", "Normal"
STATUS_COLOUR = {CONGESTED: "#a32020", WARNING: "#e07b00", NORMAL: "#0ca30c"}
# The Stage 2 models in words (the champion is read from Role 4's eval_champion.json).
MODEL_WORDS = {"transformer": "Transformer", "lstm": "LSTM", "graph_full": "Graph neural network",
               "ensemble": "Transformer + XGBoost ensemble", "xgb_lag": "XGBoost"}


@st.cache_data
def load() -> tuple:
    """Champion waiting-time forecasts for the test period with the chance of going above the alert line, plus the
    alert lines, the champion's name and p* (the validated probability threshold of Role 4's alert rule)."""
    path = stage_path(2, "predictions", "dl_forecasts.csv.gz")
    champ = ui.read_json(stage_path(2, "reports", "eval_champion.json"))
    champion = champ.get("model")
    # p* is chosen by Role 4 on validation; read it from the file so the page always follows the evaluated rule.
    p_star = champ.get("alert_rule", {}).get("p_star")
    thr_path = stage_path(2, "reports", "eval_alert_thresholds.csv")
    if not path.exists() or champion is None or not thr_path.exists():
        return pd.DataFrame(), pd.Series(dtype=float), champion, p_star
    fc = pd.read_csv(path)
    fc = fc[(fc["split"] == "test") & (fc["model_name"] == champion) & (fc["target"] == "dwell")].copy()
    fc["origin_ts"] = pd.to_datetime(fc["origin_ts"], utc=True)
    fc["target_ts"] = fc["origin_ts"] + pd.to_timedelta(fc["horizon_h"], unit="h")
    thr = pd.read_csv(thr_path, index_col=0).iloc[:, 0]
    # Chance that the waiting time ends above the port's alert line, read from the forecast's 80% range.
    fc["p_above"] = exceedance_probability(fc["y_lo"], fc["y_pred"], fc["y_hi"], fc["node_id"].map(thr))
    return fc, thr, champion, p_star


@st.cache_data
def history() -> pd.DataFrame:
    """Measured hourly waiting time per node."""
    g = pd.read_csv(stage_path(2, "data", "hourly_node_telemetry.csv.gz"), usecols=["hour_ts", "node_id", "avg_dwell_time_hours"])
    g["hour_ts"] = pd.to_datetime(g["hour_ts"], utc=True)
    return g


def node_label(node_id: str) -> str:
    """'PORT_KR_PUS' -> 'Busan (port)'."""
    kind = ui.nodes().set_index("node_id")["node_type"].map(ui.NODE_TYPE_LABEL).get(node_id, "")
    return f"{ui.node_name(node_id)} ({kind.lower()})" if kind else ui.node_name(node_id)


def status_of(current: float, line: float, chance: float, p_star: float) -> str:
    """Congested now (already above the normal upper level), Congestion warning (likely to go above it), or Normal."""
    if current > line:
        return CONGESTED
    return WARNING if chance >= p_star else NORMAL


def status_badge(word: str) -> str:
    """Coloured status word (HTML); the word is always shown, colour only helps."""
    colour = STATUS_COLOUR[word]
    return (f"<span style='display:inline-block;padding:3px 12px;border-radius:999px;font-weight:600;"
            f"border:1px solid {colour};color:{colour};background:{colour}1a'>{word}</span>")


fc, thr, champion, p_star = load()
st.title("Port forecasts")
if fc.empty or p_star is None:
    st.error("No forecasts or no validated alert rule yet: run Stage02_DL/03_dl_engineer.py and 04_evaluation_engineer.py.")
    st.stop()
st.caption(f"{MODEL_WORDS.get(champion, champion)} (deep learning) forecasts how long cargo will wait at each port, "
           "canal and hub over the next 7 days. **Congested now**: the waiting time is already above the port's normal upper "
           f"level. **Congestion warning**: it is likely to go above it (chance {p_star:.0%} or more, a level chosen on the "
           "validation period). "
           "Shown: the test period (Nov-Dec 2025), which the model never saw. The tool never changes a route by itself.")

# Per port and forecast time: the highest chance over the next 7 days, and when it is highest (Role 4's rule B),
# plus the waiting time measured at that moment and the status it gives.
peak = fc.loc[fc.groupby(["origin_ts", "node_id"])["p_above"].idxmax(), ["origin_ts", "node_id", "horizon_h", "p_above"]]
measured = history().rename(columns={"hour_ts": "origin_ts", "avg_dwell_time_hours": "now"})
peak = peak.merge(measured, on=["origin_ts", "node_id"], how="left").assign(line=lambda d: d["node_id"].map(thr))
peak["status"] = [status_of(n, ln, p, p_star) for n, ln, p in zip(peak["now"], peak["line"], peak["p_above"])]
origins = sorted(fc["origin_ts"].unique())
# Open on the date with the most EARLY warnings (likely to become congested, not congested already).
busiest = peak.assign(w=peak["status"] == WARNING).groupby("origin_ts")["w"].sum().idxmax()
rank = {WARNING: 0, CONGESTED: 1, NORMAL: 2}
opening = peak[peak["origin_ts"] == busiest].assign(r=lambda d: d["status"].map(rank))
warned = opening.sort_values(["r", "p_above"], ascending=[True, False])["node_id"].tolist()

# Step 1: check one port. Two inputs, one answer.
with st.container(border=True):
    st.subheader("Check a port")
    c1, c2 = st.columns([3, 2])
    node_ids = warned + [n for n in ui.nodes()["node_id"] if n not in warned]
    node = c1.selectbox("Port, canal or hub", node_ids, format_func=node_label,
                        help="Ports with a congestion warning on the opening date are listed first.")
    origin = c2.select_slider("Date (UTC)", options=origins, value=busiest,
                              format_func=lambda t: pd.Timestamp(t).strftime("%d %b %Y %H:%M"),
                              help="The forecast uses only data measured up to this moment.")
    path = fc[(fc["origin_ts"] == origin) & (fc["node_id"] == node)].sort_values("horizon_h")
    best = path.loc[path["p_above"].idxmax()]
    line = float(thr[node])
    now = history()
    now = now[(now["node_id"] == node) & (now["hour_ts"] <= origin)]
    current = float(now["avg_dwell_time_hours"].iloc[-1]) if len(now) else float("nan")
    status = status_of(current, line, best["p_above"], p_star)

    st.markdown(status_badge(status), unsafe_allow_html=True)
    day = path[path["horizon_h"] == 24].iloc[0]
    if status == CONGESTED:
        sentence = (f"Cargo at **{ui.node_name(node)}** waits **{current:.1f} hours** now, already above its normal upper "
                    f"level of **{line:.1f} hours**. In 1 day it is expected to wait **{day['y_pred']:.1f} hours** "
                    f"(likely {day['y_lo']:.1f} to {day['y_hi']:.1f}).")
    else:
        sentence = (f"Cargo at **{ui.node_name(node)}** waits **{current:.1f} hours** now; its normal upper level is "
                    f"**{line:.1f} hours**. In 1 day it is expected to wait **{day['y_pred']:.1f} hours** (likely "
                    f"{day['y_lo']:.1f} to {day['y_hi']:.1f}). The highest chance of going above the normal level in the "
                    f"next 7 days is **{best['p_above']:.0%}** ({AHEAD[int(best['horizon_h'])].lower()}).")
    st.markdown(sentence)

    # A small chart: the last 3 days measured, then the forecast, against the normal upper level.
    past = now[now["hour_ts"] > origin - pd.Timedelta(days=3)]
    xs = [origin] + list(path["target_ts"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=past["hour_ts"], y=past["avg_dwell_time_hours"], name="Measured",
                             line={"color": "#555", "width": 1}))
    fig.add_trace(go.Scatter(x=xs, y=[current] + list(path["y_hi"]), mode="lines", line={"width": 0},
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[current] + list(path["y_lo"]), mode="lines", fill="tonexty", line={"width": 0},
                             fillcolor="rgba(31,119,180,0.15)", name="Likely range"))
    fig.add_trace(go.Scatter(x=xs, y=[current] + list(path["y_pred"]), name="Forecast", mode="lines+markers",
                             line={"color": "#1f77b4", "width": 2}))
    fig.add_hline(y=line, line_dash="dash", line_color="#d03b3b", annotation_text="normal upper level")
    fig.update_layout(height=280, margin={"l": 10, "r": 10, "t": 10, "b": 10}, yaxis_title="Waiting time (hours)",
                      legend={"orientation": "h", "y": -0.25})
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.dataframe(pd.DataFrame({
        "When": path["horizon_h"].map(AHEAD),
        "Expected waiting time (h)": path["y_pred"].round(1),
        "Likely range (h)": [f"{lo:.1f} to {hi:.1f}" for lo, hi in zip(path["y_lo"], path["y_hi"])],
        "Above normal upper level?": ["likely" if p >= p_star else "no" for p in path["p_above"]],
    }), hide_index=True, width="stretch")

# Step 2: every port at the same date: warnings first (the early warnings to act on), then congested now, then normal.
st.subheader("All ports")
board = peak[peak["origin_ts"] == origin].set_index("node_id")
board = board.assign(rank=board["status"].map(rank)).sort_values(["rank", "p_above"], ascending=[True, False])
counts = board["status"].value_counts()
st.caption(f"At {pd.Timestamp(origin):%d %b %Y %H:%M} UTC: {counts.get(WARNING, 0)} with a congestion warning, "
           f"{counts.get(CONGESTED, 0)} congested now, {counts.get(NORMAL, 0)} normal.")
st.dataframe(pd.DataFrame({
    "Port, canal or hub": board.index.map(node_label),
    "Status": board["status"],
    "Waiting now (h)": board["now"].round(1),
    "Normal upper level (h)": board["line"].round(1),
    "Chance of congestion (next 7 days)": 100 * board["p_above"],
    "When the chance is highest": board["horizon_h"].map(AHEAD),
}).style.map(lambda s: f"color: {STATUS_COLOUR.get(s, '#333')}; font-weight: 600", subset=["Status"]),
    hide_index=True, width="stretch", height=420,
    column_config={"Chance of congestion (next 7 days)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                   "Waiting now (h)": st.column_config.NumberColumn(format="%.1f"),
                   "Normal upper level (h)": st.column_config.NumberColumn(format="%.1f")})
