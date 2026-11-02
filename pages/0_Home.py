"""CASCADE control tower - Home page: what the tool does, in plain words, with a link to each part.
Owner: Role 3 (Data & Integration Engineer). Written by Role 4 at the team lead's request; Role 3 to review.

READS : the test-period shipment scores (common/integration/ui.py), Stage01_ML and Stage02_DL reports/eval_champion.json
WRITES: nothing.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.integration import ui  # noqa: E402
from common.paths import stage_path  # noqa: E402

scored = ui.test_period_scores()
start, end = pd.to_datetime(scored["timestamp"]).agg(["min", "max"])
counts = scored["label"].value_counts()
# The two champion models, named in words (each chosen on its validation period by Role 4).
ml_model = ui.model_label(ui.read_json(stage_path(1, "reports", "eval_champion.json")).get("model_name", "STUB"))
dl_model = {"transformer": "Transformer neural network", "lstm": "LSTM neural network", "graph_full": "graph neural network",
            "ensemble": "Transformer neural network + XGBoost average"}.get(
    ui.read_json(stage_path(2, "reports", "eval_champion.json")).get("model"), "neural network")

st.title("CASCADE")
st.markdown("Early warning for **late shipments** and **congested ports** in a global shipping network. "
            "The tool gives warnings to a planner. It never changes a booking or a route by itself.")

left, right = st.columns(2, gap="large")

# Part 1: shipment status (Stage 1 model).
with left, st.container(border=True):
    st.subheader("1. Shipments: will this shipment be late?")
    st.markdown(f"**Machine learning (Stage 1)** · {ml_model}")
    st.markdown("For every shipment, the model predicts one of three outcomes:")
    for status in reversed(ui.STATUS_ORDER):
        st.markdown(f"{ui.status_chip(status)} &nbsp; {ui.STATUS_MEANING[status]} · "
                    f"**{int(counts.get(status, 0)):,}** shipments", unsafe_allow_html=True)
    st.page_link("pages/1_Shipments.py", label="Open Shipments", icon=":material/arrow_forward:")

# Part 2: port and canal forecasts (Stage 2 model).
with right, st.container(border=True):
    st.subheader("2. Forecasts: will a port or canal get congested?")
    st.markdown(f"**Deep learning (Stage 2)** · {dl_model}")
    st.markdown("For 27 ports, canals and hubs, the model forecasts the **dwell time** (how long cargo waits there) "
                "from 1 hour to 7 days ahead. It warns when dwell time is likely to go above the node's **alert line**, "
                "the level it stays below 95% of the time in normal operation.")
    st.page_link("pages/2_Forecasts.py", label="Open Forecasts", icon=":material/arrow_forward:")

st.caption(f"Data: simulated shipping data calibrated on real drivers. The screens replay the test period "
           f"({start:%d %b} to {end:%d %b %Y}), which the models never saw during training. "
           f"Shipment counts are the model's predictions for that period.")
