"""CASCADE control tower (Streamlit). OWNER: Role 3 (Data & Integration Engineer), integration hat.
Stage 1 version built by Role 4 at the team lead's request; Role 3 to review.

Start it with:   streamlit run app.py
This file only sets up the window and the page list; each page lives in pages/. Later stages add their own pages:
    Day 2  Network forecasts with intervals, propagation edges
    Day 3  Advisory search and tagging, advisory-pressure signal
    Day 4  Offline briefing client (also runnable stand-alone)
    Day 5  Scenario injection panel, wildcard live run, fused triage view

Design rules (from the team's design memory): a real product, data-dense, no marketing hero sections,
no decorative gradients. Every number on screen must come from a file written by a role script.
"""

import sys
from pathlib import Path

import streamlit as st

# Let every page and the model picker import common modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.integration import ui  # noqa: E402

st.set_page_config(page_title="CASCADE control tower", layout="wide")

# The sidebar shows these names in this order (Streamlit 1.36+ page navigation).
pages = {
    "Stage 1 · shipment delays": [
        st.Page("pages/overview.py", title="Network overview", default=True),
        st.Page("pages/1_Bottleneck_Scoring.py", title="Bottleneck scoring", url_path="scoring"),
        st.Page("pages/2_Model_Evidence.py", title="Model evidence", url_path="evidence"),
    ],
    "Stage 2 · network forecasts": [
        st.Page("pages/3_Network_Forecast.py", title="Network forecast", url_path="forecast"),
        st.Page("pages/4_Forecast_Evidence.py", title="Forecast evidence", url_path="forecast-evidence"),
    ],
}
st.sidebar.markdown("**CASCADE** control tower")
st.sidebar.caption("Decision support only · every alert needs a planner's review")

# Model picker: lets the user switch models on every page. The choice persists across page switches via
# session state. Pages read it through ui.selected_model() or pass it to validate_and_score/scored_period.
ui.model_picker()
st.sidebar.divider()

st.navigation(pages).run()
