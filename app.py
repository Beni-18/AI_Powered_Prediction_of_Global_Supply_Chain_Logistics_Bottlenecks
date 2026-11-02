"""CASCADE control tower (Streamlit). OWNER: Role 3 (Data & Integration Engineer), integration hat.
Simplified on 2 Nov 2026 by Role 4 at the team lead's request; Role 3 to review.

Start it with:   streamlit run app.py
Two pages in a menu at the top, nothing else:
    Shipments  - which shipments are likely to end 24 h or more late (Stage 1 model)
    Forecasts  - expected dwell time and throughput for the next 7 days at every node (Stage 2 model)

Design rules: show only what a planner needs to act on. No maps, no model comparisons, no evaluation charts
(those live in the reports folders). Every number on screen comes from a file written by a role script.
"""

import sys
from pathlib import Path

import streamlit as st

# Let every page import the common modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="CASCADE", layout="wide")

# Two pages, shown as a simple menu across the top of the window.
pages = [
    st.Page("pages/1_Shipments.py", title="Shipments", default=True),
    st.Page("pages/2_Forecasts.py", title="Forecasts", url_path="forecasts"),
]
st.navigation(pages, position="top").run()
