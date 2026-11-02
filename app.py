"""CASCADE control tower (Streamlit). OWNER: Role 3 (Data & Integration Engineer), integration hat.

Start it with:   streamlit run app.py
Two pages in a menu at the top, and the app opens straight on the first one (no home page to click through):
    Shipments       - which shipments will be late, and why (Stage 1, machine learning)
    Port forecasts  - which ports, canals and hubs will get congested in the next 7 days (Stage 2, deep learning)

Design rules: show only what a planner acts on, with the most urgent items first. No maps, no model comparisons,
no evaluation charts (those live in the reports folders). Every number on screen comes from a file written by a role
script. The pages read files only, so the deployed app needs no PyTorch.
"""

import sys
from pathlib import Path

import streamlit as st

# Let every page import the common modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="CASCADE", layout="wide")
# Less empty space above the page title, so the lists start higher on the screen.
st.markdown("<style>.block-container {padding-top: 2.5rem;}</style>", unsafe_allow_html=True)

# Two pages, shown as a simple menu across the top of the window; Shipments opens first.
pages = [
    st.Page("pages/1_Shipments.py", title="Shipments", default=True),
    st.Page("pages/2_Forecasts.py", title="Port forecasts", url_path="forecasts"),
]
st.navigation(pages, position="top").run()
