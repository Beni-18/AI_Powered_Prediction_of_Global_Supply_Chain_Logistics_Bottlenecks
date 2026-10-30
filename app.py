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

import streamlit as st

st.set_page_config(page_title="CASCADE control tower", layout="wide")

# The sidebar shows these names in this order (Streamlit 1.36+ page navigation).
pages = {
    "Stage 1 · shipment delays": [
        st.Page("pages/overview.py", title="Network overview", default=True),
        st.Page("pages/1_Bottleneck_Scoring.py", title="Bottleneck scoring", url_path="scoring"),
        st.Page("pages/2_Model_Evidence.py", title="Model evidence", url_path="evidence"),
    ],
}
st.sidebar.markdown("**CASCADE** control tower")
st.sidebar.caption("Decision support only · every alert needs a planner's review")
st.navigation(pages).run()
