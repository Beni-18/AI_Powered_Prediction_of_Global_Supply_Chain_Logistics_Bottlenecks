"""CASCADE control tower (Streamlit). OWNER: Role 3 (Data & Integration Engineer), integration hat.

STATUS: home page + Stage 1 page. Role 3 builds this up stage by stage (see docs/roles/ROLE_3_DATA_INTEGRATION_ENGINEER.md).

Start it with:   streamlit run app.py
Planned pages (one per stage, added on the day of that stage):
    Day 1  Bottleneck scoring + feature importance + ops field guide
    Day 2  Network map, forecasts with intervals, propagation edges
    Day 3  Advisory search and tagging, advisory-pressure signal
    Day 4  Offline briefing client (also runnable stand-alone)
    Day 5  Scenario injection panel, wildcard live run, fused triage view

Design rules (from the team's design memory): a real product, data-dense, no marketing hero sections,
no decorative gradients. Every number on screen must come from a file written by a role script.
"""

import streamlit as st

# Home page of the multi-page app; each stage adds its own page under pages/.
st.set_page_config(page_title="CASCADE control tower", layout="wide")
st.title("CASCADE control tower")
st.write("Decision support for supply-chain bottlenecks. Every alert needs a planner's review; nothing here "
         "reroutes or books a shipment. Pick a page in the sidebar.")
st.markdown("- **Bottleneck Scoring** (Stage 1): score shipments, see the alert threshold and the model-vs-baseline table.")
