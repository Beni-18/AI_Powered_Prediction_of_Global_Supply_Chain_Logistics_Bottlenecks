"""CASCADE control tower - Stage 2 page: the evidence behind the forecasters (visual prediction benchmark).
Owner: Role 3 (Data & Integration Engineer). Built by Role 4 at the team lead's request; Role 3 to review.

READS : Stage02_DL/reports/eval_*.json|csv|png|md (Role 4), Stage02_DL/artifacts/model_card.json (Role 1)
WRITES: nothing.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.integration import ui  # noqa: E402
from common.paths import stage_path  # noqa: E402

reports = stage_path(2, "reports")
metrics = ui.read_json(reports / "eval_metrics.json")
if not metrics:
    st.error("No Stage 2 evaluation yet: run python Stage02_DL/04_evaluation_engineer.py")
    st.stop()
champ = metrics["champion"]
head = metrics["headline"]

st.title("Forecast evidence")
st.caption("Every number here was computed on the held-out test period (Nov-Dec 2025) by Stage02_DL/04_evaluation_engineer.py. "
           "Ranges are 95% intervals from a day-by-day bootstrap. The champion was chosen on the validation period, never on test.")

# Step 1: headline tiles for the champion.
tiles = st.columns(5)
tiles[0].metric("Champion", champ["model"])
tiles[1].metric("Skill vs persistence, 24 h", head["skill_dwell_24h"].split(" [")[0], help=head["skill_dwell_24h"])
tiles[2].metric("Skill vs persistence, 7 days", head["skill_dwell_168h"].split(" [")[0], help=head["skill_dwell_168h"])
tiles[3].metric("Bottlenecks warned in advance", head["episode_recall"].split(" [")[0], help=head["episode_recall"])
tiles[4].metric("Median warning time", f"{head['median_lead_h'].split(' [')[0]} h", help=f"IQR {head['median_lead_h']}")
for note in metrics.get("honest_notes", []):
    st.markdown(note)

# Step 2: the charts the debate uses.
tab1, tab2, tab3, tab4 = st.tabs(["Skill by horizon", "Early warning", "Does the graph help?", "Real data (PortWatch)"])
with tab1:
    if (reports / "eval_skill_by_horizon.png").exists():
        st.image(str(reports / "eval_skill_by_horizon.png"))
    st.caption("Skill = 1 - error / error of 'repeat the last value'. Above 0 = better than persistence.")
with tab2:
    for name in ("eval_lead_time.png", "eval_visual_benchmark.png"):
        if (reports / name).exists():
            st.image(str(reports / name))
    if (reports / "eval_visual_benchmark.csv").exists():
        st.dataframe(pd.read_csv(reports / "eval_visual_benchmark.csv"), hide_index=True, width="stretch")
with tab3:
    st.markdown(f"Graph model with the real lanes vs no lanes: **{head['ablation_vs_none']}**; vs randomly relabelled lanes: "
                f"**{head['ablation_vs_shuffled']}** (cells = target x horizon, paired 95% interval).")
    if (reports / "eval_graph_ablation.csv").exists():
        st.dataframe(pd.read_csv(reports / "eval_graph_ablation.csv"), hide_index=True, width="stretch")
with tab4:
    if metrics.get("real_track"):
        st.markdown(f"Same models trained on 2019-2023, tuned on 2024 and tested on 2025-26 **real** daily container ship counts "
                    f"(IMF PortWatch). CI-positive skill horizons: **{head.get('real_positive_skill')}**.")
        for name in ("eval_real_skill_by_horizon.png", "eval_real_example.png"):
            if (reports / name).exists():
                st.image(str(reports / name))
    else:
        st.info("Real track not evaluated on this machine (PortWatch data not downloaded).")
ui.sources("Stage02_DL/reports/eval_metrics.json", "Stage02_DL/reports/eval_forecast_metrics.csv", "Stage02_DL/reports/eval_lead_time.csv",
           "Stage02_DL/reports/eval_graph_ablation.csv")
