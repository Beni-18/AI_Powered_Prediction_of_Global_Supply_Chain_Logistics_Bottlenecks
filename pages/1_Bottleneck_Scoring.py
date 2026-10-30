"""CASCADE control tower - Stage 1 page: shipment bottleneck scoring.
Owner: Role 3 (Data & Integration Engineer). Built by Role 4 at the team lead's request; Role 3 to review.

READS : Stage01_ML/reports/eval_champion.json, eval_model_comparison.csv   (Role 4)
        Stage01_ML/05_integration_engineer.py (the same scoring code the API uses)
        Stage01_ML/data/prepared_dataset.csv.gz (example shipments for the form)
WRITES: nothing. Decision support only: every row needs a planner's review.

Start with:   streamlit run app.py   (then open "Bottleneck Scoring" in the sidebar)
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.paths import stage_path  # noqa: E402

st.set_page_config(page_title="Bottleneck scoring", layout="wide")


@st.cache_resource
def api_module():
    """Load the Stage 1 API module once, so the page scores with exactly the same code as the API."""
    spec = importlib.util.spec_from_file_location("stage01_api", REPO_ROOT / "Stage01_ML" / "05_integration_engineer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.service()  # load the model now, on the main thread
    return module


def read_json(path: Path) -> dict:
    """Read a JSON report, or an empty dict if it has not been produced yet."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# Step 1: the decision rule in force, read from Role 4's champion file (never typed by hand).
champion = read_json(stage_path(1, "reports", "eval_champion.json"))
st.title("Shipment bottleneck scoring")
if not champion:
    st.error("No champion file yet (Stage01_ML/reports/eval_champion.json). Scores below come from the STUB model.")
cols = st.columns(4)
cols[0].metric("Model served", champion.get("model_name", "STUB"))
cols[1].metric("Alert when P(critical) is at least", f"{champion.get('threshold', 0.5):.2f}")
cols[2].metric("Chosen on", champion.get("selected_on", "-"))
cols[3].metric("Beats best baseline", str(champion.get("beats_baseline", "-")))
st.caption("Decision support only: every alert requires operator review. Nothing here reroutes or books a shipment.")

# Step 2: model-vs-baseline mini table from Role 4's evaluation.
comparison_file = stage_path(1, "reports", "eval_model_comparison.csv")
if comparison_file.exists():
    table = pd.read_csv(comparison_file)
    show = table[["model_name", "is_baseline", "macro_f1", "pr_auc_critical", "expected_cost_at_threshold", "threshold",
                  "at_threshold_critical_recall", "at_threshold_false_alarms_per_100_alerts"]]
    show = show.rename(columns={"expected_cost_at_threshold": "cost_per_shipment", "at_threshold_critical_recall": "critical_caught",
                                "at_threshold_false_alarms_per_100_alerts": "false_alarms_per_100"})
    st.subheader("Models on the held-out test period")
    st.dataframe(show.sort_values("cost_per_shipment").round(3), hide_index=True, use_container_width=True)

# Step 3: score shipments - pick examples or upload a CSV with the contract columns.
st.subheader("Score shipments")
api = api_module()
fields = api.NUMERIC_FIELDS + ["priority_tier", "carrier_code", "origin_port", "destination_port", "current_node_id"]
uploaded = st.file_uploader("CSV with the telemetry columns (up to 1,000 rows)", type="csv")
if uploaded is not None:
    rows = pd.read_csv(uploaded).head(api.MAX_BATCH)
else:
    n = st.slider("Example test-period shipments to score", 5, 200, 25)
    rows = pd.DataFrame(api.sample_requests(n))

# Step 4: validate every row with the API's own contract model; bad rows are listed, not silently dropped.
good, problems = [], []
for i, record in enumerate(rows.to_dict(orient="records")):
    try:
        good.append(api.Shipment(**{k: record.get(k) for k in fields + ["timestamp", "record_id"] if k in record}))
    except Exception as err:  # pydantic ValidationError: show it to the planner
        problems.append({"row": i, "problem": str(err).splitlines()[1] if "\n" in str(err) else str(err)})
if problems:
    st.warning(f"{len(problems)} row(s) break the data contract and were not scored.")
    st.dataframe(pd.DataFrame(problems), hide_index=True)

if good:
    answers = pd.DataFrame(api.service().score(good))
    answers["p_critical"] = answers["probabilities"].map(lambda p: p["CRITICAL_DELAY"])
    answers["drivers"] = answers["top_drivers"].map(", ".join)
    answers["node"] = [g.current_node_id for g in good]
    view = answers[["record_id", "node", "label", "p_critical", "risk_score", "alert", "anomaly_flag", "drivers"]]
    st.dataframe(view.sort_values("risk_score", ascending=False), hide_index=True, use_container_width=True)
    st.caption(f"{int(answers['alert'].sum())} of {len(answers)} shipments raise an alert at the current threshold.")
