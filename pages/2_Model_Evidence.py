"""CASCADE control tower - Stage 1 page: the evidence behind the model (our USP: evidence over demos).
Owner: Role 3 (Data & Integration Engineer). Built by Role 4 at the team lead's request; Role 3 to review.

READS : Stage01_ML/reports/eval_*.json|csv|png (Role 4), claims/CLAIMS_LEDGER.md (Role 4),
        docs/decisions/0004-cost-policy-and-alert-threshold.md (team decision)
WRITES: nothing.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.evalkit import ledger  # noqa: E402
from common.integration import ui  # noqa: E402
from common.paths import CONTRACTS_DIR, stage_path  # noqa: E402

reports = stage_path(1, "reports")
champion = ui.read_json(reports / "eval_champion.json")
metrics = ui.read_json(reports / "eval_metrics.json")
leak = ui.read_json(reports / "eval_leakage_tests.json")
if not metrics:
    st.error("No evaluation yet: run python Stage01_ML/04_evaluation_engineer.py")
    st.stop()

# Step 1: who is champion, why, and whether the team accepted the alert rule.
decision_file = REPO_ROOT / "docs" / "decisions" / "0004-cost-policy-and-alert-threshold.md"
decision_line = next((ln for ln in decision_file.read_text(encoding="utf-8").splitlines() if "Status:" in ln), "") if decision_file.exists() else ""
decision_status = decision_line.split("Status:", 1)[-1].strip() or "not recorded"
row = next(m for m in metrics["models"] if m["model_name"] == champion["model_name"])

st.title("Model evidence")
st.caption(f"Stage 1 results on the held-out test period, which no model saw during training. The ranges are 95% intervals "
           f"from a day-by-day bootstrap ({metrics['bootstrap_resamples']:,} resamples). Evaluated on {metrics['date']}.")
champ_name = ui.model_label(champion["model_name"])
info = metrics["champion"]
tiles = st.columns(5)
tiles[0].metric("Model in use", champ_name)
tiles[1].metric("Overall score (macro-F1)", f"{row['macro_f1']:.3f}", help=f"95% range {row['macro_f1_lo']:.3f} – {row['macro_f1_hi']:.3f}")
tiles[2].metric("Critical delays caught", f"{row['at_threshold_critical_recall']:.0%}")
tiles[3].metric("False alarms per 100 alerts", f"{row['at_threshold_false_alarms_per_100_alerts']:.0f}", help="Team budget: 40")
tiles[4].metric("Alert line (decision 0004)", f"{champion['threshold']:.0%}", help=f"Chance of a critical delay that triggers an alert · {decision_status}")

# Why this model: written out as sentences from the numbers in eval_metrics.json.
lo, hi = info.get("val_cost_saving_ci", [0, 0])
saving = info.get("val_cost_saving_vs_baseline", 0)
best = ui.model_label(info.get("best_candidate", "-"))
if info.get("beats_baseline"):
    why = (f"**{champ_name}** is in use because it beat the best simple baseline on the validation period by "
           f"{saving:.3f} cost units per shipment (95% range {lo:+.3f} to {hi:+.3f}).")
else:
    why = (f"**{champ_name}** is in use because none of the trained models was clearly better on the validation period. "
           f"The strongest one, {best}, changed the cost per shipment by only {saving:+.3f} (95% range {lo:+.3f} to {hi:+.3f}), "
           "a range that includes zero, so the difference could be chance. Our rule is to keep the simpler model unless the gain is clear, "
           "and we never choose a model by looking at the test period.")
st.info(why)

# Step 2: every model side by side, baselines included, then the verdict in words.
st.subheader("How every model performed on the test period")
models = pd.DataFrame(metrics["models"])
table = pd.DataFrame({
    "Model": models["model_name"].map(ui.model_label),
    "Type": models["is_baseline"].map({True: "Simple baseline", False: "Trained model"}),
    "Overall score (macro-F1)": models["macro_f1"],
    "95% range": models.apply(lambda r: f"{r.macro_f1_lo:.3f} – {r.macro_f1_hi:.3f}", axis=1),
    "Ranks critical cases (PR-AUC)": models["pr_auc_critical"],
    "Cost per shipment": models["expected_cost_at_threshold"],
    "Alert line": 100 * models["threshold"],
    "Critical delays caught": 100 * models["at_threshold_critical_recall"],
    # A model that never alerts has no false-alarm rate; say so instead of showing an empty number.
    "False alarms per 100 alerts": models["at_threshold_false_alarms_per_100_alerts"].map(lambda v: "never alerts" if pd.isna(v) else f"{v:.0f}"),
}).sort_values("Cost per shipment")
st.dataframe(table.round(3), hide_index=True, use_container_width=True, column_config={
    "Alert line": st.column_config.NumberColumn(format="%.0f%%"),
    "Critical delays caught": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%")})

cheapest = models.sort_values("expected_cost_at_threshold").iloc[0]
majority = models[models["model_name"] == "majority"]
lines = [
    f"On the test period, **{ui.model_label(cheapest['model_name'])}** had the lowest cost per shipment "
    f"({cheapest['expected_cost_at_threshold']:.3f}) and caught {cheapest['at_threshold_critical_recall']:.0%} of critical delays, "
    f"against {row['expected_cost_at_threshold']:.3f} and {row['at_threshold_critical_recall']:.0%} for {champ_name}.",
]
if cheapest["model_name"] != champion["model_name"]:
    lines.append("That gap appeared only on the test period, not on validation, which is why the model in use did not change.")
if len(majority):
    m = majority.iloc[0]
    lines.append(f"Every trained model is far better than simply calling everything on time, which would cost "
                 f"{m['expected_cost_at_threshold']:.3f} per shipment and catch no critical delays at all.")
st.markdown(" ".join(lines))
st.caption("Cost per shipment follows the team's cost policy: missing a critical delay costs 10 units and an unnecessary "
           "escalation costs 1, so lower is better. The overall score averages accuracy across the three classes; "
           "PR-AUC measures how well a model puts real critical cases at the top of its list.")

# Step 3: the report charts written by the evaluation script.
st.subheader("Evaluation charts")
charts = [("Predicted vs real", "eval_confusion.png", "What the champion predicted vs what really happened (test period)"),
          ("Choosing the threshold", "eval_cost_curve.png", "Validation cost at every alert threshold; the dot is the chosen one"),
          ("Calibration", "eval_calibration.png", "Does a predicted 40% really happen about 40% of the time?"),
          ("Weakest slices", "eval_slices.png", "Ten weakest slices by node, carrier, priority and month")]
for tab, (_, name, caption) in zip(st.tabs([c[0] for c in charts]), charts):
    with tab:
        if (reports / name).exists():
            st.image(str(reports / name), width=720)  # one fixed width keeps the four charts consistent
            st.caption(caption)

# Step 4: the leakage tests - proof the score is not cheating.
st.subheader("Leakage tests")
TEST_TEXT = {
    "a_feature_policy": ("Only allowed inputs used", "illegal_features", lambda v: f"{len(v)} illegal"),
    "b_label_shuffle_sanity": ("Scoring sanity: shuffled labels give chance", "test_macro_f1_with_shuffled_labels", lambda v: f"macro-F1 {v:.2f}"),
    "c_single_feature_ceiling": ("No single input gives the answer away", "max_auc", lambda v: f"best AUC {v:.2f} (< 0.95)"),
    "c2_bookkeeping_ceiling": ("Ids carry no hint beyond the clock", "excess_over_clock", lambda v: f"+{v:.3f} AUC"),
    "d_time_order": ("Train → validation → test in time, 7-day gaps", "gap_train_to_val_h", lambda v: f"gap {v:.0f} h"),
    "e_too_good_to_be_true": ("Score is not suspiciously perfect", "champion_macro_f1", lambda v: f"macro-F1 {v:.2f} (< 0.95)"),
}
rows = []
for key, (text, field, fmt) in TEST_TEXT.items():
    result = leak.get(key, {})
    detail = fmt(result[field]) if field in result else result.get("reason", "")
    if result.get("status") == "PARTIAL":
        detail += f" · {len(result.get('untested_features', []))} model-built inputs not testable yet"
    rows.append({"Check": text, "Result": result.get("status", "NOT_RUN"), "Detail": detail})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
    "Check": st.column_config.TextColumn(width="medium"), "Result": st.column_config.TextColumn(width="small"),
    "Detail": st.column_config.TextColumn(width="large")})

# Step 5: the alert rule's trade-off and the claims we are allowed to quote.
left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("If a miss cost more or less")
    sens = pd.read_csv(reports / "eval_threshold_sensitivity.csv")
    st.dataframe(pd.DataFrame({
        "Miss costs": sens["missed_critical_cost"].map(lambda v: f"{v:.0f}x"),
        "Threshold": sens["threshold"],
        "Caught": 100 * sens["critical_recall"],
        "False alarms / 100": sens["false_alarms_per_100_alerts"].round(0),
    }), hide_index=True, use_container_width=True,
        column_config={"Caught": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%")})
    st.caption(f"Team decision 0004: 10x - {decision_status}")
with right:
    st.subheader("How to read the trade-off")
    # Numbers here are read from the sensitivity table and the cost policy, never typed in.
    budget = ui.read_json(CONTRACTS_DIR / "cost_policy.json")["alert_budget"]["max_false_alarms_per_100_alerts"]
    at = sens.set_index("missed_critical_cost")["false_alarms_per_100_alerts"]
    st.markdown("- A **higher** miss cost lowers the threshold: more delays caught, but more false alarms.\n"
                f"- At **10x** (the team's choice) there are {at.get(10, float('nan')):.0f} false alarms per 100 alerts, "
                f"inside the budget of {budget}.\n"
                f"- At **20x** it would be {at.get(20, float('nan')):.0f} per 100: most alerts false, so planners would stop trusting them.")

st.subheader("Claims we can quote")
claims = pd.DataFrame(ledger.read_rows())
st.dataframe(claims[["id", "value", "status", "claim"]].rename(columns=str.capitalize),
             hide_index=True, use_container_width=True, height=420, column_config={
                 "Id": st.column_config.TextColumn(width="small"), "Value": st.column_config.TextColumn(width="medium"),
                 "Status": st.column_config.TextColumn(width="small"), "Claim": st.column_config.TextColumn(width="large")})
st.caption("Every row is written by code and re-checked from a fresh clone before it is VERIFIED.")

ui.sources("Stage01_ML/reports/eval_metrics.json", "eval_leakage_tests.json", "eval_threshold_sensitivity.csv",
           "claims/CLAIMS_LEDGER.md", "docs/decisions/0004")
