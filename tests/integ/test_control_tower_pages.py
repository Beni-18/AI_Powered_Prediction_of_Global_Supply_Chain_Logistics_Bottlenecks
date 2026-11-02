"""Smoke test for the Streamlit control tower: every page renders without an error.

Each page loads the Stage 1 model, so the pages run in a fresh process (PyTorch from the Stage 2 tests and
XGBoost clash on macOS when loaded together). Skipped where Streamlit, Plotly or the model are not available.
"""

import subprocess
import sys

import pytest

from common.paths import REPO_ROOT, stage_path

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
pytest.importorskip("xgboost")
pytestmark = pytest.mark.skipif(not stage_path(1, "artifacts", "stage01_models.joblib").exists(), reason="Stage 1 model not present")

# app.py opens on the Shipments page (its default page); both pages are also rendered on their own.
PAGES = ["app.py", "pages/1_Shipments.py", "pages/2_Forecasts.py"]
SCRIPT = """
import sys
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(sys.argv[1], default_timeout=300).run()
problems = [str(e.value) for e in at.exception] + [str(e.value) for e in at.error]
print("PROBLEMS:", problems)
sys.exit(1 if problems else 0)
"""


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(page):
    result = subprocess.run([sys.executable, "-W", "ignore", "-c", SCRIPT, page], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


RULE_SCRIPT = """
import json, sys
import pandas as pd
from streamlit.testing.v1 import AppTest
from common.evalkit.forecast import exceedance_probability
from common.paths import stage_path

# Step 1: render the Forecasts page and read its early-warning table (the first table on the page).
at = AppTest.from_file("pages/2_Forecasts.py", default_timeout=300).run()
assert not at.exception, [str(e.value) for e in at.exception]
shown = at.dataframe[0].value
origin = pd.Timestamp(at.select_slider[0].value)

# Step 2: the same rule computed independently from the files: highest P(dwell >= line) over all horizons >= p*.
champ = json.loads(stage_path(2, "reports", "eval_champion.json").read_text())
p_star = champ["alert_rule"]["p_star"]
thr = pd.read_csv(stage_path(2, "reports", "eval_alert_thresholds.csv"), index_col=0).iloc[:, 0]
fc = pd.read_csv(stage_path(2, "predictions", "dl_forecasts.csv.gz"))
fc = fc[(fc["split"] == "test") & (fc["model_name"] == champ["model"]) & (fc["target"] == "dwell")]
fc = fc[pd.to_datetime(fc["origin_ts"], utc=True) == origin]
p = pd.Series(exceedance_probability(fc["y_lo"], fc["y_pred"], fc["y_hi"], fc["node_id"].map(thr)), index=fc.index)
expected = (p.groupby(fc["node_id"]).max() >= p_star).sum()

# Step 3: same number of warnings, every shown chance >= p*, and the 'Most likely when' column is present.
print("shown", len(shown), "expected", expected, "p_star", p_star)
assert len(shown) == expected and expected > 0
assert (shown["Chance above the line"] >= 100 * p_star - 1e-9).all()
assert "Most likely when" in shown.columns
"""


def test_forecasts_page_early_warnings_follow_the_validated_p_star_rule():
    """The early-warning list equals Role 4's rule B (max chance over all horizons >= p* from eval_champion.json)."""
    if not stage_path(2, "reports", "eval_champion.json").exists():
        pytest.skip("Stage 2 evaluation not present")
    result = subprocess.run([sys.executable, "-W", "ignore", "-c", RULE_SCRIPT], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
