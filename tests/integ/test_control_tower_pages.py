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

PAGES = ["app.py", "pages/overview.py", "pages/1_Bottleneck_Scoring.py", "pages/2_Model_Evidence.py",
         "pages/3_Network_Forecast.py", "pages/4_Forecast_Evidence.py"]
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
