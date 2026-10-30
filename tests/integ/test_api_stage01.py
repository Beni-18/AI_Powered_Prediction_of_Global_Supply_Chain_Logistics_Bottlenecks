"""Tests for the Stage 1 API (Stage01_ML/05_integration_engineer.py): contract-shaped answers, 422 on bad input,
and the safety rule that every answer requires operator review.

Skipped where the API libraries or the trained model are not installed/present (for example on CI).
"""

import importlib.util
import os
import subprocess
import sys

import pytest

from common.paths import REPO_ROOT, stage_path

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("xgboost")
pytest.importorskip("joblib")

ARTEFACT = stage_path(1, "artifacts", "stage01_models.joblib")
pytestmark = pytest.mark.skipif(not ARTEFACT.exists(), reason="Stage 1 model artefact not present")


# On macOS, PyTorch (loaded by the Stage 2 tests) and XGBoost ship clashing OpenMP libraries: loading the model
# in a process that already has torch crashes it. In that case the checks below re-run in a fresh process.
FRESH_PROCESS_FLAG = "CASCADE_API_TEST_FRESH_PROCESS"


def _torch_already_loaded() -> bool:
    return "torch" in sys.modules and not os.environ.get(FRESH_PROCESS_FLAG)


def test_api_checks_in_a_fresh_process_when_torch_is_loaded():
    if not _torch_already_loaded():
        pytest.skip("torch not loaded here; the checks run directly in this process")
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q", "-p", "no:cacheprovider"],
                            env={**os.environ, FRESH_PROCESS_FLAG: "1"}, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:]


@pytest.fixture(scope="module")
def api():
    if _torch_already_loaded():
        pytest.skip("covered by test_api_checks_in_a_fresh_process_when_torch_is_loaded")
    spec = importlib.util.spec_from_file_location("stage01_api", REPO_ROOT / "Stage01_ML" / "05_integration_engineer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.service()  # load the model on the main thread, as the app's start-up hook does
    return module


@pytest.fixture(scope="module")
def client(api):
    from fastapi.testclient import TestClient

    return TestClient(api.app)


@pytest.fixture(scope="module")
def body(api):
    return api.sample_requests(1)[0]


def test_health_reports_a_real_model(client):
    answer = client.get("/health").json()
    assert answer["status"] == "ok" and answer["stub"] is False


def test_predict_answer_has_the_promised_shape(client, body):
    answer = client.post("/predict", json=body).json()
    for key in ("label", "probabilities", "risk_score", "alert", "anomaly_flag", "top_drivers", "model_version"):
        assert key in answer
    assert answer["requires_operator_review"] is True
    assert abs(sum(answer["probabilities"].values()) - 1) < 1e-3
    assert answer["alert"] == (answer["probabilities"]["CRITICAL_DELAY"] >= answer["alert_threshold"])


def test_api_serves_the_evaluation_champion(client):
    import json

    champion = json.loads(stage_path(1, "reports", "eval_champion.json").read_text(encoding="utf-8"))
    info = client.get("/model-info").json()
    assert info["model_name"] == champion["model_name"] and info["alert_threshold"] == champion["threshold"]


@pytest.mark.parametrize("field, value", [("capacity_utilization_pct", 999), ("labor_deficit_ratio", -1),
                                          ("current_node_id", "PORT_NOWHERE"), ("priority_tier", "URGENT")])
def test_bad_input_gets_a_clear_422(client, body, field, value):
    bad = dict(body, **{field: value})
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_scores_every_row_and_keeps_the_review_flag(client, api):
    rows = api.sample_requests(5)
    answer = client.post("/predict/batch", json={"shipments": rows}).json()
    assert len(answer["results"]) == 5 and all(r["requires_operator_review"] for r in answer["results"])
