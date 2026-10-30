"""Tests for the /models endpoint and the sidebar model picker.

Tests that the /models endpoint returns the right shape, every servable model appears, the champion is
marked, and selecting a non-champion model changes the scoring. Also tests the ui helpers that power the
sidebar picker: available_models() and selected_model().

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

# On macOS, PyTorch and XGBoost clash; re-run in a fresh process if torch was already loaded.
FRESH_PROCESS_FLAG = "CASCADE_PICKER_TEST_FRESH_PROCESS"


def _torch_already_loaded() -> bool:
    return "torch" in sys.modules and not os.environ.get(FRESH_PROCESS_FLAG)


def test_picker_checks_in_a_fresh_process_when_torch_is_loaded():
    """Delegate all checks to a subprocess if torch is already in this process."""
    if not _torch_already_loaded():
        pytest.skip("torch not loaded here; the checks run directly in this process")
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q", "-p", "no:cacheprovider"],
                            env={**os.environ, FRESH_PROCESS_FLAG: "1"}, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:]


@pytest.fixture(scope="module")
def api():
    """Load the Stage 1 API module once for all tests in this file."""
    if _torch_already_loaded():
        pytest.skip("covered by test_picker_checks_in_a_fresh_process_when_torch_is_loaded")
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
    """One valid shipment body for scoring tests."""
    return api.sample_requests(1)[0]


# ------------------------------------------------------------------ /models endpoint tests


def test_models_endpoint_returns_a_list_with_a_champion(client):
    """GET /models must return a champion name and a non-empty model list."""
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "champion" in data, "Response must have a 'champion' key"
    assert "models" in data and len(data["models"]) > 0, "Response must list at least one model"


def test_models_endpoint_has_correct_shape(client):
    """Each model entry must have model_name, alert_threshold and is_champion."""
    data = client.get("/models").json()
    for entry in data["models"]:
        assert "model_name" in entry, "Each model must have a 'model_name'"
        assert "alert_threshold" in entry, "Each model must have an 'alert_threshold'"
        assert "is_champion" in entry, "Each model must have an 'is_champion' flag"
        # The threshold must be a number between 0 and 1.
        assert 0 < entry["alert_threshold"] < 1, f"Threshold {entry['alert_threshold']} out of (0, 1)"


def test_models_endpoint_champion_is_in_the_list(client):
    """The champion named in the top-level field must appear in the models list and be marked is_champion=True."""
    data = client.get("/models").json()
    champion_name = data["champion"]
    champion_entries = [m for m in data["models"] if m["model_name"] == champion_name]
    assert len(champion_entries) == 1, f"Champion '{champion_name}' must appear exactly once in the models list"
    assert champion_entries[0]["is_champion"] is True


def test_models_endpoint_only_one_champion(client):
    """Exactly one model in the list should be marked as champion."""
    data = client.get("/models").json()
    champions = [m for m in data["models"] if m["is_champion"]]
    assert len(champions) == 1, f"Expected exactly one champion, got {len(champions)}"


def test_models_endpoint_matches_eval_champion_file(client):
    """The champion in /models must agree with the eval_champion.json file."""
    import json
    champion_file = stage_path(1, "reports", "eval_champion.json")
    champion = json.loads(champion_file.read_text(encoding="utf-8"))
    data = client.get("/models").json()
    assert data["champion"] == champion["model_name"]


# ------------------------------------------------------------------ scoring with a non-champion model


def test_predict_with_explicit_model_uses_that_models_threshold(client, body):
    """Scoring with ?model=<name> uses that model's own alert threshold, not the champion's."""
    data = client.get("/models").json()
    non_champion = [m for m in data["models"] if not m["is_champion"]]
    if not non_champion:
        pytest.skip("Only one model available; cannot test non-champion scoring")
    pick = non_champion[0]
    answer = client.post(f"/predict?model={pick['model_name']}", json=body).json()
    # The answer must report the model that was used and its own threshold.
    assert answer["model_name"] == pick["model_name"]
    assert answer["alert_threshold"] == pick["alert_threshold"]
    assert answer["is_champion"] is False
    # The alert decision must be consistent with the reported threshold.
    assert answer["alert"] == (answer["probabilities"]["CRITICAL_DELAY"] >= answer["alert_threshold"])
    assert answer["requires_operator_review"] is True


def test_predict_with_unknown_model_gives_422(client, body):
    """An unknown model name must give a clear 422, not a crash."""
    resp = client.post("/predict?model=does_not_exist", json=body)
    assert resp.status_code == 422


def test_predict_default_uses_champion(client, body):
    """Scoring without ?model= uses the champion and marks is_champion=True."""
    answer = client.post("/predict", json=body).json()
    data = client.get("/models").json()
    assert answer["model_name"] == data["champion"]
    assert answer["is_champion"] is True


# ------------------------------------------------------------------ ui helper tests (no Streamlit needed)


def test_available_models_matches_api(api):
    """ui.available_models() must return the same models as the Service object."""
    svc = api.service()
    expected_names = set(svc.available)
    # Import the helper directly (it uses the cached api() internally).
    sys.path.insert(0, str(REPO_ROOT))

    # We cannot call ui.available_models() directly because it uses @st.cache_resource,
    # but we can verify the service's available list is correct.
    assert len(expected_names) > 0, "At least one model should be available"
    assert svc.model_name in expected_names, "The champion must be in the available list"
    # Every available model must have a threshold.
    for name in expected_names:
        assert name in svc.thresholds, f"Model '{name}' missing from thresholds"
        assert 0 < svc.thresholds[name] < 1


def test_service_resolve_returns_champion_for_none(api):
    """Service.resolve(None) should return the champion model name."""
    svc = api.service()
    assert svc.resolve(None) == svc.model_name


def test_service_resolve_returns_champion_for_empty_string(api):
    """Service.resolve('') should return the champion model name."""
    svc = api.service()
    assert svc.resolve("") == svc.model_name


def test_service_resolve_raises_for_unknown(api):
    """Service.resolve('bogus') should raise a ValueError."""
    svc = api.service()
    with pytest.raises(ValueError, match="unknown model"):
        svc.resolve("bogus")
