"""Tests for the Stage 2 forecast API (Stage02_DL/05_integration_engineer.py): contract-shaped answers, ordered
intervals, clear 422 errors for bad windows, and the safety rule that every answer requires operator review.

Skipped where PyTorch / FastAPI or the trained Stage 2 models are not present (for example on CI).
Owner: Role 3 (integration). Written by Role 4 at the team lead's request; Role 3 to review.
"""

import importlib.util

import pandas as pd
import pytest

from common.dataio.hourly_grid import CHANNELS
from common.paths import REPO_ROOT, stage_path

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytestmark = pytest.mark.skipif(not stage_path(2, "artifacts", "lstm_forecaster.pt").exists()
                                or not stage_path(2, "data", "hourly_node_telemetry.csv.gz").exists(),
                                reason="Stage 2 models or data not present")


@pytest.fixture(scope="module")
def client():
    """Import the API module from its file (its name starts with a digit) and wrap it in a test client."""
    from fastapi.testclient import TestClient

    spec = importlib.util.spec_from_file_location("stage02_api", REPO_ROOT / "Stage02_DL" / "05_integration_engineer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return TestClient(module.app)


@pytest.fixture(scope="module")
def body():
    """72 real consecutive hours of one node from the stored grid (test period)."""
    grid = pd.read_csv(stage_path(2, "data", "hourly_node_telemetry.csv.gz"))
    rows = grid[grid["node_id"] == "PORT_US_LAX"].iloc[-200:-128]
    recs = rows[["hour_ts", *CHANNELS, "is_imputed"]].copy()
    recs["is_imputed"] = recs["is_imputed"].astype(str).str.lower().eq("true")
    return {"node_id": "PORT_US_LAX", "records": recs.to_dict(orient="records")}


def _check_answer(ans: dict) -> None:
    """Five horizons, two targets, ordered quantiles, and operator review always required."""
    assert ans["requires_operator_review"] is True
    assert [h["horizon_h"] for h in ans["horizons"]] == [1, 6, 24, 72, 168]
    for h in ans["horizons"]:
        for target in ("dwell", "throughput"):
            q = h[target]
            assert 0 <= q["q10"] <= q["q50"] <= q["q90"]


def test_forecast_with_one_node_model(client, body):
    response = client.post("/forecast?model=lstm", json=body)
    assert response.status_code == 200, response.text
    _check_answer(response.json())


def test_forecast_with_the_default_model_uses_network_history(client, body):
    response = client.post("/forecast", json=body)
    assert response.status_code == 200, response.text
    _check_answer(response.json())


def test_replay_endpoint(client):
    response = client.get("/forecast/CHOKE_SUEZ", params={"as_of": "2025-12-01T00:00:00Z", "model": "graph_full"})
    if response.status_code == 422 and "unknown model" in response.text:
        pytest.skip("graph_full artefact not present")
    assert response.status_code == 200, response.text
    _check_answer(response.json())


@pytest.mark.parametrize("change", ["too_short", "gap", "unknown_node", "out_of_range", "unknown_model"])
def test_bad_requests_get_a_clear_422(client, body, change):
    bad = {"node_id": body["node_id"], "records": [dict(r) for r in body["records"]]}
    url = "/forecast"
    if change == "too_short":
        bad["records"] = bad["records"][1:]
    elif change == "gap":
        bad["records"] = bad["records"][:30] + bad["records"][31:] + [bad["records"][-1]]
    elif change == "unknown_node":
        bad["node_id"] = "PORT_XX_NOWHERE"
    elif change == "out_of_range":
        bad["records"][5]["capacity_utilization_pct"] = 999.0
    else:
        url = "/forecast?model=no_such_model"
    assert client.post(url, json=bad).status_code == 422


def test_graph_model_outside_the_stored_period_is_refused_with_a_hint(client, body):
    shifted = {"node_id": body["node_id"], "records": [dict(r) for r in body["records"]]}
    for i, r in enumerate(shifted["records"]):
        r["hour_ts"] = (pd.Timestamp("2030-01-01T00:00:00Z") + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
    response = client.post("/forecast?model=graph_full", json=shifted)
    if response.status_code == 422 and "unknown model" in response.text:
        pytest.skip("graph_full artefact not present")
    assert response.status_code == 422 and "model=lstm" in response.text
    assert client.post("/forecast?model=lstm", json=shifted).status_code == 200


def test_nodes_and_network(client):
    assert len(client.get("/nodes").json()["nodes"]) == 27
    net = client.get("/network").json()
    assert len(net["edges"]) == 50


def test_alert_rule_follows_role4_p_star_for_the_champion_only(client, body):
    """The champion's answer carries rule B with Role 4's p*; other models get no borrowed rule."""
    champion = client.get("/models").json()["champion"]
    ans = client.post(f"/forecast?model={champion}", json=body).json()
    rule = ans["alert_rule"]
    if rule is None:
        pytest.skip("no eval_champion.json / alert thresholds on this machine")
    assert rule["alert"] == (rule["max_prob_any_horizon"] >= rule["p_star"])
    # The 24 h probability is one of the horizons, so the maximum can never be smaller.
    assert rule["max_prob_any_horizon"] >= ans["alert_prob_24h"]
    other = next(m["model_name"] for m in client.get("/models").json()["models"] if m["model_name"] != champion)
    assert client.post(f"/forecast?model={other}", json=body).json()["alert_rule"] is None


@pytest.mark.skipif(not stage_path(2, "artifacts", "ensemble_forecaster.json").exists(), reason="ensemble not built")
def test_ensemble_replay_equals_the_forecast_file(client):
    """The served ensemble (saved transformer + xgb_lag, averaged, own margins) gives exactly its rows in the file."""
    origin = "2025-12-01T00:00:00Z"
    response = client.get(f"/forecast/PORT_US_LAX?as_of={origin}&model=ensemble")
    assert response.status_code == 200, response.text
    ans = response.json()
    _check_answer(ans)
    assert ans["model_name"] == "ensemble"
    fc = pd.read_csv(stage_path(2, "predictions", "dl_forecasts.csv.gz"))
    rows = fc[(fc["model_name"] == "ensemble") & (fc["node_id"] == "PORT_US_LAX") & (fc["origin_ts"] == origin)]
    assert len(rows) == 10
    for row in rows.itertuples():
        served = next(h for h in ans["horizons"] if h["horizon_h"] == row.horizon_h)[row.target]
        assert served["q50"] == pytest.approx(row.y_pred, abs=1e-2)
        assert served["q10"] == pytest.approx(row.y_lo, abs=1e-2)
        assert served["q90"] == pytest.approx(row.y_hi, abs=1e-2)


def _outside_the_grid(body: dict) -> dict:
    """The same 72 readings, moved to 2030 (after the stored grid ends)."""
    shifted = {"node_id": body["node_id"], "records": [dict(r) for r in body["records"]]}
    for i, r in enumerate(shifted["records"]):
        r["hour_ts"] = (pd.Timestamp("2030-01-01T00:00:00Z") + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return shifted


def test_default_model_falls_back_to_the_transformer_outside_the_stored_grid(client, body):
    """No ?model and a champion that needs network history: the answer comes from the transformer and says why."""
    models = client.get("/models").json()
    champion = next(m for m in models["models"] if m["is_champion"])
    if not champion["needs_network_history"]:
        pytest.skip("the champion reads only the posted hours, so no fallback is needed")
    response = client.post("/forecast", json=_outside_the_grid(body))
    assert response.status_code == 200, response.text
    ans = response.json()
    _check_answer(ans)
    assert ans["model_name"] == "transformer"
    assert ans["evidence"]["model_fallback"].startswith(f"{champion['model_name']} needs the stored network history")
    # Inside the grid nothing is swapped.
    assert client.post("/forecast", json=body).json()["evidence"]["model_fallback"] is None


def test_explicit_ensemble_outside_the_stored_grid_is_still_refused(client, body):
    """Asking for the ensemble by name is never silently swapped: it gets the clear 422 with the hint."""
    if "ensemble" not in client.get("/health").json()["models"]:
        pytest.skip("ensemble not built")
    response = client.post("/forecast?model=ensemble", json=_outside_the_grid(body))
    assert response.status_code == 422 and "model=transformer" in response.text
