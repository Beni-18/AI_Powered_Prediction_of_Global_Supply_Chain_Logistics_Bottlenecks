"""End-to-end rehearsal of Stage01_ML/04_evaluation_engineer.py on stand-in predictions.

Why this file exists: it proves the whole evaluation pipeline runs and writes every promised file BEFORE
Role 1's real predictions arrive, and it keeps proving it every time anyone changes the script.
"""

import json
import subprocess
import sys

import pytest

from common.paths import REPO_ROOT

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

import rehearsal  # noqa: E402  (lives next to this test file)

SHIP = REPO_ROOT / "data" / "raw" / "sim" / "shipment_telemetry_raw.csv.gz"
pytestmark = pytest.mark.skipif(not SHIP.exists(), reason="raw simulator file not present")

EXPECTED_FILES = [
    "eval_metrics.json", "eval_model_comparison.csv", "eval_model_comparison.png", "eval_confusion.png",
    "eval_threshold_decision.md", "eval_cost_curve.png", "eval_threshold_sensitivity.csv", "eval_champion.json",
    "eval_calibration.png", "eval_leakage_tests.json", "eval_slices.csv", "eval_slices.png", "eval_anomaly.json",
]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("rehearsal")
    pred_path, prepared_path = rehearsal.build(folder)
    out = folder / "reports"
    from common.evalkit import ledger

    ledger_before = ledger.LEDGER_PATH.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "Stage01_ML" / "04_evaluation_engineer.py"), "--predictions", str(pred_path),
         "--prepared", str(prepared_path), "--model-card", str(folder / "model_card.json"), "--out-dir", str(out), "--n-boot", "40"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    return result, out, ledger_before


def test_script_exits_cleanly(run):
    result, _, _ = run
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_promised_file_is_written(run):
    _, out, _ = run
    missing = [f for f in EXPECTED_FILES if not (out / f).exists() or (out / f).stat().st_size == 0]
    assert not missing, missing


def test_champion_file_has_the_agreed_keys_and_a_sane_threshold(run):
    _, out, _ = run
    champ = json.loads((out / "eval_champion.json").read_text(encoding="utf-8"))
    for key in ("stage", "model_name", "model_version", "threshold", "selected_on", "cost_policy_version", "date"):
        assert key in champ
    assert champ["selected_on"] == "val" and 0 < champ["threshold"] < 1


def test_leakage_tests_run_and_pass_on_the_simulator(run):
    _, out, _ = run
    leak = json.loads((out / "eval_leakage_tests.json").read_text(encoding="utf-8"))
    assert leak["b_label_shuffle_sanity"]["status"] == "PASS"
    assert leak["c_single_feature_ceiling"]["status"] in {"PASS", "PARTIAL"}
    assert leak["c2_bookkeeping_ceiling"]["status"] == "PASS"
    assert leak["d_time_order"]["status"] == "PASS"
    assert leak["e_too_good_to_be_true"]["status"] == "PASS"


def test_dry_run_does_not_touch_the_real_ledger(run):
    from common.evalkit import ledger

    _, _, before = run
    assert ledger.LEDGER_PATH.read_text(encoding="utf-8") == before


def test_missing_predictions_file_exits_with_the_skipped_code(tmp_path):
    result = subprocess.run([sys.executable, str(REPO_ROOT / "Stage01_ML" / "04_evaluation_engineer.py"),
                             "--predictions", str(tmp_path / "nope.csv"), "--out-dir", str(tmp_path / "o")],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 3 and "WAITING" in result.stdout


# ---------- the script must refuse to publish when the input or the result cannot be trusted ----------

def _run(pred_path, prepared_path, out):
    return subprocess.run([sys.executable, str(REPO_ROOT / "Stage01_ML" / "04_evaluation_engineer.py"), "--predictions", str(pred_path),
                           "--prepared", str(prepared_path), "--model-card", str(pred_path.parent / "model_card.json"),
                           "--out-dir", str(out), "--n-boot", "20"],
                          cwd=REPO_ROOT, capture_output=True, text=True)


@pytest.fixture(scope="module")
def stand_in(tmp_path_factory):
    folder = tmp_path_factory.mktemp("integrity")
    pred_path, prepared_path = rehearsal.build(folder)
    return folder, pred_path, prepared_path


def test_split_disagreeing_with_the_frozen_dataset_stops_the_run(stand_in):
    import pandas as pd

    folder, pred_path, prepared_path = stand_in
    pred = pd.read_csv(pred_path)
    # Pretend Role 1 moved some training rows into the test split.
    swap = pred[(pred["model_name"] == "gbm") & (pred["split"] == "train")].index[:50]
    pred.loc[swap, "split"] = "test"
    bad = folder / "bad_split.csv"
    pred.to_csv(bad, index=False)
    result = _run(bad, prepared_path, folder / "out_bad_split")
    assert result.returncode == 5 and "different split" in result.stdout
    assert not (folder / "out_bad_split" / "eval_champion.json").exists()


def test_models_scoring_different_rows_stop_the_run(stand_in):
    import pandas as pd

    folder, pred_path, prepared_path = stand_in
    pred = pd.read_csv(pred_path)
    drop = pred[(pred["model_name"] == "logistic") & (pred["split"] == "test")].index[:100]
    bad = folder / "bad_rows.csv"
    pred.drop(drop).to_csv(bad, index=False)
    result = _run(bad, prepared_path, folder / "out_bad_rows")
    assert result.returncode == 5 and "different set" in result.stdout


def test_a_too_good_to_be_true_champion_is_blocked_from_publishing(stand_in):
    import pandas as pd

    folder, pred_path, prepared_path = stand_in
    pred = pd.read_csv(pred_path)
    # A "perfect" model that simply returns the truth: the classic sign of leakage.
    perfect = pred[pred["model_name"] == "gbm"].copy()
    perfect["model_name"] = "too_good"
    for col, cls in (("p_on_time", "ON_TIME"), ("p_at_risk", "AT_RISK"), ("p_critical", "CRITICAL_DELAY")):
        perfect[col] = (perfect["y_true"] == cls).astype(float)
    perfect["y_pred"] = perfect["y_true"]
    perfect["risk_score"] = perfect["p_critical"] + 0.5 * perfect["p_at_risk"]
    bad = folder / "too_good.csv"
    pd.concat([pred, perfect]).to_csv(bad, index=False)
    out = folder / "out_too_good"
    result = _run(bad, prepared_path, out)
    # The perfect model wins on cost, so the champion is it, and the script must refuse to publish.
    assert result.returncode == 4 and "BLOCKED" in result.stdout
    assert not (out / "eval_champion.json").exists()


def test_rehearsal_reads_its_own_model_card_and_the_policy_check_runs(run):
    _, out, _ = run
    leak = json.loads((out / "eval_leakage_tests.json").read_text(encoding="utf-8"))
    assert leak["a_feature_policy"]["status"] == "PASS"
    assert leak["a_feature_policy"]["n_one_hot_columns"] == 2
