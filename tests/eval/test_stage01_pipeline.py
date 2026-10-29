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
         "--prepared", str(prepared_path), "--out-dir", str(out), "--n-boot", "40"],
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
    assert leak["b_shuffle_label"]["status"] == "PASS"
    assert leak["c_single_feature_ceiling"]["status"] == "PASS"
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
