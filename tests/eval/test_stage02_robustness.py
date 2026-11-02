"""Tests for the Stage 2 robustness checks in Stage02_DL/04_evaluation_engineer.py: the seed-variance verdict.

Owner: Role 4 (Evaluation Engineer).
"""

import importlib.util

import pandas as pd
import pytest

from common.paths import REPO_ROOT

pytest.importorskip("matplotlib")


def load_eval_script():
    """Import Stage02_DL/04_evaluation_engineer.py as a module (its file name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("stage02_eval", REPO_ROOT / "Stage02_DL" / "04_evaluation_engineer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def per_seed_table(test_skills: dict, val_skills: dict) -> pd.DataFrame:
    """Rows for dwell 24 h: {seed: {model: skill}} for test and val."""
    rows = []
    for split, table in (("test", test_skills), ("val", val_skills)):
        for seed, by_model in table.items():
            for model, skill in by_model.items():
                rows.append({"seed": seed, "model": model, "split": split, "target": "dwell", "horizon_h": 24,
                             "mae": 1.0, "skill_vs_persistence": skill})
    return pd.DataFrame(rows)


def test_seed_verdict_reports_lead_and_validation_pick_per_seed():
    """The champion's lead over xgb_lag is computed per seed, and the validation pick is reported for each seed."""
    ev = load_eval_script()
    test = {42: {"ensemble": 0.17, "transformer": 0.16, "xgb_lag": 0.13},
            7: {"ensemble": 0.16, "transformer": 0.17, "xgb_lag": 0.14}}
    val = {42: {"ensemble": 0.18, "transformer": 0.17, "xgb_lag": 0.16},
           7: {"ensemble": 0.17, "transformer": 0.18, "xgb_lag": 0.16}}
    v = ev.seed_verdict(per_seed_table(test, val), "ensemble")
    assert v["by_seed"][42]["champion_lead_over_xgb_dwell_24h"] == pytest.approx(0.04)
    assert v["by_seed"][7]["champion_lead_over_xgb_dwell_24h"] == pytest.approx(0.02)
    assert v["lead_positive_every_seed"] is True
    assert v["by_seed"][42]["val_pick"] == "ensemble" and v["by_seed"][7]["val_pick"] == "transformer"
    assert v["same_val_pick_every_seed"] is False


def test_seed_verdict_flags_a_lead_that_is_not_positive_for_every_seed():
    """If one seed has the champion behind xgb_lag, the verdict says so instead of hiding it."""
    ev = load_eval_script()
    test = {42: {"ensemble": 0.15, "xgb_lag": 0.13}, 7: {"ensemble": 0.12, "xgb_lag": 0.13}}
    val = {42: {"ensemble": 0.18, "xgb_lag": 0.16}, 7: {"ensemble": 0.17, "xgb_lag": 0.16}}
    v = ev.seed_verdict(per_seed_table(test, val), "ensemble")
    assert v["lead_positive_every_seed"] is False
    assert v["champion_lead_min"] == pytest.approx(-0.01)
