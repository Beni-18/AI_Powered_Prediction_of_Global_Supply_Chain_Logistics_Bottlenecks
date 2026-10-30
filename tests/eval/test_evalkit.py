"""Unit tests for common/evalkit. Each test checks a small case whose right answer we can work out by hand."""

import numpy as np
import pandas as pd
import pytest

from common.evalkit import (
    average_precision, balanced_accuracy, bayes_decision, block_bootstrap_ci, brier, choose_threshold,
    coverage_and_width, decide_with_threshold, ece, episodes_from_series, expected_cost, lead_time, ledger,
    load_cost_matrix, log_loss, macro_f1, paired_bootstrap_diff, pinball_loss, reliability_table, roc_auc,
    scaled_cost_matrix, skill_score,
)
from common.evalkit import leakage, lead_time_by_node

CM = load_cost_matrix()


# ---------- costs ----------

def test_cost_matrix_matches_the_policy():
    assert CM.shape == (3, 3)
    assert CM[2, 0] == 10.0 and CM[0, 2] == 1.0 and np.trace(CM) == 0


def test_expected_cost_by_hand():
    # One missed critical (10) + one false alarm (1) over four shipments = 11 / 4.
    y_true = ["CRITICAL_DELAY", "ON_TIME", "ON_TIME", "AT_RISK"]
    y_pred = ["ON_TIME", "CRITICAL_DELAY", "ON_TIME", "AT_RISK"]
    assert expected_cost(y_true, y_pred, CM) == pytest.approx(11 / 4)


def test_bayes_decision_prefers_alerting_when_a_miss_is_expensive():
    # 15% critical is far below 50%, yet saying ON_TIME costs 1.6 on average, so the cheap hedge is AT_RISK (0.69).
    assert bayes_decision(np.array([[0.80, 0.05, 0.15]]), CM)[0] == 1
    # At 25% critical a full CRITICAL alert (0.725) beats the hedge (0.96).
    assert bayes_decision(np.array([[0.70, 0.05, 0.25]]), CM)[0] == 2
    # A shipment that is 97% on time needs no action (0.14 vs 0.32 for the hedge).
    assert bayes_decision(np.array([[0.97, 0.02, 0.01]]), CM)[0] == 0


def test_decide_with_threshold():
    proba = np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]])
    assert list(decide_with_threshold(proba, 0.5)) == [0, 1, 2]
    assert list(decide_with_threshold(proba, 0.15)) == [0, 2, 2]


def test_choose_threshold_is_below_half_when_misses_are_costly():
    rng = np.random.default_rng(0)
    n = 4000
    truth = rng.choice(3, size=n, p=[0.65, 0.24, 0.11])
    # A reasonably informative but noisy probability model.
    logits = rng.normal(size=(n, 3)) + 2.0 * np.eye(3)[truth]
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y_true = np.array(["ON_TIME", "AT_RISK", "CRITICAL_DELAY"])[truth]
    thr, table = choose_threshold(y_true, proba, CM)
    assert thr < 0.5
    assert table["expected_cost"].min() == pytest.approx(table.loc[table["threshold"] == thr, "expected_cost"].iloc[0])


def test_higher_miss_cost_never_raises_the_threshold():
    rng = np.random.default_rng(1)
    n = 3000
    truth = rng.choice(3, size=n, p=[0.65, 0.24, 0.11])
    logits = rng.normal(size=(n, 3)) + 1.5 * np.eye(3)[truth]
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y_true = np.array(["ON_TIME", "AT_RISK", "CRITICAL_DELAY"])[truth]
    thresholds = [choose_threshold(y_true, proba, scaled_cost_matrix(m))[0] for m in (3, 5, 10, 20, 40)]
    assert thresholds == sorted(thresholds, reverse=True)


def test_scaled_matrix_keeps_other_cells():
    scaled = scaled_cost_matrix(40)
    assert scaled[2, 0] == 40 and scaled[2, 1] == pytest.approx(12.0)
    assert scaled[0, 2] == CM[0, 2]


# ---------- classification metrics ----------

def test_macro_f1_and_balanced_accuracy_on_a_tiny_case():
    t = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 0, 1, 2, 2, 2])
    # class0 F1=1; class1 P=1 R=.5 F1=2/3; class2 P=2/3 R=1 F1=.8
    assert macro_f1(t, p) == pytest.approx((1 + 2 / 3 + 0.8) / 3)
    assert balanced_accuracy(t, p) == pytest.approx((1 + 0.5 + 1) / 3)


def test_average_precision_and_auc_extremes():
    pos = np.array([1, 1, 0, 0, 0])
    assert average_precision(pos, np.array([0.9, 0.8, 0.3, 0.2, 0.1])) == pytest.approx(1.0)
    assert roc_auc(pos, np.array([0.9, 0.8, 0.3, 0.2, 0.1])) == pytest.approx(1.0)
    assert roc_auc(pos, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == pytest.approx(0.0)
    assert roc_auc(pos, np.full(5, 0.5)) == pytest.approx(0.5)


def test_log_loss_is_small_for_confident_correct_answers():
    t = np.array([0, 2])
    good = np.array([[0.98, 0.01, 0.01], [0.01, 0.01, 0.98]])
    bad = np.array([[0.01, 0.01, 0.98], [0.98, 0.01, 0.01]])
    assert log_loss(t, good) < 0.05 < log_loss(t, bad)


# ---------- calibration ----------

def test_perfectly_calibrated_model_has_tiny_ece():
    rng = np.random.default_rng(3)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    assert ece(y, p) < 0.02


def test_overconfident_model_has_large_ece():
    rng = np.random.default_rng(4)
    p = np.full(5000, 0.9)
    y = (rng.uniform(size=5000) < 0.5).astype(int)
    assert ece(y, p) == pytest.approx(0.4, abs=0.03)


def test_reliability_table_counts_every_row_and_handles_probability_one():
    table = reliability_table(np.array([1, 0, 1]), np.array([1.0, 0.0, 0.55]))
    assert table["n"].sum() == 3
    assert brier(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0


# ---------- bootstrap ----------

def test_block_bootstrap_interval_contains_the_point_and_is_repeatable():
    rng = np.random.default_rng(5)
    x = rng.normal(loc=2.0, size=600)
    blocks = np.repeat(np.arange(60), 10)
    a = block_bootstrap_ci(np.mean, x, blocks=blocks, n=300, seed=42)
    b = block_bootstrap_ci(np.mean, x, blocks=blocks, n=300, seed=42)
    assert a == b
    assert a[1] <= a[0] <= a[2]


def test_block_bootstrap_is_wider_when_blocks_are_correlated():
    rng = np.random.default_rng(6)
    blocks = np.repeat(np.arange(50), 20)
    day_effect = rng.normal(size=50)[blocks]
    correlated = day_effect + 0.1 * rng.normal(size=1000)
    independent = rng.normal(size=1000)
    w_corr = np.subtract(*block_bootstrap_ci(np.mean, correlated, blocks=blocks, n=300)[:0:-1])
    w_ind = np.subtract(*block_bootstrap_ci(np.mean, independent, blocks=blocks, n=300)[:0:-1])
    assert w_corr > w_ind


def test_paired_bootstrap_detects_a_clearly_better_model():
    rng = np.random.default_rng(7)
    truth = rng.choice(3, size=900, p=[0.6, 0.25, 0.15])
    good = np.where(rng.uniform(size=900) < 0.9, truth, rng.integers(0, 3, 900))
    weak = np.where(rng.uniform(size=900) < 0.5, truth, rng.integers(0, 3, 900))
    blocks = np.repeat(np.arange(90), 10)
    res = paired_bootstrap_diff(macro_f1, (truth, good), (truth, weak), blocks, n=200)
    assert res["lo"] > 0 and res["p_a_not_better"] < 0.05


# ---------- forecast scores ----------

def test_skill_score_basic_cases():
    assert skill_score(2.0, 4.0) == pytest.approx(0.5)
    assert skill_score(4.0, 4.0) == 0.0  # persistence against itself must be zero skill
    assert skill_score(8.0, 4.0) == pytest.approx(-1.0)


def test_coverage_width_and_pinball():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    cov, width = coverage_and_width(y, y - 1, y + 1)
    assert cov == 1.0 and width == 2.0
    cov, _ = coverage_and_width(y, np.array([0, 0, 0, 0.0]), np.array([5, 5, 5, 5.0]))
    assert cov == 0.75
    # Perfect forecast has zero pinball loss; under-forecast at q=0.9 costs 0.9 per unit.
    assert pinball_loss(y, y, 0.9) == 0.0
    assert pinball_loss(np.array([10.0]), np.array([9.0]), 0.9) == pytest.approx(0.9)


# ---------- episodes and lead time ----------

def _hourly(n, start="2025-01-01"):
    return pd.date_range(start, periods=n, freq="h")


def test_episode_needs_six_consecutive_hours():
    t = _hourly(48)
    v = np.zeros(48)
    v[10:15] = 9  # only 5 hours: not an episode
    v[30:40] = 9  # 10 hours: an episode starting at hour 30
    onsets = episodes_from_series(t, v, threshold=5)
    assert onsets == [t[30]]


def test_close_runs_are_merged_into_one_episode():
    t = _hourly(120)
    v = np.zeros(120)
    v[10:20] = 9
    v[30:40] = 9   # starts 10 h after the previous run ended: merged
    v[90:100] = 9  # far away: new episode
    assert episodes_from_series(t, v, threshold=5) == [t[10], t[90]]


def test_lead_time_catches_misses_and_false_alarms():
    onsets = [pd.Timestamp("2025-01-10 00:00"), pd.Timestamp("2025-02-10 00:00")]
    alerts = [pd.Timestamp("2025-01-08 00:00"),   # 48 h before onset 1: caught
              pd.Timestamp("2025-01-09 12:00"),   # also before onset 1; first alert wins
              pd.Timestamp("2025-03-01 00:00")]   # no onset follows: false alarm
    res = lead_time(alerts, onsets)
    assert res["n_episodes"] == 2 and res["n_caught"] == 1
    assert res["episode_recall"] == 0.5
    assert res["median_lead_h"] == pytest.approx(48.0)
    assert res["false_alarms"] == 1


def test_alert_outside_lookback_does_not_count_as_a_catch():
    onsets = [pd.Timestamp("2025-01-20 00:00")]
    alerts = [pd.Timestamp("2025-01-01 00:00")]  # 19 days earlier, beyond 168 h
    res = lead_time(alerts, onsets)
    assert res["n_caught"] == 0 and res["false_alarms"] == 1


# ---------- ledger ----------

def _blank_ledger(tmp_path):
    path = tmp_path / "LEDGER.md"
    path.write_text("# Ledger\n\n" + ledger.BEGIN + "\n" + "\n".join(ledger.HEADER) + "\n" + ledger.END + "\n\nfooter\n", encoding="utf-8")
    return path


def test_ledger_upsert_inserts_then_replaces_one_row(tmp_path):
    path = _blank_ledger(tmp_path)
    ledger.upsert("C-ML-02", 1, "second claim", "0.5", "a.json", "python run.py", path=path)
    ledger.upsert("C-ML-01", 1, "first claim", "0.4 [0.3, 0.5]", "b.json", "python run.py", path=path)
    ledger.upsert("C-ML-01", 1, "first claim", "0.45 [0.4, 0.5]", "b.json", "python run.py", status="VERIFIED", path=path)
    rows = ledger.read_rows(path)
    assert [r["id"] for r in rows] == ["C-ML-01", "C-ML-02"]
    assert rows[0]["value"] == "0.45 [0.4, 0.5]" and rows[0]["status"] == "VERIFIED" and rows[0]["verified_on"]
    assert path.read_text(encoding="utf-8").endswith("footer\n")


def test_ledger_rejects_unknown_status_and_survives_pipes(tmp_path):
    path = _blank_ledger(tmp_path)
    with pytest.raises(ValueError):
        ledger.upsert("X", 1, "c", "v", "s", "r", status="MAYBE", path=path)
    ledger.upsert("X", 1, "a | b", "v", "s", "r", path=path)
    assert ledger.read_rows(path)[0]["claim"] == "a / b"


def test_real_ledger_file_has_markers():
    text = ledger.LEDGER_PATH.read_text(encoding="utf-8")
    assert ledger.BEGIN in text and ledger.END in text


# ---------- fixes from the code review ----------

def test_macro_f1_ignores_a_class_that_never_occurs():
    # No CRITICAL anywhere in this small slice: it must not drag the average down with a fake zero.
    t = np.array([0, 0, 1, 1])
    p = np.array([0, 0, 1, 1])
    assert macro_f1(t, p) == 1.0
    # But predicting a class that never occurs is a real error and does count.
    assert macro_f1(np.array([0, 0, 1, 1]), np.array([0, 2, 1, 1])) < 1.0


def test_average_precision_does_not_depend_on_row_order_when_scores_tie():
    is_pos = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    score = np.array([0.7, 0.7, 0.7, 0.7, 0.2, 0.2, 0.2, 0.2])
    reference = average_precision(is_pos, score)
    rng = np.random.default_rng(0)
    for _ in range(10):
        order = rng.permutation(len(score))
        assert average_precision(is_pos[order], score[order]) == pytest.approx(reference)
    # Hand check: 2 of 4 rows at 0.7 are positive -> precision 0.5 at recall 2/3; then 3 of 8 at recall 1.
    assert reference == pytest.approx(0.5 * (2 / 3) + (3 / 8) * (1 / 3))


def test_a_missing_hour_inside_an_episode_does_not_split_it():
    t = pd.date_range("2025-01-01", periods=48, freq="h")
    v = np.zeros(48)
    v[10:20] = 9
    keep = np.ones(48, dtype=bool)
    keep[14] = False  # one hour missing from the feed
    assert episodes_from_series(t[keep], v[keep], threshold=5) == [t[10]]


def test_alerts_at_one_node_are_not_credited_to_another_node():
    onset = pd.Timestamp("2025-01-10")
    alerts = pd.DataFrame({"node_id": ["PORT_A"], "time": [pd.Timestamp("2025-01-08")]})
    onsets = pd.DataFrame({"node_id": ["PORT_B"], "time": [onset]})
    res = lead_time_by_node(alerts, onsets)
    assert res["n_caught"] == 0 and res["false_alarms"] == 1
    same = lead_time_by_node(alerts.assign(node_id="PORT_B"), onsets)
    assert same["n_caught"] == 1 and same["median_lead_h"] == pytest.approx(48.0)


def test_single_feature_ceiling_is_partial_when_features_could_not_be_tested():
    df = pd.DataFrame({"a": np.arange(100.0), "carrier": ["X", "Y"] * 50})
    crit = (np.arange(100) % 7 == 0)
    res = leakage.single_feature_ceiling(df, ["a", "carrier", "lag_a"], crit)
    assert res["status"] == "PARTIAL" and set(res["untested_features"]) == {"carrier", "lag_a"}


def test_a_feature_that_nearly_equals_the_answer_fails_the_ceiling():
    rng = np.random.default_rng(1)
    crit = rng.uniform(size=2000) < 0.15
    df = pd.DataFrame({"leaky": crit + rng.normal(scale=0.05, size=2000), "honest": rng.normal(size=2000)})
    assert leakage.single_feature_ceiling(df, ["leaky", "honest"], crit)["status"] == "FAIL"


def test_ids_that_follow_the_clock_pass_but_ids_that_encode_the_label_fail():
    n = 3000
    rng = np.random.default_rng(2)
    crit = rng.uniform(size=n) < 0.2
    ts = pd.date_range("2025-01-01", periods=n, freq="h").strftime("%Y-%m-%dT%H:%M:%SZ")
    honest = pd.DataFrame({"record_id": [f"R{i:06d}" for i in range(n)], "timestamp": ts})
    assert leakage.bookkeeping_ceiling(honest, crit)["status"] == "PASS"
    # Ids handed out so that critical rows get the high numbers.
    order = np.argsort(crit.astype(float) + rng.uniform(size=n) * 0.5)
    leaky = honest.copy()
    leaky["record_id"] = [f"R{i:06d}" for i in np.argsort(order)]
    assert leakage.bookkeeping_ceiling(leaky, crit)["status"] == "FAIL"


# ---------- fixes after Role 1's real predictions arrived ----------

def test_one_hot_columns_of_text_features_are_allowed():
    res = leakage.feature_policy_check(["dwell_time_hours", "carrier_code_MSC", "current_node_id_CHOKE_SUEZ",
                                        "priority_tier_EXPEDITED", "cal_month", "roll_node_dwell_mean_24h"])
    assert res["status"] == "PASS" and res["n_one_hot_columns"] == 3


def test_one_hot_trick_cannot_smuggle_in_forbidden_or_numeric_expansions():
    res = leakage.feature_policy_check(["dwell_time_hours_next", "shipment_id_SHP-100001", "split_test", "realized_delay_hours"])
    assert res["status"] == "FAIL" and len(res["illegal_features"]) == 4


def test_role_1_baseline_names_are_recognised():
    import importlib.util

    from common.paths import REPO_ROOT

    spec = importlib.util.spec_from_file_location("eval04", REPO_ROOT / "Stage01_ML" / "04_evaluation_engineer.py")
    eval04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval04)
    assert all(eval04.is_baseline(n) for n in ("majority", "rule_dwell", "logreg"))
    assert not any(eval04.is_baseline(n) for n in ("xgboost", "xgboost_cal", "random_forest"))
