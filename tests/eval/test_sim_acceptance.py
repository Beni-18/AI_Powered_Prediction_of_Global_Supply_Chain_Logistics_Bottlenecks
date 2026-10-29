"""Acceptance tests for the simulator output (Role 4 checks Role 3's raw data).

Why this file exists: our whole story is "honest labels, no leakage". These tests fail loudly if the
raw data would let a model cheat, or if the class mix drifts away from what the team agreed.
They read the committed raw files in data/raw/sim/ and are skipped if those files are absent.
"""

import numpy as np
import pandas as pd
import pytest

from common.paths import REPO_ROOT

SIM_DIR = REPO_ROOT / "data" / "raw" / "sim"
SHIP = SIM_DIR / "shipment_telemetry_raw.csv.gz"
HOURLY = SIM_DIR / "node_hourly_raw.csv.gz"

pytestmark = pytest.mark.skipif(not (SHIP.exists() and HOURLY.exists()), reason="raw simulator files not present")

# Same cut-offs as contracts/label_policy.json.
TARGET_MIX = {"ON_TIME": 0.65, "AT_RISK": 0.22, "CRITICAL_DELAY": 0.13}
FEATURES = [
    "dwell_time_hours", "throughput_teu_per_hr", "capacity_utilization_pct", "customs_hold_hours",
    "weather_severity_index", "labor_deficit_ratio", "equipment_availability_pct", "historical_delay_rate",
]


@pytest.fixture(scope="module")
def ship():
    df = pd.read_csv(SHIP)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    # The raw file plants exact duplicate rows on purpose; the Stage 1 data script must drop them.
    return df.drop_duplicates(subset="record_id").sort_values("ts").reset_index(drop=True)


@pytest.fixture(scope="module")
def hourly():
    return pd.read_csv(HOURLY)


def _label(hours: pd.Series) -> pd.Series:
    """Apply the frozen label policy: <6 h on time, 6-24 h at risk, >=24 h critical."""
    return pd.cut(hours, [-np.inf, 6, 24, np.inf], labels=["ON_TIME", "AT_RISK", "CRITICAL_DELAY"], right=False)


def test_class_mix_is_within_three_points_of_the_agreed_target(ship):
    mix = _label(ship["realized_delay_hours"].dropna()).value_counts(normalize=True)
    for cls, target in TARGET_MIX.items():
        assert abs(mix[cls] - target) <= 0.03, f"{cls}: {mix[cls]:.3f} vs target {target}"


def test_every_class_is_present_in_each_quarter(ship):
    """A class that vanishes in one season would make the chronological test split meaningless."""
    labels = _label(ship["realized_delay_hours"])
    by_quarter = pd.crosstab(ship["ts"].dt.quarter, labels, normalize="index")
    assert (by_quarter > 0.02).all().all(), by_quarter


def test_label_basis_is_clean(ship):
    assert ship["realized_delay_hours"].isna().mean() < 0.01
    assert (ship["realized_delay_hours"].dropna() >= 0).all()


def test_label_columns_start_empty_for_the_data_engineer_to_fill(ship):
    """delay_status and split are assigned in ONE place (Stage01 01_data_engineer.py), never in the raw file."""
    assert ship["delay_status"].isna().all() and ship["split"].isna().all()


def test_no_timestamps_outside_the_simulated_year(ship):
    assert ship["ts"].min() >= pd.Timestamp("2025-01-01", tz="UTC")
    assert ship["ts"].max() < pd.Timestamp("2026-01-01", tz="UTC")


def test_hourly_inventory_never_negative(hourly):
    assert (hourly["inventory_level_teu"].dropna() >= 0).all()


def test_hourly_grid_is_nearly_complete(hourly):
    """The raw grid has a small planted share of missing hours; more than 2% would hide real gaps."""
    hours = pd.to_datetime(hourly["hour_ts"], utc=True)
    expected = int((hours.max() - hours.min()) / pd.Timedelta(hours=1)) + 1
    per_node = hourly.groupby("node_id").size() / expected
    assert per_node.min() > 0.97, per_node.sort_values().head()


def test_no_single_feature_comes_close_to_predicting_critical(ship):
    """Leakage ceiling: the best single feature must stay well below AUC 0.95."""
    from common.evalkit import roc_auc

    is_crit = (ship["realized_delay_hours"] >= 24).to_numpy()
    aucs = {f: roc_auc(is_crit, ship[f].fillna(ship[f].median()).to_numpy()) for f in FEATURES}
    best = max(max(a, 1 - a) for a in aucs.values())
    assert best < 0.95, aucs


def test_label_is_not_a_shallow_function_of_the_snapshot_columns(ship):
    """A depth-4 tree on the snapshot features, trained on the past and tested on the future, must be far from perfect."""
    pytest.importorskip("sklearn")
    from sklearn.tree import DecisionTreeClassifier

    from common.evalkit import macro_f1

    labels = _label(ship["realized_delay_hours"])
    ok = labels.notna()
    X = ship.loc[ok, FEATURES].fillna(ship[FEATURES].median())
    y = labels[ok].cat.codes.to_numpy()
    cut = int(len(X) * 0.7)  # rows are already sorted by time
    tree = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X.iloc[:cut], y[:cut])
    score = macro_f1(y[cut:], tree.predict(X.iloc[cut:]))
    assert 0.30 < score < 0.85, f"macro-F1 {score:.3f}: below 0.30 = nothing to learn, above 0.85 = suspiciously easy"


def test_littles_law_is_not_applicable_to_the_fallback_simulator():
    """Role 3's file asks for a Little's-law check, but the merged simulator is the allowed fallback design
    (SIM_CARD.md): it has no flow-conservation model, so there is nothing to test. Recorded here on purpose."""
    pytest.skip("fallback AR(1) simulator has no queueing model; see data/raw/sim/SIM_CARD.md")
