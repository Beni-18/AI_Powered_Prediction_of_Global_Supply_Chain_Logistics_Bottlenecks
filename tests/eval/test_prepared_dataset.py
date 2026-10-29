"""Checks on Role 3's prepared Stage 1 dataset: labels, split order and the purge gap.

Why this file exists: every Stage 1 number depends on the split being clean. These tests are skipped
until Stage01_ML/data/prepared_dataset.csv[.gz] exists, then they guard the split policy automatically.
"""

import json

import pandas as pd
import pytest

from common.paths import CONTRACTS_DIR, stage_path

CANDIDATES = [stage_path(1, "data", "prepared_dataset.csv.gz"), stage_path(1, "data", "prepared_dataset.csv")]
PREPARED = next((p for p in CANDIDATES if p.exists()), None)

pytestmark = pytest.mark.skipif(PREPARED is None, reason="Stage 1 prepared dataset not published yet")

POLICY = json.loads((CONTRACTS_DIR / "label_policy.json").read_text(encoding="utf-8"))
PURGE = pd.Timedelta(hours=POLICY["split_policy"]["purge_gap_hours"])


@pytest.fixture(scope="module")
def df():
    data = pd.read_csv(PREPARED)
    data["ts"] = pd.to_datetime(data["timestamp"], utc=True)
    return data


def test_labels_are_filled_and_follow_the_policy(df):
    assert df["delay_status"].notna().all()
    cuts = POLICY["thresholds_hours"]
    hours = df["realized_delay_hours"]
    assert (df.loc[df["delay_status"] == "ON_TIME", "realized_delay_hours"] < cuts["ON_TIME_below"]).all()
    assert (df.loc[df["delay_status"] == "CRITICAL_DELAY", "realized_delay_hours"] >= cuts["CRITICAL_DELAY_at_or_above"]).all()
    assert set(df["delay_status"]) <= set(POLICY["classes"]) and hours.notna().all()


def test_three_splits_exist_with_roughly_70_15_15(df):
    share = df["split"].value_counts(normalize=True)
    assert set(share.index) == {"train", "val", "test"}
    # Rows removed by the purge gap shift the shares slightly, so allow a few points either way.
    assert abs(share["train"] - 0.70) < 0.05 and abs(share["val"] - 0.15) < 0.05 and abs(share["test"] - 0.15) < 0.05


def test_splits_are_in_time_order_with_the_purge_gap(df):
    last_train = df.loc[df["split"] == "train", "ts"].max()
    first_val, last_val = df.loc[df["split"] == "val", "ts"].agg(["min", "max"])
    first_test = df.loc[df["split"] == "test", "ts"].min()
    assert first_val - last_train >= PURGE, "train and val are closer than the purge gap"
    assert first_test - last_val >= PURGE, "val and test are closer than the purge gap"


def test_no_shipment_appears_in_two_splits(df):
    assert df.groupby("shipment_id")["split"].nunique().max() == 1


def test_no_duplicate_record_ids(df):
    assert not df["record_id"].duplicated().any()
