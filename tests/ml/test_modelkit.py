"""ROLE: ML ENGINEER  |  TESTS
Unit tests for common/modelkit utilities (feature guard, model card, split loader, latency probe).
Owner: Role 1 - ML Engineer
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from common.modelkit import (
    assert_features_allowed,
    get_feature_policy,
    load_split,
    save_model_card,
    latency_probe,
)


def test_assert_features_allowed_accepts_valid_base():
    """Verify that allowed base features pass validation without error."""
    valid_cols = ["dwell_time_hours", "throughput_teu_per_hr", "capacity_utilization_pct", "carrier_code"]
    # This should not raise an exception
    assert_features_allowed(valid_cols)


def test_assert_features_allowed_accepts_valid_derived():
    """Verify that approved prefixes (cal_, lag_, roll_, eng_) pass validation."""
    valid_derived = [
        "cal_hour",
        "cal_dow",
        "lag_dwell_1h",
        "roll_node_dwell_mean_24h",
        "eng_util_x_customs",
        "eng_throughput_per_util",
    ]
    assert_features_allowed(valid_derived)


def test_assert_features_allowed_rejects_forbidden_columns():
    """Verify that target and leakage columns trigger ValueError."""
    forbidden_examples = [
        ["dwell_time_hours", "realized_delay_hours"],
        ["dwell_time_hours", "delay_status"],
        ["dwell_time_hours", "record_id"],
        ["dwell_time_hours", "split"],
        ["dwell_time_hours", "timestamp"],
    ]
    for cols in forbidden_examples:
        with pytest.raises(ValueError, match="Zero-leakage policy violation"):
            assert_features_allowed(cols)


def test_assert_features_allowed_rejects_unapproved_columns():
    """Verify that random unknown columns not in policy fail validation."""
    unapproved = ["dwell_time_hours", "random_unregistered_feature"]
    with pytest.raises(ValueError, match="Unapproved feature violation"):
        assert_features_allowed(unapproved)


def test_load_split():
    """Verify that load_split isolates rows cleanly without cross-partition bleeding."""
    data = pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R4"],
        "dwell_time_hours": [10.0, 20.0, 30.0, 40.0],
        "split": ["train", "train", "val", "test"],
    })

    train_df = load_split(data, "train")
    assert len(train_df) == 2
    assert set(train_df["record_id"]) == {"R1", "R2"}

    val_df = load_split(data, "val")
    assert len(val_df) == 1
    assert list(val_df["record_id"]) == ["R3"]

    test_df = load_split(data, "test")
    assert len(test_df) == 1
    assert list(test_df["record_id"]) == ["R4"]

    with pytest.raises(ValueError, match="is invalid"):
        load_split(data, "invalid_split")


def test_save_model_card(tmp_path: Path):
    """Verify that save_model_card writes a complete, parseable JSON document."""
    card_path = save_model_card(
        output_dir=tmp_path,
        name="test_model",
        version="v1.0",
        features=["dwell_time_hours", "capacity_utilization_pct"],
        train_window={"rows": 100},
        metrics_on_val={"macro_f1": 0.85, "expected_cost": 1.2},
        notes="Unit test card",
    )
    assert card_path.exists()

    with open(card_path, "r", encoding="utf-8") as f:
        card = json.load(f)

    assert card["model_name"] == "test_model"
    assert card["model_version"] == "v1.0"
    assert card["metrics_on_val"]["macro_f1"] == 0.85
    assert "library_versions" in card
    assert card["seed"] == 42


def test_latency_probe():
    """Verify that latency_probe accurately computes timing percentiles."""
    def dummy_task():
        # Quick computation
        _ = sum(i * i for i in range(100))

    stats = latency_probe(dummy_task, n=20, warmup=2)
    assert "p50_ms" in stats
    assert "p95_ms" in stats
    assert stats["p95_ms"] >= stats["p50_ms"]
    assert stats["iterations"] == 20
