"""Tests for common/dataio/node_context.py: node history must come from the PAST only."""

import numpy as np
import pandas as pd

from common.dataio.node_context import MAX_GAP_HOURS, NODE_HISTORY_FEATURES, node_history_features, node_lag


def _hourly(values: dict) -> pd.DataFrame:
    """A tiny hourly table for one node, PORT_X, from {hour_offset: utilisation}."""
    start = pd.Timestamp("2025-03-01", tz="UTC")
    return pd.DataFrame({"hour_ts": [start + pd.Timedelta(hours=h) for h in values],
                         "node_id": "PORT_X", "capacity_utilization_pct": list(values.values())})


def test_uses_the_last_complete_hour_never_the_snapshot_hour():
    hourly = _hourly({8: 50.0, 9: 60.0, 10: 99.0})
    # Snapshot at 10:20: hour 10 is still running, so the newest usable reading is hour 9.
    shipment = pd.DataFrame({"timestamp": ["2025-03-01T10:20:00Z"], "current_node_id": ["PORT_X"]})
    out = node_lag(shipment, ["capacity_utilization_pct"], hourly=hourly)
    assert out.iloc[0, 0] == 60.0


def test_missing_hour_falls_back_to_an_older_reading_but_not_a_stale_one():
    hourly = _hourly({0: 40.0})
    near = pd.DataFrame({"timestamp": ["2025-03-01T03:05:00Z"], "current_node_id": ["PORT_X"]})
    far = pd.DataFrame({"timestamp": [f"2025-03-01T{MAX_GAP_HOURS + 3:02d}:05:00Z"], "current_node_id": ["PORT_X"]})
    assert node_lag(near, ["capacity_utilization_pct"], hourly=hourly).iloc[0, 0] == 40.0
    assert np.isnan(node_lag(far, ["capacity_utilization_pct"], hourly=hourly).iloc[0, 0])


def test_unknown_node_stays_missing_and_order_is_kept():
    hourly = _hourly({0: 40.0, 1: 41.0, 2: 42.0})
    rows = pd.DataFrame({"timestamp": ["2025-03-01T03:00:00Z", "2025-03-01T02:30:00Z", "2025-03-01T03:00:00Z"],
                         "current_node_id": ["PORT_X", "PORT_X", "PORT_NOWHERE"]})
    out = node_lag(rows, ["capacity_utilization_pct"], hourly=hourly)["lag_capacity_utilization_pct"].tolist()
    assert out[:2] == [42.0, 41.0] and np.isnan(out[2])


def test_real_table_gives_every_history_feature_for_a_known_shipment():
    shipment = pd.DataFrame({"timestamp": ["2025-06-10T12:30:00Z"], "current_node_id": ["PORT_CN_SHA"],
                             "destination_port": ["PORT_US_LAX"]})
    out = node_history_features(shipment)
    assert all(col in out.columns for col in NODE_HISTORY_FEATURES)
    assert out[NODE_HISTORY_FEATURES].notna().all(axis=None)
