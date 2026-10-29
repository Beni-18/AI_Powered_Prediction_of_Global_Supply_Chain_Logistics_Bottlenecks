"""Unit tests for Role 3's data code: labels, the split, the adjacency matrix and the window builder."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from common.dataio.graph import load_adjacency
from common.dataio.windows import build_windows
from common.paths import NODES_CSV, REPO_ROOT

# The Stage 1 script name starts with a digit, so it is loaded by file path instead of "import".
_spec = importlib.util.spec_from_file_location("stage01_data", REPO_ROOT / "Stage01_ML" / "01_data_engineer.py")
stage01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage01)

NODE_IDS = pd.read_csv(NODES_CSV)["node_id"].tolist()


def test_delay_status_uses_policy_thresholds():
    """6 h and 24 h are the edges: below 6 on time, 6 to under 24 at risk, 24 and up critical."""
    delays = pd.Series([0.0, 5.99, 6.0, 23.99, 24.0, 500.0])
    expected = ["ON_TIME", "ON_TIME", "AT_RISK", "AT_RISK", "CRITICAL_DELAY", "CRITICAL_DELAY"]
    assert stage01.derive_delay_status(delays).tolist() == expected


def test_split_is_chronological_with_purge_gap():
    """Every train row is earlier than every val row, val earlier than test, and the gap is respected."""
    ts = pd.Series(pd.date_range("2025-01-01", periods=24 * 365, freq="h", tz="UTC"))
    split, in_gap, b = stage01.assign_split(ts)
    assert ts[split == "train"].max() < ts[split == "val"].min() < ts[split == "test"].min()
    # The first val row starts at least 168 h after train_end.
    assert ts[split == "val"].min() >= pd.Timestamp(b["val_start"])
    assert ts[split == "test"].min() >= pd.Timestamp(b["test_start"])
    # Nothing is dropped and the fractions are close to 70 / 15 / 15.
    assert len(split) == len(ts)
    assert abs((split == "train").mean() - 0.70 - in_gap.mean() / 2) < 0.02


def test_adjacency_shape_and_symmetry():
    """Matrix is n_nodes x n_nodes, symmetric when asked, with self-loops on the diagonal."""
    adj = load_adjacency(symmetric=True, self_loops=True, normalise=False)
    assert adj.shape == (len(NODE_IDS), len(NODE_IDS))
    assert np.allclose(adj, adj.T)
    assert np.all(np.diag(adj) == 1.0)
    directed = load_adjacency(symmetric=False, self_loops=False, normalise=False)
    # Shanghai -> Los Angeles is a lane in the contract; the reverse direction is not.
    i, j = NODE_IDS.index("PORT_CN_SHA"), NODE_IDS.index("PORT_US_LAX")
    assert directed[i, j] == 1.0 and directed[j, i] == 0.0


def _tiny_grid(days: int = 60) -> pd.DataFrame:
    """A complete fake grid where every value equals the hour number, so shapes and offsets are easy to check."""
    hours = pd.date_range("2025-01-01", periods=24 * days, freq="h", tz="UTC")
    df = pd.DataFrame([(h, n) for h in hours for n in NODE_IDS], columns=["hour_ts", "node_id"])
    hour_no = (df["hour_ts"] - hours[0]) // pd.Timedelta(hours=1)
    for col in ["avg_dwell_time_hours", "throughput_teu_per_hr"]:
        df[col] = hour_no.astype(float)
    return df


def test_windows_shapes_offsets_and_no_leakage():
    """Shapes match the promise, Y really is h hours ahead, and no window crosses a split boundary."""
    df = _tiny_grid()
    b = {"train_end": "2025-02-01T00:00:00Z", "val_start": "2025-02-08T00:00:00Z",
         "val_end": "2025-02-18T00:00:00Z", "test_start": "2025-02-25T00:00:00Z"}
    feats, targets = ("avg_dwell_time_hours",), ("avg_dwell_time_hours", "throughput_teu_per_hr")
    ws = build_windows(df, input_len=24, horizons=(1, 24), feature_cols=feats, target_cols=targets, boundaries=b)

    n = len(ws.origin_ts)
    assert ws.X.shape == (n, 24, len(NODE_IDS), 1)
    assert ws.Y.shape == (n, 2, len(NODE_IDS), 2)
    # The last input hour + 1 (or + 24) equals the target, because value = hour number.
    assert np.allclose(ws.Y[:, 0, :, 0], ws.X[:, -1, :, 0] + 1)
    assert np.allclose(ws.Y[:, 1, :, 0], ws.X[:, -1, :, 0] + 24)
    assert set(ws.split) == {"train", "val", "test"}

    # Leakage checks: train targets end before train_end; val/test inputs start after their gap.
    origins = pd.to_datetime(ws.origin_ts, utc=True)
    assert (origins[ws.split == "train"] + pd.Timedelta(hours=24) < pd.Timestamp(b["train_end"])).all()
    assert (origins[ws.split == "val"] - pd.Timedelta(hours=23) >= pd.Timestamp(b["val_start"])).all()
    assert (origins[ws.split == "test"] - pd.Timedelta(hours=23) >= pd.Timestamp(b["test_start"])).all()


def test_windows_refuse_incomplete_grid():
    """A missing hour must raise an error instead of silently gluing two days together."""
    df = _tiny_grid(days=10)
    df = df[df["hour_ts"] != df["hour_ts"].iloc[500]]
    b = {"train_end": "2025-01-05T00:00:00Z", "val_start": "2025-01-06T00:00:00Z",
         "val_end": "2025-01-08T00:00:00Z", "test_start": "2025-01-09T00:00:00Z"}
    with pytest.raises(ValueError):
        build_windows(df, input_len=24, horizons=(1,), feature_cols=("avg_dwell_time_hours",),
                      target_cols=("avg_dwell_time_hours",), boundaries=b)
