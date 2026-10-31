"""ROLE: ML ENGINEER  |  TESTS
Leakage and correctness tests for the Stage 2 data pipeline (common/modelkit/dl_data.py, dl_train.py).

The most important test here: changing EVERY value after a forecast origin must not change what the model sees.
Owner: Role 1 - ML Engineer. Written by Role 4 with the new pipeline; Role 1 to review.
"""

import numpy as np
import pandas as pd
import pytest

# The training helpers import PyTorch (Stage 02 only); skip on machines and CI runners without it.
pytest.importorskip("torch")

from common.dataio.hourly_grid import CHANNELS  # noqa: E402
from common.dataio.windows import build_windows  # noqa: E402
from common.modelkit.dl_data import (  # noqa: E402
    SIM_TRACK, ForecastData, shuffle_permutation, split_origins, symmetric_adjacency, transit_edges,
)
from common.modelkit.dl_train import apply_margins, conformal_margins, naive_point  # noqa: E402
from common.paths import EDGES_CSV, NODES_CSV  # noqa: E402

NODES = pd.read_csv(NODES_CSV)["node_id"].tolist()
EDGES = pd.read_csv(EDGES_CSV)
T_HOURS = 24 * 60  # 60 days of hours


def _scaler():
    """Identity scaler (median 0, IQR 1) so scaled values equal raw values in the checks."""
    return {"nodes": {n: {c: {"median": 0.0, "iqr": 1.0} for c in CHANNELS} for n in NODES}}


def _data(values=None) -> ForecastData:
    """60 days x 27 nodes of random data with an identity scaler."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=(T_HOURS, len(NODES), len(CHANNELS))) if values is None else values
    times = pd.date_range("2025-01-01", periods=T_HOURS, freq="h", tz="UTC")
    return ForecastData(values, np.zeros(values.shape[:2], bool), times, NODES, SIM_TRACK, _scaler())


def test_inputs_never_contain_the_future():
    """Scramble every hour AFTER the origin: x, ctx and anchor must be bit-for-bit identical."""
    base = _data()
    origin = 24 * 40
    changed_values = base.raw.copy()
    changed_values[origin + 1:] = 999.0
    changed = _data(changed_values)
    a, b = base.batch(np.array([origin])), changed.batch(np.array([origin]))
    for key in ("x", "ctx", "anchor"):
        assert np.array_equal(a[key], b[key]), f"{key} reads hours after the origin"
    assert not np.array_equal(a["y"], b["y"])  # the targets, of course, do change


def test_batch_shapes_and_targets_line_up():
    """y at horizon h is the raw value h hours after the origin."""
    d = _data()
    origin = 24 * 30
    batch = d.batch(np.array([origin]))
    assert batch["x"].shape == (1, len(NODES), 72, d.n_inputs)
    assert batch["ctx"].shape == (1, len(NODES), SIM_TRACK.n_blocks, len(CHANNELS))
    for hi, h in enumerate(SIM_TRACK.horizons):
        assert np.allclose(batch["y"][0, :, hi, 0], d.raw[origin + h, :, 0])
    assert np.allclose(batch["anchor"][0, :, 0], d.raw[origin, :, 0])


def test_long_context_block_zero_is_the_last_24_hours():
    """ctx block 0 = mean of the 24 hours ending at the origin (inclusive)."""
    d = _data()
    origin = 24 * 30
    ctx = d.batch(np.array([origin]))["ctx"]
    assert np.allclose(ctx[0, :, 0, :], d.raw[origin - 23:origin + 1].mean(axis=0), atol=1e-5)


def test_split_rule_matches_the_shared_window_builder():
    """split_origins gives the same split labels as common/dataio/windows.build_windows for every origin."""
    times = pd.date_range("2025-01-01", periods=T_HOURS, freq="h", tz="UTC")
    b = {"train_end": "2025-01-25T00:00:00Z", "val_start": "2025-02-01T00:00:00Z",
         "val_end": "2025-02-12T00:00:00Z", "test_start": "2025-02-19T00:00:00Z"}
    rows = [{"hour_ts": t, "node_id": n, **{c: 1.0 for c in CHANNELS}} for t in times for n in NODES]
    ws = build_windows(pd.DataFrame(rows), boundaries=b)
    ours = split_origins(times, SIM_TRACK, b)
    for split in ("train", "val", "test"):
        expected = set(pd.DatetimeIndex(ws.origin_ts[ws.split == split]))
        assert set(times[ours[split]]) == expected


def test_transit_messages_read_the_right_past_block():
    """A 14-day lane: for h = 1 h the source state from 335 h ago (block 13); for h = 168 h, 168 h ago (block 7)."""
    edges = pd.DataFrame([{"src": NODES[0], "dst": NODES[1], "typical_transit_days": 14, "bidirectional": False},
                          {"src": NODES[2], "dst": NODES[3], "typical_transit_days": 1, "bidirectional": False}])
    e = transit_edges(NODES, edges, SIM_TRACK)
    assert e["lag_steps"].tolist() == [336, 24]
    assert e["block"][0].tolist() == [335 // 24, 330 // 24, 312 // 24, 264 // 24, 168 // 24]
    # 1-day lane: congestion for h >= 72 h has not left the source yet -> flagged as a nowcast.
    assert e["future"][1].tolist() == [0, 0, 0, 1, 1]


def test_shuffled_graph_keeps_degrees_but_moves_every_lane():
    """The shuffled ablation has the same degree distribution but no node keeps its own label."""
    perm = shuffle_permutation(len(NODES))
    assert not np.any(perm == np.arange(len(NODES)))
    full = symmetric_adjacency(NODES, EDGES) > 0
    shuf = symmetric_adjacency(NODES, EDGES, permutation=perm) > 0
    assert sorted(full.sum(axis=1)) == sorted(shuf.sum(axis=1))
    assert not np.array_equal(full, shuf)
    assert np.allclose(symmetric_adjacency(NODES, EDGES), symmetric_adjacency(NODES, EDGES).T)


def test_baselines_use_only_the_past():
    """Persistence = value at the origin; seasonal naive for h <= 168 reads t + h - 168 (never after t)."""
    d = _data()
    origin = np.array([24 * 30])
    pers = naive_point(d, origin, "persistence")
    assert np.allclose(pers[0, :, 0, 0], d.raw[origin[0], :, 0])
    seas = naive_point(d, origin, "seasonal_naive")
    for hi, h in enumerate(SIM_TRACK.horizons):
        assert np.allclose(seas[0, :, hi, 0], d.raw[origin[0] + h - 168, :, 0])


def test_conformal_margins_reach_the_target_coverage():
    """Intervals that are far too narrow get widened until about 80% of calibration targets are inside."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=(400, 3, 1, 1))
    q = np.stack([np.full_like(y, -0.1), np.zeros_like(y), np.full_like(y, 0.1)], axis=-1)
    mask = np.ones((400, 3, 1), dtype=bool)
    m = conformal_margins(q, y, mask)
    q2 = apply_margins(q, m)
    coverage = ((y >= q2[..., 0]) & (y <= q2[..., 2])).mean()
    assert coverage == pytest.approx(0.80, abs=0.02)
