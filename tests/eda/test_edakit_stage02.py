"""Unit tests for the Stage 2 EDA helpers (common.edakit.timeseries, common.edakit.attribution) and the
lead-lag scan of Stage02_DL/02_eda_engineer.py, on small synthetic series where the right answer is known.
"""

import importlib.util
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest

from common.edakit.timeseries import (
    classify_excursion, deseasonalise, lagged_correlation, mad_sigma, remove_common_factor, run_lengths,
    trailing_residual, weekly_profile,
)
from common.paths import REPO_ROOT


def hourly_index(hours: int) -> pd.DatetimeIndex:
    """UTC hourly timestamps starting on a Monday."""
    return pd.date_range("2025-01-06", periods=hours, freq="h", tz="UTC")


def test_weekly_profile_removes_a_pure_weekly_pattern():
    """A series that is only a weekly pattern becomes flat zero after deseasonalising."""
    idx = hourly_index(168 * 6)
    pattern = np.sin(2 * np.pi * np.arange(len(idx)) / 168) + 0.5 * np.sin(2 * np.pi * np.arange(len(idx)) / 24)
    frame = pd.DataFrame({"a": pattern}, index=idx)
    train = np.arange(len(idx)) < 168 * 4
    out = deseasonalise(frame, weekly_profile(frame, train))
    assert np.abs(out["a"].to_numpy()).max() < 1e-9


def test_trailing_residual_never_uses_future_hours():
    """Changing a later hour must not change the residual of earlier hours (past-only rule)."""
    idx = hourly_index(600)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=600)}, index=idx)
    before = trailing_residual(frame, window_hours=48)
    changed = frame.copy()
    changed.iloc[500:, 0] += 100.0
    after = trailing_residual(changed, window_hours=48)
    pd.testing.assert_series_equal(before["a"].iloc[:500], after["a"].iloc[:500])


def test_mad_sigma_matches_standard_deviation_for_normal_noise_and_ignores_spikes():
    """MAD x 1.4826 is ~1 for standard normal noise, and a few huge spikes barely move it."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=20000)
    assert abs(mad_sigma(x) - 1.0) < 0.03
    spiky = x.copy()
    spiky[:50] = 1000.0
    assert abs(mad_sigma(spiky) - mad_sigma(x)) < 0.03


def test_run_lengths():
    """Runs of True are found with their start and length."""
    assert run_lengths([False, True, True, False, True]) == [(1, 2), (4, 1)]
    assert run_lengths([False, False]) == []


@pytest.mark.parametrize("residual,raw,label", [
    (np.r_[np.zeros(10), np.full(30, 5.0), np.zeros(10)], np.full(50, 5.0), "STRUCTURAL"),
    (np.r_[np.zeros(10), np.full(5, 5.0), np.zeros(35)], np.full(50, 5.0), "TRANSIENT"),
    (np.zeros(50), np.full(50, 20.0), "SEASONAL_PEAK"),
    (np.zeros(50), np.full(50, 5.0), "NORMAL"),
])
def test_classify_excursion_follows_the_rule(residual, raw, label):
    """3 x noise floor for >= 24 h = structural; shorter = transient; alert line crossed without a jump = seasonal."""
    out = classify_excursion(residual, raw, noise_floor=1.0, alert_line=10.0)
    assert out["label"] == label


def test_lagged_correlation_finds_a_planted_delay():
    """b copies a 7 steps later, so the correlation peaks at lag 7."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=3000)
    b = np.r_[rng.normal(size=7), a[:-7]] + 0.3 * rng.normal(size=3000)
    assert int(np.argmax(lagged_correlation(a, b, 30))) == 7


def test_remove_common_factor_leaves_rows_that_average_to_zero():
    """After removing the network average, every hour's values average to zero."""
    frame = pd.DataFrame(np.random.default_rng(3).normal(size=(50, 4)))
    assert np.allclose(remove_common_factor(frame).mean(axis=1), 0.0)


def test_integrated_gradients_is_exact_for_a_linear_model():
    """For f(x) = w . x and a zero baseline, the attribution of each input is w_i * x_i, and completeness holds."""
    torch = pytest.importorskip("torch")
    from common.edakit.attribution import completeness_gap, integrated_gradients

    w = torch.tensor([1.0, -2.0, 0.0, 3.0])
    x = torch.tensor([[0.5, 1.0, 7.0, -1.0], [2.0, 0.0, 1.0, 1.0]])

    def fn(x):
        """Linear model: one number per row."""
        return (x * w).sum(-1)

    att = integrated_gradients(fn, {"x": x}, steps=8)["x"]
    assert torch.allclose(att, x * w, atol=1e-6)
    assert completeness_gap(fn, {"x": x}, {"x": att}) < 1e-6


def load_eda_script():
    """Import Stage02_DL/02_eda_engineer.py as a module (its file name starts with a digit)."""
    pytest.importorskip("statsmodels")
    pytest.importorskip("matplotlib")
    path = REPO_ROOT / "Stage02_DL" / "02_eda_engineer.py"
    spec = importlib.util.spec_from_file_location("stage02_eda", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lead_lag_scan_recovers_planted_transit_times():
    """A small network where each downstream node copies its upstream node after the lane's transit time."""
    eda = load_eda_script()
    rng = np.random.default_rng(4)
    n = 4000
    src = pd.Series(rng.normal(size=n)).rolling(5, min_periods=1).mean().to_numpy()
    lags = {"B": 10, "C": 25, "D": 40}
    x = {"A": src}
    for node, lag in lags.items():
        x[node] = np.r_[rng.normal(size=lag) * 0.1, src[:-lag]] + 0.2 * rng.normal(size=n)
    x["E"] = rng.normal(size=n)        # an unrelated node for the chance level
    x = pd.DataFrame(x)
    edges = pd.DataFrame({"src": ["A", "A", "A"], "dst": list(lags), "mode": "SEA",
                          "typical_transit_days": [lag / 24 for lag in lags.values()], "bidirectional": False})
    table, curves, summary = eda.lead_lag(x, edges, step_h=1, max_lag=60, tol=2, track="test")
    assert list(table["best_lag_h"]) == list(lags.values())
    assert summary["lag_hits"] == 3
    assert curves.shape == (3, 61)
