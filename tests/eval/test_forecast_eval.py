"""Tests for Role 4's Stage 2 evaluation helpers (common/evalkit/forecast.py and leadtime.py additions)."""

import numpy as np
import pandas as pd
import pytest

from common.evalkit import (
    block_skill_ci, diebold_mariano, episode_hours, exceedance_probability, interval_score, smape, wilson_interval,
)


def test_skill_of_a_model_against_itself_is_zero_and_halving_errors_gives_half():
    """Skill = 1 - MAE_model / MAE_baseline, computed on the same rows."""
    rng = np.random.default_rng(0)
    err = rng.normal(size=500)
    blocks = np.repeat(np.arange(50), 10)
    assert block_skill_ci(err, err, blocks, n=200)[0] == pytest.approx(0.0)
    s, lo, hi = block_skill_ci(err / 2, err, blocks, n=200)
    assert s == pytest.approx(0.5) and lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_skill_interval_is_wider_for_noisier_models():
    """A model that is better only on average gets an interval that includes values below its point skill."""
    rng = np.random.default_rng(1)
    base = np.abs(rng.normal(size=1000))
    model = base * rng.uniform(0.2, 1.6, size=1000)
    s, lo, hi = block_skill_ci(model, base, np.repeat(np.arange(100), 10), n=500)
    assert lo < s < hi


def test_diebold_mariano_flags_a_clearly_better_forecast():
    """Loss A is systematically 0.5 lower than B: negative statistic, tiny p-value."""
    rng = np.random.default_rng(2)
    b = rng.uniform(1, 2, size=300)
    res = diebold_mariano(b - 0.5, b + rng.normal(0, 0.05, size=300), max_lag=4)
    assert res["dm_stat"] < 0 and res["p_value"] < 1e-6


def test_exceedance_probability_is_half_at_the_median_and_monotone():
    """P(value >= median) = 0.5; a higher threshold gives a lower probability."""
    p = exceedance_probability(np.array([8.0, 8.0]), np.array([10.0, 10.0]), np.array([12.0, 12.0]), np.array([10.0, 12.0]))
    assert p[0] == pytest.approx(0.5)
    assert p[1] == pytest.approx(0.10, abs=1e-3)  # the 90th percentile -> 10% chance above


def test_interval_score_penalises_misses():
    """Same width, but a miss adds 2/alpha times the distance outside."""
    y = np.array([5.0])
    inside = interval_score(y, np.array([4.0]), np.array([6.0]))
    outside = interval_score(y, np.array([6.0]), np.array([8.0]))
    assert inside == pytest.approx(2.0) and outside == pytest.approx(2.0 + 10 * 1.0)


def test_smape_handles_zeros():
    """Both zero counts as a perfect forecast, not a division by zero."""
    assert smape(np.array([0.0, 10.0]), np.array([0.0, 10.0])) == 0.0


def test_episode_hours_marks_only_long_runs():
    """A 6-hour run above the threshold counts; a 3-hour blip does not."""
    times = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    values = np.zeros(48)
    values[5:8] = 10      # 3-hour blip
    values[30:36] = 10    # 6-hour episode
    inside = episode_hours(times, values, threshold=5)
    assert inside.iloc[30:36].all() and not inside.iloc[5:8].any() and inside.sum() == 6


def test_wilson_interval_contains_the_estimate():
    """For 7 of 10 the interval surrounds 0.7 and stays inside [0, 1]."""
    lo, hi = wilson_interval(7, 10)
    assert 0 <= lo < 0.7 < hi <= 1
