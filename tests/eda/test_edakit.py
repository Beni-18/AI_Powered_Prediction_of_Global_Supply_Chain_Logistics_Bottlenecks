"""Unit tests for the shared EDA toolkit (common.edakit).

Verifies styling, saving, PSI calculation, robust noise floor, KS drift, and leakage detection.
"""

from pathlib import Path
import pytest

# Skip this test module in environments (like CI) where only requirements-dev.txt is installed
pytest.importorskip("matplotlib")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.edakit import (
    apply_style,
    save_fig,
    psi,
    robust_noise_floor,
    ks_drift,
    univariate_auc,
)


def test_apply_style_runs_without_error():
    """Verify that applying style sets valid matplotlib rcParams without errors."""
    apply_style()
    assert plt.rcParams["axes.titlesize"] == 12
    assert plt.rcParams["figure.titlesize"] == 14


def test_save_fig(tmp_path: Path):
    """Verify that save_fig saves an image file cleanly and creates parent folders."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out_file = tmp_path / "subfolder" / "test_plot.png"

    saved = save_fig(fig, out_file)
    assert saved.exists()
    assert saved.is_file()
    assert saved.stat().st_size > 0


def test_psi_identical_distributions():
    """Verify that two identical samples produce a near-zero PSI value."""
    np.random.seed(42)
    sample_a = np.random.normal(loc=10.0, scale=2.0, size=1000)
    sample_b = np.random.normal(loc=10.0, scale=2.0, size=1000)

    psi_val = psi(sample_a, sample_b, bins=10)
    # Identical underlying distributions should have PSI well below 0.1
    assert psi_val < 0.05


def test_psi_shifted_distribution():
    """Verify that a significantly shifted distribution produces PSI > 0.2."""
    np.random.seed(42)
    baseline = np.random.normal(loc=10.0, scale=1.0, size=1000)
    shifted = np.random.normal(loc=15.0, scale=1.0, size=1000)

    psi_val = psi(baseline, shifted, bins=10)
    # Significant shift must exceed 0.2 threshold
    assert psi_val > 0.2


def test_psi_edge_cases():
    """Verify PSI calculation handles empty or constant inputs safely."""
    assert psi([], []) == 0.0
    assert psi([1.0] * 50, [1.0] * 50) == 0.0


def test_robust_noise_floor_resists_outliers():
    """Verify that robust noise floor using MAD ignores rare extreme shock spikes."""
    np.random.seed(42)
    # 200 normal points with unit variance
    nominal = np.random.normal(loc=20.0, scale=1.0, size=200)
    # Add a massive shock spike simulating a typhoon or port shutdown
    nominal[50] = 500.0
    nominal[51] = 600.0

    s = pd.Series(nominal)
    noise = robust_noise_floor(s, window=24)

    # Unit Gaussian noise with MAD scaling should yield noise floor ≈ 1.0 (between 0.7 and 1.4)
    assert 0.7 <= noise <= 1.5


def test_robust_noise_floor_constant():
    """Verify constant series has zero noise floor."""
    s = pd.Series([10.0] * 100)
    assert robust_noise_floor(s, window=24) == 0.0


def test_ks_drift():
    """Verify Kolmogorov-Smirnov test detects distributional drift."""
    np.random.seed(42)
    a = np.random.normal(0, 1, 500)
    b = np.random.normal(0, 1, 500)
    c = np.random.normal(3, 1, 500)

    res_same = ks_drift(a, b)
    assert res_same["p_value"] > 0.05

    res_diff = ks_drift(a, c)
    assert res_diff["p_value"] < 0.001
    assert res_diff["statistic"] > 0.5


def test_univariate_auc_leak_detection():
    """Verify univariate ROC-AUC flags perfect predictors and neutral noise."""
    np.random.seed(42)
    n = 200
    # Binary label: 0 or 1
    y = np.random.binomial(1, 0.3, size=n)
    # Feature 1 is an intentional leak (strongly correlated with y)
    leaky_feature = y * 5.0 + np.random.normal(0, 0.1, size=n)
    # Feature 2 is pure Gaussian noise
    noise_feature = np.random.normal(0, 1, size=n)

    df = pd.DataFrame({
        "target": y,
        "leak": leaky_feature,
        "noise": noise_feature,
    })

    auc_scores = univariate_auc(df, ["leak", "noise"], target="target")

    # The leaking feature should have AUC near 1.0
    assert auc_scores["leak"] > 0.95
    # The noise feature should be near 0.5 (random chance)
    assert 0.4 <= auc_scores["noise"] <= 0.65
