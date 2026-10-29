"""ROLE: EDA ENGINEER | COMMON EDA TOOLKIT
Shared statistical and plotting utilities for exploratory data analysis across all stages.

OWNER : Role 2 - EDA Engineer
PROVIDES: apply_style, save_fig, psi, robust_noise_floor, ks_drift, univariate_auc
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    mpl = None
    plt = None

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    roc_auc_score = None


def apply_style() -> None:
    """Set uniform matplotlib and seaborn styling with colour-blind safe palettes."""
    if plt is None or mpl is None:
        raise ImportError("matplotlib is required for apply_style(). Install with: pip install matplotlib")

    # Use standard sans-serif font family and readable font sizes
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Customise matplotlib settings for clean, readable engineering reports
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]
    mpl.rcParams["axes.edgecolor"] = "#cccccc"
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["grid.color"] = "#e5e5e5"
    mpl.rcParams["grid.linestyle"] = "--"
    mpl.rcParams["grid.alpha"] = 0.7
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["xtick.labelsize"] = 9
    mpl.rcParams["ytick.labelsize"] = 9
    mpl.rcParams["figure.titlesize"] = 14
    mpl.rcParams["figure.titleweight"] = "bold"


def save_fig(fig: mpl.figure.Figure, path: Union[str, Path], dpi: int = 150) -> Path:
    """Save a matplotlib figure cleanly to disk, creating parent folders if missing."""
    out_path = Path(path)

    # Create the destination directory if it does not already exist
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure with tight bounding box so labels and titles are never clipped
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) to measure distribution shift between two samples.

    A PSI < 0.1 indicates no notable change.
    A PSI between 0.1 and 0.2 indicates moderate shift.
    A PSI > 0.2 indicates significant shift requiring model or operational review.
    """
    # Convert inputs to clean 1D numpy arrays excluding null values
    exp = np.asarray(pd.Series(expected).dropna(), dtype=float)
    act = np.asarray(pd.Series(actual).dropna(), dtype=float)

    # If either array is empty or has zero variance, no meaningful shift can be computed
    if len(exp) == 0 or len(act) == 0:
        return 0.0

    # Determine quantile bin edges based on the expected (baseline) distribution
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(exp, quantiles)

    # Deduplicate bin edges in case data contains many repeated values
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    # Ensure the outer edges cover the full range of both expected and actual values
    bin_edges[0] = min(bin_edges[0], act.min()) - 1e-5
    bin_edges[-1] = max(bin_edges[-1], act.max()) + 1e-5

    # Count observed samples in each bin for both expected and actual datasets
    exp_counts, _ = np.histogram(exp, bins=bin_edges)
    act_counts, _ = np.histogram(act, bins=bin_edges)

    # Convert counts to proportions with Laplace-style epsilon smoothing to avoid log(0)
    eps = 1e-4
    exp_pct = (exp_counts / len(exp)) + eps
    act_pct = (act_counts / len(act)) + eps

    # Re-normalise so probabilities sum to 1 after adding epsilon
    exp_pct = exp_pct / np.sum(exp_pct)
    act_pct = act_pct / np.sum(act_pct)

    # Compute standard PSI formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_value)


def robust_noise_floor(series: pd.Series, window: int = 24) -> float:
    """Calculate the typical random wobble (noise floor) using rolling-median residual and MAD.

    MAD (Median Absolute Deviation) ignores extreme shock spikes (e.g. rare strikes or typhoons),
    ensuring that the calculated background noise floor reflects normal operational fluctuations.
    Multiplication by 1.4826 makes MAD scale-comparable to standard deviation for Gaussian noise.
    """
    # Drop missing values and make sure the series is numeric
    clean = pd.Series(series).dropna().astype(float)
    if len(clean) < max(4, window // 4):
        return 0.0

    # Calculate smooth signal using rolling median (centered window where possible)
    min_periods = max(1, window // 4)
    smooth = clean.rolling(window, min_periods=min_periods, center=True).median()

    # The residual is the raw observation minus the underlying smooth signal
    residual = clean - smooth
    clean_residual = residual.dropna()

    if len(clean_residual) == 0:
        return 0.0

    # Compute median absolute deviation of the residuals
    med_res = clean_residual.median()
    mad = (clean_residual - med_res).abs().median()

    # Scale factor 1.4826 calibrates MAD to normal standard deviation
    return float(mad * 1.4826)


def ks_drift(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Perform a two-sample Kolmogorov-Smirnov test to detect changes in continuous distributions."""
    # Strip nulls from both samples
    s_a = pd.Series(a).dropna().astype(float)
    s_b = pd.Series(b).dropna().astype(float)

    # If either sample is empty, return zero statistic and non-significant p-value
    if len(s_a) == 0 or len(s_b) == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    # Execute the two-sample KS test using scipy
    res = stats.ks_2samp(s_a, s_b)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}


def univariate_auc(df: pd.DataFrame, features: Sequence[str], target: str) -> Dict[str, float]:
    """Calculate the univariate ROC-AUC of each feature against a binary target as a leak detector.

    A feature with ROC-AUC > 0.95 is flagged as a potential target leak.
    We compute max(auc, 1 - auc) so that inverse relationships are equally detected.
    """
    results: Dict[str, float] = {}

    # Ensure the target column is binary (0 or 1)
    y_raw = df[target].dropna()
    unique_vals = np.unique(y_raw)
    if len(unique_vals) != 2:
        # If target is not binary, return empty or dummy dictionary
        return results

    # Map target values to binary 0 and 1
    y = (df[target] == unique_vals[1]).astype(int)

    for feat in features:
        if feat not in df.columns:
            continue

        # Extract feature values aligned with the target
        subset = df[[feat]].copy()
        subset["_y"] = y
        subset = subset.dropna()

        if len(subset) == 0 or subset[feat].nunique() <= 1:
            results[feat] = 0.5
            continue

        try:
            # Handle numeric columns directly
            if pd.api.types.is_numeric_dtype(subset[feat]):
                auc = roc_auc_score(subset["_y"], subset[feat])
            else:
                # For categorical features, calculate mean target rate per category
                cat_ranks = subset.groupby(feat)["_y"].mean()
                mapped_feat = subset[feat].map(cat_ranks)
                auc = roc_auc_score(subset["_y"], mapped_feat)

            # Symmetrise ROC-AUC so strong inverse associations also show as near 1.0
            results[feat] = float(max(auc, 1.0 - auc))
        except Exception:
            # Fallback to neutral 0.5 if scoring fails due to numerical edge cases
            results[feat] = 0.5

    return results
