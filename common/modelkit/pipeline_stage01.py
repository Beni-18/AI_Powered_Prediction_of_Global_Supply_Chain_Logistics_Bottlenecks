"""ROLE: ML ENGINEER | COMMON MODELKIT
Standard feature preprocessing and model bundle for Stage 01 ML predictions.

Owner: Role 1 - ML Engineer
READS : In-memory pandas DataFrames during training and API inference
WRITES: Standardized numeric feature arrays for tree models and baselines
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Stage01Pipeline:
    """Encapsulates data preprocessing, feature encoding, and fitted models for Stage 01."""

    def __init__(self, fast_mode: bool = False):
        """Initialize empty preprocessors, imputer dict, and models registry."""
        self.fast_mode = fast_mode
        self.numeric_imputer_medians: dict[str, float] = {}
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names_out: list[str] = []

        # Models dictionary storing trained classifiers and anomaly detectors
        self.models: dict[str, Any] = {}
        # Dwell time split points in hours (e.g. 12h for ON_TIME/AT_RISK, 36h for CRITICAL_DELAY)
        self.rule_thresholds: tuple[float, float] = (12.0, 36.0)

    def fit_preprocessor(self, train_df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> None:
        """Fit imputation, standard scaling, and one-hot encoding on train partition only."""
        # 1. Compute train medians for numeric imputation (never look at val/test to prevent leakage)
        for c in num_cols:
            self.numeric_imputer_medians[c] = float(train_df[c].median())

        # Impute missing values with the training median before computing scaling statistics
        X_num = train_df[num_cols].fillna(self.numeric_imputer_medians)
        self.scaler.fit(X_num)

        # 2. Fit one-hot encoder on categorical columns
        # Unknown or missing categories are mapped to 'UNKNOWN'
        X_cat = train_df[cat_cols].fillna("UNKNOWN").astype(str)
        self.encoder.fit(X_cat)

        # Save output feature names in exact column order
        cat_names = list(self.encoder.get_feature_names_out(cat_cols))
        self.feature_names_out = list(num_cols) + cat_names

    def transform(self, df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> np.ndarray:
        """Transform features into standardized numeric array using the train-fitted statistics."""
        # Standardize numeric columns using previously computed train medians and scalers
        X_num = df[num_cols].fillna(self.numeric_imputer_medians)
        X_num_scaled = self.scaler.transform(X_num)

        # Encode categorical columns with one-hot encoding (unseen categories become all zeros)
        X_cat = df[cat_cols].fillna("UNKNOWN").astype(str)
        X_cat_encoded = self.encoder.transform(X_cat)

        # Horizontally stack numeric and one-hot encoded columns into one design matrix
        return np.hstack([X_num_scaled, X_cat_encoded])
