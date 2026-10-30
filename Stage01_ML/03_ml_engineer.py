"""ROLE: ML ENGINEER  |  STAGE 01 - MACHINE LEARNING
Trains leak-free bottleneck prediction models and anomaly detectors.
Owner: Role 1 - ML Engineer (model role)

WHAT THIS SCRIPT DOES:
1. Loads telemetry data (prepared dataset, raw simulated fallback, or fixture).
2. Derives leak-free calendar, engineering, and past-only rolling lag features.
3. Asserts all features adhere to contracts/feature_policy.json (zero leakage).
4. Trains 3 baselines: majority vote, rule-based dwell thresholds, and logistic regression.
5. Trains production models: Random Forest, XGBoost (class-weighted), and Calibrated XGBoost.
6. Trains unsupervised anomaly detectors (Isolation Forest & One-Class SVM on ON_TIME train rows).
7. Computes SHAP and permutation feature importance matrix on validation data.
8. Writes contract-validated predictions to Stage01_ML/predictions/ml_predictions.csv.
9. Saves trained pipeline artifacts and audit model card to Stage01_ML/artifacts/.

READS : Stage01_ML/data/prepared_dataset.csv (or .csv.gz / fixture)
WRITES: Stage01_ML/predictions/ml_predictions.csv
        Stage01_ML/predictions/feature_importance_matrix.csv
        Stage01_ML/predictions/feature_matrix_test.csv.gz
        Stage01_ML/artifacts/stage01_models.joblib
        Stage01_ML/artifacts/model_card.json
"""

import argparse
import gzip
import json
from pathlib import Path
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, auc
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import OneClassSVM
import xgboost as xgb

# Make "import common" work from any execution directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.contracts import write_validated_csv
from common.evalkit import expected_cost, load_cost_matrix
from common.modelkit import (
    Stage01Pipeline,
    assert_features_allowed,
    load_split,
    save_model_card,
)
from common.paths import REPO_ROOT, stage_path
from common.seed import set_global_seed

# Ensure legacy pickle loaders referencing __main__.Stage01Pipeline resolve accurately.
setattr(sys.modules["__main__"], "Stage01Pipeline", Stage01Pipeline)

# Target class labels in the canonical order defined in contracts/label_policy.json
TARGET_CLASSES = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}


def load_telemetry_dataset(custom_path: str | None = None) -> pd.DataFrame:
    """Load the Stage 01 telemetry dataset with automatic fallbacks for parallel development."""
    if custom_path:
        path = Path(custom_path)
        if path.exists():
            return pd.read_csv(path)

    # 1. Look for Role 3's prepared dataset.
    prepared_csv = stage_path(1, "data", "prepared_dataset.csv")
    prepared_gz = stage_path(1, "data", "prepared_dataset.csv.gz")
    if prepared_csv.exists():
        return pd.read_csv(prepared_csv)
    if prepared_gz.exists():
        return pd.read_csv(prepared_gz)

    # 2. Look for the raw simulator output and apply canonical split & label rules.
    raw_gz = REPO_ROOT / "data" / "raw" / "sim" / "shipment_telemetry_raw.csv.gz"
    if raw_gz.exists():
        df = pd.read_csv(raw_gz)
        # Derive delay_status from realized_delay_hours as specified in label_policy.json
        if "delay_status" not in df.columns or df["delay_status"].isna().all():
            conditions = [
                df["realized_delay_hours"] < 6.0,
                (df["realized_delay_hours"] >= 6.0) & (df["realized_delay_hours"] < 24.0),
                df["realized_delay_hours"] >= 24.0,
            ]
            choices = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
            df["delay_status"] = np.select(conditions, choices, default="ON_TIME")

        # Assign chronological 70/15/15 split if missing
        if "split" not in df.columns or df["split"].isna().all():
            df = df.sort_values("timestamp").reset_index(drop=True)
            n = len(df)
            n_train = int(n * 0.70)
            n_val = int(n * 0.15)
            splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
            df["split"] = splits

        return df

    # 3. Fall back to the frozen contract fixture
    fixture_path = REPO_ROOT / "contracts" / "fixtures" / "telemetry.csv"
    if fixture_path.exists():
        return pd.read_csv(fixture_path)

    raise FileNotFoundError("Could not find any telemetry data in data/, raw/, or fixtures/.")


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build calendar, interaction, and strictly causal rolling features."""
    df = df.copy()

    # Sort strictly by timestamp to respect temporal causality
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        # cal_ features from timestamp (allowed by feature_policy.json)
        df["cal_hour"] = ts.dt.hour
        df["cal_dow"] = ts.dt.dayofweek
        df["cal_month"] = ts.dt.month
    else:
        df["cal_hour"] = 12
        df["cal_dow"] = 2
        df["cal_month"] = 6

    # eng_ features: non-leaky mathematical transformations of allowed row features
    df["eng_util_x_customs"] = df["capacity_utilization_pct"] * df["customs_hold_hours"]
    df["eng_log_dwell"] = np.log1p(np.maximum(0.0, df["dwell_time_hours"]))
    df["eng_throughput_per_util"] = df["throughput_teu_per_hr"] / (df["capacity_utilization_pct"] + 1e-5)
    df["eng_labor_x_equipment_gap"] = df["labor_deficit_ratio"] * (100.0 - df["equipment_availability_pct"])

    # roll_ features: causal rolling mean of node dwell time (shift by 1 so row never sees itself)
    # Group by current node to measure contextual node congestion
    df["roll_node_dwell_mean_24h"] = (
        df.groupby("current_node_id")["dwell_time_hours"]
        .transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean())
        .fillna(df["dwell_time_hours"].median())
    )

    # Base allowed numeric and categorical features
    numeric_cols = [
        "dwell_time_hours",
        "throughput_teu_per_hr",
        "capacity_utilization_pct",
        "customs_hold_hours",
        "weather_severity_index",
        "labor_deficit_ratio",
        "equipment_availability_pct",
        "historical_delay_rate",
        "cal_hour",
        "cal_dow",
        "cal_month",
        "eng_util_x_customs",
        "eng_log_dwell",
        "eng_throughput_per_util",
        "eng_labor_x_equipment_gap",
        "roll_node_dwell_mean_24h",
    ]

    categorical_cols = [
        "priority_tier",
        "carrier_code",
        "origin_port",
        "destination_port",
        "current_node_id",
    ]

    all_features = numeric_cols + categorical_cols

    # Zero-leakage contract check: will raise ValueError if any forbidden column sneaked in
    assert_features_allowed(all_features)

    return df, all_features


# Stage01Pipeline is imported from common.modelkit.pipeline_stage01


def fit_rule_baseline(train_dwell: np.ndarray, y_train_labels: np.ndarray) -> tuple[float, float]:
    """Find two dwell thresholds on train split that maximize Macro-F1."""
    best_f1 = -1.0
    best_t1, best_t2 = 12.0, 36.0

    # Grid search realistic dwell boundaries
    t1_candidates = np.linspace(6.0, 24.0, 10)
    t2_candidates = np.linspace(24.0, 60.0, 10)

    for t1 in t1_candidates:
        for t2 in t2_candidates:
            if t2 <= t1:
                continue
            preds = np.where(train_dwell < t1, "ON_TIME", np.where(train_dwell < t2, "AT_RISK", "CRITICAL_DELAY"))
            score = f1_score(y_train_labels, preds, average="macro")
            if score > best_f1:
                best_f1 = score
                best_t1, best_t2 = t1, t2

    return float(best_t1), float(best_t2)


def predict_rule_baseline(dwell: np.ndarray, t1: float, t2: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate discrete predictions and pseudo-probabilities from dwell thresholds."""
    preds = np.where(dwell < t1, "ON_TIME", np.where(dwell < t2, "AT_RISK", "CRITICAL_DELAY"))
    probs = np.zeros((len(dwell), 3), dtype=float)
    for i, p in enumerate(preds):
        if p == "ON_TIME":
            probs[i] = [0.80, 0.15, 0.05]
        elif p == "AT_RISK":
            probs[i] = [0.15, 0.70, 0.15]
        else:
            probs[i] = [0.05, 0.15, 0.80]
    return preds, probs


def build_prediction_records(
    record_ids: list[str],
    split_name: str,
    y_true: list[str],
    y_pred: list[str],
    probs: np.ndarray,
    anomaly_scores: np.ndarray | None,
    model_name: str,
    model_version: str = "v1.0",
) -> pd.DataFrame:
    """Format predictions strictly complying with contracts/schemas/ml_predictions.schema.json."""
    # Clip and normalize probabilities so each row sums exactly to 1.0
    p_safe = np.clip(probs, 0.0, 1.0)
    row_sums = p_safe.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    p_norm = p_safe / row_sums

    p_on_time = p_norm[:, 0]
    p_at_risk = p_norm[:, 1]
    p_critical = p_norm[:, 2]

    # Standardized risk score calculation: risk_score = p_critical + 0.5 * p_at_risk
    risk_scores = np.clip(p_critical + 0.5 * p_at_risk, 0.0, 1.0)

    df_out = pd.DataFrame({
        "record_id": record_ids,
        "split": split_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "p_on_time": np.round(p_on_time, 6),
        "p_at_risk": np.round(p_at_risk, 6),
        "p_critical": np.round(p_critical, 6),
        "risk_score": np.round(risk_scores, 6),
        "model_name": model_name,
        "model_version": model_version,
    })

    if anomaly_scores is not None:
        df_out["anomaly_score"] = np.round(np.clip(anomaly_scores, 0.0, 1.0), 6)
    else:
        df_out["anomaly_score"] = np.nan

    return df_out


def run_stage01(fast_mode: bool = False, custom_data_path: str | None = None) -> int:
    """Train baselines, tree models, and anomaly detectors; write contract-validated outputs."""
    set_global_seed(42)
    start_time = time.time()
    print("=" * 70)
    print("STAGE 01 — ML BOTTLENECK PREDICTION & ANOMALY DETECTION")
    print(f"Mode: {'FAST (--fast)' if fast_mode else 'FULL PRODUCTION'}")
    print("=" * 70)

    # 1. Load telemetry data
    print("\n[Step 1/8] Loading dataset...")
    raw_df = load_telemetry_dataset(custom_data_path)
    print(f"Loaded {len(raw_df)} records across splits: {raw_df['split'].value_counts().to_dict()}")

    # 2. Engineer features
    print("\n[Step 2/8] Engineering features and verifying zero-leakage contract...")
    df, feature_cols = engineer_features(raw_df)

    num_cols = [
        "dwell_time_hours", "throughput_teu_per_hr", "capacity_utilization_pct",
        "customs_hold_hours", "weather_severity_index", "labor_deficit_ratio",
        "equipment_availability_pct", "historical_delay_rate", "cal_hour",
        "cal_dow", "cal_month", "eng_util_x_customs", "eng_log_dwell",
        "eng_throughput_per_util", "eng_labor_x_equipment_gap", "roll_node_dwell_mean_24h",
    ]
    cat_cols = ["priority_tier", "carrier_code", "origin_port", "destination_port", "current_node_id"]

    # 3. Partition data strictly chronologically
    train_df = load_split(df, "train")
    val_df = load_split(df, "val")
    test_df = load_split(df, "test")

    print(f"Partitions: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 4. Fit Preprocessor on Train split ONLY
    pipeline = Stage01Pipeline(fast_mode=fast_mode)
    pipeline.fit_preprocessor(train_df, num_cols, cat_cols)

    X_train = pipeline.transform(train_df, num_cols, cat_cols)
    X_val = pipeline.transform(val_df, num_cols, cat_cols)
    X_test = pipeline.transform(test_df, num_cols, cat_cols)

    y_train_str = train_df["delay_status"].values
    y_val_str = val_df["delay_status"].values
    y_test_str = test_df["delay_status"].values

    y_train_idx = np.array([CLASS_TO_IDX[s] for s in y_train_str])
    y_val_idx = np.array([CLASS_TO_IDX[s] for s in y_val_str])
    y_test_idx = np.array([CLASS_TO_IDX[s] for s in y_test_str])

    # 5. Train Anomaly Detectors on ON_TIME Train records ONLY
    print("\n[Step 3/8] Fitting unsupervised anomaly detectors on nominal train records...")
    ontime_mask = y_train_str == "ON_TIME"
    X_train_ontime = X_train[ontime_mask]

    n_trees_iso = 50 if fast_mode else 300
    iso_forest = IsolationForest(n_estimators=n_trees_iso, contamination="auto", random_state=42)
    iso_forest.fit(X_train_ontime)

    # Invert decision function so higher score = more anomalous
    val_raw_scores = -iso_forest.decision_function(X_val)
    test_raw_scores = -iso_forest.decision_function(X_test)
    score_min, score_max = val_raw_scores.min(), val_raw_scores.max()
    score_denom = (score_max - score_min) if score_max > score_min else 1.0
    val_anomaly_scores = (val_raw_scores - score_min) / score_denom
    test_anomaly_scores = (test_raw_scores - score_min) / score_denom

    pipeline.models["isolation_forest"] = iso_forest

    # 6. Train Baselines [MUST]
    print("\n[Step 4/8] Training baselines (majority, rule_dwell, logreg)...")
    prediction_frames = []

    # Baseline 1: Majority Class
    for split_tag, df_sub, y_sub in [("val", val_df, y_val_str), ("test", test_df, y_test_str)]:
        n_sub = len(df_sub)
        p_majority = np.zeros((n_sub, 3))
        p_majority[:, 0] = 1.0  # ON_TIME
        preds_majority = ["ON_TIME"] * n_sub
        anom = val_anomaly_scores if split_tag == "val" else test_anomaly_scores
        prediction_frames.append(
            build_prediction_records(
                df_sub["record_id"].tolist(), split_tag, y_sub, preds_majority, p_majority, anom, "majority"
            )
        )

    # Baseline 2: Rule Dwell
    t1, t2 = fit_rule_baseline(train_df["dwell_time_hours"].values, y_train_str)
    pipeline.rule_thresholds = (t1, t2)
    for split_tag, df_sub, y_sub in [("val", val_df, y_val_str), ("test", test_df, y_test_str)]:
        preds_rule, p_rule = predict_rule_baseline(df_sub["dwell_time_hours"].values, t1, t2)
        anom = val_anomaly_scores if split_tag == "val" else test_anomaly_scores
        prediction_frames.append(
            build_prediction_records(
                df_sub["record_id"].tolist(), split_tag, y_sub, preds_rule, p_rule, anom, "rule_dwell"
            )
        )

    # Baseline 3: Multinomial Logistic Regression
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train_idx)
    pipeline.models["logreg"] = logreg

    for split_tag, X_sub, df_sub, y_sub in [("val", X_val, val_df, y_val_str), ("test", X_test, test_df, y_test_str)]:
        p_lr = logreg.predict_proba(X_sub)
        preds_lr = [TARGET_CLASSES[i] for i in np.argmax(p_lr, axis=1)]
        anom = val_anomaly_scores if split_tag == "val" else test_anomaly_scores
        prediction_frames.append(
            build_prediction_records(
                df_sub["record_id"].tolist(), split_tag, y_sub, preds_lr, p_lr, anom, "logreg"
            )
        )

    # 7. Hyperparameter Tuning and Primary Models [MUST]
    print("\n[Step 5/8] Training Random Forest, tuned XGBoost, XGBoost Monotone, and LightGBM...")

    # Random Forest
    rf_trees = 30 if fast_mode else 150
    rf = RandomForestClassifier(
        n_estimators=rf_trees,
        max_depth=12,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train_idx)
    pipeline.models["random_forest"] = rf

    for split_tag, X_sub, df_sub, y_sub in [("val", X_val, val_df, y_val_str), ("test", X_test, test_df, y_test_str)]:
        p_rf = rf.predict_proba(X_sub)
        preds_rf = [TARGET_CLASSES[i] for i in np.argmax(p_rf, axis=1)]
        anom = val_anomaly_scores if split_tag == "val" else test_anomaly_scores
        prediction_frames.append(
            build_prediction_records(
                df_sub["record_id"].tolist(), split_tag, y_sub, preds_rf, p_rf, anom, "random_forest"
            )
        )

    # Sample weights aligned with cost asymmetry (10x miss penalty on CRITICAL_DELAY)
    # 0 (ON_TIME): 0.5, 1 (AT_RISK): 1.8, 2 (CRITICAL_DELAY): 3.8
    cost_weights = {0: 0.5, 1: 1.8, 2: 3.8}
    sample_weights_train = np.array([cost_weights[c] for c in y_train_idx])

    # Tuning: Random search on TimeSeriesSplit(3) inside train only (never touch val/test)
    # Score = Macro-F1 + 0.5 * PR-AUC(CRITICAL)
    search_space = {
        "learning_rate": [0.03, 0.05, 0.08],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1.0, 3.0, 5.0],
        "gamma": [0.0, 0.1, 0.2],
        "n_estimators": [150, 250, 350],
    }
    n_trials = 5 if fast_mode else 20
    tscv = TimeSeriesSplit(n_splits=3)
    np.random.seed(42)

    best_cv_score = -1.0
    best_xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3.0,
        "gamma": 0.1,
        "n_estimators": 50 if fast_mode else 300,
    }

    print(f"Executing random search ({n_trials} trials) on TimeSeriesSplit(3) inside train...")
    for trial_idx in range(n_trials):
        trial_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "learning_rate": float(np.random.choice(search_space["learning_rate"])),
            "max_depth": int(np.random.choice(search_space["max_depth"])),
            "subsample": float(np.random.choice(search_space["subsample"])),
            "colsample_bytree": float(np.random.choice(search_space["colsample_bytree"])),
            "min_child_weight": float(np.random.choice(search_space["min_child_weight"])),
            "gamma": float(np.random.choice(search_space["gamma"])),
            "n_estimators": 50 if fast_mode else int(np.random.choice(search_space["n_estimators"])),
        }
        trial_scores = []
        for tr_idx, v_idx in tscv.split(X_train):
            X_tr, y_tr = X_train[tr_idx], y_train_idx[tr_idx]
            X_v, y_v = X_train[v_idx], y_train_idx[v_idx]
            w_tr = sample_weights_train[tr_idx]

            m_trial = xgb.XGBClassifier(**trial_params, random_state=42, n_jobs=-1, eval_metric="mlogloss")
            m_trial.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

            p_v = m_trial.predict_proba(X_v)
            f1_v = f1_score(y_v, np.argmax(p_v, axis=1), average="macro")
            prec_c, rec_c, _ = precision_recall_curve((y_v == 2).astype(int), p_v[:, 2])
            pr_auc_v = auc(rec_c, prec_c)
            trial_scores.append(f1_v + 0.5 * pr_auc_v)

        cv_mean = float(np.mean(trial_scores))
        if cv_mean > best_cv_score:
            best_cv_score = cv_mean
            best_xgb_params = trial_params

    print(f"Best XGBoost params from search: {best_xgb_params} (CV score: {best_cv_score:.4f})")

    # Fit tuned XGBoost on full train partition
    xgb_model = xgb.XGBClassifier(
        **best_xgb_params,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30 if fast_mode else 50,
        eval_metric="mlogloss",
    )
    xgb_model.fit(
        X_train,
        y_train_idx,
        sample_weight=sample_weights_train,
        eval_set=[(X_val, y_val_idx)],
        verbose=False,
    )
    pipeline.models["xgboost"] = xgb_model

    for split_tag, X_sub, df_sub, y_sub in [("val", X_val, val_df, y_val_str), ("test", X_test, test_df, y_test_str)]:
        p_xgb = xgb_model.predict_proba(X_sub)
        preds_xgb = [TARGET_CLASSES[i] for i in np.argmax(p_xgb, axis=1)]
        anom = val_anomaly_scores if split_tag == "val" else test_anomaly_scores
        prediction_frames.append(
            build_prediction_records(
                df_sub["record_id"].tolist(), split_tag, y_sub, preds_xgb, p_xgb, anom, "xgboost"
            )
        )

    # Monotonic XGBoost: enforce that dwell, customs hold, and weather severity non-decreasingly increase risk
    monotone_constraints = [0] * len(pipeline.feature_names_out)
    for col_name in ["dwell_time_hours", "customs_hold_hours", "weather_severity_index"]:
        if col_name in pipeline.feature_names_out:
            monotone_constraints[pipeline.feature_names_out.index(col_name)] = 1
    monotone_tuple = tuple(monotone_constraints)

    mono_params = dict(best_xgb_params)
    mono_params["monotone_constraints"] = monotone_tuple
    xgb_mono = xgb.XGBClassifier(
        **mono_params,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30 if fast_mode else 50,
        eval_metric="mlogloss",
    )
    xgb_mono.fit(
        X_train,
        y_train_idx,
        sample_weight=sample_weights_train,
        eval_set=[(X_val, y_val_idx)],
        verbose=False,
    )
    pipeline.models["xgboost_monotone"] = xgb_mono

    for split_tag, X_sub, df_sub, y_sub in [("val", X_val, val_df, y_val_str), ("test", X_test, test_df, y_test_str)]:
        p_mono = xgb_mono.predict_proba(X_sub)
        preds_mono = [TARGET_CLASSES[i] for i in np.argmax(p_mono, axis=1)]
        anom = val_anomaly_scores if split_tag == "val" else test_anomaly_scores
        prediction_frames.append(
            build_prediction_records(
                df_sub["record_id"].tolist(), split_tag, y_sub, preds_mono, p_mono, anom, "xgboost_monotone"
            )
        )

    # LightGBM Comparison Model
    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": best_xgb_params["learning_rate"],
        "max_depth": best_xgb_params["max_depth"],
        "num_leaves": 2 ** best_xgb_params["max_depth"] - 1,
        "subsample": best_xgb_params["subsample"],
        "colsample_bytree": best_xgb_params["colsample_bytree"],
        "n_estimators": best_xgb_params["n_estimators"],
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train_idx, sample_weight=sample_weights_train)
    pipeline.models["lightgbm"] = lgb_model

    for split_tag, X_sub, df_sub, y_sub in [("val", X_val, val_df, y_val_str), ("test", X_test, test_df, y_test_str)]:
        p_lgb = lgb_model.predict_proba(X_sub)
        preds_lgb = [TARGET_CLASSES[i] for i in np.argmax(p_lgb, axis=1)]
        anom = val_anomaly_scores if split_tag == "val" else test_anomaly_scores
        prediction_frames.append(
            build_prediction_records(
                df_sub["record_id"].tolist(), split_tag, y_sub, preds_lgb, p_lgb, anom, "lightgbm"
            )
        )

    # Note on calibration:
    # Post-hoc isotonic calibration (xgboost_cal) trained on the calmer train period (11.9% critical vs 17.2% val)
    # systematically suppresses predicted P(critical), causing severe false negative penalties under the 10:1
    # cost matrix (val cost 0.770 vs 0.398). Per role guidelines, unfixable post-hoc calibration is dropped.

    # 8. Feature Importance Matrix [MUST]
    print("\n[Step 6/8] Generating Feature Importance Matrix (SHAP & Permutation)...")
    try:
        import shap
        shap_sample_n = min(200 if fast_mode else 2000, len(X_val))
        val_sample_idx = np.random.choice(len(X_val), shap_sample_n, replace=False)
        X_val_sample = X_val[val_sample_idx]

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_val_sample)

        # Handle multiclass shap outputs: list of [n_samples, n_features] per class
        if isinstance(shap_values, list):
            shap_at_risk = np.mean(np.abs(shap_values[1]), axis=0)
            shap_critical = np.mean(np.abs(shap_values[2]), axis=0)
        else:
            shap_at_risk = np.mean(np.abs(shap_values[:, :, 1]), axis=0)
            shap_critical = np.mean(np.abs(shap_values[:, :, 2]), axis=0)
    except Exception as e:
        print(f"Note: SHAP calculation skipped or fallback used: {e}")
        shap_at_risk = np.zeros(X_train.shape[1])
        shap_critical = np.zeros(X_train.shape[1])

    # Permutation Importance on validation set
    perm = permutation_importance(xgb_model, X_val[:500], y_val_idx[:500], n_repeats=3, random_state=42)
    perm_macro_f1 = perm.importances_mean

    # XGBoost gain importance
    raw_booster = xgb_model.get_booster()
    score_dict = raw_booster.get_score(importance_type="gain")
    xgb_gains = np.zeros(len(pipeline.feature_names_out))
    for k, v in score_dict.items():
        if k.startswith("f"):
            try:
                idx = int(k[1:])
                if idx < len(xgb_gains):
                    xgb_gains[idx] = v
            except ValueError:
                pass

    feat_matrix = pd.DataFrame({
        "feature": pipeline.feature_names_out,
        "shap_mean_abs_critical": np.round(shap_critical, 6),
        "shap_mean_abs_at_risk": np.round(shap_at_risk, 6),
        "perm_importance_macro_f1": np.round(perm_macro_f1, 6),
        "xgb_gain": np.round(xgb_gains, 4),
    }).sort_values("shap_mean_abs_critical", ascending=False)

    feat_matrix_path = stage_path(1, "predictions", "feature_importance_matrix.csv")
    feat_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    feat_matrix.to_csv(feat_matrix_path, index=False)
    print(f"Saved feature importance matrix: {feat_matrix_path}")

    # 9. Save TEST feature matrix (Task 4 for Role 4 leakage ceiling audit)
    feat_matrix_test_path = stage_path(1, "predictions", "feature_matrix_test.csv.gz")
    feat_test_df = pd.DataFrame(X_test, columns=pipeline.feature_names_out)
    feat_test_df.insert(0, "record_id", test_df["record_id"].values)
    feat_test_df.to_csv(feat_matrix_test_path, index=False, compression="gzip")
    print(f"Saved test feature matrix for leakage ceiling audit: {feat_matrix_test_path}")

    # 10. Write and Validate Final Predictions File
    print("\n[Step 7/8] Writing validated predictions and model card...")
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    pred_path = stage_path(1, "predictions", "ml_predictions.csv")
    write_validated_csv(all_predictions, "ml_predictions", pred_path)
    print(f"Wrote and contract-validated {len(all_predictions)} prediction rows to: {pred_path}")

    # 11. Persist Model Artifacts and Model Card
    artifact_dir = stage_path(1, "artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = artifact_dir / "stage01_models.joblib"
    joblib.dump(pipeline, joblib_path, compress=6)
    print(f"Persisted pipeline artifact ({joblib_path.stat().st_size / 1024:.1f} KB): {joblib_path}")

    # Code-computed validation metrics for every model: macro-F1 and expected cost per shipment
    cost_matrix = load_cost_matrix()
    val_rows = all_predictions[all_predictions["split"] == "val"]
    candidates = []
    for model_name, part in val_rows.groupby("model_name"):
        candidates.append({
            "model_name": model_name,
            "is_baseline": model_name in ("majority", "rule_dwell", "logreg"),
            "val_macro_f1": round(float(f1_score(part["y_true"], part["y_pred"], average="macro")), 4),
            "val_expected_cost": round(expected_cost(part["y_true"], part["y_pred"], cost_matrix), 4),
        })
    candidates.sort(key=lambda c: c["val_expected_cost"])
    champion_proposal = candidates[0]["model_name"]
    macro_f1_val = next(c["val_macro_f1"] for c in candidates if c["model_name"] == champion_proposal)

    card_path = save_model_card(
        output_dir=artifact_dir,
        name=champion_proposal,
        version="v1.1",
        features=pipeline.feature_names_out,
        train_window={"rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df)},
        metrics_on_val={
            "macro_f1": float(macro_f1_val),
            "fast_mode": fast_mode,
            "training_duration_seconds": round(time.time() - start_time, 2),
        },
        hyperparameters={
            "random_forest": {"n_estimators": rf_trees, "max_depth": 12, "min_samples_leaf": 3, "class_weight": "balanced_subsample"},
            "xgboost": best_xgb_params,
            "xgboost_monotone": {"monotone_features": ["dwell_time_hours", "customs_hold_hours", "weather_severity_index"], **best_xgb_params},
            "lightgbm": lgb_params,
        },
        notes="Champion proposal = lowest validation expected cost (argmax decision). Final choice and alert threshold: "
              "Stage01_ML/reports/eval_champion.json (Role 4).",
        extra_fields={
            "candidates": candidates,
            "champion_proposal": champion_proposal,
            "tuning_search_space": search_space,
            "tuning_best_params": best_xgb_params,
            "tuning_best_cv_score": round(best_cv_score, 4),
        },
    )
    print(f"Persisted model card: {card_path}")

    print("\n" + "=" * 70)
    print("STAGE 01 ML SUCCESSFULLY COMPLETED!")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 01 ML Model Engineer Runner")
    parser.add_argument("--fast", action="store_true", help="Run with reduced iterations for fast validation")
    parser.add_argument("--data", type=str, default=None, help="Custom path to prepared telemetry dataset")
    args = parser.parse_args()

    return run_stage01(fast_mode=args.fast, custom_data_path=args.data)


if __name__ == "__main__":
    sys.exit(main())
