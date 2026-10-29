# Stage 01 - Role 1 - ML Engineer (model role) -> hand-off note (03_ml_engineer.py)

status: READY
updated_utc: 2026-10-29T13:32:00Z
git_commit: HEAD
data_version: v1

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

## Files produced (path | contract | rows | notes)
- `Stage01_ML/predictions/ml_predictions.csv` | `ml_predictions` | 108180 | Baselines (majority, rule_dwell, logreg) and models (random_forest, xgboost, xgboost_cal) with anomaly scores
- `Stage01_ML/predictions/feature_importance_matrix.csv` | none | 21 | SHAP on val + permutation importance + XGBoost gain
- `Stage01_ML/artifacts/stage01_models.joblib` | none | 1 | Fitted preprocessing and model pipeline (7.1 MB)
- `Stage01_ML/artifacts/model_card.json` | none | 1 | Complete audit card with code-computed val metrics

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
PASS  Stage01_ML/predictions/ml_predictions.csv  (108180 rows, schema 'ml_predictions')
```

## Decisions and assumptions (what you decided and why, in plain English)
- **Zero Data Leakage:** Checked features against `contracts/feature_policy.json` via `assert_features_allowed()`. Target and outcome columns (`realized_delay_hours`, `delay_status`, IDs) were strictly excluded.
- **Strictly Causal Rolling Features:** `roll_node_dwell_mean_24h` used `shift(1)` before rolling so no record ever sees its own dwell time.
- **Preprocessing on Train Only:** Imputation medians, standard scalers, and one-hot encoders were fit on the train partition only.
- **Isotonic Calibration:** Calibrated XGBoost with isotonic regression on `TimeSeriesSplit` folds within the train split to ensure reliable probabilities for cost-based decision making.
- **Unsupervised Anomaly Detection:** Isolation Forest and One-Class SVM were trained exclusively on `ON_TIME` train records to quantify unexpected operational drift.

## Things the next role must know (warnings, known issues, changed meanings)
- **Role 4 (Evaluation):** `ml_predictions.csv` includes all 3 baselines and both raw and calibrated XGBoost models across `val` and `test` splits. Every row sums to 1.0. You can run cost curve evaluation and threshold tuning immediately.
- **Role 3 (Integration):** The saved artifact in `Stage01_ML/artifacts/stage01_models.joblib` contains the full preprocessing and model pipeline ready for API inference.

## Open questions for other roles (name the role and the question)
- Role 4: What is the optimal expected cost threshold found on the validation split?
