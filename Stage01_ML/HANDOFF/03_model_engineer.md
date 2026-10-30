# Stage 01 - Role 1 - ML Engineer (model role) -> hand-off note (03_ml_engineer.py)

status: READY
updated_utc: 2026-10-30T10:22:00Z
git_commit: HEAD
data_version: v1

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

## Files produced (path | contract | rows | notes)
- `Stage01_ML/predictions/ml_predictions.csv` | `ml_predictions` | 110033 | Baselines (majority, rule_dwell, logreg) and models (random_forest, xgboost, xgboost_monotone, lightgbm) with anomaly scores
- `Stage01_ML/predictions/feature_importance_matrix.csv` | none | 68 | SHAP on val + permutation importance + XGBoost gain
- `Stage01_ML/predictions/feature_matrix_test.csv.gz` | none | 7899 | Exact test feature matrix (68 features + record_id) for Role 4 leakage ceiling audit
- `Stage01_ML/artifacts/stage01_models.joblib` | none | 1 | Fitted preprocessing and model pipeline (14.4 MB)
- `Stage01_ML/artifacts/model_card.json` | none | 1 | Complete audit card with tuning search space and candidate metrics

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
PASS  Stage01_ML/predictions/ml_predictions.csv  (110033 rows, schema 'ml_predictions')
```

## Decisions and assumptions (what you decided and why, in plain English)
- **Tuning on Train Only:** Executed random search (20 trials) on `TimeSeriesSplit(3)` strictly inside the training partition (score = Macro-F1 + 0.5 * PR-AUC). Neither validation nor test splits were touched during tuning. Best hyperparameters: `learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7, min_child_weight=5.0, gamma=0.2, n_estimators=250`.
- **Beating LogReg Champion:** Cost-aligned sample weights (0.5 for ON_TIME, 1.8 for AT_RISK, 3.8 for CRITICAL) were incorporated into tree training to directly penalize the 10x cost of missed critical delays. This enabled `xgboost_monotone` to beat `logreg` on validation expected cost (0.3862 vs 0.4002), with paired block bootstrap confirming statistical significance (95% CI cost saving: [0.0037, 0.0346]).
- **Monotonic Constraints:** Added `xgboost_monotone` with monotonic non-decreasing constraints (+1) on `dwell_time_hours`, `customs_hold_hours`, and `weather_severity_index`. It achieves the lowest validation expected cost and highest test PR-AUC (0.814).
- **LightGBM Comparison:** Added `lightgbm` as a comparative tree architecture. It achieves validation expected cost of 0.4067 (and 0.3673 at optimal threshold), confirming tree-based advantage over linear baselines.
- **Why `xgboost_cal` was dropped:** Post-hoc isotonic calibration on `TimeSeriesSplit` inside train caused severe probability suppression for CRITICAL_DELAY because the early training period had a calmer regime (11.9% critical rate vs 17.2% in val and 17.9% in test). Under the 10:1 cost policy, squashing P(critical) down caused severe false negative penalties (cost 0.770 vs 0.386). We verified that calibrating on late-train slices still yielded an inferior cost (0.675). Following role guidance, unfixable post-hoc calibration was dropped in favor of cost-weighted loss training.
- **Pipeline in `common.modelkit`:** Migrated `Stage01Pipeline` into `common/modelkit/pipeline_stage01.py` and exported it from `common.modelkit` so Role 3's API can unpickle and serve models directly without importing `03_ml_engineer.py`.

## Things the next role must know (warnings, known issues, changed meanings)
- **Role 4 (Evaluation):**
  - Predictions now contain `xgboost_monotone` and `lightgbm` alongside `xgboost`, `random_forest`, and all 3 baselines.
  - `xgboost_monotone` beats the `logreg` baseline on validation cost (0.3862 vs 0.4002), satisfying the champion requirement (`beats_baseline: True`).
  - Saved `Stage01_ML/predictions/feature_matrix_test.csv.gz` containing the exact 68 derived feature columns on the test split for the single-feature leakage ceiling audit.
  - Re-run `python Stage01_ML/04_evaluation_engineer.py` to regenerate the official evaluation reports and claims ledger.
- **Role 3 (Integration):**
  - Model pipeline class `Stage01Pipeline` is now in `common.modelkit` (and aliased to `__main__` for backward compatibility).

## Open questions for other roles (name the role and the question)
- Role 4: Please re-run `python Stage01_ML/04_evaluation_engineer.py` with the updated predictions.
