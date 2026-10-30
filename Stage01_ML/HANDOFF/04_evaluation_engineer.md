# Stage 01 - Role 4 - Evaluation Engineer -> hand-off note (04_evaluation_engineer.py)

status: READY
updated_utc: 2026-10-30
git_commit: see PR from branch eval/stage01-reeval
data_version: v1 (prepared_dataset.csv.gz, train 42,000 / val 7,820 / test 7,899); predictions from Role 1's tuning (PR 14)

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

## Files produced (path | contract | rows | notes)
- `Stage01_ML/reports/eval_champion.json` | - | 1 | **Role 3 reads this**: model `xgboost_monotone`, alert threshold `0.32`, chosen on val
- `Stage01_ML/reports/eval_metrics.json`, `eval_model_comparison.csv/png` | - | 7 models | test metrics with 95% day-block bootstrap intervals (1,000 resamples)
- `Stage01_ML/reports/eval_threshold_decision.md`, `eval_cost_curve.png`, `eval_threshold_sensitivity.csv` | - | - | alert threshold and its sensitivity to the miss cost
- `Stage01_ML/reports/eval_confusion.png`, `eval_calibration.png`, `eval_slices.csv/png`, `eval_anomaly.json`, `eval_leakage_tests.json`, `eval_api_latency.json`
- `claims/CLAIMS_LEDGER.md` | - | 15 rows C-ML-01..15 | all **VERIFIED** from a fresh clone on 2026-10-30

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
Input check inside 04_evaluation_engineer.py: ml_predictions.csv passes the ml_predictions contract;
split, labels and row sets match the frozen prepared dataset for all 7 models (no errors, no warnings).
```

## Decisions and assumptions (what you decided and why, in plain English)
- **New champion: `xgboost_monotone`** (Role 1's tuned XGBoost whose risk can only rise with dwell, customs hold and weather). It beats the best baseline (`logreg`) on **validation** cost by 0.024 per shipment, 95% paired bootstrap 0.007 to 0.040, so the gain is clear. The previous champion was `logreg`, because no model beat it before the tuning.
- **Alert threshold 0.32** (re-chosen on validation, as decision 0004 requires when the champion changes). On test: catches 82% of critical delays, 38 false alarms per 100 alerts (budget 40), cost 0.408 per shipment vs 0.486 for `logreg`.
- **Leakage tests: PASS.** The single-feature ceiling now reads Role 1's `feature_matrix_test.csv.gz`, so all 68 model inputs (including `cal_`, `eng_`, `roll_` and one-hot columns) are checked; the strongest is dwell time at AUC 0.905 (< 0.95).
- **Reproducibility:** a fresh clone reproduces every prediction, score and claim **exactly** with Role 1's library versions (Python 3.13, numpy 2.3.4, pandas 2.3.3, scikit-learn 1.8.0, xgboost 3.1.3, lightgbm 4.7.0; listed in `model_card.json`). With newer versions (xgboost 3.4, scikit-learn 1.9) the tree models shift slightly: same champion, still clearly better than the baseline, but threshold 0.30, 83% caught, 40.6 false alarms per 100 alerts.

## Things the next role must know (warnings, known issues, changed meanings)
- **Role 3:** the API switches to the new champion by itself (it reads `eval_champion.json`); checked over real HTTP: p95 17.7 ms, 0 errors, every answer has `requires_operator_review`. Please regenerate the field guide (`python Stage01_ML/05_integration_engineer.py`): it still says 37% / 79%. Please also pin the ML library versions in `requirements.txt` (see above) so every laptop reproduces the same numbers.
- **Role 1:** weakest slices are still the chokepoints; `eval_slices.csv` has the details.
- **Role 2:** compare `eval_slices.csv` with your `eda_error_slices.csv` when it lands.

## Open questions for other roles (name the role and the question)
- **Team:** note the new champion and threshold in `docs/decisions/0004` (the rule is unchanged; the numbers moved).
- **Role 3:** pin library versions (decision for the team; it affects everyone's install).
