# Stage 01 - Role 4 - Evaluation Engineer -> hand-off note (04_evaluation_engineer.py)

status: READY
updated_utc: 2026-10-30
git_commit: see PR from branch ml/stage01-improve
data_version: v1.1 (prepared_dataset.csv.gz with dwell corrections; train 42,000 / val 7,820 / test 7,899)

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

## Files produced (path | contract | rows | notes)
- `Stage01_ML/reports/eval_champion.json` | - | 1 | **Role 3 reads this**: model `logreg`, alert threshold `0.34`, chosen on val
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
- **What changed (decision 0009):** corrected the two dwell reporting faults, removed `cal_month`, added node-relative and past-hour node-history inputs, and moved XGBoost's early stopping off validation. Each change came from a controlled experiment that changed one thing at a time and judged it on validation cost only.
- **Champion: `logreg`** (lowest validation cost, 0.310). The tree models are clearly worse on validation with the new inputs (xgboost_monotone +0.039 per shipment, 95% range +0.023 to +0.057), so the rule keeps the simpler model.
- **Alert threshold 0.34** (re-chosen on validation, as decision 0004 requires). On test: cost **0.369** per shipment (previous champion 0.408), **82.8%** of critical delays caught (was 81.8%), **39.6** false alarms per 100 alerts (budget 40), PR-AUC **0.829** (was 0.814).
- **Leakage tests: PASS** on all 77 model inputs (Role 1's test feature matrix); the strongest single input is the node's previous-hour inventory at AUC 0.917 (< 0.95). It comes from a completed past hour, which the feature policy's `lag_` rule allows.
- **Reproducibility:** a fresh clone reproduces the data, every prediction, the champion and every claim exactly with Role 1's library versions (Python 3.13, numpy 2.3.4, pandas 2.3.3, scikit-learn 1.8.0, xgboost 3.1.3, lightgbm 4.7.0).

## Things the next role must know (warnings, known issues, changed meanings)
- **Role 3:** the API serves `logreg` at 0.34 automatically; checked over real HTTP: p95 14 ms, 0 errors, every answer has `requires_operator_review`. The field guide is regenerated; the PDF needs re-exporting. Please pin the ML library versions in `requirements.txt`.
- **Role 1:** the tree models may now over-fit (learning rate 0.08, all 250 trees used); a re-tune with a smaller learning rate could beat logreg again. Any new model is re-evaluated in about 10 seconds.
- **Role 2:** chokepoints are still the weak spot; compare `eval_slices.csv` with your `eda_error_slices.csv`.

## Open questions for other roles (name the role and the question)
- **Team:** vote on decision 0009 (the hourly node table becomes a model input; the cleaning rules).
- **Role 3:** pin library versions (affects everyone's install).
