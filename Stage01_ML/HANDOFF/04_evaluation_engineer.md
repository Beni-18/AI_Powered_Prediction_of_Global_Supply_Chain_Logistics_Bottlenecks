# Stage 01 - Role 4 - Evaluation Engineer -> hand-off note (04_evaluation_engineer.py)

status: READY
updated_utc: 2026-10-30
git_commit: see PR from branch eval/stage01-complete
data_version: v1 (prepared_dataset.csv.gz, train 42,000 / val 7,820 / test 7,899)

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

## Files produced (path | contract | rows | notes)
- `Stage01_ML/reports/eval_champion.json` | - | 1 | **Role 3 reads this**: model `logreg`, alert threshold `0.37`, chosen on val
- `Stage01_ML/reports/eval_metrics.json`, `eval_model_comparison.csv/png` | - | 6 models | test metrics with 95% day-block bootstrap CIs (1,000 resamples)
- `Stage01_ML/reports/eval_threshold_decision.md`, `eval_cost_curve.png`, `eval_threshold_sensitivity.csv` | - | - | Team Huddle answer: false-alarm threshold and sensitivity to the miss cost
- `Stage01_ML/reports/eval_confusion.png`, `eval_calibration.png`, `eval_slices.csv/png`, `eval_anomaly.json`, `eval_leakage_tests.json`
- `claims/CLAIMS_LEDGER.md` | - | 14 rows C-ML-01..14 | status PENDING until a clean-clone re-run

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
Input check inside 04_evaluation_engineer.py: ml_predictions.csv passes the ml_predictions contract;
split, labels and row sets match the frozen prepared dataset for all 6 models (no errors, no warnings).
```

## Decisions and assumptions (what you decided and why, in plain English)
- **Champion = `logreg` (a baseline).** By our frozen rule a model is champion only if it beats the best baseline on **validation** cost with a 95% paired bootstrap. `xgboost` does not: val cost saving -0.004 [-0.023, +0.014]. On **test** `xgboost` is cheaper (0.413 vs 0.485 per shipment), but we never choose on test.
- **Alert threshold 0.37** on P(critical), minimum expected cost on val with the cost policy (miss = 10, false alarm = 1). On test: catches 79% of critical delays, 38 false alarms per 100 alerts (inside the 40 budget).
- Leakage tests: feature policy PASS, time order and 7-day purge PASS, id-vs-clock PASS, too-good alarm PASS; single-feature ceiling PARTIAL (max AUC 0.874; model-built `cal_`/`eng_`/`roll_` features cannot be tested without the feature matrix).

## Things the next role must know (warnings, known issues, changed meanings)
- **Role 3:** serve `model_name` and `threshold` from `reports/eval_champion.json`; do not hard-code them. Every response must carry `requires_operator_review: true`.
- **Role 1:** `xgboost_cal` is the worst candidate on cost (val 0.770): isotonic calibration fitted on the calmer train period pushes P(critical) down (test threshold 0.03). Weakest slices are the chokepoints: critical recall Suez 5%, Malacca 6%, Panama 0%.
- **Role 2:** please compare `eval_slices.csv` with your noise-floor findings (chokepoints vs gateway ports).

## Open questions for other roles (name the role and the question)
- **Team Huddle:** ratify `contracts/cost_policy.json` and the 0.37 threshold (record in `docs/decisions/`).
- **Role 1:** can a tuned tree model beat `logreg` on validation cost? If yes, re-run and I re-evaluate (about 10 seconds).
- **Role 1:** please save the test feature matrix so the single-feature ceiling can cover derived features.
