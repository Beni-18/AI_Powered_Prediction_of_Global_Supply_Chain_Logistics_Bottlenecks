# Stage 01 - Role 4 - Evaluation Engineer -> hand-off note (04_evaluation_engineer.py)

status: READY
updated_utc: 2026-11-03
git_commit: branch eval/stage01-xgboost
data_version: v2 (SIM v0.3 driven by real weather, PortWatch traffic and size, GSCPI; train 42,000 / val 7,824 / test 7,894)

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

## Files produced (path | contract | rows | notes)
- `Stage01_ML/reports/eval_champion.json` | - | 1 | **Role 3 reads this**: model `xgboost`, alert threshold `0.39`, chosen on val (decision 0012)
- `Stage01_ML/reports/eval_metrics.json`, `eval_model_comparison.csv/png` | - | 7 models | test metrics with 95% day-block bootstrap intervals (1,000 resamples)
- `Stage01_ML/reports/eval_threshold_decision.md`, `eval_cost_curve.png`, `eval_threshold_sensitivity.csv` | - | - | alert threshold and its sensitivity to the miss cost
- `Stage01_ML/reports/eval_confusion.png`, `eval_calibration.png`, `eval_slices.csv/png`, `eval_anomaly.json`, `eval_leakage_tests.json`, `eval_api_latency.json`
- `claims/CLAIMS_LEDGER.md` | - | 15 rows C-ML-01..15 | recomputed for XGBoost and **VERIFIED** from a fresh clone on 2026-11-03 (every eval file identical; API p95 22.05 ms there)

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
Input check inside 04_evaluation_engineer.py: ml_predictions.csv passes the ml_predictions contract;
split, labels and row sets match the frozen prepared dataset for all 7 models (no errors, no warnings).
```

## Decisions and assumptions (what you decided and why, in plain English)
- **Data v2 (decision 0010):** the simulator is now driven by real data (Open-Meteo hourly weather, IMF PortWatch busy months and port size, NY Fed GSCPI). Shipment outcomes are still simulated, so rows stay `SYNTHETIC_CALIBRATED`. Numbers are not comparable with the earlier data.
- **Champion: `xgboost`** at alert threshold **0.39** (decision 0012, 3 Nov, team lead). It has the lowest validation cost of every model (0.3160 per shipment vs 0.3222 for logreg). Its saving over logreg on validation, 0.006 [-0.006, +0.019], is **not statistically clear**; `eval_champion.json`, the memo and C-ML-12 say so. Before 3 Nov the rule kept the baseline (logreg) in that case; `CHAMPION_RULE = "must_beat_baseline"` restores it.
- **Test period (xgboost):** 79.8% of critical delays caught, **37.4** false alarms per 100 alerts (within the soft budget of 40), cost 0.322 [0.291, 0.356] per shipment, PR-AUC 0.781, macro-F1 0.680. Logreg: 77.3%, 40.1, 0.337, 0.762, 0.666. Calibration is worse for xgboost (ECE 0.083 vs 0.049): its probabilities are less exact as percentages, while the alert threshold was set on validation for xgboost itself.
- **Leakage tests: PASS** on all 77 model inputs; strongest single input: the node's previous-hour utilisation, AUC 0.908.
- **Reproducibility:** a fresh clone regenerates the simulator data and reproduces the cleaned data, every prediction, the champion and every claim exactly with Role 1's library versions.

## Things the next role must know (warnings, known issues, changed meanings)
- **Role 3:** the API serves `xgboost` at 0.39 automatically; checked over real HTTP on 3 Nov: p95 20.3 ms, 0 errors, every answer has `requires_operator_review` and matches the champion. The field guide (.md/.html) is regenerated and says `xgboost`; the PDF needs re-exporting.
- **Role 1:** XGBoost is now the deployed model; its gain over logreg is not clear on validation, so more tuning or more training data that makes it clear would strengthen the claim. Any new model is re-evaluated in about 10 seconds.
- **Role 2:** chokepoints are still the weak spot; compare `eval_slices.csv` with your `eda_error_slices.csv`.

## Open questions for other roles (name the role and the question)
- **Team:** vote on decision 0009 (the hourly node table becomes a model input; the cleaning rules).
- **Role 3:** pin library versions (affects everyone's install).
