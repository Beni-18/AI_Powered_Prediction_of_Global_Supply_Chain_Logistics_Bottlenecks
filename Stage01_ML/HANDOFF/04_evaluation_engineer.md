# Stage 01 - Role 4 - Evaluation Engineer -> hand-off note (04_evaluation_engineer.py)

status: READY
updated_utc: 2026-10-30
git_commit: see PR from branch data/real-calibration
data_version: v2 (SIM v0.3 driven by real weather, PortWatch traffic and size, GSCPI; train 42,000 / val 7,824 / test 7,894)

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

## Files produced (path | contract | rows | notes)
- `Stage01_ML/reports/eval_champion.json` | - | 1 | **Role 3 reads this**: model `logreg`, alert threshold `0.39`, chosen on val
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
- **Data v2 (decision 0010):** the simulator is now driven by real data (Open-Meteo hourly weather, IMF PortWatch busy months and port size, NY Fed GSCPI). Shipment outcomes are still simulated, so rows stay `SYNTHETIC_CALIBRATED`. Numbers are not comparable with the earlier data.
- **Champion: `logreg`** at alert threshold **0.39** (chosen on validation). XGBoost is slightly cheaper on validation (0.006 per shipment) but its 95% range (-0.006 to +0.019) includes zero, so the simpler model stays.
- **Test period:** 77.3% of critical delays caught, **40.1** false alarms per 100 alerts (0.1 over the soft budget of 40), cost 0.337 per shipment, PR-AUC 0.762. XGBoost would sit at 79.8% caught and 37.4 false alarms per 100 alerts.
- **Leakage tests: PASS** on all 77 model inputs; strongest single input: the node's previous-hour utilisation, AUC 0.908.
- **Reproducibility:** a fresh clone regenerates the simulator data and reproduces the cleaned data, every prediction, the champion and every claim exactly with Role 1's library versions.

## Things the next role must know (warnings, known issues, changed meanings)
- **Role 3:** the API serves `logreg` at 0.34 automatically; checked over real HTTP: p95 14 ms, 0 errors, every answer has `requires_operator_review`. The field guide is regenerated; the PDF needs re-exporting. Please pin the ML library versions in `requirements.txt`.
- **Role 1:** the tree models may now over-fit (learning rate 0.08, all 250 trees used); a re-tune with a smaller learning rate could beat logreg again. Any new model is re-evaluated in about 10 seconds.
- **Role 2:** chokepoints are still the weak spot; compare `eval_slices.csv` with your `eda_error_slices.csv`.

## Open questions for other roles (name the role and the question)
- **Team:** vote on decision 0009 (the hourly node table becomes a model input; the cleaning rules).
- **Role 3:** pin library versions (affects everyone's install).
