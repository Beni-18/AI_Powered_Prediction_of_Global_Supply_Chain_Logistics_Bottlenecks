# Claims ledger

**Rule:** we only say a number aloud in a debate if it has a row here with status `VERIFIED`.
Owner: Role 4 (Evaluation Engineer). Rows are written by code (`common/evalkit` `ledger.upsert(...)`), not by hand.
`Reproduce command` must work from a clean clone. `Verified on` = date a clean-clone run reproduced the value.

Status values: `PENDING` (computed, not yet re-run clean) · `VERIFIED` · `RETRACTED` (kept for honesty, never quoted).

<!-- LEDGER:BEGIN -->
| ID | Day | Claim wording | Value [95% CI] | Source file | Reproduce command | Verified on | Status |
|---|---|---|---|---|---|---|---|
| C-ML-01 | 1 | xgboost_monotone macro-F1 on the held-out test period (most-likely-class rule) | 0.686 [0.668, 0.706] | Stage01_ML/reports/eval_metrics.json | `python Stage01_ML/04_evaluation_engineer.py` |  | PENDING |
| C-ML-02 | 1 | logreg (best baseline) macro-F1 on test (most-likely-class rule) | 0.658 [0.635, 0.681] | Stage01_ML/reports/eval_metrics.json | `python Stage01_ML/04_evaluation_engineer.py` |  | PENDING |
| C-ML-03 | 1 | xgboost_monotone PR-AUC for CRITICAL_DELAY on test | 0.814 [0.778, 0.849] | Stage01_ML/reports/eval_metrics.json | `python Stage01_ML/04_evaluation_engineer.py` |  | PENDING |
| C-ML-04 | 1 | Alert threshold on P(critical), chosen on validation by minimum cost | 0.32 | Stage01_ML/reports/eval_champion.json | `python Stage01_ML/04_evaluation_engineer.py` |  | PENDING |
| C-ML-05 | 1 | xgboost_monotone expected cost per shipment at that threshold (test) | 0.408 [0.373, 0.441] | Stage01_ML/reports/eval_metrics.json | `python Stage01_ML/04_evaluation_engineer.py` |  | PENDING |
| C-ML-06 | 1 | Share of CRITICAL delays caught at the frozen threshold (test) | 0.818 | Stage01_ML/reports/eval_metrics.json | `python Stage01_ML/04_evaluation_engineer.py` |  | PENDING |
| C-ML-07 | 1 | False alarms per 100 alerts at the frozen threshold (test) | 38.1 | Stage01_ML/reports/eval_metrics.json | `python Stage01_ML/04_evaluation_engineer.py` |  | PENDING |
| C-ML-08 | 1 | Scoring sanity check: model trained on shuffled labels scores near chance (does not prove absence of leakage) | 0.236 | Stage01_ML/reports/eval_leakage_tests.json | `python Stage01_ML/04_evaluation_engineer.py` |  | PENDING |
| C-ML-09 | 1 | Strongest single feature's AUC for CRITICAL (must stay under 0.95) | 0.905 | Stage01_ML/reports/eval_leakage_tests.json | `python Stage01_ML/04_evaluation_engineer.py` |  | PENDING |
| C-ML-10 | 1 | Time-order and purge-gap check between train, val and test | PASS | Stage01_ML/reports/eval_leakage_tests.json | `python Stage01_ML/04_evaluation_engineer.py` |  | PENDING |
| C-ML-11 | 1 | xgboost_monotone calibration error (ECE) for P(critical) on test | 0.076 | Stage01_ML/reports/eval_calibration.png | `python Stage01_ML/04_evaluation_engineer.py` |  | PENDING |
| C-ML-12 | 1 | Does the champion beat the best baseline on validation cost? | True | Stage01_ML/reports/eval_champion.json | `python Stage01_ML/04_evaluation_engineer.py` |  | PENDING |
| C-ML-13 | 1 | xgboost_monotone macro-F1 at the frozen alert threshold (the decision the operator gets) | 0.657 [0.636, 0.678] | Stage01_ML/reports/eval_metrics.json | `python Stage01_ML/04_evaluation_engineer.py` |  | PENDING |
| C-ML-14 | 1 | Id-column check: how much more than the clock an id's numeric order says about CRITICAL, in AUC (must stay under 0.05) | 0.000 | Stage01_ML/reports/eval_leakage_tests.json | `python Stage01_ML/04_evaluation_engineer.py` |  | PENDING |
| C-ML-15 | 1 | Stage 1 API latency per shipment over real HTTP, 1,000 calls (p95) | 17.7 ms | Stage01_ML/reports/eval_api_latency.json | `uvicorn 05_integration_engineer:app --port 8001 + python Stage01_ML/04_evaluation_engineer.py --api-url http://127.0.0.1:8001` |  | PENDING |
<!-- LEDGER:END -->
