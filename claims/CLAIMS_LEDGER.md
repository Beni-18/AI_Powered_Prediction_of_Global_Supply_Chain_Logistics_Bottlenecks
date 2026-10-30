# Claims ledger

**Rule:** we only say a number aloud in a debate if it has a row here with status `VERIFIED`.
Owner: Role 4 (Evaluation Engineer). Rows are written by code (`common/evalkit` `ledger.upsert(...)`), not by hand.
`Reproduce command` must work from a clean clone. `Verified on` = date a clean-clone run reproduced the value.

Status values: `PENDING` (computed, not yet re-run clean) · `VERIFIED` · `RETRACTED` (kept for honesty, never quoted).

<!-- LEDGER:BEGIN -->
| ID | Day | Claim wording | Value [95% CI] | Source file | Reproduce command | Verified on | Status |
|---|---|---|---|---|---|---|---|
| C-ML-01 | 1 | logreg macro-F1 on the held-out test period (most-likely-class rule) | 0.666 [0.638, 0.695] | Stage01_ML/reports/eval_metrics.json | `python Stage01_ML/04_evaluation_engineer.py` | 2026-10-30 | VERIFIED |
| C-ML-02 | 1 | logreg (best baseline) macro-F1 on test (most-likely-class rule) | 0.666 [0.638, 0.695] | Stage01_ML/reports/eval_metrics.json | `python Stage01_ML/04_evaluation_engineer.py` | 2026-10-30 | VERIFIED |
| C-ML-03 | 1 | logreg PR-AUC for CRITICAL_DELAY on test | 0.762 [0.707, 0.814] | Stage01_ML/reports/eval_metrics.json | `python Stage01_ML/04_evaluation_engineer.py` | 2026-10-30 | VERIFIED |
| C-ML-04 | 1 | Alert threshold on P(critical), chosen on validation by minimum cost | 0.39 | Stage01_ML/reports/eval_champion.json | `python Stage01_ML/04_evaluation_engineer.py` | 2026-10-30 | VERIFIED |
| C-ML-05 | 1 | logreg expected cost per shipment at that threshold (test) | 0.337 [0.306, 0.370] | Stage01_ML/reports/eval_metrics.json | `python Stage01_ML/04_evaluation_engineer.py` | 2026-10-30 | VERIFIED |
| C-ML-06 | 1 | Share of CRITICAL delays caught at the frozen threshold (test) | 0.773 | Stage01_ML/reports/eval_metrics.json | `python Stage01_ML/04_evaluation_engineer.py` | 2026-10-30 | VERIFIED |
| C-ML-07 | 1 | False alarms per 100 alerts at the frozen threshold (test) | 40.1 | Stage01_ML/reports/eval_metrics.json | `python Stage01_ML/04_evaluation_engineer.py` | 2026-10-30 | VERIFIED |
| C-ML-08 | 1 | Scoring sanity check: model trained on shuffled labels scores near chance (does not prove absence of leakage) | 0.270 | Stage01_ML/reports/eval_leakage_tests.json | `python Stage01_ML/04_evaluation_engineer.py` | 2026-10-30 | VERIFIED |
| C-ML-09 | 1 | Strongest single feature's AUC for CRITICAL (must stay under 0.95) | 0.908 | Stage01_ML/reports/eval_leakage_tests.json | `python Stage01_ML/04_evaluation_engineer.py` | 2026-10-30 | VERIFIED |
| C-ML-10 | 1 | Time-order and purge-gap check between train, val and test | PASS | Stage01_ML/reports/eval_leakage_tests.json | `python Stage01_ML/04_evaluation_engineer.py` | 2026-10-30 | VERIFIED |
| C-ML-11 | 1 | logreg calibration error (ECE) for P(critical) on test | 0.049 | Stage01_ML/reports/eval_calibration.png | `python Stage01_ML/04_evaluation_engineer.py` | 2026-10-30 | VERIFIED |
| C-ML-12 | 1 | Does the champion beat the best baseline on validation cost? | False | Stage01_ML/reports/eval_champion.json | `python Stage01_ML/04_evaluation_engineer.py` | 2026-10-30 | VERIFIED |
| C-ML-13 | 1 | logreg macro-F1 at the frozen alert threshold (the decision the operator gets) | 0.652 [0.624, 0.682] | Stage01_ML/reports/eval_metrics.json | `python Stage01_ML/04_evaluation_engineer.py` | 2026-10-30 | VERIFIED |
| C-ML-14 | 1 | Id-column check: how much more than the clock an id's numeric order says about CRITICAL, in AUC (must stay under 0.05) | 0.000 | Stage01_ML/reports/eval_leakage_tests.json | `python Stage01_ML/04_evaluation_engineer.py` | 2026-10-30 | VERIFIED |
| C-ML-15 | 1 | Stage 1 API latency per shipment over real HTTP, 1,000 calls (p95) | 15.36 ms (fresh-clone re-run: 15.6 ms; target < 50 ms) | Stage01_ML/reports/eval_api_latency.json | `uvicorn 05_integration_engineer:app --port 8001 + python Stage01_ML/04_evaluation_engineer.py --api-url http://127.0.0.1:8001` | 2026-10-30 | VERIFIED |
<!-- LEDGER:END -->
