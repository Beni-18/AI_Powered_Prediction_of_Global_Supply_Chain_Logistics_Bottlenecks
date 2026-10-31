# Claims ledger

**Rule:** we only say a number aloud in a debate if it has a row here with status `VERIFIED`.
Owner: Role 4 (Evaluation Engineer). Rows are written by code (`common/evalkit` `ledger.upsert(...)`), not by hand.
`Reproduce command` must work from a clean clone. `Verified on` = date a clean-clone run reproduced the value.

Status values: `PENDING` (computed, not yet re-run clean) · `VERIFIED` · `RETRACTED` (kept for honesty, never quoted).

<!-- LEDGER:BEGIN -->
| ID | Day | Claim wording | Value [95% CI] | Source file | Reproduce command | Verified on | Status |
|---|---|---|---|---|---|---|---|
| C-DL-01 | 2 | transformer skill vs persistence, dwell, 24 h ahead (test) | 0.168 [0.147, 0.189] | Stage02_DL/reports/eval_metrics.json | `python Stage02_DL/04_evaluation_engineer.py` |  | PENDING |
| C-DL-02 | 2 | transformer skill vs persistence, dwell, 168 h (7 days) ahead (test) | 0.216 [0.191, 0.241] | Stage02_DL/reports/eval_metrics.json | `python Stage02_DL/04_evaluation_engineer.py` |  | PENDING |
| C-DL-03 | 2 | Horizons (of 5) where transformer has skill > 0 with the 95% interval above 0, dwell / throughput (test) | 5/5 dwell, 5/5 throughput | Stage02_DL/reports/eval_forecast_metrics.csv | `python Stage02_DL/04_evaluation_engineer.py` |  | PENDING |
| C-DL-04 | 2 | transformer skill vs the best baseline (chosen on val), dwell, 24 h (test) | 0.048 [0.035, 0.063] vs xgb_lag | Stage02_DL/reports/eval_metrics.json | `python Stage02_DL/04_evaluation_engineer.py` |  | PENDING |
| C-DL-05 | 2 | transformer warns about dwell bottleneck episodes in advance: recall [95% CI] (n episodes), test, alert rule chosen on val (B_prob_ge_0.70) | 0.69 [0.42, 0.87] (n = 13) | Stage02_DL/reports/eval_lead_time.csv | `python Stage02_DL/04_evaluation_engineer.py` |  | PENDING |
| C-DL-06 | 2 | transformer median lead time before onset [IQR], hours (test, same rule) | 32 [10, 89] | Stage02_DL/reports/eval_lead_time.csv | `python Stage02_DL/04_evaluation_engineer.py` |  | PENDING |
| C-DL-07 | 2 | transformer false alarms per node-week (test, same rule) | 0.06 | Stage02_DL/reports/eval_lead_time.csv | `python Stage02_DL/04_evaluation_engineer.py` |  | PENDING |
| C-DL-08 | 2 | transformer 80% interval coverage, dwell, all horizons (test; target 0.80) | 0.798 | Stage02_DL/reports/eval_interval_quality.csv | `python Stage02_DL/04_evaluation_engineer.py` |  | PENDING |
| C-DL-09 | 2 | Graph ablation: target-horizons where graph_full beats graph_none (paired 95% CI above 0), test | better in 0/10, worse in 3 | Stage02_DL/reports/eval_graph_ablation.csv | `python Stage02_DL/04_evaluation_engineer.py` |  | PENDING |
| C-DL-10 | 2 | Graph ablation: target-horizons where graph_full beats graph_shuffled (paired 95% CI above 0), test | better in 0/10, worse in 6 | Stage02_DL/reports/eval_graph_ablation.csv | `python Stage02_DL/04_evaluation_engineer.py` |  | PENDING |
| C-DL-11 | 2 | Leakage and time-order checks on the forecast file (origins, purge gap, y_true re-read from the grid) | PASS | Stage02_DL/reports/eval_leakage_tests.json | `python Stage02_DL/04_evaluation_engineer.py` |  | PENDING |
| C-DL-12 | 2 | REAL data (IMF PortWatch daily ship counts, test 2025-26): transformer skill vs persistence by horizon | 24 h: 0.253 [0.235, 0.270]; 72 h: 0.257 [0.241, 0.272]; 168 h: 0.215 [0.200, 0.230] | Stage02_DL/reports/eval_real_forecast_metrics.csv | `python Stage02_DL/04_evaluation_engineer.py` |  | PENDING |
| C-DL-13 | 2 | Early-warning cost on test (10 per missed bottleneck + 1 per false alarm; each model's rule chosen on val), lower is better | transformer 49 vs lstm 54, xgb_lag 98, persistence 92, stage1_risk 401 | Stage02_DL/reports/eval_lead_time.csv | `python Stage02_DL/04_evaluation_engineer.py` |  | PENDING |
| C-DL-14 | 2 | REAL data: transformer skill vs the best baseline chosen on val, by horizon (test 2025-26) | 24 h: 0.033 [0.021, 0.046]; 72 h: 0.017 [-0.005, 0.040]; 168 h: 0.035 [0.023, 0.049] | Stage02_DL/reports/eval_real_forecast_metrics.csv | `python Stage02_DL/04_evaluation_engineer.py` |  | PENDING |
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
