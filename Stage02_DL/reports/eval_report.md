# Stage 2 evaluation (deep learning forecasts)

> **What this shows:** how well each forecaster predicts node dwell and throughput 1 h to 7 days ahead, how early
> it warns about dwell bottlenecks, and whether the graph (trade lanes) really helps. **Reproduce:**
> `python Stage02_DL/04_evaluation_engineer.py`. Every number below is in `eval_metrics.json` or a CSV next to it.

**Champion (chosen on validation, never on test):** `ensemble`. Beats the best baseline on validation: True (best baseline `xgb_lag`).

## Headline numbers (test period)
- Skill vs persistence, dwell 24 h: **0.165 [0.140, 0.190]**; 168 h: **0.231 [0.205, 0.257]**.
- Horizons with CI-positive skill (dwell / throughput): **5/5 dwell, 5/5 throughput** (target: >= 4 of 5).
- Skill vs the best baseline (dwell 24 h): **0.037 [0.030, 0.044] vs xgb_lag**.
- Early warning (rule B_prob_ge_0.65, chosen on validation): recall **0.77 [0.50, 0.92] (n = 13)**, median lead **22 [7, 76] h**, false alarms per node-week **0.06**.
- Early-warning cost on test (10 per miss + 1 per false alarm): **ensemble 38 vs transformer 38, lstm 55, xgb_lag 52, persistence 93, stage1_risk 401**.
- 80% interval coverage (dwell): **0.799**.
- Graph ablation vs graph_none: better in 0/10, worse in 10; vs graph_shuffled: better in 0/10, worse in 8.
- Seed check (test, dwell 24 h): **ensemble 0.168 (sd 0.005, range 0.165 to 0.173, 3 seeds)**; lead over xgb_lag across seeds: **+0.030 to +0.038 (positive for every seed)**.
- Backtest, 3 rolling-origin folds (dwell 24 h): **ensemble 0.151 (sd 0.015, range 0.136 to 0.165, 3 folds) vs xgb_lag 0.132**; pooled early warning (rule A): ensemble 5/6 caught, cost 25; transformer 5/6 caught, cost 27; xgb_lag 4/6 caught, cost 43; persistence 4/6 caught, cost 77.
- REAL data (PortWatch, test 2025-26), skill vs persistence: **24 h: 0.256 [0.241, 0.270]; 72 h: 0.243 [0.229, 0.257]; 168 h: 0.218 [0.205, 0.230]**; vs best baseline: **24 h: 0.020 [0.013, 0.026]; 72 h: -0.002 [-0.022, 0.019]; 168 h: 0.025 [0.019, 0.033]**.

## Honest reading
- The champion was picked on validation; its test skill is reported as is, even where it is not significant.
- **Recall alone is not a fair score.** With rule A even persistence catches 12/13 test episodes (noisy readings crossing the line), at 71 false alarms. Compare models on the warning cost of their validation-chosen rule: ensemble 38 vs transformer 38, lstm 55, xgb_lag 52, persistence 93, stage1_risk 401.
- **The graph does not clearly help** on the simulated track: graph_full vs graph_none better in 0/10, worse in 10, vs graph_shuffled better in 0/10, worse in 8 (target x horizon cells, paired 95% CI). That is a finding, not a failure.
- The graph does not help on the REAL track either (eval_real_graph_ablation.csv).
- **Seed check:** retrained with 3 seeds, ensemble's test skill at 24 h (dwell) is ensemble 0.168 (sd 0.005, range 0.165 to 0.173, 3 seeds); its lead over xgb_lag ranges +0.030 to +0.038 (positive for every seed). Validation does NOT pick the same model for every seed, so the choice between those models is within seed noise: seed 7: transformer, seed 13: transformer, seed 42: ensemble.
- **Backtest (3 rolling-origin folds, Jun-Aug):** skill at 24 h (dwell) ensemble 0.151 (sd 0.015, range 0.136 to 0.165, 3 folds) vs xgb_lag 0.132. Early warning pooled over the folds (rule A; 6 episodes, separate from the 13 test episodes): ensemble 5/6 caught, cost 25; transformer 5/6 caught, cost 27; xgb_lag 4/6 caught, cost 43; persistence 4/6 caught, cost 77.
- Data are simulated (SYNTHETIC_CALIBRATED); the real-data track checks the same models on IMF PortWatch counts.

## Visual benchmark episodes (test)
- PORT_US_NYC onset 2025-12-22T19:00:00+00:00: first-alert lead (h) {'xgb_lag': 1.0, 'lstm': 1.0, 'graph_full': 7.0, 'ensemble': 1.0, 'stage1_risk': 73.0}
- CHOKE_HORMUZ onset 2025-12-05T06:00:00+00:00: first-alert lead (h) {'persistence': 66.0, 'xgb_lag': 36.0, 'graph_full': 72.0, 'ensemble': 36.0, 'stage1_risk': 144.0}
- PORT_NL_RTM onset 2025-12-07T23:00:00+00:00: first-alert lead (h) {'xgb_lag': 23.0, 'lstm': 11.0, 'graph_full': 23.0, 'ensemble': 11.0}

## Files
- `eval_forecast_metrics.csv`, `eval_skill_by_horizon.png`: accuracy and skill with 95% CIs
- `eval_interval_quality.csv`: coverage, width, interval score
- `eval_lead_time.csv`, `eval_lead_time.png`: early warning
- `eval_graph_ablation.csv`: does the real graph help
- `eval_visual_benchmark.png/.csv/.md`: problem-statement benchmark
- `eval_slices.csv`, `eval_leakage_tests.json`, `eval_champion.json`, `eval_alert_thresholds.csv`
- `eval_real_*`: the same on REAL IMF PortWatch data
- `eval_backtest_folds.csv`, `eval_backtest_summary.csv`, `eval_backtest_warning.csv`: rolling-origin folds
- `eval_seed_runs.csv`, `eval_seed_summary.csv`: the same models trained with 3 seeds
