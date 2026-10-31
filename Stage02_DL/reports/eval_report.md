# Stage 2 evaluation (deep learning forecasts)

> **What this shows:** how well each forecaster predicts node dwell and throughput 1 h to 7 days ahead, how early
> it warns about dwell bottlenecks, and whether the graph (trade lanes) really helps. **Reproduce:**
> `python Stage02_DL/04_evaluation_engineer.py`. Every number below is in `eval_metrics.json` or a CSV next to it.

**Champion (chosen on validation, never on test):** `transformer`. Beats the best baseline on validation: True (best baseline `xgb_lag`).

## Headline numbers (test period)
- Skill vs persistence, dwell 24 h: **0.168 [0.147, 0.189]**; 168 h: **0.216 [0.191, 0.241]**.
- Horizons with CI-positive skill (dwell / throughput): **5/5 dwell, 5/5 throughput** (target: >= 4 of 5).
- Skill vs the best baseline (dwell 24 h): **0.048 [0.035, 0.063] vs xgb_lag**.
- Early warning (rule B_prob_ge_0.70, chosen on validation): recall **0.69 [0.42, 0.87] (n = 13)**, median lead **32 [10, 89] h**, false alarms per node-week **0.06**.
- Early-warning cost on test (10 per miss + 1 per false alarm): **transformer 49 vs lstm 54, xgb_lag 98, persistence 92, stage1_risk 401**.
- 80% interval coverage (dwell): **0.798**.
- Graph ablation vs graph_none: better in 0/10, worse in 3; vs graph_shuffled: better in 0/10, worse in 6.
- REAL data (PortWatch, test 2025-26), skill vs persistence: **24 h: 0.253 [0.235, 0.270]; 72 h: 0.257 [0.241, 0.272]; 168 h: 0.215 [0.200, 0.230]**; vs best baseline: **24 h: 0.033 [0.021, 0.046]; 72 h: 0.017 [-0.005, 0.040]; 168 h: 0.035 [0.023, 0.049]**.

## Honest reading
- The champion was picked on validation; its test skill is reported as is, even where it is not significant.
- **Recall alone is not a fair score.** With rule A even persistence catches 12/13 test episodes (noisy readings crossing the line), at 71 false alarms. Compare models on the warning cost of their validation-chosen rule: transformer 49 vs lstm 54, xgb_lag 98, persistence 92, stage1_risk 401.
- **The graph does not clearly help** on the simulated track: graph_full vs graph_none better in 0/10, worse in 3, vs graph_shuffled better in 0/10, worse in 6 (target x horizon cells, paired 95% CI). That is a finding, not a failure.
- The graph does not help on the REAL track either (eval_real_graph_ablation.csv).
- These numbers come from the --fast training run (hidden 32, <= 10 epochs); claims stay PENDING until the full run.
- Data are simulated (SYNTHETIC_CALIBRATED); the real-data track checks the same models on IMF PortWatch counts.

## Visual benchmark episodes (test)
- PORT_US_NYC onset 2025-12-22T19:00:00+00:00: first-alert lead (h) {'graph_full': 1.0, 'stage1_risk': 73.0}
- CHOKE_HORMUZ onset 2025-12-05T06:00:00+00:00: first-alert lead (h) {'persistence': 66.0, 'xgb_lag': 36.0, 'lstm': 36.0, 'graph_full': 72.0, 'transformer': 36.0, 'stage1_risk': 144.0}
- PORT_NL_RTM onset 2025-12-07T23:00:00+00:00: first-alert lead (h) {'xgb_lag': 11.0, 'lstm': 23.0, 'graph_full': 23.0, 'transformer': 11.0}

## Files
- `eval_forecast_metrics.csv`, `eval_skill_by_horizon.png`: accuracy and skill with 95% CIs
- `eval_interval_quality.csv`: coverage, width, interval score
- `eval_lead_time.csv`, `eval_lead_time.png`: early warning
- `eval_graph_ablation.csv`: does the real graph help
- `eval_visual_benchmark.png/.csv/.md`: problem-statement benchmark
- `eval_slices.csv`, `eval_leakage_tests.json`, `eval_champion.json`, `eval_alert_thresholds.csv`
- `eval_real_*`: the same on REAL IMF PortWatch data
