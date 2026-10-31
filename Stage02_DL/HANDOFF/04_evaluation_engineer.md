# Stage 02 - Role 4 - Evaluation Engineer -> hand-off note (04_evaluation_engineer.py)

status: IN_PROGRESS
updated_utc: 2026-10-31
git_commit: not committed yet
data_version: v1 (sim) + real track v1; forecasts v2.0-fast

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

**Owner: Bennish (Role 4).** Evaluated on the **--fast** forecasts (hidden 32, <= 10 epochs). Every C-DL claim stays PENDING until
Role 1's full run, a re-run of this script, and a clean-clone check. Status becomes READY after that.

## Files produced (path | contract | rows | notes)
- `Stage02_DL/reports/eval_report.md`: plain-English summary (read this first)
- `eval_metrics.json`: headline, champion, Diebold-Mariano tests, honest notes, real track
- `eval_forecast_metrics.csv` + `eval_skill_by_horizon.png`: MAE/RMSE/sMAPE, skill vs persistence and vs the best baseline (picked on val), with day-block bootstrap 95% CIs (1,000 resamples)
- `eval_interval_quality.csv`: 80% coverage, width, Winkler interval score
- `eval_lead_time.csv` + `eval_lead_time.png`: early warning (rules A/B per model, val and test, the validation-chosen rule, warning cost)
- `eval_alert_thresholds.csv`: per-node dwell alert line = p95 of TRAIN dwell (used by the API and the map)
- `eval_graph_ablation.csv`: graph_full vs graph_none / graph_shuffled / lstm, paired CIs, plus lead time and parameter counts
- `eval_visual_benchmark.png/.csv/.md`: the problem-statement benchmark (3 test episodes; xgb_lag and the Stage 1 risk score as "baseline ML detection")
- `eval_slices.csv` (node type and node), `eval_leakage_tests.json`, `eval_champion.json`
- `eval_real_*`: forecast metrics, intervals, ablation and figures on REAL IMF PortWatch data
- `claims/CLAIMS_LEDGER.md`: C-DL-01..14 (PENDING)
- New evalkit helpers: `block_skill_ci`, `diebold_mariano`, `exceedance_probability`, `interval_score`, `rmse`, `smape`, `episode_hours`, `wilson_interval` (tests: `tests/eval/test_forecast_eval.py`)

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
PASS  Stage02_DL/predictions/dl_forecasts.csv.gz  (729000 rows, schema 'dl_forecasts')
Trust checks (eval_leakage_tests.json): PASS. Test inputs inside test, val inside val, no train rows, every y_true re-read
from Role 3's grid (0 mismatches), persistence self-skill = 0, 802 imputed target hours excluded from scoring.
```

## Results (test period, fast mode; champion chosen on VALIDATION)
- **Champion: `transformer`.** Val skill at 24 h is 0.180 vs 0.146 for the best baseline (xgb_lag).
- **Skill vs persistence, dwell:**
  - 24 h: 0.168 [0.147, 0.189]
  - 168 h: 0.216 [0.191, 0.241]
  - CI-positive at **5/5 horizons for both dwell and throughput** (target was >= 4 of 5).
- **Vs the best baseline (xgb_lag), dwell 24 h:** +0.048 [0.035, 0.063]. Diebold-Mariano p = 1.6e-10 at 24 h, **but not significant at 168 h (p = 0.24)**.
- **Early warning:** the Transformer's own rule, chosen on validation, is B (P >= 0.70).
  - Recall 0.69 [0.42, 0.87], **n = 13 episodes only**.
  - Median lead 32 h [IQR 10-89].
  - 0.06 false alarms per node-week.
  - Warning cost (10 per miss + 1 per false alarm): **transformer 49, lstm 54, persistence 92, xgb_lag 98, Stage 1 risk score 401.**
- **80% interval coverage:** 0.798 (dwell). Every model sits at 0.76-0.80 after the conformal step.
- **Graph ablation: negative.**
  - graph_full is never better than graph_none (0/10 cells; worse in 3) or graph_shuffled (0/10; worse in 6).
  - The plain LSTM and the Transformer beat all three graph variants.
  - The same holds on the real track.
- **REAL data (PortWatch, test 2025-03..2026-07), Transformer:**
  - Skill vs persistence: 0.253 (1 day), 0.257 (3 days), 0.215 (7 days), all CIs above 0.
  - Vs the best baseline: +0.033 [0.021, 0.046], +0.017 [-0.005, 0.040], +0.035 [0.023, 0.049]. The 3-day CI includes 0.
- **API** (Role 3's script, this machine): service-level p95 of 3.3 ms (Transformer), 4.8 ms (LSTM), ~21 ms (graph). This was measured without the HTTP layer, because `httpx2` is missing. The API's answers match the evaluated file exactly (max difference 0 over 120 checked values).

## Decisions and assumptions (what you decided and why, in plain English)
- Episode = dwell above the node's p95 of TRAIN dwell for >= 6 h, episodes >= 24 h apart. An onset only counts when the full 7-day look-back is covered by origins.
- Alerts raised while a node is already inside an episode are not early warnings, so they are dropped.
- **Recall alone was misleading.** With rule A (median crosses), plain persistence catches 12/13, because noisy readings cross the line, at the price of 71 false alarms. So each model's rule (A, or B with p*) is chosen on VALIDATION by warning cost, and only that rule is reported on test.
- The best baseline is picked on validation per target and horizon. Skill CIs resample whole forecast days (sim) or ISO weeks (real).
- Imputed target hours are never scored.

## Things the next role must know (warnings, known issues, changed meanings)
- Only 13 eligible test episodes (9 in val). Never quote recall without n and its interval.
- The graph result is negative in fast mode. Re-check after Role 1's full run before the debate. If it stays negative, we say it first: it is a finding about this simulator, not a failure.
- `httpx` / `httpx2` are not in `requirements-dev.txt`, so the Stage 1 and Stage 2 API tests and the model-picker test **skip** on a clean install (Role 3).

## Open questions for other roles (name the role and the question)
- Role 1: please run the full mode (and `--fold k` if possible) so the C-DL claims can be verified.
- Role 2: your lead-lag edges (`eda_leadlag_edges.csv`) are the test of WHY the graph does not help. Do the residual cross-correlations even show the transit lags?
- Role 3: add `httpx` + `httpx2` to `requirements-dev.txt` so the API tests run, then re-measure latency over HTTP.

## Remaining (Bennish)
1. Re-run `python Stage02_DL/04_evaluation_engineer.py` after the full run; clean-clone verify C-DL-01..14 -> VERIFIED.
2. Debate notes: "skill at every horizon", "beats XGBoost at 24 h but not clearly at 7 days", "the graph did not help, and here is the ablation".
