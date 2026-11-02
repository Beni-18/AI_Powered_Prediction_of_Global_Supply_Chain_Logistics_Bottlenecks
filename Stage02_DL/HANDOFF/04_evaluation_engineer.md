# Stage 02 - Role 4 - Evaluation Engineer -> hand-off note (04_evaluation_engineer.py)

status: READY
updated_utc: 2026-11-02
git_commit: branch eval/stage02-improvements
data_version: v1 (sim) + real track v1; forecasts v2.0 (full run) + ensemble

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

**Owner: Bennish (Role 4).** Evaluated on the **full-run** forecasts (v2.0, hidden 64, <= 30 epochs), plus the
transformer + XGBoost ensemble, 3 training seeds and 3 rolling-origin backtest folds.

## Files produced (path | contract | rows | notes)
- `Stage02_DL/reports/eval_report.md`: plain-English summary (read this first)
- `eval_metrics.json`: headline, champion, Diebold-Mariano tests, honest notes, real track, seed study, backtest
- `eval_forecast_metrics.csv` + `eval_skill_by_horizon.png`: MAE/RMSE/sMAPE, skill vs persistence and vs the best baseline (picked on val), with day-block bootstrap 95% CIs (1,000 resamples)
- `eval_interval_quality.csv`: 80% coverage, width, Winkler interval score
- `eval_lead_time.csv` + `eval_lead_time.png`: early warning (rules A/B per model, val and test, the validation-chosen rule, warning cost)
- `eval_alert_thresholds.csv`: per-node dwell alert line = p95 of TRAIN dwell (used by the API and the dashboard)
- `eval_graph_ablation.csv`: graph_full vs graph_none / graph_shuffled / lstm, paired CIs, plus lead time and parameter counts
- `eval_visual_benchmark.png/.csv/.md`: the problem-statement benchmark (3 test episodes; xgb_lag and the Stage 1 risk score as "baseline ML detection")
- `eval_backtest_folds.csv`, `eval_backtest_summary.csv`, `eval_backtest_warning.csv`: Role 1's 3 rolling-origin folds (skill per fold, spread, pooled early warning with rule A)
- `eval_seed_runs.csv`, `eval_seed_summary.csv`: xgb_lag, lstm, transformer and the ensemble trained with seeds 42, 7 and 13
- `eval_slices.csv` (node type and node), `eval_leakage_tests.json`, `eval_champion.json`
- `eval_real_*`: forecast metrics, intervals, ablation and figures on REAL IMF PortWatch data
- `claims/CLAIMS_LEDGER.md`: C-DL-01..16, **VERIFIED 2026-11-02** from a fresh clone
- Tests: `tests/eval/test_forecast_eval.py` (evalkit helpers), `tests/eval/test_stage02_robustness.py` (seed verdict)

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
PASS  Stage02_DL/predictions/dl_forecasts.csv.gz  (810000 rows, schema 'dl_forecasts')
Trust checks (eval_leakage_tests.json): PASS. Test inputs inside test, val inside val, no train rows, every y_true re-read
from Role 3's grid (0 mismatches), persistence self-skill = 0, imputed target hours excluded from scoring.
```

## Results (test period; champion chosen on VALIDATION)
- **Champion: `ensemble`** (mean of the transformer's and xgb_lag's quantiles). Val skill at 24 h (dwell) 0.184 vs
  transformer 0.182 and xgb_lag 0.166.
- **Skill vs persistence, dwell:** 24 h 0.165 [0.140, 0.190]; 168 h 0.231 [0.205, 0.257]. CI-positive at **5/5 horizons
  for dwell and throughput**.
- **Vs the best baseline (xgb_lag), dwell 24 h:** +0.037 [0.030, 0.044]; Diebold-Mariano p < 0.001 at 24 h, p = 0.047 at 168 h.
- **Vs the transformer alone:** no significant difference (DM p = 0.19 at 24 h, 0.07 at 168 h). The ensemble is better at
  1 h (0.252 vs 0.235) and 168 h (0.231 vs 0.222), slightly worse at 24 h (0.165 vs 0.171).
- **Seeds (3):** ensemble 24 h skill 0.168 (sd 0.005); lead over xgb_lag +0.030 to +0.038, **positive for every seed**.
  Validation would pick the transformer for seeds 7 and 13 and the ensemble for seed 42: the choice between those two is
  within seed noise. Both beat XGBoost robustly.
- **Backtest (3 folds, Jun-Aug):** ensemble 0.151 (sd 0.015) vs xgb_lag 0.132 at 24 h. Pooled early warning (rule A,
  6 separate episodes): ensemble 5/6, cost 25; transformer 5/6, cost 27; xgb_lag 4/6, cost 43; persistence 4/6, cost 77.
- **Early warning (test):** rule B (P >= 0.65, chosen on validation). Recall 0.77 [0.50, 0.92], **n = 13 episodes only**;
  median lead 22 h [IQR 7-76]; 0.06 false alarms per node-week. Warning cost: ensemble 38, transformer 38, xgb_lag 52,
  lstm 55, persistence 93, Stage 1 risk score 401.
- **80% interval coverage (dwell):** 0.799.
- **Graph ablation: negative** on both tracks (graph_full better than graph_none in 0/10 cells, worse in 10).
- **REAL data (PortWatch), ensemble:** skill vs persistence 0.256 (1 day), 0.243 (3 days), 0.218 (7 days), all CIs above 0;
  vs the best baseline +0.020 [0.013, 0.026], -0.002 [-0.022, 0.019], +0.025 [0.019, 0.033]. The 3-day CI includes 0.
- **API** (Role 3's script, in-process TestClient): p95 39.6 ms (ensemble), 9.0 ms (transformer), 19.7 ms (LSTM).
  The served ensemble equals its rows in the forecast file (`tests/integ/test_api_stage02.py`).

## Decisions and assumptions (what you decided and why, in plain English)
- Episode = dwell above the node's p95 of TRAIN dwell for >= 6 h, episodes >= 24 h apart. An onset only counts when the full 7-day look-back is covered by origins.
- Alerts raised while a node is already inside an episode are not early warnings, so they are dropped.
- Each model's alert rule (A, or B with p*) is chosen on VALIDATION by warning cost; recall alone flatters noisy models.
- The champion rule was NOT changed when the ensemble arrived: best non-baseline model on validation skill at 24 h.
- Backtest folds have no conformal intervals, so their early warning uses rule A only. Fold windows lie inside the main training period; that is fine for a backtest (each fold trained only on data before its cutoff).
- Imputed target hours are never scored.

## Things the next role must know (warnings, known issues, changed meanings)
- Only 13 eligible test episodes (9 in val, 6 in the backtest folds). Never quote recall without n and its interval.
- Ensemble vs transformer is a tie within seed noise; the robust claim is "the DL models beat XGBoost at 24-72 h".
- The ensemble needs the network's stored history (its XGBoost part reads up to 4 weeks), like the graph models.

## Open questions for other roles (name the role and the question)
- Role 2: the structural rule (29% recall, 93% of episodes near an event) vs the p95 rule (71%, 62%): keep p95 for lead time? (`eda_episode_rule_check.csv`)

## Clean-clone verification (2 Nov 2026)
A fresh clone of `eval/stage02-improvements` (+ the git-ignored PortWatch download) re-ran `01_data_engineer.py`,
`03_dl_engineer.py --add-ensemble`, `02_eda_engineer.py`, `04_evaluation_engineer.py` and `05_integration_engineer.py`.
Every data file, forecast file, EDA and evaluation output (CSV, JSON, Markdown and PNG) and the ledger came out identical;
only `integ_latency.json` differs (timings). The check found and fixed two reproducibility bugs: the EDA's neighbour order
(set order between Python runs) and the real-track forecast file's gzip timestamp.
Note: re-running `04_evaluation_engineer.py` writes the ledger rows as PENDING again until the next clean-clone check.
