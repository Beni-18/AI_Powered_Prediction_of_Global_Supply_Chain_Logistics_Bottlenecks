# Stage 02 - Role 3 - Data & Integration Engineer -> hand-off note (05_integration_engineer.py)

status: READY
updated_utc: 2026-11-02
git_commit: main b4d26df (PR #28) + branch integ/stage02-final-check-v2
data_version: v1, models v2.0 (full run)

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

**Owner: Pravin (Role 3).** First version written by Role 4 (Bennish) at the team lead's request, the same way as Stage 1.
**Pravin: you run it, share the results with the team, deploy, and take it over** (see "Your tasks").

## Files produced (path | contract | rows | notes)
- `Stage02_DL/05_integration_engineer.py` | - | - | FastAPI app. Run from `Stage02_DL/` with `uvicorn 05_integration_engineer:app --port 8002`. Endpoints:
  - `GET /health`, `/nodes`, `/network`, `/models`
  - `POST /forecast` (exactly 72 consecutive hourly records of ONE node, ranges from the contract schema)
  - `GET /forecast/{node_id}?as_of=...` (replay from the stored grid)
  - optional `?model=ensemble|transformer|lstm|graph_full|graph_shuffled|graph_none`; the default is Role 4's champion (`eval_champion.json`, now `ensemble`)
  - default model outside the stored grid: a POST whose hours are after the grid falls back to the transformer and says so in `evidence.model_fallback`; an explicit `?model=ensemble` (or graph model) gets a 422 with a hint
- Answers carry, for each of the 5 horizons, q10/q50/q90 for dwell and throughput, plus `alert_prob_24h` (P(dwell above the node's alert line 24 h ahead)), `alert_rule` (Role 4's rule B with p*, champion only), `alert_threshold_dwell_h`, `model_version`, `evidence` (incl. `model_fallback`) and **`requires_operator_review: true`**
- Script mode `python 05_integration_engineer.py` -> `Stage02_DL/reports/integ_latency.json` (200 in-process calls per model)
- Dashboard simplified on 2 Nov 2026 (team lead's request, written by Role 4): `app.py` now has three pages in a top menu, champion models only, no model picker, no maps, no evidence pages.
  - `pages/0_Home.py`: what the tool does in plain words: the three shipment statuses (On time < 6 h late, At risk 6-24 h, Critical 24 h+) with counts, and what the forecasts warn about; links to the two pages
  - `pages/1_Shipments.py`: counts per predicted status; list filtered by status (default Critical) and location, status in colour; click a row for the chance of each status, the planner-check decision (chance of Critical >= the champion threshold) and the main reasons; a form to check a new shipment
  - `pages/2_Forecasts.py`: pick a forecast time; **Early warnings (next 7 days)** with Role 4's validated rule (highest P(dwell >= alert line) over all horizons >= `alert_rule.p_star` read from `eval_champion.json`, 0.65 now) and a "Most likely when" column; one node's last 7 days + 7-day forecast with the 80% range and the alert line, and the same as a small table
  - The evaluation charts stay in `Stage0N/reports/eval_*` and `eda_*` (not on screen)
- `tests/integ/test_api_stage02.py`: answer shape, ordered intervals, review flag, 422 on short/gapped/unknown-node/out-of-range/unknown-model, refusal with a hint outside the stored period; `tests/integ/test_control_tower_pages.py` renders every page and checks the early-warning list against the p* rule computed independently from the files; the API tests also cover the transformer fallback and the explicit-ensemble 422

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
(no hand-over file of my own; API request ranges come from contracts/schemas/hourly_node_telemetry.schema.json)
Checked on 2026-11-02 against the files the API and pages read:
PASS  Stage02_DL/predictions/dl_forecasts.csv.gz  (729000 rows, schema 'dl_forecasts')   <- all rows model_version v2.0
PASS  Stage02_DL/predictions/dl_backtest_folds.csv.gz  (249480 rows, schema 'dl_forecasts')  note: extra column 'fold'
PASS  Stage02_DL/data/hourly_node_telemetry.csv.gz  (236520 rows, schema 'hourly_node_telemetry')
```

## Decisions and assumptions (what you decided and why, in plain English)
- The ensemble and the graph models need every node's recent history (up to 4 weeks). The API takes the other nodes from the stored grid for the same hour and swaps in the posted node's 72 hours. Outside the stored period the DEFAULT model falls back to the transformer (reads one node's 72 hours) and says so; an explicitly requested ensemble/graph model gets a 422, because silently swapping a model the caller asked for would be wrong.
- The dashboard's early warnings use the evaluated rule (p* chosen by Role 4 on validation), not an untested 50% line: a planner sees the same alerts the evaluation scored.
- The pages read files only (no PyTorch), so the Streamlit Cloud app does not need torch. Live model calls go through the API.
- `alert_prob_24h` comes from the forecast's 80% band, assuming a bell-shaped spread (`common.evalkit.forecast.exceedance_probability`). The alert line is the node's 95th percentile of TRAIN dwell (`eval_alert_thresholds.csv`, Role 4).

## Things the next role must know (warnings, known issues, changed meanings)
- Latency (2 Nov, close-out, `reports/integ_latency.json`, in-process TestClient, 200 POST calls per model, 0 failures): p95 ensemble 31.52 ms, transformer 7.98 ms, LSTM 11.06 ms, graph_full 27.09 ms, graph_shuffled 32.17 ms, graph_none 27.10 ms. A first run while another training job loaded the laptop (load average 15) gave 2-3x higher numbers and was discarded.
- API answers equal the evaluated forecast file exactly (5 models x 3 nodes x 5 horizons x 2 targets x 3 quantiles = 450 values, max difference 0; the ensemble is checked by `test_ensemble_replay_equals_the_forecast_file`).
- Fixed on 31 Oct: the 72-hour check compared raw integers in nanoseconds, but pandas 3 parses these times at seconds resolution, so it rejected every valid POST. It now compares real time differences. The API tests did not catch this because they skip without `httpx`.
- 2 Nov (Role 4, at the team lead's request): the champion is now `ensemble` (transformer + xgb_lag, `eval_champion.json`). The API loads it from `artifacts/ensemble_forecaster.json` (`EnsembleForecaster` in `dl_infer.py`). Like the graph models it needs the stored network history, so a POST outside the stored grid gets a 422 that points to `?model=transformer` or `?model=lstm`. Latency p95 (in-process TestClient, 200 calls): ensemble 39.6 ms, transformer 9.0 ms, LSTM 19.7 ms, graph models 31-34 ms. `test_ensemble_replay_equals_the_forecast_file` checks the served ensemble against the file.
- Until `04_evaluation_engineer.py` has run, the Forecasts page shows an error (no alert lines or p* yet) and the API defaults to the first loaded model.
- `requirements-lock.txt` (close-out): `pip freeze` of the Python 3.12 environment where every test passes. It pins scikit-learn 1.9.1 / xgboost 3.4.1, NOT the 1.8.0 / 3.1.3 the Stage 1 model was saved with, so loading that model still logs `InconsistentVersionWarning` (known limitations row 10). No model was retrained.
- After a change to `common/integration/ui.py` the live Streamlit app kept the old module in memory (Home and Shipments failed with `AttributeError` until the app was rebooted on 2 Nov). Reboot the app after merging changes to `common/`.

## Review and full-run verification by Role 3 (2026-11-02)
- **Environment:** `.venv` with torch 2.15.0, fastapi 0.162.0, starlette 1.10.0, httpx 0.28.1, httpx2 2.16.0.
- **Evaluation was stale:** the models in `artifacts/` were v2.0 (full run) but `eval_champion.json` said `v2.0-fast`, so the
  API used a fast-run champion and alert rule. `04_evaluation_engineer.py` was re-run unchanged (Cross-role commit). Champion
  is still `transformer`, chosen on validation; p* moved from 0.70 to 0.65; alert thresholds (`eval_alert_thresholds.csv`) unchanged.
- **httpx + httpx2** added to `requirements-dev.txt`. `httpx2` is the Pydantic team's package that starlette 1.x's TestClient
  uses. The 11 Stage 2 API tests now RUN: 11 passed, 0 skipped.
- **Latency over HTTP** (`reports/integ_latency.json`): 200 POST /forecast calls per model through FastAPI's in-process
  TestClient, real 72-hour test-period windows, 1 warm-up call not counted, 0 failures (every answer 200 + review flag):

  | model | p50 ms | p95 ms |
  |---|---|---|
  | transformer (champion) | 7.35 | 8.57 |
  | lstm | 9.66 | 10.11 |
  | graph_full | 27.31 | 30.86 |
  | graph_shuffled | 27.09 | 31.69 |
  | graph_none | 27.00 | 30.29 |

  In-process means no network hop; a real deployment adds network time. Earlier 3-5 ms numbers had no HTTP layer at all.
- **Live API** (`uvicorn 05_integration_engineer:app --port 8002`): `/health`, `/models` (5 models, all v2.0, champion
  transformer), `/nodes` (27), `/network` (27 nodes, 50 lanes), `/docs` and `/openapi.json` (200),
  `GET /forecast/PORT_US_LAX?as_of=2025-12-01T00:00:00Z` (200), `POST /forecast` with 72 real hours (200, identical to the
  replay). Errors: 71 rows, unknown node, unknown model and a graph model outside the stored grid all give 422 with a message.
  Every forecast answer carries `requires_operator_review: true` (`/nodes` and `/network` are registry data, not predictions).
- **Consistency:** origin 2025-12-01T00:00Z (test), 5 models x 3 nodes (PORT_US_LAX, PORT_NL_RTM, CHOKE_SUEZ) x 5 horizons x
  2 targets x 3 quantiles = 450 values; max |API - dl_forecasts.csv.gz| = 0.0000.
- **Dashboard** (`streamlit run app.py`, after PR #24 simplified it to two pages): Shipments and Forecasts both open with
  no exception. Forecasts reads the champion (`transformer`, from the regenerated `eval_champion.json`) and lists nodes
  likely above their alert line in 24 h. The Stage 1 page log shows one scikit-learn `InconsistentVersionWarning`
  (Stage 1 model pickled with 1.8.0, loaded with 1.9.1); see open questions.
- **Done in the close-out:** the Forecasts page now uses Role 4's validated rule (p* from `eval_champion.json`) instead of
  P(24 h) >= 0.50; agreed with Role 4.

## Open questions for other roles (name the role and the question)
- Role 4: answered. The API now returns `alert_rule` for the champion: rule B from `eval_champion.json` (highest
  P(dwell >= threshold) over all horizons >= p*, the same calculation as `04_evaluation_engineer.py`). Other models get
  `null`, because p* was chosen on validation for the champion only.
- Roles 1 and 4: the lock file pins the tested environment, not Role 1's Stage 1 versions. Re-saving the Stage 1 model with the locked versions would remove the warning but needs Role 4 to re-verify the Stage 1 claims; not done (no retraining in the close-out).
- Contracts: `requirements-lock.txt` is not in `contracts/ownership.json`; its commit carries a Cross-role line. Please add it to the `integ` paths.

## Your tasks (Pravin) - status
1. [MUST] Review the API, the pages and the tests - **done** (2 Nov; Role 4's ensemble changes reviewed and kept).
2. [MUST] Full-run outputs - **done** (PR #22 models, PR #25 evaluation, PR #27 ensemble). Fresh-clone run 2 Nov: every Stage 2 csv/json/md output IDENTICAL (`reports/integ_final_check.md`).
3. [MUST] httpx + httpx2, latency over HTTP, `/docs` - **done**.
4. [MUST] Close-out fixes - **done and merged** (PR #28): API fallback to the transformer, p* early warnings on the Forecasts page, `requirements-lock.txt`.
5. [MUST] Deploy and check on the live URL - **done** (2 Nov, after PR #28): Home, Shipments (ML) and Forecasts (DL) load with no error and show the ensemble and the p* early warnings.
6. [SHOULD] Polish the map - **dropped**: the team lead's simplified dashboard (PR #24) has no map.
7. [STRETCH] Animated propagation replay - **not done**.
