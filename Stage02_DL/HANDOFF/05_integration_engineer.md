# Stage 02 - Role 3 - Data & Integration Engineer -> hand-off note (05_integration_engineer.py)

status: IN_PROGRESS
updated_utc: 2026-11-02
git_commit: branch integ/stage02-wrapup
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
  - optional `?model=lstm|transformer|graph_full|graph_shuffled|graph_none`; the default is Role 4's champion (`eval_champion.json`)
- Answers carry, for each of the 5 horizons, q10/q50/q90 for dwell and throughput, plus `alert_prob_24h` (P(dwell above the node's alert line 24 h ahead)), `alert_threshold_dwell_h`, `model_version`, `evidence` and **`requires_operator_review: true`**
- Script mode `python 05_integration_engineer.py` -> `Stage02_DL/reports/integ_latency.json` (200 in-process calls per model)
- Dashboard simplified on 2 Nov 2026 (team lead's request, written by Role 4): `app.py` now has two pages in a top menu, champion models only, no model picker, no maps, no evidence pages.
  - `pages/1_Shipments.py`: test-period shipments with the chance of a 24 h+ delay (alerts only by default, filter by location), click a row for the reasons, and a form to check a new shipment
  - `pages/2_Forecasts.py`: pick a forecast time; nodes likely above their dwell alert line in 24 h; one node's last 7 days + 7-day forecast with the 80% range and the alert line, and the same as a small table
  - The evaluation charts stay in `Stage0N/reports/eval_*` and `eda_*` (not on screen)
- `tests/integ/test_api_stage02.py`: answer shape, ordered intervals, review flag, 422 on short/gapped/unknown-node/out-of-range/unknown-model, refusal with a hint outside the stored period; `tests/integ/test_control_tower_pages.py` now renders the two new pages too

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
(no hand-over file of my own; API request ranges come from contracts/schemas/hourly_node_telemetry.schema.json)
Checked on 2026-11-02 against the files the API and pages read:
PASS  Stage02_DL/predictions/dl_forecasts.csv.gz  (729000 rows, schema 'dl_forecasts')   <- all rows model_version v2.0
PASS  Stage02_DL/predictions/dl_backtest_folds.csv.gz  (249480 rows, schema 'dl_forecasts')  note: extra column 'fold'
PASS  Stage02_DL/data/hourly_node_telemetry.csv.gz  (236520 rows, schema 'hourly_node_telemetry')
```

## Decisions and assumptions (what you decided and why, in plain English)
- Graph models need every node's recent history and 4 weeks of context. The API takes the other nodes from the stored grid for the same hour and swaps in the posted node's 72 hours. Outside the stored period it answers 422 with "use ?model=lstm" (the LSTM and Transformer read one node only).
- The pages read files only (no PyTorch), so the Streamlit Cloud app does not need torch. Live model calls go through the API.
- `alert_prob_24h` comes from the forecast's 80% band, assuming a bell-shaped spread (`common.evalkit.forecast.exceedance_probability`). The alert line is the node's 95th percentile of TRAIN dwell (`eval_alert_thresholds.csv`, Role 4).

## Things the next role must know (warnings, known issues, changed meanings)
- Latency (31 Oct, fast models, this laptop), `reports/integ_latency.json`: p95 3.3 ms Transformer, 4.8 ms LSTM, ~21 ms graph models. Measured on the **service without HTTP**, because the test client needs `httpx2`, which is not installed.
- API answers equal the evaluated forecast file exactly (checked 4 models x 3 nodes x 5 horizons x 2 targets x 3 quantiles, max difference 0).
- Fixed on 31 Oct: the 72-hour check compared raw integers in nanoseconds, but pandas 3 parses these times at seconds resolution, so it rejected every valid POST. It now compares real time differences. The API tests did not catch this because they skip without `httpx`.
- The champion is `transformer` (`eval_champion.json`), so the default model needs only the node's own 72 hours.
- Until `04_evaluation_engineer.py` has run, the map shows a warning and no alert probabilities (no thresholds yet), and the API defaults to the first loaded model.
- Fast-mode models are in `artifacts/` now. Re-run the latency script after the full run.

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
- **Dashboard** (`streamlit run app.py`): all 5 pages open with no exception. Network forecast shows the map coloured by
  P(above line); Forecast evidence shows transformer, skill 0.171 (24 h) / 0.222 (7 d), recall 0.77, median lead 22 h,
  which equal `eval_report.md`. The log shows one scikit-learn `InconsistentVersionWarning` (Stage 1 model pickled with
  1.8.0, loaded with 1.9.1); see open items.

## Open questions for other roles (name the role and the question)
- Role 4: answered. The API now returns `alert_rule` for the champion: rule B from `eval_champion.json` (highest
  P(dwell >= threshold) over all horizons >= p*, the same calculation as `04_evaluation_engineer.py`). Other models get
  `null`, because p* was chosen on validation for the champion only.
- Role 1: Stage 1 artefacts were saved with scikit-learn 1.8.0; should we pin `scikit-learn<1.9` or re-save them?

## Your tasks (Pravin) - status
1. [MUST] Review the API, both pages and the tests - **done** (see review above; one addition: `alert_rule`).
2. [MUST] Run the stage on the full-run outputs - **done locally** (Role 1 committed the full-run artefacts in PR #22).
   Clean-clone reproduction of 02/04/05 is **not done yet**.
3. [MUST] httpx + httpx2, latency over HTTP, `/docs` - **done**.
4. [MUST] Deploy and check on the live URL - **not done**: the live app deploys from `main`, so it needs this PR merged.
5. [SHOULD] Polish the map (lane colours by mode, node labels) and add a link to Role 4's `eval_visual_benchmark.png` on the forecast page.
6. [STRETCH] Animated propagation replay along the lanes for one test episode.
