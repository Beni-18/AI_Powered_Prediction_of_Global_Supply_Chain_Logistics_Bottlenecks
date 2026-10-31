# Stage 02 - Role 3 - Data & Integration Engineer -> hand-off note (05_integration_engineer.py)

status: IN_PROGRESS
updated_utc: 2026-10-31
git_commit: not committed yet (branch to be opened by Bennish)
data_version: v1

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
- `pages/3_Network_Forecast.py`: world map with lanes, nodes coloured by P(dwell above alert line) at the chosen horizon; top-10 table; model selector (persistence / XGBoost / LSTM / Transformer / graph variants); node fan chart with the 80% band and "what really happened"
- `pages/4_Forecast_Evidence.py`: champion tiles, skill by horizon, early warning, graph ablation, real-data track (reads Role 4's `eval_*` files)
- `app.py`: new sidebar group "Stage 2 · network forecasts"
- `tests/integ/test_api_stage02.py`: answer shape, ordered intervals, review flag, 422 on short/gapped/unknown-node/out-of-range/unknown-model, refusal with a hint outside the stored period; `tests/integ/test_control_tower_pages.py` now renders the two new pages too

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
(no hand-over file; API request ranges come from contracts/schemas/hourly_node_telemetry.schema.json)
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

## Open questions for other roles (name the role and the question)
- Role 4: should the API add the probability alert rule (p* from `eval_champion.json`) to its answer, next to `alert_prob_24h`?

## Your tasks (Pravin)
1. [MUST] Review the API, both pages and the tests.
2. [MUST] Run the stage from a clean clone (commands in `Stage02_DL/README.md`), share the results with the team, and fix what breaks. Commit the full-run artefacts with Role 1.
3. [MUST] Add `httpx` and `httpx2` to `requirements-dev.txt` so the API tests run; re-run `python 05_integration_engineer.py` for latency over HTTP and paste p95 here. Start the API and try `/docs`.
4. [MUST] Deploy the two pages to the live Streamlit app and check them on the deployed URL.
5. [SHOULD] Polish the map (lane colours by mode, node labels) and add a link to Role 4's `eval_visual_benchmark.png` on the forecast page.
6. [STRETCH] Animated propagation replay along the lanes for one test episode.
