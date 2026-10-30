# Stage 01 - Role 3 - Data & Integration Engineer -> hand-off note (05_integration_engineer.py)

status: READY
updated_utc: 2026-10-30
git_commit: see PR from branch eval/stage01-complete
data_version: v1

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

**Written by Role 4 (Bennish) at the team lead's request so Stage 1 could close; owner Pravin (Role 3) to review and take over.**

## Files produced (path | contract | rows | notes)
- `Stage01_ML/05_integration_engineer.py` | - | - | FastAPI app: `/health`, `/model-info`, `/predict`, `/predict/batch` (up to 1,000). Run from the stage folder: `uvicorn 05_integration_engineer:app --port 8001`
- `pages/1_Bottleneck_Scoring.py` + `app.py` home | - | - | Streamlit control tower: champion + threshold, model-vs-baseline table, example or uploaded shipments scored with the same code as the API. `streamlit run app.py`
- `Stage01_ML/reports/integ_ops_field_guide.md` | - | - | one page for planners; every number read from a team file and footnoted
- `Stage01_ML/reports/integ_latency.json` | - | 1,000 calls | in-process p50 17 ms, p95 27 ms (target < 50 ms)
- `tests/integ/test_api_stage01.py` | - | 8 tests | answer shape, 422 on bad input, review flag always true, serves the champion

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
Request bodies are validated by a pydantic model built from contracts/schemas/telemetry.schema.json
(numeric ranges, allowed priority tiers and carriers, node ids from contracts/network_nodes.csv): bad input -> HTTP 422.
Role 4's independent check (eval_api_latency.json): 1,000 real HTTP calls, 0 errors, 0 answers without
requires_operator_review, 0 not matching eval_champion.json.
```

## Decisions and assumptions (what you decided and why, in plain English)
- The served model and alert threshold are read from `reports/eval_champion.json` (currently `logreg`, 0.37), never hard-coded.
- `roll_node_dwell_mean_24h` needs the node's recent history; a single request has none, so the API uses the mean of the node's last 24 records in the prepared dataset.
- `top_drivers`: per-shipment contributions (weight x standardised value) for a linear champion; the global SHAP ranking from Role 1's importance matrix for tree models.
- `anomaly_flag` = Isolation Forest's own outlier decision on the request.
- The model is loaded at start-up on the main thread: loading it inside a request thread, or after PyTorch, crashes on macOS (clashing OpenMP libraries).

## Things the next role must know (warnings, known issues, changed meanings)
- **Everyone on macOS with PyTorch installed:** run `pytest tests --ignore=tests/ml` and `pytest tests/ml` separately; in one process PyTorch and scikit-learn/XGBoost crash (clashing OpenMP libraries). Linux CI is not affected.
- The artefact was pickled from `03_ml_engineer.py` run as a script, so the API imports that file to unpickle `Stage01Pipeline`. If Role 1 moves that class into `common/modelkit`, simplify `_load_role1_module`.

## Open questions for other roles (name the role and the question)
- **Pravin (Role 3):** please review and own this code; printable HTML/PDF of the field guide is still to do.
- **Role 2:** the field guide reuses `eda_field_guide_text.md`; it mentions "capacity utilization above 90%" - please confirm that figure from your reports.
