# Stage 01 - Role 3 - Data & Integration Engineer -> hand-off note (05_integration_engineer.py)

status: READY
updated_utc: 2026-10-30
git_commit: see PR from branch integ/stage01-review
data_version: v1

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

**Owner: Pravin (Role 3).** First version written by Role 4 (Bennish) at the team lead's request so Stage 1 could close; reviewed, fixed and taken over by Role 3 on 2026-10-30.

## Files produced (path | contract | rows | notes)
- `Stage01_ML/05_integration_engineer.py` | - | - | FastAPI app: `/health`, `/model-info`, `/predict`, `/predict/batch` (up to 1,000). Run from the stage folder: `uvicorn 05_integration_engineer:app --port 8001`
- `pages/1_Bottleneck_Scoring.py` + `app.py` home | - | - | Streamlit control tower: champion + threshold, model-vs-baseline table, example or uploaded shipments scored with the same code as the API. `streamlit run app.py`
- `Stage01_ML/reports/integ_ops_field_guide.md` (+ `.html`, `.pdf`) | - | - | one page for planners; every number read from a team file and footnoted. The PDF is the HTML printed from Chrome and fits one A4 page
- `Stage01_ML/reports/integ_latency.json` | - | 1,000 calls | in-process p50 14.7 ms, p95 15.8 ms (target < 50 ms)
- `tests/integ/test_api_stage01.py` | - | 10 tests | answer shape, 422 on bad input, review flag always true, serves the champion

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
- `timestamp` is optional; any ISO date-time is accepted and turned into one UTC format (no zone = UTC). A bad one gets a 422.
- The model is loaded at start-up on the main thread: loading it inside a request thread, or after PyTorch, crashes on macOS (clashing OpenMP libraries).

## Things the next role must know (warnings, known issues, changed meanings)
- **Everyone on macOS with PyTorch installed:** run `pytest tests --ignore=tests/ml` and `pytest tests/ml` separately; in one process PyTorch and scikit-learn/XGBoost crash (clashing OpenMP libraries). Linux CI is not affected.
- The artefact was pickled from `03_ml_engineer.py` run as a script, so the API imports that file to unpickle `Stage01Pipeline`. If Role 1 moves that class into `common/modelkit`, simplify `_load_role1_module` (checked 2026-10-30: not moved yet).
- **Real-data calibration (`data/calibration/`, PR 9) is NOT wired into the simulator yet, on purpose.** Doing it changes every dataset and invalidates all Stage 1 results. Role 3 will announce it to the team and do it only after the Stage 1 debate.

## Review by Role 3 (2026-10-30) - what changed
- **Bug fixed:** a malformed `timestamp` crashed `/predict` with HTTP 500; it now gets a clear 422.
- **Bug fixed:** a batch mixing a given timestamp with a missing one (or different ISO styles) crashed with HTTP 500; every timestamp is now normalised before scoring.
- Dashboard: validation messages now say what is wrong (they only showed the field name); blank CSV cells count as missing instead of failing as "not a string"; a file over 1,000 rows now gets a warning instead of being cut silently; priority and carrier lists are read from the contract.
- Field guide: print styles added so the printable copy fits one page; `integ_ops_field_guide.pdf` added.
- Two regression tests added to `tests/integ/test_api_stage01.py`; latency re-measured after the changes.

## Update 2026-10-30 (written by Role 4 at the team lead's request; Pravin to review)
- The API now passes the pipeline's train `node_medians` to `engineer_features` and gets the node-history inputs from `common/dataio/node_context.py`, so it scores exactly like training (max difference 0.00005 on val and test; a shipment dated after the data ends still scores).
- Node history uses a per-node binary search: in-process p95 14 ms (it was 54 ms with a table join per request).
- Field guide regenerated (34% / 83% / 40 per 100). **The PDF is now stale**: please re-export it from `integ_ops_field_guide.html`.

## Open questions for other roles (name the role and the question)
- **All roles:** please vote on `docs/decisions/0008-missing-feature-values.md` (keep the median fill for Stage 1; allow `was_missing_*` and `overload_flag` as features from Stage 2). It needs a second approver.
- **Role 2:** the field guide reuses `eda_field_guide_text.md`; it mentions "capacity utilization above 90%" - please confirm that figure from your reports.
