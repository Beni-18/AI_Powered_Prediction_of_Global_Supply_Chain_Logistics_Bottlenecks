# Stage 02 - Deep Learning: status and task board

> **What this is:** the one page that says what Stage 2 already does, how to run it, and what is left for each role.
> Built end to end by Role 4 at the team lead's request (31 Oct 2026). Every file keeps its owner (`contracts/ownership.json`);
> owners review, run, and set their HANDOFF to READY. Design decisions: `docs/decisions/0011-stage02-forecasting-design.md`.

## Run it (from the repo root, inside the venv)

```bash
pip install -r requirements.txt -r requirements-dev.txt -r Stage02_DL/requirements.txt
python scripts/fetch_real_data.py            # once: PortWatch daily data for the real track (~10 min download, git-ignored)
cd Stage02_DL
python 01_data_engineer.py                   # ~10 s: hourly grid, splits, scaler, network, real track
python 03_dl_engineer.py --fast              # ~11 min CPU (hidden 32, <= 10 epochs). Full run: drop --fast (hidden 64, <= 30 epochs, a few hours)
                                             # --models / --epochs / --track = partial run -> artifacts/scratch, predictions/scratch only
python 04_evaluation_engineer.py             # ~1 min: every eval_* file + claims ledger rows
python 05_integration_engineer.py            # API latency -> reports/integ_latency.json
uvicorn 05_integration_engineer:app --port 8002     # the forecast API
cd .. && streamlit run app.py                # pages "Network forecast" and "Forecast evidence"
python -m pytest tests -q
```

On macOS, PyTorch and XGBoost clash over OpenMP in one process. The scripts and `tests/conftest.py` set
`OMP_NUM_THREADS=1` before importing them; keep that line if you copy code.

## What exists now

| Step | File(s) | Owner | State |
|---|---|---|---|
| Complete hourly grid, splits, train-only scaler, network matrices | `01_data_engineer.py`, `common/dataio/hourly_grid.py` | Role 3 | built, contract PASS, to review |
| **Real-data track** (IMF PortWatch, 23 nodes, 2019-2026, daily) | `common/dataio/real_activity.py`, `data/real_*` | Role 3 | built, to review |
| Baselines (persistence, seasonal naive, rolling mean, XGBoost on lags) | `common/modelkit/dl_train.py` | Role 1 | built, to review |
| LSTM, Transformer, graph model (GCN + transit-delayed lane messages) and the 3 graph ablations, each trained separately | `common/modelkit/dl_models.py`, `03_dl_engineer.py` | Role 1 | **full run done** (v2.0, PR #22), hparam study, 3 backtest folds |
| Inference helper with the exact training preprocessing (API + attribution) | `common/modelkit/dl_infer.py` | Role 1 | built |
| **Ensemble** (mean of transformer + xgb_lag quantiles, own conformal margins) and the seed-variance study (seeds 42/7/13) | `03_dl_engineer.py --add-ensemble`, `--seeds`; `dl_infer.EnsembleForecaster` | Role 1 | built 2 Nov by Role 4, to review |
| Evaluation: skill + CIs, intervals, lead time, ablation, visual benchmark, slices, DM tests, real track, champion, ledger | `04_evaluation_engineer.py`, `common/evalkit/forecast.py`, `leadtime.py` | Role 4 | **run on the full-run forecasts**: champion `ensemble` (chosen on val), skill > 0 at 5/5 horizons, beats XGBoost at 24 h for every seed, graph ablation negative; backtest folds and seeds scored; see `reports/eval_report.md`; claims C-DL-01..16 |
| Forecast API (`POST /forecast`, `GET /forecast/{node}`, `/nodes`, `/network`, `/models`) + latency | `05_integration_engineer.py` | Role 3 | built, to review |
| Dashboard (simplified 2 Nov: Home, Shipments with On time / At risk / Critical, Forecasts; champion models only, no maps or evidence) | `app.py`, `pages/0_Home.py`, `pages/1_Shipments.py`, `pages/2_Forecasts.py` | Role 3 | built, to review + deploy |
| Tests | `tests/{data,ml,eval,integ}/` (grid, leakage, models, eval helpers, API) | each role | passing |

## Left for each role (in priority order)

**Role 3 - Data & Integration (Pravin)**
1. [MUST] Review `01_*`, `05_*` (now also serves the ensemble), the dashboard pages, `common/dataio/{hourly_grid,real_activity}.py`; take them over in your HANDOFF files.
2. Done (PR #25): clean-clone run and full-run artefacts.
3. [MUST] Redeploy the Streamlit app from main after PR #26 (Home / Shipments (ML) / Forecasts (DL); files only, no PyTorch on the server). `httpx` + `httpx2` added in PR #25.
4. [SHOULD] Document the chokepoint "throughput" conversion in your HANDOFF (schema note); add a "Visual benchmark" panel polish; pin library versions.
5. [STRETCH] Animated propagation replay along the lanes for one test episode.

**Role 1 - ML (Santhosh)** - full step-by-step runbook (also for Antigravity): [docs/roles/ROLE_1_STAGE02_RUNBOOK.md](../docs/roles/ROLE_1_STAGE02_RUNBOOK.md)
1. Done: full mode run, HANDOFF READY. [MUST] Review the ensemble and `--seeds` additions (HANDOFF 03, 2 Nov).
2. Done: hyper-parameter study (`artifacts/hparam_log.csv`) and 3 rolling-origin folds (`predictions/dl_backtest_folds.csv.gz`).
4. [STRETCH] 1D-CNN baseline; per-node-type heads.

**Role 2 - EDA (Isha)** - `02_eda_engineer.py` is built (1 Nov, by Role 4) and re-run on the full-run models (2 Nov): every Day 2 MUST + the
graph-attribution SHOULD, plus lead-lag on the real PortWatch data. Results: `reports/eda_findings.md`; details in `HANDOFF/02_eda_engineer.md`.
1. [MUST] Review the findings and charts (already re-run on the full models; the noise-channel test passes).
2. [MUST] Post the three best findings in the chat; answer Role 4's alert-rule question (p95 vs structural rule).
3. [SHOULD] Granger-causality between linked nodes, compared with `eda_leadlag_edges.csv`.

**Role 4 - Evaluation (Bennish)**
1. Done 2 Nov: evaluation re-run on the full run + ensemble + seeds + backtest folds; `docs/KNOWN_LIMITATIONS.md` rows 13-22 updated.
2. Clean-clone verify the C-DL ledger rows (PENDING -> VERIFIED), then HANDOFF 04 READY.
