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
python 03_dl_engineer.py --fast              # ~15 min CPU (hidden 32, <= 10 epochs). Full run: drop --fast (hidden 64, <= 30 epochs, ~1.5-2.5 h)
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
| LSTM, Transformer, graph model (GCN + transit-delayed lane messages) and the 3 graph ablations, each trained separately | `common/modelkit/dl_models.py`, `03_dl_engineer.py` | Role 1 | built, fast run done, **full run to do** |
| Inference helper with the exact training preprocessing (API + attribution) | `common/modelkit/dl_infer.py` | Role 1 | built |
| Evaluation: skill + CIs, intervals, lead time, ablation, visual benchmark, slices, DM tests, real track, champion, ledger | `04_evaluation_engineer.py`, `common/evalkit/forecast.py`, `leadtime.py` | Role 4 | **run on the fast forecasts**: champion transformer, skill > 0 at 5/5 horizons, graph ablation negative; see `reports/eval_report.md`; claims C-DL-01..14 PENDING |
| Forecast API (`POST /forecast`, `GET /forecast/{node}`, `/nodes`, `/network`, `/models`) + latency | `05_integration_engineer.py` | Role 3 | built, to review |
| Dashboard pages (map + fan chart, evidence) | `pages/3_Network_Forecast.py`, `pages/4_Forecast_Evidence.py`, `app.py` | Role 3 | built, to review + deploy |
| Tests | `tests/{data,ml,eval,integ}/` (grid, leakage, models, eval helpers, API) | each role | passing |

## Left for each role (in priority order)

**Role 3 - Data & Integration (Pravin)**
1. [MUST] Review `01_*`, `05_*`, the two pages, `common/dataio/{hourly_grid,real_activity}.py`; take them over in your HANDOFF files.
2. [MUST] Clean-clone run of the commands above; run the **full** `03_dl_engineer.py` (or ask Role 1) and commit its artefacts and forecasts.
3. [MUST] Deploy the two new pages to the Streamlit app (they read files only; no PyTorch needed on the server). Add `httpx` + `httpx2` to `requirements-dev.txt` (API tests skip without them), then re-run `05_integration_engineer.py` for latency over HTTP.
4. [SHOULD] Document the chokepoint "throughput" conversion in your HANDOFF (schema note); add a "Visual benchmark" panel polish; pin library versions.
5. [STRETCH] Animated propagation replay along the lanes for one test episode.

**Role 1 - ML (Santhosh)**
1. [MUST] Review `common/modelkit/dl_*.py` and `03_dl_engineer.py`; run the full mode; set your HANDOFF READY.
2. [SHOULD] Hyper-parameter study (hidden 32/64/128, window 48/72/168) logged to `artifacts/hparam_log.csv`.
3. [SHOULD] Rolling-origin backtest: a `--fold k` option (3 expanding folds) so Role 4 can report skill mean +- spread.
4. [STRETCH] 1D-CNN baseline; per-node-type heads.

**Role 2 - EDA (Isha)** - `02_eda_engineer.py` is still a stub; everything it needs exists:
1. [MUST] ACF/PACF + ADF, STL (24/168), the 5 ambiguous graphs, node-to-node lead-lag (`eda_leadlag_edges.csv`, compare with `typical_transit_days`), imputed-hour report, recommendations.
2. [MUST] Attribution on the saved LSTM: `Stage2Forecaster.from_artifact("lstm_forecaster")` + `sim_window(...)` give the exact input tensors (see the docstring of `common/modelkit/dl_infer.py`).
3. [MUST] Noise-channel test: `python 03_dl_engineer.py --noise-channel` saves `artifacts/noise_check/lstm_noise.pt`; its last two inputs are pure noise / time-shuffled - attribution there must be ~0.
4. [MUST] Attention maps: `TransformerForecaster.attention_maps(x)` for 3 episodes.
5. [SHOULD] graph_full vs graph_none attribution.

**Role 4 - Evaluation (Bennish)**
1. Re-run `04_evaluation_engineer.py` after the full run; clean-clone verify the C-DL ledger rows (PENDING -> VERIFIED).
2. Update `docs/KNOWN_LIMITATIONS.md` and the debate notes with the final numbers.
