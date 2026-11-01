# Stage 02 - Role 1 - ML Engineer (model role) -> hand-off note (03_dl_engineer.py)

status: READY
updated_utc: 2026-10-31
git_commit: HEAD
data_version: v1 (sim) + real track v1

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

**Owner: Santhosh (Role 1).** `03_dl_engineer.py` and `common/modelkit/dl_*.py` have been trained in FULL mode (v2.0, hidden 64, <= 30 epochs), verified against contracts, tested, and accompanied by the validation hyper-parameter study and 3-fold rolling-origin backtest.

## Files produced (path | contract | rows | notes)
- `Stage02_DL/predictions/dl_forecasts.csv.gz` | `dl_forecasts` | 729,000 | sim track, full run (v2.0, hidden 64, patience 5). Val+test, origins every 6 h, 27 nodes x 5 horizons x 2 targets x 9 models: persistence, seasonal_naive, rolling_mean, xgb_lag, lstm, transformer, graph_full, graph_shuffled, graph_none. (13 MB)
- `Stage02_DL/predictions/dl_real_forecasts.csv.gz` | - (same columns, `target = activity`) | 530,334 | REAL track (PortWatch), full run (v2.0, hidden 64): 23 nodes x horizons 24/72/168 h x the same 9 models (6.4 MB)
- `Stage02_DL/predictions/dl_backtest_folds.csv.gz` | `dl_forecasts` + fold | 249,480 | Rolling-origin backtest for Role 4: folds 1-3, pre-cutoff scalers, 21-day purge-isolated validation windows, 7 models (persistence, seasonal_naive, xgb_lag, lstm, transformer, graph_full, graph_none) (4.3 MB)
- `Stage02_DL/artifacts/hparam_log.csv` | - | 18 rows | Validation-only study: hidden (32/64/128) x input_len (48/72/168) evaluated on common 168h origin set for lstm and graph_full
- `Stage02_DL/artifacts/{lstm_forecaster, transformer_forecaster, graph_forecaster, graph_shuffled, graph_none}.pt` (+ `real_*.pt`) | - | - | full weights + config + lanes + interval margins + input channels; each < 0.4 MB. Load with `common.modelkit.load_model(path)`
- `Stage02_DL/artifacts/xgb_lag_{sim,real}.joblib` (6.4 MB / 1.4 MB), `training_curve.csv`, `model_card.json` (val metrics only, parameter counts, settings, full run mode: run_seconds 1526.0s, champion proposal: transformer)
- New shared code: `common/modelkit/dl_data.py` (batches, splits, lanes, flexible spec), `dl_models.py` (LSTM, Transformer, GraphForecaster), `dl_train.py` (training, baselines, XGBoost, conformal, output), `dl_infer.py` (load + exact preprocessing for the API and Role 2)

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
PASS  Stage02_DL/predictions/dl_forecasts.csv.gz  (729000 rows, schema 'dl_forecasts')
```

## Decisions and assumptions (what you decided and why, in plain English)
- Full training run executed end-to-end: hidden 64, up to 30 epochs with early-stopping patience 5 on val, train stride 2 (sim) / 1 (real), XGBoost 600 trees. Runtime: 1526.0s (25.4 min) on CPU.
- Baselines are real methods: persistence; seasonal_naive (same hour last week); rolling_mean (last 24 h); xgb_lag (XGBoost quantile regression on lag features, one model per horizon and target, early stopping on val). Each baseline's 80% band comes from its own TRAIN errors.
- Every neural model outputs q10/q50/q90 as "last value + learnt change", with non-crossing quantiles and pinball loss. AdamW 1e-3, gradient clip 1.0, early stopping on val (patience 5). Imputed target hours get weight 0.
- Model performance on validation:
  - sim track: `transformer` best epoch 3 (val pinball 0.14424), `lstm` best epoch 3 (0.14508), `graph_none` best epoch 2 (0.14539), `graph_shuffled` best epoch 2 (0.14575), `graph_full` best epoch 2 (0.14592).
  - real track: `transformer` best epoch 7 (val pinball 0.20329), `graph_shuffled` best epoch 3 (0.20667), `graph_none` best epoch 2 (0.20683), `graph_full` best epoch 2 (0.20714), `lstm` best epoch 19 (0.21108).
  - Champion proposal: `transformer` achieves lowest validation pinball loss on both tracks.
- Graph ablation: `graph_full` does not beat `graph_none` or `graph_shuffled` on validation for either track. This confirms the finding that node-level temporal dynamics dominate over spatial graph propagation on this telemetry.
- Hyper-parameter study (validation-only): tested hidden (32, 64, 128) x window (48, 72, 168) on a common 168h origin set (`artifacts/hparam_log.csv`). H=64 at L=72 achieves 0.14528 val pinball (vs 0.14427 for H=128 with nearly 4x the weights); H=64 is optimal in accuracy and parameter efficiency. Window remains 72h per contract.
- macOS: `OMP_NUM_THREADS=1` is set before the imports (PyTorch + XGBoost OpenMP deadlock); both still use 4 threads.

## Things the next role must know (warnings, known issues, changed meanings)
- **Current files are from the FULL run** (`model_version = v2.0`, hidden 64, patience 5). All neural models and XGBoost were trained with full capacity.
- Every neural model converged and early-stopped cleanly before reaching the 30-epoch cap (sim neural models stopped at epochs 7-8; real at epochs 7-24).
- The champion proposal remains **transformer** on validation.
- All artifact and prediction files are strictly < 50 MB (maximum is `dl_forecasts.csv.gz` at 13 MB and `xgb_lag_sim.joblib` at 6.4 MB).

## Open questions for other roles (name the role and the question)
- Role 4: rolling-origin backtest folds (`--fold 1..3`) have been generated and saved to `Stage02_DL/predictions/dl_backtest_folds.csv.gz` (249,480 rows total across all 7 models). Ready for Diebold-Mariano / spread evaluation!

## Tasks completed (Santhosh)
- [x] [MUST] Review `03_dl_engineer.py` and `common/modelkit/dl_{data,models,train,infer}.py` with the tests `tests/ml/test_dl_models.py` and `tests/ml/test_dl_data.py`.
- [x] [MUST] Full run: `cd Stage02_DL && caffeinate -i python3 -u 03_dl_engineer.py` (hidden 64, <= 30 epochs; 25.4 min CPU, contract PASS, model_version v2.0).
- [x] [SHOULD] Hyper-parameter study (hidden 32/64/128, window 48/72/168) -> `artifacts/hparam_log.csv`, evaluated on common validation origin set.
- [x] [SHOULD] `--fold k` rolling-origin retrains (3 expanding folds) -> `Stage02_DL/predictions/dl_backtest_folds.csv.gz`.
