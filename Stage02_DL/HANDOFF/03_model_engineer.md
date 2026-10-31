# Stage 02 - Role 1 - ML Engineer (model role) -> hand-off note (03_dl_engineer.py)

status: IN_PROGRESS
updated_utc: 2026-10-31
git_commit: not committed yet (branch to be opened by Bennish)
data_version: v1 (sim) + real track v1

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

**Owner: Santhosh (Role 1).** `03_dl_engineer.py` and `common/modelkit/dl_*.py` were rebuilt by Role 4 (Bennish) at the team lead's request.
**The v1.0 hand-off of 29 Oct is WITHDRAWN.** Its baselines were not real methods: "seasonal_naive" was the last value plus a sine offset, and "xgb_lag" was 0.85 x last + 0.15 x mean with no XGBoost. Its intervals were invented (+-15%). It split windows 70/15/15 by count instead of using the shared dates. And its graph ablations were one model trained on the real graph and then *run* with other matrices. Details are in `docs/decisions/0011-stage02-forecasting-design.md`. **Santhosh: review, run the FULL mode, then set READY.**

## Files produced (path | contract | rows | notes)
- `Stage02_DL/predictions/dl_forecasts.csv.gz` | `dl_forecasts` | 729,000 | sim track. Val+test, origins every 6 h, 27 nodes x 5 horizons x 2 targets x 9 models: persistence, seasonal_naive, rolling_mean, xgb_lag, lstm, transformer, graph_full, graph_shuffled, graph_none. Replaces `dl_forecasts.csv` (deleted)
- `Stage02_DL/predictions/dl_real_forecasts.csv.gz` | - (same columns, `target = activity`) | 530,334 | REAL track (PortWatch): 23 nodes x horizons 24/72/168 h x the same 9 models
- `Stage02_DL/predictions/dl_backtest_folds.csv.gz` | `dl_forecasts` + fold | 249,480 | Rolling-origin backtest for Role 4: folds 1-3, pre-cutoff scalers, 21-day purge-isolated validation windows, 7 models (persistence, seasonal_naive, xgb_lag, lstm, transformer, graph_full, graph_none)
- `Stage02_DL/artifacts/hparam_log.csv` | - | 18 rows | Validation-only study: hidden (32/64/128) x input_len (48/72/168) evaluated on common 168h origin set for lstm and graph_full
- `Stage02_DL/artifacts/{lstm_forecaster, transformer_forecaster, graph_forecaster, graph_shuffled, graph_none}.pt` (+ `real_*.pt`) | - | - | weights + config + lanes + interval margins + input channels; each < 0.2 MB. Load with `common.modelkit.load_model(path)`
- `Stage02_DL/artifacts/xgb_lag_{sim,real}.joblib` (~7.6 MB / 2.3 MB), `training_curve.csv`, `model_card.json` (val metrics only, parameter counts, settings, run mode)
- New shared code: `common/modelkit/dl_data.py` (batches, splits, lanes, flexible spec), `dl_models.py` (LSTM, Transformer, GraphForecaster), `dl_train.py` (training, baselines, XGBoost, conformal, output), `dl_infer.py` (load + exact preprocessing for the API and Role 2)

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
PASS  Stage02_DL/predictions/dl_forecasts.csv.gz  (729000 rows, schema 'dl_forecasts')   <- fast run, 31 Oct; re-paste after your full run
```

## Decisions and assumptions (what you decided and why, in plain English)
- Baselines are real methods: persistence; seasonal_naive (same hour last week); rolling_mean (last 24 h); xgb_lag (XGBoost quantile regression on lag features, one model per horizon and target, early stopping on val). Each baseline's 80% band comes from its own TRAIN errors.
- Every neural model outputs q10/q50/q90 as "last value + learnt change", with the quantiles kept from crossing, and is trained with pinball loss. AdamW 1e-3, gradient clip 1.0, early stopping on val (patience 5; 3 in --fast). Imputed target hours get weight 0.
- Graph model = LSTM + 4-week daily context + 2 GCN layers + **transit-delayed lane messages**: for lane s->d and horizon h, the model reads s's state from the time whose congestion reaches d at t+h. The head reads [LSTM state, graph state, messages].
- The ablations are trained **separately** from the same seed, with identical size; only the lanes differ (none / relabelled by a fixed-point-free permutation).
- Every model, baselines included, gets the same split-conformal (CQR) interval correction on VAL.
- Training origins: every 6th hour in --fast, every 2nd in full. Batch = 16 origins (432 node histories).
- Hyper-parameter study (validation-only): tested hidden (32, 64, 128) x window (48, 72, 168) on a common 168h origin set. H=64 at L=72 achieves 0.14528 val pinball (vs 0.14427 for H=128 with nearly 4x the weights); H=64 is selected for full training as the optimal efficiency-capacity balance. Window remains 72h per contract.
- macOS: `OMP_NUM_THREADS=1` is set before the imports (PyTorch + XGBoost OpenMP deadlock); both still use 4 threads.

## Things the next role must know (warnings, known issues, changed meanings)
- **Current files are from `--fast` mode** (hidden 32, <= 10 epochs; `model_version = v2.0-fast`). Most neural models were still improving at epoch 10. Only full-run numbers become VERIFIED claims.
- Role 4's evaluation of the fast run (`Stage02_DL/reports/eval_report.md`):
  - **champion = transformer**, with CI-positive skill at all 5 horizons;
  - **the graph ablation is negative**: graph_full never beats graph_none or graph_shuffled;
  - xgb_lag is the strongest baseline.
  For the full run, you may tune the graph model on VAL only (e.g. message dropout, fewer GCN layers), but never pick settings by looking at test numbers.
- A first graph design overfit (train loss well below val); it was fixed by concatenating the LSTM state into the head and adding dropout on the context path. The curves are in `training_curve.csv`.

## Open questions for other roles (name the role and the question)
- Role 4: rolling-origin backtest folds (`--fold 1..3`) have been generated and saved to `Stage02_DL/predictions/dl_backtest_folds.csv.gz` (249,480 rows total across all 7 models). Ready for Diebold-Mariano / spread evaluation!

## Your tasks (Santhosh)
**Step-by-step instructions (also for your Antigravity agent): [docs/roles/ROLE_1_STAGE02_RUNBOOK.md](../../docs/roles/ROLE_1_STAGE02_RUNBOOK.md).**
It covers setup, the full run, the hparam study and fold backtest specs, the hand-off, the commit and the PR. Partial runs (`--models`, `--epochs`, `--track`) now write only to `artifacts/scratch/` and `predictions/scratch/` (git-ignored).

1. [x] [MUST] Review `03_dl_engineer.py` and `common/modelkit/dl_{data,models,train,infer}.py` with the tests `tests/ml/test_dl_models.py` and `tests/ml/test_dl_data.py`.
2. [x] [SHOULD] Hyper-parameter study (hidden 32/64/128, window 48/72/168) -> `artifacts/hparam_log.csv`, chosen on VAL only (DONE).
3. [x] [SHOULD] `--fold k` rolling-origin retrains (3 expanding folds) for Role 4 -> `Stage02_DL/predictions/dl_backtest_folds.csv.gz` (DONE).
4. [ ] [MUST] Full run: `cd Stage02_DL && caffeinate -i python3 -u 03_dl_engineer.py` (hidden 64, <= 30 epochs; run in background terminal). Paste the contract check above and set READY.
5. [ ] [STRETCH] 1D-CNN baseline; per-node-type heads; use Role 3's augmentation helpers when they exist.
