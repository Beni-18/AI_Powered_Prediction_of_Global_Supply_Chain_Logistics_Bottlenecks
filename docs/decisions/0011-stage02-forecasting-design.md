# 0011 - Stage 2 forecasting design (rebuild of the DL stage)
Date: 2026-10-31   Deciders: Role 4 (written at the team lead's request), to be confirmed by Roles 1, 2, 3   Status: PROPOSED

## Context
The first Stage 2 hand-off (v1.0, 29-30 Oct) could not be defended in a debate:
- `seasonal_naive` was "last value + a sine offset" and `xgb_lag` was "0.85 x last value + 0.15 x mean", not XGBoost;
- persistence/baseline intervals were invented (+-15%, +-10%);
- windows were split 70/15/15 by count, ignoring the shared split dates and the 168 h purge gap;
- the graph "ablations" were one model trained on the real graph and then *run* with other matrices, so
  `graph_none`/`graph_shuffled` measured a broken model, not a model without a graph;
- no early stopping, no Role 3 data (`hourly_node_telemetry` did not exist), zeros treated as missing values.
R5 ("honest numbers only") rules all of this out, so Stage 2 was rebuilt end to end.

## Options
1. Patch v1 (fix the baselines only). Cheap, but the ablation and the split would still be wrong.
2. Rebuild on the shared contracts with honest baselines, separately trained ablations and a real-data check. More work, but every number becomes defensible.

## Decision (option 2)
**Data (Role 3 lane).** `Stage02_DL/01_data_engineer.py` rebuilds the complete hourly grid. Missing hours are filled from the past (forward-fill of at most 6 h) or from the TRAIN hour-of-week median, and flagged `is_imputed`. It uses the Stage 1 split dates. Scaling is robust (median/IQR) per node and channel, fitted on TRAIN only. **New: a real-data track.** Daily IMF PortWatch container port calls and chokepoint transits, 2019-01-01 to 2026-07-24, for 23 nodes. LA/LB is one PortWatch port and is kept once. Rail and DC nodes are absent. Split by calendar year: train up to 2023, val 2024, test 2025-26.

**Baselines.** All are real, stated methods, and each gets intervals from its own TRAIN errors:
- `persistence`
- `seasonal_naive` (same hour last week)
- `rolling_mean` (last 24 h)
- `xgb_lag` (XGBoost quantile regression on lag features, one model per horizon and target, early-stopped on val)

**Neural models.** All share one output design: q10/q50/q90 as "last value + learnt change", with non-crossing quantiles and pinball loss.
- `lstm`: the role-file spec.
- `transformer`: 2 layers, 4 heads, d = 64. Attention maps are exposed for Role 2.
- `graph_full`: LSTM plus 4 weeks of daily context, plus 2 GCN layers `H' = H + ReLU(Â H W)`, plus **transit-delayed lane messages**. For lane s→d with transit L and horizon h, the model reads s's state from the time whose congestion will reach d at t+h, because the simulator (and real shipping) moves congestion along lanes after the transit time.
- `graph_shuffled`: identical weights and size, with the lanes relabelled by a fixed-point-free permutation (seed 42).
- `graph_none`: identical weights and size, with no lanes.

**Each graph variant is trained separately** from the same seed.

**Training and intervals.** Fit on TRAIN origins and early-stop on VAL (patience 5). Every model, baselines included, gets the same split-conformal (CQR) widening on VAL, so interval widths compare fairly.

**Evaluation (Role 4).**
- Skill vs persistence and vs the best baseline (chosen on val), with day-block bootstrap CIs.
- Episodes: dwell above the node's p95 of TRAIN dwell for at least 6 h, at least 24 h apart.
- Alerts raised while the node is already in an episode are not early warnings, so they are dropped.
- Rule A: the median crosses the threshold. Rule B: probability ≥ p*, with p* chosen on VAL by cost (miss 10, false alarm 1, from `cost_policy.json`).
- The champion is chosen on VAL (skill at 24 h for dwell, then recall).

## Consequences
- Role 1: review `common/modelkit/dl_*.py` and `03_dl_engineer.py`. Own the full run (hidden 64, ≤ 30 epochs), the [SHOULD] hyper-parameter study and the rolling-origin folds.
- Role 2: the EDA script (`02_eda_engineer.py`) is still to do. Attribution uses `common/modelkit/dl_infer.py` (the exact preprocessing), `03_dl_engineer.py --noise-channel` gives the noise-channel test model, and `TransformerForecaster.attention_maps()` gives the attention maps.
- Role 3: review the data and integration files. Run the pipeline on a clean clone. Commit the full-run artefacts. Deploy the two new pages. The live app reads files only, with no PyTorch.
- `contracts/` unchanged. The real-track forecasts are an extra file (`dl_real_forecasts.csv.gz`) with the `dl_forecasts` columns and `target = activity`. They are not a contract file.
- We give up a one-command 20-minute full run. `--fast` is about 15 min on a laptop CPU and the full run takes longer; numbers state which mode produced them (`model_version` ends in `-fast`).
