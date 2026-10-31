# Stage 02 - Role 2 - EDA Engineer -> hand-off note (02_eda_engineer.py)

status: NOT_STARTED
updated_utc: 2026-10-31
git_commit: -
data_version: v1

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

**Owner: Isha (Role 2).** `02_eda_engineer.py` is still a stub, and **this is your piece of Stage 2**. Everything else in the stage
(data, models, evaluation, API, pages) was built by Role 4 at the team lead's request, so all of your inputs already exist.
The section "Prepared for you" was written by Role 4; replace it with your own notes as you go.

## Prepared for you (inputs that exist now)
- Data: `Stage02_DL/data/hourly_node_telemetry.csv.gz` (27 nodes x 8,760 h, complete grid, `is_imputed` flag, contract PASS) and `splits.json`. Also `real_daily_activity.csv.gz`, real IMF PortWatch daily ship counts for 23 nodes, 2019-2026, which you can use for a "real vs simulated" comparison.
- Network: `contracts/network_edges.csv` (`typical_transit_days`), `Stage02_DL/data/adjacency_directed_transit_days.csv`.
- Events for the ambiguous-graph huddle: `data/raw/sim/events_ground_truth.csv` (67 events).
- Models (fast-mode now, full run later): `Stage02_DL/artifacts/*.pt`. Load them **with the exact training preprocessing**:
  ```python
  from common.modelkit.dl_infer import Stage2Forecaster, sim_window
  f = Stage2Forecaster.from_artifact("lstm_forecaster")          # or "transformer_forecaster", "graph_forecaster", "graph_none"
  data, origin = sim_window(grid, f.node_ids, f.scaler, as_of="2025-12-01T00:00:00Z")
  batch = data.batch([origin])          # x [1, 27, 72, 13], ctx [1, 27, 28, 8], anchor -> what the model reads
  data.input_channel_names              # the 13 input channels in order (8 measures, is_imputed, 4 clock features)
  ```
  The model is a normal PyTorch module: `f.model(x, ctx, anchor)` gives [1, nodes, horizons, targets, 3] (q10/q50/q90, scaled).
- Noise-channel test model: `cd Stage02_DL && python 03_dl_engineer.py --noise-channel` (about 3 min). It saves `artifacts/noise_check/lstm_noise.pt`. Its last two input channels are pure noise and a time-shuffled copy of dwell, so attribution to them must be about 0.
- Attention maps: `TransformerForecaster.attention_maps(x)` returns one [batch*nodes, 72, 72] matrix per layer (rows = query hour).
- Role 4's alert line per node (p95 of train dwell): `Stage02_DL/reports/eval_alert_thresholds.csv`, once the evaluation has run.

## Files produced (path | contract | rows | notes)
-

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
(not run yet - EDA writes reports only)
```

## Decisions and assumptions (what you decided and why, in plain English)
-

## Things the next role must know (warnings, known issues, changed meanings)
-

## Open questions for other roles (name the role and the question)
- From Role 4: the episode rule used for lead time is "dwell above the node's p95 of TRAIN dwell for >= 6 h, episodes >= 24 h apart". Your structural-bottleneck rule (residual > 3x noise floor for >= 24 h) may replace it. Tell us before the debate.

## Your tasks (Isha) - from docs/roles/ROLE_2_EDA_ENGINEER.md, Day 2
1. [MUST] ADF + ACF/PACF up to 168 lags -> `eda_acf_summary.csv`, `eda_acf.png`.
2. [MUST] STL (periods 24 and 168) -> `eda_stl_components.png`, with the variance shares.
3. [MUST] The 5 ambiguous delay graphs (seasonal peak vs transient shock vs structural, explicit rule) -> `eda_ambiguous_graphs.md` + 5 PNGs.
4. [MUST] Node-to-node lead-lag over lags of 0-14 days -> `eda_leadlag_edges.csv` (`src, dst, best_lag_h, corr, n`) + a heat-map; compare with `typical_transit_days`. This is the evidence for (or against) the graph model's transit-delayed messages.
5. [MUST] Imputed-hour report (2,379 hours; are they clustered?), plus recommendations -> `eda_findings.md`.
6. [MUST] Attribution on the LSTM (integrated gradients, or permutation by channel and by time step) -> `eda_attribution_channels.csv/png`, `eda_attribution_timesteps.png`.
7. [MUST] Noise-channel test -> `eda_noise_channel_test.json`.
8. [MUST] Transformer attention heat-maps for 3 episodes.
9. [SHOULD] graph_full vs graph_none attribution: where does the graph help?
