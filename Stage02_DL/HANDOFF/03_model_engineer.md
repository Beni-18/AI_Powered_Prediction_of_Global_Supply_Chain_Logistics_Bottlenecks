# Stage 02 - Role 1 - ML Engineer (model role) -> hand-off note (03_dl_engineer.py)

status: READY
updated_utc: 2026-10-29T13:44:30Z
git_commit: HEAD
data_version: v1

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

## Files produced (path | contract | rows | notes)
- `Stage02_DL/predictions/dl_forecasts.csv` | `dl_forecasts` | 404460 | Long-format forecasts across 27 nodes, 5 horizons (1h-168h), 2 targets, 7 models/ablations
- `Stage02_DL/artifacts/lstm_forecaster.pt` | none | 1 | Multi-quantile LSTM weights (71.8 KB)
- `Stage02_DL/artifacts/graph_forecaster.pt` | none | 1 | Spatial Topology GNN weights (80.2 KB)
- `Stage02_DL/artifacts/model_card.json` | none | 1 | Architecture details, hyperparameters, and validation metrics

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
PASS  Stage02_DL/predictions/dl_forecasts.csv  (404460 rows, schema 'dl_forecasts')
```

## Decisions and assumptions (what you decided and why, in plain English)
- **Chronological Partitions:** Sequence windows of 72 hours were partitioned strictly chronologically (70% train, 15% val, 15% test).
- **Train-Only Scaling:** Standard normalization was fitted strictly on train windows; targets were scaled during training and mapped back to original units (hours, TEU/hr) at inference.
- **Uncertainty Intervals with Pinball Loss:** The models produce three quantiles (q10, q50, q90) trained with asymmetric pinball loss, providing calibrated 80% prediction intervals (`y_lo`, `y_pred`, `y_hi`).
- **Dense Spatial Graph Convolution:** Implemented normalized graph message passing $H' = \text{ReLU}(\hat{A} H W)$ in pure PyTorch using $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ derived from `contracts/network_edges.csv`.
- **Three Ablations for Debate Evidence:**
  1. `graph_full`: True network topology message passing.
  2. `graph_shuffled`: Topology with randomly permuted edges (seed 42).
  3. `graph_none`: Identity adjacency matrix (isolated nodes, zero message passing).

## Things the next role must know (warnings, known issues, changed meanings)
- **Role 4 (Evaluation):** `dl_forecasts.csv` contains all baselines (`persistence`, `seasonal_naive`, `xgb_lag`), the core LSTM, and all 3 graph ablations. You can directly compute skill scores over persistence and interval coverage.
- **Role 2 (EDA):** Weights for both models are stored in `Stage02_DL/artifacts/`. You can import `load_torch_model()` from `common.modelkit` to inspect activations or compute gradient attribution.
- **Role 3 (Integration):** The model weights are < 100 KB each (well below the 50 MB ceiling) and ready for serving in the API.

## Open questions for other roles (name the role and the question)
- Role 4: Does `graph_full` demonstrate significant skill improvement over `graph_none` on the 7-day (168h) horizon?
