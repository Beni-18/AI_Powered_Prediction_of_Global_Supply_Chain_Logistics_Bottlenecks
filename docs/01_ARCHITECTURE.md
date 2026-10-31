# 01 - Architecture (low-level design, kept current daily)

## 1. System picture

```
                   REAL-WORLD ANCHORS (public)                       SIMULATOR (Role 3)
        PortWatch transits | GSCPI | Open-Meteo | advisories  ──►  calibrated, event-driven, outcome-labelled
                              │                                              │
                              └────────────── data/raw/{real,sim} ───────────┘
                                                      │
        ┌───────────────┬──────────────┬──────────────┼───────────────┬──────────────────┐
        ▼               ▼              ▼              ▼               ▼                  ▼
  Stage 1  ML      Stage 2  DL    Stage 3  NLP   Stage 4  SLM    Stage 5  GenAI    (each stage = 5 role scripts,
  shipment risk    7-day node     advisory NER    offline 2-line   scenario          01 Data → 02 EDA → 03 Model
  + anomalies      forecast+GNN   + urgency       briefs (LoRA)    generator +       → 04 Evaluation → 05 Integration)
        │               │              │              │            stress test
        └───────────────┴──────┬───────┴──────────────┘                 │
                               ▼                                        │ synthetic disruptions are
                 FUSION  "CASCADE triage score" per node ◄──────────────┘ replayed through Stages 1-4
                               ▼
                 Control tower (Streamlit) + APIs (FastAPI)  ──►  HUMAN OPERATOR APPROVES (no auto-reroute)
```

Everything between boxes travels as a **contract file** (`contracts/schemas/*.schema.json`), checked by
`scripts/validate_contract.py`. That is the whole integration story.

## 2. Repository layout

```
.
├── AGENTS.md CLAUDE.md GEMINI.md        rules every AI assistant reads (identical content, different file names)
├── README.md
├── app.py  run_pipeline.py              control tower and one-command pipeline            (Role 3)
├── requirements*.txt  .env.example
├── contracts/                           FROZEN shared agreements                         (Role 3 owns, changes need approval)
│   ├── network_nodes.csv  network_edges.csv  jargon_glossary.csv
│   ├── label_policy.json  cost_policy.json  feature_policy.json  ownership.json
│   ├── schemas/*.schema.json            one per hand-over file type
│   └── fixtures/*.csv                   tiny VALID example of every hand-over file (work against these on minute 0)
├── common/                              shared python code
│   ├── paths.py seed.py contracts.py    (everyone imports; Role 3 maintains)
│   ├── dataio/                          windowing + loaders          (Role 3)
│   ├── edakit/                          plot style + EDA helpers      (Role 2)
│   ├── modelkit/                        model cards, split helpers    (Role 1)
│   ├── evalkit/                         metrics, costs, bootstrap     (Role 4)
│   └── integration/                     API helpers, stub predictors  (Role 3)
├── data/                                shared data foundation        (Role 3)
│   ├── raw/real/    git-ignored, re-fetched by scripts/fetch_*.py
│   ├── raw/sim/     simulator output used by all stages
│   └── calibration/ numbers learnt from real data
├── Stage01_ML/ Stage02_DL/ Stage03_NLP/ Stage04_SLM/ Stage05_GenAI/
│   ├── 01_data_engineer.py … 05_integration_engineer.py      one script per role
│   ├── data/          prepared datasets                       (Role 3 writes)
│   ├── artifacts/     trained models + model_card.json        (Role 1 writes)
│   ├── predictions/   model outputs in contract format        (Role 1 writes, Role 4 reads)
│   ├── reports/       eda_* (Role 2)  eval_* (Role 4)  integ_* (Role 3)   ← prefix decides the owner
│   └── HANDOFF/       one status note per role                (each role writes its own)
├── claims/CLAIMS_LEDGER.md              every number we will say in a debate + how to reproduce it (Role 4)
├── docs/                                this documentation, role files, decisions log
├── scripts/                             validate_contract, check_ownership, board, strip_ai_trailers, fetch_*
├── tests/                               shared tests + one folder per role (tests/ml, eda, data, eval, integ)
├── notebooks/<role>/                    scratch only, never an input to anyone else
├── .githooks/ .github/                  safety hooks and CI
└── _reference/                          local only (git-ignored): example project + problem statement
```

## 3. Ownership map (who may edit what)

The machine-readable version is `contracts/ownership.json`, enforced by `scripts/check_ownership.py` and CI.
A pattern in the *file name* decides the owner, so two people never write the same file.

| Path pattern | Owner | Commit type |
|---|---|---|
| `Stage0N/01_data_engineer.py`, `Stage0N/data/**`, `data/**`, `contracts/**`, `common/dataio/**`, `scripts/fetch_*.py`, `Stage03_NLP/annotation/**` | Role 3 | `data` |
| `Stage0N/02_eda_engineer.py`, `Stage0N/reports/eda_*`, `Stage05_GenAI/scenarios/**`, `common/edakit/**`, `contracts/jargon_glossary.csv` | Role 2 | `eda` |
| `Stage0N/03_*_engineer.py`, `Stage0N/artifacts/**`, `Stage0N/predictions/**`, `Stage0N/requirements.txt`, `common/modelkit/**` | Role 1 | `ml` |
| `Stage0N/04_evaluation_engineer.py`, `Stage0N/reports/eval_*`, `common/evalkit/**`, `claims/**` | Role 4 | `eval` |
| `Stage0N/05_integration_engineer.py`, `Stage0N/integration/**`, `Stage0N/reports/integ_*`, `app.py`, `pages/**`, `run_pipeline.py`, root `requirements*.txt` | Role 3 | `integ` |
| `docs/**`, `*.md` | anyone | `docs` |

## 4. The contract catalogue (producer → consumer)

| Contract (schema) | File | Produced by | Read by | Day |
|---|---|---|---|---|
| `telemetry` | `Stage01_ML/data/prepared_dataset.csv` | Role 3 `01` | Roles 1, 2, 4; Stage 4-5 calibration | 1 |
| `ml_predictions` | `Stage01_ML/predictions/ml_predictions.csv` | Role 1 `03` | Role 4, Role 3 `05` | 1 |
| `hourly_node_telemetry` | `Stage02_DL/data/hourly_node_telemetry.csv` | Role 3 `01` | Roles 1, 2, 4; Stage 5 | 2 |
| `dl_forecasts` | `Stage02_DL/predictions/dl_forecasts.csv` | Role 1 `03` | Role 4, Role 3 | 2 |
| (no contract) real-data track | `Stage02_DL/predictions/dl_real_forecasts.csv.gz` (same columns, `target = activity`) | Role 1 `03` | Role 4, Role 3 | 2 |
| `incident_messages` | `Stage03_NLP/data/incident_messages.csv` | Role 3 `01` | Roles 1, 2, 4 | 3 |
| `nlp_predictions` (+ `nlp_entities_pred.jsonl`) | `Stage03_NLP/predictions/` | Role 1 `03` | Role 4, Role 3 | 3 |
| `slm_pairs` | `Stage04_SLM/data/slm_pairs.csv` | Role 3 `01` | Roles 1, 2, 4 | 4 |
| `slm_generations` | `Stage04_SLM/predictions/slm_generations.csv` | Role 1 `03` | Role 4, Role 3 | 4 |
| `scenario_telemetry` | `Stage05_GenAI/predictions/scenario_telemetry.csv` | Role 1 `03` | Role 4, Role 3 | 5 |
| `stress_results` | `Stage05_GenAI/reports/eval_stress_results.csv` | Role 4 `04` | Role 3 `05`, debate | 5 |

Policies (JSON, frozen): `label_policy.json` (what ON_TIME / AT_RISK / CRITICAL mean and how the split is made),
`cost_policy.json` (what each kind of mistake costs), `feature_policy.json` (which columns a model may see).

## 5. Canonical entities

**Node registry** (`contracts/network_nodes.csv`): 27 nodes - 18 ports, 1 rail hub, 2 distribution centres,
6 chokepoints (Suez, Panama, Malacca, Hormuz, Bab el-Mandeb, Cape of Good Hope). Every `node_id` in any file must exist
here. `aliases` (pipe-separated) feed the NER linker. `network_edges.csv` holds 50 lanes (SEA, RAIL, TRUCK, BARGE,
ADJACENT) with typical transit days; the graph model uses them symmetrically, the generator uses the direction and delays.

**Label policy:** `realized_delay_hours` < 6 → ON_TIME; 6-24 → AT_RISK; ≥ 24 → CRITICAL_DELAY.
`delay_status` is derived **once**, in `Stage01_ML/01_data_engineer.py`.

**Split policy:** chronological 70/15/15 with a 168-hour purge gap, grouped by `shipment_id`; assigned **once** in
the Stage 1 Data Engineer script and stored in the `split` column. Stage 2 uses calendar-hour boundaries from the
same dates (`splits.json`). Nobody re-splits.

**File size rule:** a data file larger than 10 MB is written as `<name>.csv.gz` (same base name plus `.gz`); `pandas.read_csv` and `scripts/validate_contract.py` read it unchanged. Nothing above 50 MB is ever committed. Role files write `name.csv[.gz]` to mean "either".

**Time:** all timestamps are ISO-8601 UTC. **Units:** hours, TEU per hour, percent (0-100, overload allowed up to 150),
indexes 0-10, ratios 0-1.

## 6. Shared code interfaces (agree now, implement on Day 0)

| Module | Owner | Function (signature = the promise) |
|---|---|---|
| `common.paths` | Role 3 | `stage_dir(n)`, `stage_path(n, sub, filename)`, `NODES_CSV`, `EDGES_CSV` - **already implemented** |
| `common.seed` | Role 3 | `set_global_seed(seed=42)` - **already implemented** |
| `common.contracts` | Role 3 | `validate_csv(schema, path)`, `validate_dataframe(df, schema)`, `write_validated_csv(df, schema, path)` - **already implemented** |
| `common.dataio.windows` | Role 3 | `build_windows(hourly_df, input_len=72, horizons=(1,6,24,72,168), feature_cols=..., target_cols=("avg_dwell_time_hours","throughput_teu_per_hr"), boundaries=None, stride=1) -> WindowSet` (boundaries default to `Stage01_ML/data/split_boundaries.json`) with `.X [n, input_len, n_nodes, n_features]`, `.Y [n, n_horizons, n_nodes, n_targets]`, `.origin_ts [n]`, `.split [n]` (train/val/test by `origin_ts` and the purge gap), `.node_ids` |
| `common.dataio.graph` | Role 3 | `load_adjacency(symmetric=True, self_loops=True, normalise=True) -> np.ndarray [n_nodes, n_nodes]` in the node order of `network_nodes.csv` |
| `common.modelkit` | Role 1 | `save_model_card(dir, name, version, features, train_window, metrics_on_val, seed, notes)`; `load_split(df, "train")`; `predictions_frame(...)` helpers |
| `common.edakit` | Role 2 | `apply_style()` (one plotting style for all charts), `save_fig(fig, path)` (consistent size/dpi), `psi(a, b)`, `robust_noise_floor(series)` |
| `common.evalkit` | Role 4 | `expected_cost(y_true, y_pred, cost_matrix)`, `choose_threshold(y_val, p_val, cost_matrix)`, `ece(y, p, bins)`, `bootstrap_ci(metric, *arrays, n=1000)`, `skill_score(err_model, err_baseline)`, `lead_time(alert_times, onset_times)`, `grounding_check(summary, facts)` |
| `common.dataio.hourly_grid` / `real_activity` | Role 3 | `complete_grid(raw, node_ids, train_end)`, `fit_robust_scaler(grid, train_end)`, `scale_array(...)`; `load_real_daily(node_order)` (IMF PortWatch daily counts), `real_split_boundaries(days)` - **implemented Day 2** |
| `common.modelkit` (Stage 2) | Role 1 | `dl_data.ForecastData` (+ `sim_data_from_grid`, `real_data_from_grid`, `split_origins`, `transit_edges`, `symmetric_adjacency`); `dl_models.{LSTMForecaster, TransformerForecaster, GraphForecaster, PinballLoss, save_model, load_model}`; `dl_train.{train_neural, train_xgb_lag, naive_point, conformal_margins, long_frame}`; `dl_infer.{Stage2Forecaster, sim_window, single_node_window}` - **implemented Day 2** |
| `common.evalkit` (Stage 2) | Role 4 | `block_skill_ci`, `diebold_mariano`, `exceedance_probability`, `interval_score`, `rmse`, `smape`, `episode_hours`, `wilson_interval` - **implemented Day 2** |
| `common.integration` | Role 3 | `StubPredictor` classes that return contract-shaped answers; `review_flag(...)` helper that always sets `requires_operator_review=True` |

If you implement a function with a different signature, you must update this table in the same PR (`docs` commit).

## 7. Fusion: the CASCADE triage score (Day 5, Role 3 + Role 4)

Per node and hour, three transparent inputs (no black box on top of black boxes):

1. `ml_risk` - mean Stage 1 `risk_score` of shipments currently at that node,
2. `dl_exceedance` - probability that forecast dwell within 24 hours exceeds the node's alert threshold (from the forecast interval),
3. `advisory_pressure` - count of URGENT advisories linked to the node in the last 24 hours (Stage 3 `nlp_signals.csv`).

Combined with a logistic model fitted on the **validation** window (weights are shown in the UI, calibration curve in the report).
If the fitted model does not beat the best single input on the test window, we report that honestly and fall back to the single input.

## 8. Safety principles (all stages)

- Advisory only: no function in the repo executes a reroute or booking.
- Every API/UI response contains `requires_operator_review: true`, `model_version`, and the evidence used.
- Missing values are "unknown", never zero.
- Anything below the confidence threshold is flagged, not hidden.
- Synthetic data is always labelled (`data_origin`), in files and on screen.

## 9. Targets we hold ourselves to

| Area | Target |
|---|---|
| Stage 1 API latency | p95 < 50 ms per single prediction on a laptop CPU |
| Stage 2 forecast | beats persistence baseline (positive skill) at ≥ 4 of 5 horizons |
| Stage 3 NER | entity-level F1 reported per type on the **hand-labelled** test set; node-linking accuracy reported |
| Stage 4 SLM | p95 < 5 s per brief on laptop CPU; exactly 2 sentences; hallucination rate measured and below the template-base-prompt rate |
| Stage 5 generator | realism audit passes (ranges, conservation); discriminator AUC reported honestly; 20 scenarios + wildcard run end to end |
| Repo | clean clone → `python run_pipeline.py` works; CI green; no AI attribution in history |

## 10. Technology choices (and why)

| Need | Choice | Why |
|---|---|---|
| Tabular ML | scikit-learn, XGBoost, LightGBM, SHAP | fast, explainable, laptop-friendly |
| Deep learning | PyTorch (CPU is enough) | one framework for LSTM, Transformer, GNN, VAE, LoRA; no graph library required (dense GCN) |
| NLP | scikit-learn baseline, spaCy NER, sentence-transformers / small BERT | strong baseline first, small upgrade second |
| SLM | Hugging Face `transformers` + `peft` (LoRA) on FLAN-T5; optional decoder SLM | runs offline on CPU after one download |
| Serving | FastAPI + Uvicorn | typed, auto-docs, trivial to test |
| UI | Streamlit + Plotly | quick, data-dense; no front-end build chain |
| Quality | pytest, contract validator, GitHub Actions | catches integration breaks before the debate |
