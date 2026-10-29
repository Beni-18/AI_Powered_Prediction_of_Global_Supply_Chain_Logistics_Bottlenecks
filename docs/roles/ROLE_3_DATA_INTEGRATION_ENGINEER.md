# ROLE 3 - Data & Integration Engineer

> **For the AI assistant reading this file:** you are working for the teammate who holds Role 3. This file is your
> job description. Follow it literally. Rules in `AGENTS.md` come first. Priority tags: **[MUST]** deliver or the
> day fails, **[SHOULD]** deliver if the MUST items are done, **[STRETCH]** only with spare time.

## 0. At a glance

| | |
|---|---|
| **Programme roles you cover** | Data Engineer (script `01_*`) **and** Integration Engineer (script `05_*`) in all five stages |
| **Commit types** | `data` (01, data, contracts, annotation, fetch) and `integ` (05, API/app, pipeline, requirements) |
| **Your scripts** | `StageNN/01_data_engineer.py`, `StageNN/05_integration_engineer.py`, `common/dataio/*`, `common/integration/*`, `app.py`, `pages/*`, `run_pipeline.py` |
| **Why the project depends on you** | Every other role starts from your files and ends in your app. You also run the repository (CI, hooks, tags) |
| **You are the busiest role** | The load-shedding plan in section 9 is pre-agreed: use it at the first gate you miss, do not suffer silently |
| **Your two jobs each day** | Morning: publish clean data v1 within the first hour. Afternoon: wrap the day's models in an API/UI page |

## 1. Mission

1. Build and own the **data foundation**: real public data → calibration numbers → a calibrated, outcome-labelled simulator → stage datasets that obey the contracts.
2. Guarantee **no leakage by construction**: labels from outcomes, one split in one place, forbidden columns never reach models.
3. Turn each stage's best model into something a planner can use: **API + control-tower page**, always with `requires_operator_review: true`.
4. Keep four devices in sync: contracts, hooks, CI, tags, `run_pipeline.py`.

## 2. Reading order (30 minutes, Day 0)

1. `AGENTS.md` → `docs/00_PROJECT_BRIEF.md` → `docs/01_ARCHITECTURE.md` → `docs/09_HANDOFF_PROTOCOL.md`
2. `docs/04_GITHUB_WORKFLOW.md` (you also do the repo-admin checklist in its section 11) → `docs/06_DATA_SOURCES.md`
3. `contracts/` - read every schema and policy; these are your products
4. Problem statement PDF (Your Data blocks of each stage) and the reference project's `01_data_engineer.py` + `05_integration_engineer.py` of Stage 1 (structure only)

## 3. Your lane

| You OWN (may edit) | You READ (never edit) | You must NEVER touch |
|---|---|---|
| `StageNN/01_data_engineer.py`, `StageNN/05_integration_engineer.py` | `StageNN/03_*`, `artifacts/`, `predictions/` (to serve them) | `StageNN/02_*`, `reports/eda_*` (Role 2) |
| `StageNN/data/**`, `data/**`, `StageNN/annotation/**` | `reports/eval_*` (to show metrics, read the champion/threshold) | `StageNN/03_*`, `artifacts/**`, `predictions/**` (Role 1) |
| `StageNN/integration/**`, `reports/integ_*`, `HANDOFF/01_*`, `HANDOFF/05_*` | `reports/eda_*` (to show findings) | `StageNN/04_*`, `reports/eval_*`, `claims/**` (Role 4) |
| `contracts/**` (except `jargon_glossary.csv` which Role 2 extends), `common/dataio/**`, `common/integration/**`, `common/paths.py seed.py contracts.py` | | |
| `app.py`, `pages/**`, `run_pipeline.py`, root `requirements*.txt`, `.github/**`, `.githooks/**`, `scripts/**` | | |

## 4. Restrictions (hard)

1. **Never blank, clip, or silently drop** a value. Flag it with a `*_flag` column or a `was_*` indicator and log counts in `data_quality_report.json`. (Utilisation above 100% is real overload: flag, keep.)
2. **Never create a label from the feature columns.** `delay_status` is derived only from `realized_delay_hours` using `contracts/label_policy.json`.
3. **One split, one place**: the `split` column is assigned in `Stage01_ML/01_data_engineer.py` and nowhere else. Later stages read it; they never re-split.
4. **Fit scalers/encoders on train rows only** and store them (`scaler.json`). Never fit on all data.
5. **Contracts are frozen.** Additive change only through a PR approved by one other teammate. Never rename a column. Never change a unit.
6. **Every file you hand over passes** `python scripts/validate_contract.py <schema> <file>`; use `common.contracts.write_validated_csv` so a broken file cannot be written.
7. **No scraping behind logins, no ignoring robots.txt/terms, no personal data.** Real text is stored with `source_url` and date. Record every source in `docs/06_DATA_SOURCES.md`.
8. **Label data honestly**: every row has `data_origin` (and `text_origin` / `label_origin` / `summary_origin` for text stages).
9. **Never put a secret in the repo.** Files over 10 MB are written as `.csv.gz` (same base name + `.gz`); nothing over 50 MB is ever committed.
10. **Advisory only:** no endpoint or button may execute a reroute, booking or change. Every response carries `requires_operator_review: true`, `model_version`, and the evidence fields.
11. **Do not wait silently.** If an upstream file is late, work on the fixture and write the blocker in your HANDOFF file.

## 5. How you work in parallel with the others

| You wait for | Meanwhile you do |
|---|---|
| Role 1's model artefact (Day 1-5 afternoons) | Build the API/UI against `common.integration.StubPredictor` (returns fixture-shaped answers). When the artefact arrives, swap one line |
| Role 4's `eval_champion.json` (model name + threshold) | Show "threshold pending" in the UI; read it dynamically so you never hard-code it |
| Role 2's findings | Not needed for your build; link their charts in the UI when they land |
| Anyone's review of a contract change | Keep working on the previous contract version; do not merge until approved |

Others wait on **you** for: data v1 (target 60 minutes after the forenoon guide), `common/dataio` helpers, the repo/CI being green.
When you publish, post one line in the group chat.

## 6. Day 0 - Data foundation (6-8 hours; others help as noted)

### 6.1 Repository administration [MUST] (30 min)
- [ ] Do the owner checklist in `docs/04_GITHUB_WORKFLOW.md` §11 (collaborators, branch rule requiring the `guard` check, merge-commit only).
- [ ] Open a throwaway PR to see CI run green; fix `guard` if it fails on a fresh runner.
- [ ] Ask each teammate to run the "dress rehearsal" (clone, hooks, one-line PR merged).

### 6.2 Fetch real data and calibrate [MUST] (2 h) - `scripts/fetch_real_data.py`, `scripts/build_calibration.py` (commit type `data`)
Goal: numbers that make the simulation believable. Use `docs/06_DATA_SOURCES.md`; verify each endpoint and licence first and fill the "Verified on" notes.

| Source | Output (committed, small) | How it is used |
|---|---|---|
| IMF PortWatch daily port calls + chokepoint transits | `data/calibration/activity_profile.csv` (per node: weekday profile, annual trend, relative size); `data/calibration/real_event_anchors.csv` (event, node, start, end, % change vs trailing 8-week baseline) for: Suez blockage Mar 2021, Panama drought 2023-24, Red Sea diversions 2024, plus 2-3 port-level events you can find | Arrival rates and capacity scale; **scenario magnitudes in Stage 5** |
| Open-Meteo archive (hourly wind gust, precipitation at each port's lat/lon, simulation year) | `data/calibration/weather_hourly.csv.gz` | `weather_severity_index = 10 * min(1, 0.6*gust_kmh/90 + 0.4*precip_mm_h/20)` per node-hour (real weather in the simulator) |
| NY Fed GSCPI | `data/calibration/gscpi_monthly.csv` | Monthly global stress multiplier on baseline congestion |
| CPPI / published port statistics | `data/calibration/port_scale.csv` (mean dwell scale per port, with citation) | Per-port baseline dwell and capacity scale |

Fallback if any source fails: typed numbers with citation in `data/calibration/manual_anchors.csv`; state this in the HANDOFF file. Fetch scripts must be re-runnable and write raw downloads to `data/raw/real/` (git-ignored).

### 6.3 Calibrated simulator [MUST] - `common/dataio/simulator.py`, `scripts/build_sim_data.py` (type `data`)
This is the engine behind every dataset. Required design (a discrete-time **fluid queue network**; this is what makes cascades emerge from physics rather than from hand-written coupling):

**State per node and hour** (all derived/observed columns follow from these):
- inventory `I_t` (TEU waiting in the node), arrivals `A_t`, departures `D_t`.
- effective capacity `cap_t = cap0 * (1 - labor_deficit_t) * equipment_availability_t * weather_factor_t`, where `weather_factor_t = 1 - 0.8 * (weather_severity_t/10)^2`.
- departures `D_t = min(I_t + A_t - customs_held_t, cap_t)`; customs hold retains a fraction of the queue: `customs_held_t = I_t * min(0.9, customs_hold_hours_t / 48)`.
- update `I_{t+1} = I_t + A_t - D_t`. **Little's law** gives dwell: `avg_dwell_time_hours = I_t / max(D_t, 1e-6)` (clip 0-400) and `capacity_utilization_pct = 100 * I_t / yard_capacity` (allowed up to 150).
- cascade: `D_t` of a node feeds the arrivals of its downstream nodes `typical_transit_days*24` hours later (`contracts/network_edges.csv`), split equally across outgoing edges unless you calibrate shares. End nodes (e.g. DCs) leak a fraction out of the system.
- exogenous arrivals at origin ports follow `activity_profile` × GSCPI multiplier × peak-season factor × demand-surge events.
- drivers `labor_deficit_t`, `customs_hold_hours_t`, `equipment_availability_t`: mean-reverting AR(1) processes plus **events**.

**Event catalogue** (`data/calibration/event_catalogue.json`; Role 2 proposes frequencies/durations/magnitudes in `docs/decisions/0001-event-catalogue.md` on Day 0 - transcribe them): `CUSTOMS_SURGE`, `LABOR_STRIKE`, `STORM`, `EQUIPMENT_FAILURE`, `CHOKEPOINT_BLOCKAGE`, `DEMAND_SURGE`, `CYBER_OUTAGE`, `ROUTE_DIVERSION`. Each has: affected node types, rate per year, duration range, ramp/decay shape, parameter effects, **cause tag** (CUSTOMS / WEATHER / LABOR / EQUIPMENT / CONGESTION / SECURITY). Write every sampled event to `data/raw/sim/events_ground_truth.csv` (`event_id, type, node_id, start_ts, end_ts, severity, cause`) - Stage 3 text and Stage 5 scenarios are generated from it, so text and telemetry agree.

**Observed telemetry** = latent values + measurement noise (multiplicative lognormal, sigma 0.05-0.15) + **reporting issues** (see 6.5).

**Shipments:** sample about 60,000 shipments over the simulation year (origin/destination from major lanes, carrier from the 10-carrier list, priority tier with realistic proportions). A shipment's route is the shortest path over `network_edges.csv`. Its *planned* transit time is the sum of `typical_transit_days` + nominal node dwell. Its *snapshot* is taken at a random time at a random node on the route (columns: current node's dwell, throughput, utilisation, customs, weather, labour, equipment, node-level historical delay rate = trailing 30-day delay rate of earlier shipments, priority, carrier). Its **`realized_delay_hours`** = (sum over the *remaining* route of waiting beyond plan, from the future simulated node states) + **hidden-factor noise** (heavy-tailed: missed vessel connection, random inspection, yard-gang shortage - none of these appear in any column). Because the label depends on the *future* and on hidden factors, no model can reach 100%: expected range is a macro-F1 of roughly 0.6-0.85. If a simple baseline scores above 0.95 your simulator leaks; fix it.

**Simulation year:** 2025-01-01 to 2025-12-31 (real weather and GSCPI exist for it). Seed 42. One command: `python scripts/build_sim_data.py --seed 42`.

**Targets:** class mix about 65% ON_TIME, 22% AT_RISK, 13% CRITICAL_DELAY by the label policy thresholds - tune the hidden-noise scale **once** to get this mix and record it; never tune it to make models look better.

**Outputs [MUST]:**
- `data/raw/sim/shipment_telemetry_raw.csv.gz` (telemetry contract before cleaning: may contain the injected issues; `split` and `delay_status` not yet set)
- `data/raw/sim/node_hourly_raw.csv.gz` (hourly node view, full grid, before cleaning)
- `data/raw/sim/events_ground_truth.csv`
- `data/raw/sim/SIM_CARD.md` (every parameter, source, seed, version, class mix, known simplifications)

**Fallback if time is short [allowed]:** replace the fluid network by per-node AR(1) latent congestion with edge-lagged coupling - but keep outcome-based labels, hidden factors, events and the same output files. Say so in `SIM_CARD.md`.

### 6.4 Independent acceptance of the simulator [MUST] (Role 4 helps, 45 min)
Role 4 writes `tests/eval/test_sim_acceptance.py`; you make it pass: non-negative inventory, Little's-law error < 10% on average, class mix within 3 points of target, label not computable from snapshot columns (a depth-limited tree on them stays well below perfect), no future timestamps used in lag features, node hourly grid has no missing hours after cleaning.

### 6.5 Planted issues for the EDA role to discover [SHOULD]
Inject, and write to a **private** list that you commit only after Role 2 reports their Day 1 findings (`data/raw/sim/INJECTED_ISSUES.json`): a carrier whose dwell is reported in a different unit/offset after a date; a weekly reporting-lag pattern for one port; 0.5% sensor outliers; 1-2% missing values; 2 duplicated blocks. This lets Role 4 score the EDA ("found 5 of 6 planted issues"), which is a great debate fact.

### 6.6 Shared helpers [MUST] (1.5 h)
- `common/dataio/windows.py`: `build_windows(...)` exactly as promised in `docs/01_ARCHITECTURE.md` §6, with unit tests in `tests/data/` (shapes, no overlap across the purge gap, no test row in train windows).
- `common/dataio/graph.py`: `load_adjacency(...)`.
- `common/integration/stubs.py`: `StubPredictor` for each stage, `review_flag()` helper.
- `scripts/fetch_*.py` documented; `docs/06_DATA_SOURCES.md` provenance table filled.

**Day 0 exit gate:** `data/raw/sim/*` merged to `main`, `SIM_CARD.md` written, `tests/data` green, tag `day0-foundations`.

## 7. Day-by-day (build days)

Standard rhythm each day: **hour 1** data v1 on `main` (G1); **hours 2-4** integration against stub; **after the debate** swap in real artefacts and finish pages; **end of day** tag.

### Day 1 - Stage 01 ML

**`Stage01_ML/01_data_engineer.py` [MUST]** (input `data/raw/sim/shipment_telemetry_raw.csv.gz`)
1. Load as text; check required columns; stop with a clear error if any is missing.
2. Parse timestamps to UTC ISO; coerce numerics; every coercion failure is counted, never silently turned to a number.
3. Remove **exact** duplicate records (all columns equal); log count. Near-duplicates are only flagged (`dup_flag`).
4. Missing numerics: keep as missing in the file and add `was_missing_<col>` flags; the model code imputes with train medians (stored in the model). Do not pre-fill here.
5. Range rules from the schema: out-of-range → `range_flag_<col>`; **keep the value**. Utilisation above 100 → `overload_flag`, keep.
6. Derive `delay_status` from `realized_delay_hours` with `label_policy.json` thresholds (single function `derive_delay_status`, unit-tested).
7. Assign `split` chronologically per `label_policy.json` (70/15/15, 168-hour purge: rows whose timestamp is inside the purge gap get `split` = the earlier split but the *shipment_id group* is kept together; drop nothing - document the rule you implement and unit-test that no shipment appears in two splits).
8. Write `Stage01_ML/data/prepared_dataset.csv[.gz]` with `write_validated_csv(..., "telemetry", ...)`, and `data_quality_report.json` (row counts in/out, duplicates, missing per column, flags per column, class mix per split, date range per split).
9. HANDOFF `01_data_engineer.md` → READY, merge, post in chat. **Target: 60 minutes after the guide session.**

**`Stage01_ML/05_integration_engineer.py` [MUST]** FastAPI app (`uvicorn 05_integration_engineer:app --port 8001`, run from the stage folder):
- `GET /health`; `GET /model-info` (reads `artifacts/model_card.json` + `reports/eval_champion.json`);
- `POST /predict` (single shipment; pydantic model with the contract ranges and clear 422 errors);
- `POST /predict/batch` (up to 1,000 rows);
- response: `label`, `probabilities`, `risk_score`, `alert` (bool at the Role 4 threshold), `anomaly_flag`, `top_drivers` (top 3 SHAP features if available), `model_version`, `requires_operator_review: true`.
- Measure latency with 1,000 calls and write `reports/integ_latency.json` (p50/p95). Target p95 < 50 ms.
- If the model artefact does not exist yet, serve the stub and say `"model_version": "STUB"` loudly.

**Control-tower page [MUST]** `pages/1_Bottleneck_Scoring.py` (Streamlit): shipment form + batch upload, risk table sorted by risk, drivers per row, threshold shown, model-vs-baseline mini table read from `eval_*` files.
**Ops Field Guide [MUST]** `Stage01_ML/reports/integ_ops_field_guide.md` (+ printable HTML/PDF): ONE page for planners - what the three statuses mean, the top five drivers (from Role 1's importance matrix, wording by Role 2), the alert threshold and what a false alarm costs (Role 4), what the tool will not do. Every number copied from a file, with its path in a footnote.
[SHOULD] `tests/integ/test_api_stage01.py` (contract-shaped responses, 422 on bad input, review flag always true).
[STRETCH] Server-sent live feed replaying the test split as "real-time" shipment stream.

### Day 2 - Stage 02 DL

**`Stage02_DL/01_data_engineer.py` [MUST]** (input `data/raw/sim/node_hourly_raw.csv.gz`)
1. Build the **complete grid**: every node × every hour of the simulation year; mark missing hours `is_imputed=True`, fill with forward-fill limited to 6 hours, then node-hour-of-week median; log counts.
2. Keep `n_records`; add no new meaning to existing columns.
3. `splits.json`: calendar boundaries identical in meaning to Stage 1's split dates; 168-hour purge gap between splits.
4. `scaler.json`: per-node and per-channel median/IQR (robust) **fitted on train hours only**; document that inverse-transform is available in `common/dataio`.
5. `adjacency_matrix.csv` (27×27, node order of `network_nodes.csv`, symmetric version and a directed version).
6. Publish `hourly_node_telemetry.csv[.gz]` (validated, schema `hourly_node_telemetry`), `data_quality_report.json`.
7. **Windows [MUST]**: `common/dataio/windows.py` final and merged *before* Role 1 starts training (target: within the first 90 minutes).
[SHOULD] Training-time augmentation helpers (jitter, scaling, window-warp) as **functions Role 1 may call**; applied to train windows only; unit-tested for no leakage.

**`Stage02_DL/05_integration_engineer.py` [MUST]** `POST /forecast`: body = last 72 hourly records for one node (validated: exactly 72, ordered, no gaps) → horizons 1/6/24/72/168 with lower/upper bounds, `alert_prob_24h`, `requires_operator_review`. `GET /nodes` (registry), `GET /network` (nodes+edges).
**Page [MUST]** `pages/2_Network_Forecast.py`: world map (Plotly) with nodes coloured by current risk and lanes drawn from the edges; click a node → history + forecast fan chart with intervals; table of top-10 nodes by 24h exceedance probability; selector for model (persistence / LSTM / graph model) so the audience sees the skill difference.
[SHOULD] Visual Prediction Benchmark page reading Role 4's `eval_visual_benchmark.png/csv`.
[STRETCH] Animated propagation replay along edges for a historical episode.

### Day 3 - Stage 03 NLP

**`Stage03_NLP/01_data_engineer.py` [MUST]** builds `incident_messages.csv[.gz]`:
1. **Corpus sources** (record each in `docs/06_DATA_SOURCES.md`): (a) **real** advisories/notices you may lawfully use, with `source_url`, date, `text_origin=REAL` - target ≥ 150, minimum 60; (b) **event-linked synthetic messages** generated from `events_ground_truth.csv` using ≥ 40 phrasing templates across 4 source types with realistic shorthand ("LA/LB", "NYNJ", "omitted call", "blank sailing", "VACIS exam", "gate-out") - each message carries the event's real node(s), time, cause; `text_origin=TEMPLATE_SYNTH`; (c) [SHOULD] LLM paraphrases of (b) to broaden wording, `text_origin=LLM_PARAPHRASE` (key in `.env`, never committed; review a sample).
2. **Cleaning that keeps NER alive**: unicode normalise, strip HTML, mask e-mails/phones, collapse whitespace; **keep original casing and punctuation** (do not lower-case); replace nothing that is an entity.
3. De-duplicate: exact + near-duplicate (token-set similarity ≥ 0.92 via rapidfuzz) - keep one, log the rest.
4. **Group-aware split**: all messages from the same `template_id` or real source/day stay in one split, so near-duplicates never straddle train and test. Hold out at least 12 templates entirely for test.
5. Weak labels: `urgency_label`, `risk_factor`, `trade_lane`, `entities_json` from event attributes and the gazetteer → `label_origin=WEAK_RULE` or `TEMPLATE_SYNTH`.
6. **Gold set [MUST]**: set up the annotation pipeline in `Stage03_NLP/annotation/` (a JSONL import/export plus a local labelling tool, e.g. Label Studio or doccano, or a simple Streamlit labeller you write); export 150+ messages (≥ 60 real) from the test split for **human** labelling by Role 2 and Role 4 independently (40 overlapping); you compute Cohen's kappa (urgency) and entity-level agreement, and merge adjudicated labels into `label_origin=HUMAN_GOLD`. The gold rows are the **only** rows used for headline test numbers.
7. Validate (`incident_messages`), write `data_quality_report.json`, HANDOFF READY.

**`Stage03_NLP/05_integration_engineer.py` [MUST]** `POST /analyze` (text → urgency + confidence, risk factor, entities with `node_id`, `ambiguous` flag, `requires_operator_review`; review forced when confidence < 0.70 or urgency = URGENT); `GET /search?q=` over the corpus (BM25 via `rank_bm25`, filters node/urgency/date).
**Page [MUST]** `pages/3_Advisory_Desk.py`: paste/upload advisory → entities highlighted by type with node badges, urgency badge, "similar past tickets", map jump; search tab with filters.
**Signal bridge [MUST]** `Stage03_NLP/data/nlp_signals.csv` (`hour_ts, node_id, urgent_count_24h, watch_count_24h`) from the model's predictions on the whole corpus (used by the fused score on Day 5).
[SHOULD] Correlation panel: advisory pressure vs next-24h dwell change per node (numbers from Role 2/4 files).

### Day 4 - Stage 04 SLM

**`Stage04_SLM/01_data_engineer.py` [MUST]** builds `slm_pairs.csv`:
1. Sample **alert traces**: a hub (node) at a time with k = 1-30 shipments (statuses and key metrics from Stage 1 predictions/telemetry), optional forecast flag (Stage 2) and advisory flag (Stage 3). Include hard cases: all fine, missing values ("unknown", never 0), contradictory signals, > 20 shipments (truncation test), non-English-looking codes.
2. Compact **serialisation** (Role 2 decides the final format on Day 4 morning): e.g. `[12] PORT_US_LAX|CRIT|dwell=58h|cust=15h|util=104% ; ...`.
3. `source_facts_json` computed by code (worst node, counts per status, max dwell, main cause, list of numbers and node ids that are true) - this is the grading key for faithfulness.
4. Targets: deterministic fact selector → ≥ 20 planner-style paraphrase templates, exactly two sentences. [SHOULD] A teacher LLM drafts alternatives; the team reviews ≥ 10%; `summary_origin=TEACHER_LLM_REVIEWED`. **100 test pairs are hand-written by Roles 2 and 4 (50 each)** with `summary_origin=HUMAN`; give them the trace list and a shared sheet.
5. Split by **scenario/time/hub** (not by random row); ≈ 1,800 train / 300 val / 300 test pairs.
6. Validate (`slm_pairs`), report token-length statistics, HANDOFF READY.

**`Stage04_SLM/05_integration_engineer.py` [MUST]** the **offline briefing client**: local app (Streamlit or a small FastAPI+HTML page) that loads the model from a local folder with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; select an alert batch → shows (left) the raw alert table, (right) the 2-sentence brief with **evidence highlighting** (numbers/node ids in the brief matched to the table); a guard runs Role 4's `grounding_check` - if it fails, show "Manual review required" plus the deterministic template brief; shows latency per brief; always `requires_operator_review`.
- `scripts/package_offline.py`: copies app + merged model + a sample alert file into `dist/offline_hub/` with a one-command launcher (`run_offline.bat` / `run_offline.sh`) and a README.
- **Offline proof [MUST]**: `tests/integ/test_offline_demo.py` monkeypatches `socket.socket.connect` to raise, sets offline env vars, loads the model and generates a brief; the test must pass. Also demo it live with Wi-Fi off.
[STRETCH] Side-by-side operator view: 100 raw alerts vs the brief + a timer for the reading-time study (Role 4 runs it).

### Day 5 - Stage 05 GenAI and the whole system

**`Stage05_GenAI/01_data_engineer.py` [MUST]** `data/baseline_profiles.json` (per node: mean/std/quantiles per channel per hour-of-week, channel correlation matrix, capacity, Little's-law ratio), `real_event_anchors.csv` (from Day 0), training windows manifest for the generator (nominal-only windows, train period, 168 h).
**`common/dataio/shipment_sampler.py` [MUST, needed by Role 4 for the stress test]** `sample_shipments_from_nodes(node_df, n, seed)` turns ANY node-hour table (real-nominal, simulator or generated scenario) into shipment snapshots in the `telemetry` contract, reusing the simulator's delay logic (including hidden-factor noise) so `realized_delay_hours` is defined the same way everywhere. Unit-test it and publish it in the first 90 minutes of Day 5.

**`Stage05_GenAI/05_integration_engineer.py` + app [MUST]**
- `pages/5_Scenario_Lab.py`: choose one of the 20 catalogue scenarios or build the **Wildcard** (sliders: blockage severity/duration, strike severity, demand multiplier; node pickers); press *Run* → loads/generates the scenario stream → runs it through Stage 1 and Stage 2 models and the Stage 3 advisory generator → live charts (node dwell, alerts raised, lead time, stage that degrades first) + recorded fallback if live generation is slow.
- `pages/6_Triage.py`: the **fused CASCADE triage score** (architecture §7) per node with the weights visible; click → evidence from each stage.
- `run_pipeline.py` updated so `python run_pipeline.py` runs all stages end to end on a clean clone; `tests/integ/test_pipeline_smoke.py`.
- Final documentation: refresh `docs/01_ARCHITECTURE.md` (as built), write `docs/LLD.md` (module list, data flow, API tables, run commands, known limitations), update `README.md`.
- Tag `day5-complete`; make a `final-submission` tag after the last fix.
[SHOULD] Docker-free "one-click" launcher `run_demo.bat/.sh`.
[STRETCH] A recorded 3-minute demo video saved outside git (link in README).

## 8. Hand-off checklist (every day, before READY)

- [ ] All produced files pass `validate_contract.py`; output pasted into your HANDOFF file
- [ ] `data_quality_report.json` lists every change you made to the data
- [ ] No absolute paths, no secrets, nothing over 50 MB, `.gz` for files over 10 MB
- [ ] Unit tests for new helpers in `tests/data/` or `tests/integ/`
- [ ] PR merged, tag created at day end, one-line chat post
- [ ] `python run_pipeline.py --stage N` runs (skipped scripts allowed, failures not)

## 9. Load-shedding plan (pre-agreed; use it without guilt)

If at gate G1 or G2 you are behind: post it in the chat immediately. Default re-allocation:
1. Role 4 takes the Ops Field Guide layout (Day 1) and the offline-proof test (Day 4).
2. Role 2 takes the annotation tool setup (Day 3) and the reading-time study design (Day 4).
3. Role 1 takes `StubPredictor` swapping inside the API when the model is theirs.
4. You keep: data v1, contracts, API core, CI/tags.
Anything dropped is written in `docs/KNOWN_LIMITATIONS.md` by Role 4 that day.

## 10. Debate duties (you open and you demo)

You must be able to explain, alone and without notes:
- Where each dataset comes from, what is real and what is simulated, and why the simulation is **not** just random numbers (fluid queue, events, calibration).
- Why the label is an outcome with hidden factors, and how the split avoids leakage (chronological + purge + grouped by shipment).
- What the Little's-law/conservation relation is and why it matters for the Stage 5 realism audit.
- The API request/response, the 50 ms latency target and the measured number, and why `requires_operator_review` is always true.
- How four devices stay conflict-free (ownership lanes, contracts, hooks, CI) - show `python scripts/board.py`.
- What we do **not** claim (live feeds, production accuracy).

## 11. Cover-for-me checklist (if you are absent)

Another teammate can keep the system alive with: `python scripts/build_sim_data.py --seed 42` (regenerates raw data), `python Stage0N/01_data_engineer.py`, `python scripts/validate_contract.py ...`, `uvicorn` command in each integration script header, `python scripts/board.py`. These commands must stay in `README.md` and in each script header.

## 12. Common mistakes to avoid

- Editing a column "just a little" in an already published file (bump `data_version`, tell the chat).
- Filling missing values in the shared file (models must impute with train-only statistics).
- Lower-casing text in Stage 3 data (NER needs case).
- Hard-coding the alert threshold in the API (read it from `eval_champion.json`).
- Letting the Streamlit page compute a metric itself (it must read the value from an `eval_*` file).
- Letting an API call raise a 500 on bad input (return 422 with a helpful message).
- Merging your own contract change without a second approver.
