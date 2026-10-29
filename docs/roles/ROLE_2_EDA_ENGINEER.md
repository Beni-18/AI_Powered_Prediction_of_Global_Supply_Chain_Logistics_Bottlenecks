# ROLE 2 - EDA Engineer (also Prompt / Workflow Engineer)

> **For the AI assistant reading this file:** you are working for the teammate who holds Role 2. This file is your
> job description. Rules in `AGENTS.md` come first. Priority tags: **[MUST]** deliver or the day fails,
> **[SHOULD]** if MUSTs are done, **[STRETCH]** only with spare time.

## 0. At a glance

| | |
|---|---|
| **Programme role** | EDA / Prompt / Workflow Engineer: Day 1 distributions and correlations, Day 2 time-series + attention analysis, Day 3 text and word frequency, Day 4 data quality + tokens, Day 5 prompts and boundary conditions |
| **Commit type** | `eda` |
| **Your scripts** | `StageNN/02_eda_engineer.py` (all five stages) |
| **Your folders** | `StageNN/reports/eda_*` (file-name prefix `eda_` is your marker), `Stage05_GenAI/scenarios/`, `common/edakit/`, `tests/eda/`, `notebooks/eda/`, `contracts/jargon_glossary.csv` (the only contract file you may extend) |
| **What you hand over** | Findings that **change decisions** for the other three roles: a `eda_findings.md` per stage with "so we do X" recommendations, plus charts and tables |
| **Your golden habit** | A chart without a decision attached is decoration. Every finding ends with *"so we decided / recommend ..."* |
| **You are judged on** | Insight, rigour (noise vs signal), clear charts, and being able to explain statistics in plain words |

## 1. Mission

Find out what is **really in** the data - noise, drift, seasonality, leaks, propagation, jargon, token problems - before
anyone trusts a model. You are the team's early-warning system for bad data and for models that are right for the wrong
reasons. You also design the **scenarios** (Day 5) that make our stress test credible.

## 2. Reading order (Day 0)

`AGENTS.md` → `docs/00_PROJECT_BRIEF.md` → `docs/01_ARCHITECTURE.md` → `docs/09_HANDOFF_PROTOCOL.md` →
`docs/04_GITHUB_WORKFLOW.md` → `docs/08_COMMENTING_STANDARD.md` → contracts (`telemetry`, `hourly_node_telemetry`,
`incident_messages`, `slm_pairs`) and `contracts/jargon_glossary.csv`. Skim the reference EDA scripts for structure
and see what they do not check (leakage, drift, noise floor).

## 3. Your lane

| You OWN | You READ | You must NEVER touch |
|---|---|---|
| `StageNN/02_eda_engineer.py` | `StageNN/data/*`, `data/raw/sim/*`, `StageNN/artifacts/*`, `StageNN/predictions/*` (for attribution/error analysis) | `StageNN/data/**`, `contracts/**` except the glossary (Role 3) |
| `StageNN/reports/eda_*` | `reports/eval_*` | `StageNN/03_*`, `artifacts/`, `predictions/` (Role 1) |
| `StageNN/HANDOFF/02_eda_engineer.md` | | `StageNN/04_*`, `reports/eval_*`, `claims/**` (Role 4) |
| `Stage05_GenAI/scenarios/**`, `common/edakit/**`, `tests/eda/**`, `notebooks/eda/**` | | `StageNN/01_*`, `05_*`, `app.py`, `pages/**` (Role 3) |

## 4. Restrictions (hard)

1. **EDA never changes data.** You record findings and recommend; Role 3 cleans (and logs). Never "fix" a CSV in place, never save a modified copy of a hand-over file.
2. **Look at train (and val) only for anything that influences modelling decisions.** You may look at the test split to check *drift* or *shift*, but never to choose features or thresholds, and say so in the report.
3. **Every finding states: what, evidence (a number/chart path), so-what (decision).** No orphan charts.
4. **Every chart is produced by code** with `common.edakit.apply_style()` and `save_fig()`; titles state the finding ("Dwell noise is 3x higher at LAX than at Rotterdam"), axes have units, and the colour scheme is consistent (colour-blind safe).
5. **Distinguish noise from signal explicitly**: report a noise floor with a robust estimator (MAD), never use the standard deviation of a heavy-tailed series.
6. **No causal language** from correlation. Say "associated with".
7. **Annotation guidelines and gold labelling are done independently** from Role 4 (Day 3), then adjudicated; do not look at the other's labels before finishing yours.
8. **Plain-English comments on every step** (`docs/08_COMMENTING_STANDARD.md`).
9. Keep notebooks in `notebooks/eda/` as scratch; the script version is the deliverable.
10. When a chart reveals something alarming (leak, broken unit, impossible values) **tell the group chat immediately**, do not wait for the report.

## 5. How you work in parallel

| You wait for | Meanwhile |
|---|---|
| Role 3's prepared data v1 | Start on the **raw** simulator files in `data/raw/sim/` (available since Day 0); re-run on the prepared file and note differences |
| Role 1's model (only Day 2 attribution; Day 3-4 error analysis) | Do all model-independent analysis first; the model-dependent part is the last 25% of your day |
| Role 4's metrics (for the error analysis) | Prepare the slicing code; run it when predictions exist |

Who waits on you: Role 1 (feature recommendations, serialisation decision), Role 4 (annotation guidelines, boundary values for checks), Role 3 (cleaning rules, Field Guide wording, planted-issue scoring).
**Post your first three findings in the chat within 90 minutes of data v1**; the full report follows later.

## 6. Day 0 - Foundations (3 hours)

- [ ] **[MUST]** Environment as in `docs/04_GITHUB_WORKFLOW.md` §2 plus `statsmodels`, `seaborn`.
- [ ] **[MUST]** `common/edakit/`: `apply_style()`, `save_fig(fig, path)`, `psi(expected, actual, bins=10)` (population stability index), `robust_noise_floor(series, window=24)` (rolling-median residual, MAD×1.4826), `ks_drift(a, b)`, `univariate_auc(df, features, target)` (leak detector), with unit tests in `tests/eda/`.
- [ ] **[MUST]** **Event catalogue proposal** `docs/decisions/0001-event-catalogue.md` for Role 3's simulator: for each event type (customs surge, labour strike, storm, equipment failure, chokepoint blockage, demand surge, cyber outage, route diversion) give frequency per year, duration range, affected node types, parameter effects and **a real-world precedent with date and source** (e.g. Ever Given blockage of Suez, March 2021; Panama Canal drought restrictions 2023-24; Red Sea diversions from late 2023; 2021-22 LA/LB anchorage backlog; 2022 Felixstowe port strike and the US rail labour dispute - verify each one before citing). Cite sources; mark what you could not verify.
- [ ] **[SHOULD]** Draft the Day 3 **annotation guidelines** (`Stage03_NLP/reports/eda_annotation_guidelines.md`): urgency definitions (URGENT = disruption in effect or imminent within 24 h with cargo impact; WATCH = potential/partial/regional or announced for 24-72 h; ROUTINE = administrative/informational), risk-factor definitions, entity span rules (PORT, CARRIER, DISRUPTION_TYPE, MITIGATION_ACTION), 10 worked examples including edge cases and negations.
- [ ] **[SHOULD]** Read about: STL decomposition, ACF/PACF, PSI, KS test, cross-correlation, integrated gradients/permutation attribution, token vs subword, Cohen's kappa.

## 7. Day-by-day

### Day 1 - Stage 01 ML (`Stage01_ML/02_eda_engineer.py`)

Input: `Stage01_ML/data/prepared_dataset.csv[.gz]` (and raw for comparison). Outputs in `Stage01_ML/reports/` with prefix `eda_`.

**[MUST] Noise floor and seasonality** (problem statement: "identify noise floor variations across seasonal demand cycles and reporting drift")
1. Hour-of-day, day-of-week, month profiles of dwell, throughput, utilisation per node type → `eda_seasonality.csv` + `eda_seasonality.png`.
2. Per-node **noise floor** (robust: rolling median residual → MAD × 1.4826) per month → `eda_noise_floor.csv` + heat-map `eda_noise_floor.png`. Finding example: "noise floor doubles in Q4 at US west-coast ports → alert thresholds must be seasonal."
3. **Reporting drift**: PSI and KS between first and last third of the period per column, and per carrier; check unit/offset shifts and reporting-lag patterns (timestamp minute/second distributions, weekday gaps) → `eda_drift_report.json`, `eda_drift.png`. State which differences are genuine operations changes and which look like reporting artefacts.
4. Outliers and impossible values: counts per column using IQR and robust-z; list of overload (>100% utilisation) cases kept as real.

**[MUST] Target and leakage audit**
5. Class balance overall and per node/carrier/month → `eda_class_balance.csv`.
6. **Leakage detector:** univariate ROC-AUC of every allowed feature for CRITICAL vs rest, and a shallow decision tree (depth 3) on allowed features → if any single feature or the depth-3 tree is near-perfect (> 0.95), raise the alarm immediately (the simulator may leak). Also check that forbidden columns do not correlate trivially (they must not be in any feature list). → `eda_leakage_audit.json`.
7. Feature associations: Spearman correlation matrix, mutual information with the target (train only), VIF for multicollinearity → `eda_correlations.csv`, `eda_mutual_info.csv`, `eda_vif.csv`.

**[MUST] Recommendations to Role 1** in `eda_findings.md` (this is the deliverable others read): features to add (e.g. `eng_` ratios), transforms (log of dwell), categorical handling, expected hard cases, which node/period will be hard, whether thresholds must be seasonal.
**[MUST] Field Guide wording** for Role 3: a 120-word plain-English explanation of the top five drivers (after Role 1's importance matrix lands), in `eda_field_guide_text.md`.
**[SHOULD]** Score yourself against the planted issues once Role 3 reveals `INJECTED_ISSUES.json` (recall of planted issues) → `eda_planted_issue_recall.md`; compare raw vs prepared data (did cleaning remove or keep the issues?). 
**[SHOULD]** Error analysis after Role 4's first evaluation: which segments (node, carrier, hour, class) have the worst errors → `eda_error_slices.csv` and recommended fixes.
**[STRETCH]** Change-point detection (PELT, `ruptures`) on node dwell to date regime changes; compare with `events_ground_truth.csv`.

### Day 2 - Stage 02 DL (`Stage02_DL/02_eda_engineer.py`)

Input: `Stage02_DL/data/hourly_node_telemetry.csv[.gz]`, node registry/edges; after Role 1's model exists, its artefact.

**[MUST] Before the model (first 60% of the day)**
1. Stationarity and memory: ADF test per node/channel, ACF/PACF up to 168 lags → `eda_acf_summary.csv`, `eda_acf.png`. Finding example: "daily and weekly cycles dominate; use 72 h input and seasonal-naive baseline."
2. Seasonal decomposition (STL, period 24 and 168) → `eda_stl_components.png`: seasonal vs trend vs residual shares of variance per node.
3. **Peak-season congestion vs structural bottleneck** (the problem statement's team huddle: review 5 ambiguous delay graphs): choose 5 episodes (from `events_ground_truth.csv` plus two non-event spikes), plot raw, seasonal-adjusted residual, and neighbours; classify each as seasonal peak, transient shock, or structural, with an explicit rule (residual above 3× noise floor for ≥ 24 h = structural candidate) → `eda_ambiguous_graphs.md` + 5 PNGs. This rule feeds Role 4's alert definition.
4. **Node-to-node lead-lag** (evidence for the GNN): cross-correlation of residuals between every pair of linked nodes over lags 0-14 days; estimate delay and strength per edge → `eda_leadlag_edges.csv` (columns `src, dst, best_lag_h, corr, n`) and a heat-map. Compare estimated lags with `typical_transit_days`. Hand to Role 1 for propagation weights (Stage 5).
5. Missing/imputed hours: how many per node, are they clustered → informs Role 3.
6. Recommendations to Role 1: window length, horizons, whether node embeddings are needed, which channels carry signal → `eda_findings.md`.

**[MUST] After the model (last 40%)** (problem statement: "analyse model attention maps to confirm focus on genuine disruption signals over reporting noise")
7. **Attribution**: for the LSTM, compute integrated gradients (or permutation-by-channel and by-time-step if gradients are awkward) over the 72-hour window for the test origins; aggregate per channel and per lag → `eda_attribution_channels.csv/png`, `eda_attribution_timesteps.png`. Check the model leans on genuine drivers (customs hold, utilisation, weather, labour) rather than the noisy/reporting channels.
8. **Noise-channel sanity test**: add a column of pure random noise (and a copy of a channel shuffled in time) as extra input to a small re-training (Role 1 provides a `--noise-channel` flag or you train a tiny model yourself on the same windows); attribution to the noise channel must be ≈ 0. → `eda_noise_channel_test.json`. This is a strong debate point.
9. If a Transformer exists: attention-weight heat-maps by lag for 3 episodes.
**[SHOULD]** Compare attribution of `graph_full` vs `graph_none` to show *where* the graph helps (which nodes' histories matter for which target nodes).
**[STRETCH]** Granger-causality test between linked nodes; compare with the learnt attention.

### Day 3 - Stage 03 NLP (`Stage03_NLP/02_eda_engineer.py`)

Input: `incident_messages.csv[.gz]` (train + val for modelling decisions).

**[MUST]**
1. Corpus profile per `source_type` and `text_origin`: length (chars/tokens), vocabulary size, type/token ratio, share of uppercase, digits, shorthand tokens → `eda_text_profile.csv/png`.
2. **Terminology and shorthand** (problem statement: "analyse freight terminology and shorthand across carrier and customs sources"): top n-grams per urgency and per risk factor (TF-IDF log-odds), shorthand inventory (LA/LB, NYNJ, VACIS, LFD, omitted call, blank sailing ...), out-of-vocabulary rate of real vs synthetic text (domain gap!) → `eda_top_terms_by_urgency.csv`, `eda_shorthand_inventory.csv`, `eda_domain_gap.json`.
3. **Gazetteer coverage:** share of messages in which each node/carrier alias is found; list of capitalised unknown tokens that look like ports/carriers but are not in the registry → `eda_gazetteer_coverage.csv`. Extend `contracts/jargon_glossary.csv` with new terms and ambiguities (this is the one contract file you may edit; commit type `eda`).
4. **Label noise and class balance**: balance per split; confident-learning-style check (cross-val predicted vs given label disagreements, sample 30 for manual review) → `eda_label_noise.csv`.
5. **Annotation**: finalise `eda_annotation_guidelines.md`; label your share of the gold set **independently** (Role 3 provides the tool and the item list); 40 items overlap with Role 4 for agreement.
6. **Terminology Misinterpretation catalogue (input to Role 4's audit):** ≥ 25 ambiguous terms with plain meaning, why they confuse, and the correct mapping (e.g. "hold" = customs vs carrier vs terminal; "terminal" = marine vs rail; "ramp"; "LAX"; "Red Sea" → two chokepoints) → `eda_ambiguous_terms.csv` (Role 4 turns it into tests).
7. Recommendations to Role 1 → `eda_findings.md`: tokenisation choices (keep case for NER), n-gram ranges, negation handling, expected hard classes.
**[SHOULD]** Duplicate/near-duplicate leakage check between train and test (token-set similarity) and template-share in each split. 
**[STRETCH]** Topic model (NMF) of tickets to propose new risk factors; time-lag analysis of advisory counts vs node dwell change (cross-correlation) - evidence for the fused score.

### Day 4 - Stage 04 SLM (`Stage04_SLM/02_eda_engineer.py`)

Input: `slm_pairs.csv`, tokenizer of the chosen model (FLAN-T5 SentencePiece).

**[MUST]** (problem statement: "audit token distributions to guarantee mission-critical port and carrier codes are retained")
1. **Tokenisation audit:** for every node id, carrier code and typical number format, count the subword pieces; list codes that split into ≥ 4 pieces or contain `<unk>` → `eda_token_audit.csv`. Propose **short aliases** (e.g. `LAX` for `PORT_US_LAX`) if it helps and test round-trip mapping.
2. **Length and truncation:** distribution of source/target token lengths; % of pairs truncated at 384/512; **critical-node-loss rate** (a CRITICAL shipment falls beyond the truncation point) → `eda_length_truncation.csv/png`. Decision: maximum shipments per trace or sorting by severity before truncation.
3. **Serialisation decision** `eda_serialisation_decision.md`: compare 3 candidate serialisations (verbose key=value, compact pipe format, JSON) by token count and critical-info retention; recommend one **within the first two hours of Day 4** so Role 3 and Role 1 can build on it.
4. **Data quality of pairs:** missing-value handling ("unknown" never 0), duplicate traces, balance of statuses/hubs/lengths, target sentence-count and length, templates share per split → `eda_pairs_quality.csv`.
5. **Prompt workflow (the "Prompt Engineer" part):** write 3 prompt variants for the `base_prompt` system and (if used) the teacher LLM; run them on 50 validation pairs with Role 4's metric functions (`grounding_check`, sentence-count) → `eda_prompt_comparison.csv`; recommend the best.
6. **Hand-written test pairs:** write 50 of the 100 human test pairs (Role 4 writes the other 50) - two sentences each, only facts in the trace, varied voice.
**[SHOULD]** Salience study: which facts do hand-written summaries mention first (worst node? counts? cause?) → informs the template/teacher selector.
**[STRETCH]** Error taxonomy of generated summaries once Role 1 has outputs (omission, wrong number, invented cause).

### Day 5 - Stage 05 GenAI (`Stage05_GenAI/02_eda_engineer.py`)

Input: `Stage05_GenAI/data/baseline_profiles.json`, `real_event_anchors.csv`, Stage 2 lead-lag results.

**[MUST]** (problem statement: "define boundary conditions for extreme edge cases, e.g. canal blockage plus demand shock")
1. **Boundary conditions:** per channel and node type, the plausible envelope (99.5th percentile of nominal, and real-event extremes from the anchors) → `eda_boundary_conditions.csv` and `.md`.
2. **Scenario catalogue (20 + wildcard)** `Stage05_GenAI/scenarios/scenario_catalog.json`: for each scenario `id (S01..S20)`, name, narrative (2 sentences), affected nodes, event types, start offset, duration range, severity range, ramp shape, expected first-affected stage, **real-world precedent and magnitude anchor** where one exists. Must cover: single-node events (port closure, customs surge, labour strike, storm, equipment failure, cyber outage), chokepoint events (Suez blockage, Panama restriction, Red Sea diversion, Malacca congestion), demand events (peak surge, pre-holiday surge, tariff front-loading), compound events (strike + storm; blockage + demand), and slow burns (gradual equipment decay, persistent customs inspection regime).
3. **Wildcard design** `eda_wildcard_design.md`: simultaneous canal blockage (Suez), port labour strike (a named major port, e.g. Rotterdam or LA), and regional demand spike; give parameters, narrative, hypothesis of which stage degrades first and why, and the question we want the live demo to answer.
4. **Prompts for scenario narratives / advisories** (the "Prompt Engineer" part): templates that turn a scenario into 3-5 realistic advisory messages for Stage 3 (used to test NLP under stress), with explicit constraints (use only given facts, include node names with real aliases).
5. Recommendations to Role 1: propagation weights from Stage 2 lead-lag (`eda_propagation_weights.csv`: src, dst, weight, lag_h), severity ranges, what the generator must not do.
**[SHOULD]** Compare real anchors with scenario severity: plot each scenario's magnitude against the nearest real event ("our Suez scenario removes X% of transits for Y days; the real 2021 blockage removed A% for B days" - X, Y, A, B come from the scenario catalogue and the anchors file, never from memory).
**[STRETCH]** Sensitivity-analysis grid design for Role 4 (which two parameters to sweep for the wildcard).

## 8. Hand-off checklist (every day)

- [ ] `eda_findings.md` exists with numbered findings, each with *evidence path* and *so-what*
- [ ] Every chart is reproducible by `python 02_eda_engineer.py` from the stage folder and uses `apply_style()`
- [ ] No data file was modified; all tables are new `eda_*` files
- [ ] Recommendations to Role 1, 3 and 4 listed under their names
- [ ] First three findings posted in chat; alarming findings escalated immediately
- [ ] Plain-English comments on every step; no commented-out code

## 9. Debate duties

Without notes you must explain:
- What a noise floor is and how MAD differs from standard deviation; what drift/PSI means and the threshold you used (PSI > 0.2 = notable shift).
- Seasonality vs structural bottleneck and the rule we use to tell them apart.
- How you tested for leakage (univariate AUC, depth-3 tree) and the number you found.
- What the lead-lag analysis showed and why it justifies a graph model.
- What the attribution/noise-channel test showed (model looks at the right signals).
- How the gold set was built (guidelines, two annotators, kappa value) and what the domain gap between real and synthetic text is.
- Why we serialise traces the way we do (token audit), and how the 20 scenarios were chosen and anchored to real events.
- **Your three best findings per stage and the decision each one caused.**

## 10. Cover-for-me checklist

Every `02_*` script regenerates all its outputs with one command and writes `eda_findings.md` from a template filled by code (numbers inserted by code, prose by you). Another teammate can re-run and edit the prose.

## 11. Common mistakes to avoid

- Standard deviation on spiky data (use MAD / quantiles).
- Plotting without a decision (every chart: "so we ...").
- Looking at the test split to pick features.
- Reporting a correlation as a cause.
- Overwriting or "cleaning" shared data files (not your lane).
- Lower-casing text before the NER analysis (case carries signal).
- Declaring the attribution "looks fine" without the noise-channel test.
- Writing scenarios with no real-world anchor.
