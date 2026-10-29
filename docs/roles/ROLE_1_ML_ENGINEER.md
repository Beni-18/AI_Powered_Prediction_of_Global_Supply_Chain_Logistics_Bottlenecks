# ROLE 1 - ML Engineer (the model role in all five stages)

> **For the AI assistant reading this file:** you are working for the teammate who holds Role 1. This file is your
> job description. Rules in `AGENTS.md` come first. Priority tags: **[MUST]** deliver or the day fails,
> **[SHOULD]** if MUSTs are done, **[STRETCH]** only with spare time.

## 0. At a glance

| | |
|---|---|
| **Programme role** | Model / AI Engineer. Day 1 trains ML models, Day 2 DL, Day 3 NLP, Day 4 fine-tunes the SLM, Day 5 builds the generator |
| **Commit type** | `ml` |
| **Your scripts** | `Stage01_ML/03_ml_engineer.py`, `Stage02_DL/03_dl_engineer.py`, `Stage03_NLP/03_nlp_engineer.py`, `Stage04_SLM/03_slm_engineer.py`, `Stage05_GenAI/03_genai_engineer.py` |
| **Your folders** | `StageNN/artifacts/`, `StageNN/predictions/`, `common/modelkit/`, `tests/ml/`, `notebooks/ml/`, `StageNN/requirements.txt` |
| **What you hand over** | Trained artefacts + `model_card.json` + a **predictions file in the contract format** (including baselines) |
| **Your golden habit** | Baseline first. Then the model. Then show what the model adds over the baseline |
| **You are judged on** | Choosing sensible models, training them without leakage, knowing why they work, explaining them in plain words |

## 1. Mission

Build, per stage, a small set of models that **beat clearly-stated baselines on a leak-free split**, hand them over in a form
Role 4 can evaluate and Role 3 can serve, and be able to explain every choice. Fancy models that do not beat the baseline
are reported honestly - that is still a good result.

## 2. Reading order (Day 0, 30 min)

`AGENTS.md` → `docs/00_PROJECT_BRIEF.md` → `docs/01_ARCHITECTURE.md` (esp. §5 labels/splits and §6 interfaces) →
`docs/09_HANDOFF_PROTOCOL.md` → `docs/04_GITHUB_WORKFLOW.md` → `docs/08_COMMENTING_STANDARD.md` → all of `contracts/`
(schemas `telemetry`, `ml_predictions`, `dl_forecasts`, `nlp_predictions`, `slm_generations`, `scenario_telemetry`; policies `label`, `feature`, `cost`).
Then skim the reference project's `Stage0N/03_*` scripts for structure and notice what they lack (no baselines, leaky labels).

## 3. Your lane

| You OWN | You READ | You must NEVER touch |
|---|---|---|
| `StageNN/03_*_engineer.py` | `StageNN/data/*` (inputs from Role 3), `contracts/**` | `contracts/**` (frozen), `StageNN/data/**` (Role 3) |
| `StageNN/artifacts/**`, `StageNN/predictions/**` | `StageNN/reports/eda_*` (recommendations from Role 2) | `StageNN/02_*`, `reports/eda_*` (Role 2) |
| `StageNN/HANDOFF/03_model_engineer.md` | `StageNN/reports/eval_*` (feedback from Role 4) | `StageNN/04_*`, `reports/eval_*`, `claims/**` (Role 4) |
| `StageNN/requirements.txt`, `common/modelkit/**`, `tests/ml/**`, `notebooks/ml/**` | `common/dataio`, `common/evalkit` (import, don't edit) | `StageNN/01_*`, `05_*`, `app.py`, `pages/**` (Role 3) |

## 4. Restrictions (hard)

1. **Never touch the test split** for fitting, early stopping, feature selection, calibration or threshold choice. Train on `train`; tune/early-stop on `val`; the test split is opened by Role 4's evaluation script (you may produce predictions on it but must not look at its metrics to change anything).
2. **Only features allowed by `contracts/feature_policy.json`.** Never `realized_delay_hours`, `delay_status`, ids, `scenario_id`, `data_origin`, `split`, raw `timestamp`. Write a `assert_features_allowed(cols)` check in `common/modelkit` and call it in every training script.
3. **Every model ships with its baseline** in the same predictions file (`model_name` differs). No baseline → no hand-off.
4. **Fixed seed 42** via `set_global_seed()`; log library versions in `model_card.json`.
5. **Fit every preprocessor (imputer, scaler, encoder) on train only** and store it inside the saved model object/pipeline.
6. **Lag/rolling features use only earlier timestamps** (shift before rolling). A unit test in `tests/ml/` proves this.
7. **Write predictions only through** `common.contracts.write_validated_csv(df, "<schema>", path)`.
8. **Do not tune on Role 4's feedback by peeking at test numbers.** If Role 4 reports a problem, fix the cause (e.g. calibration), re-train, re-hand-off as v2.
9. **Model files < 50 MB** (hook blocks bigger). Prefer LoRA adapters, quantised weights, or smaller models; never commit checkpoints.
10. **Every training script runs end to end from the stage folder in < 20 minutes on a laptop CPU** (state the exact command at the top). If it cannot, add a `--fast` flag that uses fewer trials/epochs and document which numbers came from which mode.
11. **Plain-English comments on every step** (`docs/08_COMMENTING_STANDARD.md`). A teammate must be able to read your script top to bottom like a story.
12. No hand-typed metrics anywhere: metrics in `model_card.json` come from code.

## 5. How you work in parallel

| You wait for | Meanwhile |
|---|---|
| Role 3's prepared data v1 (≈ 60 min after the guide) | Build the training script on the **fixture** (`contracts/fixtures/<schema>.csv`): feature builder, baselines, training loop, prediction writer, model-card writer. Switch the input path to the real file when it lands |
| Role 2's recommendations (EDA) | Start with the feature policy base set; add Role 2's recommended features as v2 |
| Role 4's first evaluation | Keep improving; read `reports/eval_*`; fix calibration/threshold-related findings |

Who waits on you: Role 4 (needs `predictions/*`), Role 3 (needs `artifacts/*` for the API), Role 2 (needs the model for attribution on Day 2, and for the error analysis on later days).
**Publish a "v0 predictions" file with the baselines only within 45 minutes of data v1** so Role 4 can start immediately.

## 6. Day 0 - Foundations (3-4 hours)

- [ ] **[MUST]** Environment: `pip install -r requirements.txt -r Stage02_DL/requirements.txt -r Stage03_NLP/requirements.txt -r Stage04_SLM/requirements.txt -r Stage05_GenAI/requirements.txt`; run `python -m spacy download en_core_web_sm`.
- [ ] **[MUST]** Pre-download models while you have internet (cache stays on your disk): `google/flan-t5-small`, `google/flan-t5-base`, `sentence-transformers/all-MiniLM-L6-v2`, `distilbert-base-uncased`.
- [ ] **[MUST]** Hardware decision, written in `docs/decisions/0002-compute.md`: CPU-only? A GPU? If CPU-only, Day 4 uses FLAN-T5 + LoRA (CPU-feasible); decoder SLMs only if a free GPU session (Colab/Kaggle) is available. Time one epoch of a tiny LSTM and one LoRA step to know your budget.
- [ ] **[MUST]** `common/modelkit/`: `assert_features_allowed(cols)`, `save_model_card(...)`, `load_split(df, split)`, `predictions_frame(...)`, `latency_probe(fn, n=200)`; with unit tests in `tests/ml/`.
- [ ] **[SHOULD]** Help Role 3: review `common/dataio/windows.py` interface; try it on the fixture.
- [ ] **[SHOULD]** Read about: class weights, calibration (reliability curve), SHAP, LSTM sequence windows, graph convolution (`H' = ReLU(Â H W)`), LoRA (low-rank adapters), VAE (ELBO = reconstruction + KL). You will have to explain each in plain words.

## 7. Day-by-day

### Day 1 - Stage 01 ML (`Stage01_ML/03_ml_engineer.py`)

Input: `Stage01_ML/data/prepared_dataset.csv[.gz]` (contract `telemetry`; columns `split` train/val/test).
Output: `predictions/ml_predictions.csv`, `predictions/feature_importance_matrix.csv`, `artifacts/*`, `artifacts/model_card.json`.

**Feature set [MUST]**
- Base features exactly as in `feature_policy.json` (categoricals one-hot or target-free ordinal; carrier/origin/destination/node/priority).
- `cal_hour`, `cal_dow`, `cal_month` from the timestamp.
- `eng_`: `eng_util_x_customs`, `eng_log_dwell`, `eng_throughput_per_util`, `eng_labor_x_equipment_gap`.
- `lag_/roll_` node context from **earlier** records only, e.g. `roll_node_dwell_mean_24h`, `roll_node_critical_rate_7d` using the train/val/test-appropriate past (for val/test rows the past may include train/val *features and labels that were already realised before that timestamp* - but never the row's own or later labels). Unit-test it.
- Imputation: train medians (numeric), "UNKNOWN" (categorical); keep `was_missing_*` flags as features if Role 3 provided them.

**Baselines [MUST]** (each writes rows to `ml_predictions.csv` with its own `model_name`):
1. `majority` - always ON_TIME.
2. `rule_dwell` - dwell thresholds chosen on train only (grid search two thresholds to maximise macro-F1).
3. `logreg` - multinomial logistic regression, standardised, `class_weight="balanced"`.

**Models [MUST]**
- `random_forest` - 300 trees, `class_weight="balanced_subsample"`, `min_samples_leaf=3`.
- `xgboost` - `objective="multi:softprob"`, `learning_rate=0.05`, `max_depth` 4-6, `subsample=0.8`, `colsample_bytree=0.8`, up to 800 trees with early stopping (50 rounds) on **val**, sample weights = inverse class frequency.
- Tuning: random search ≤ 25 trials, `TimeSeriesSplit(3)` **inside train**, score = macro-F1 + 0.5 × PR-AUC(CRITICAL). Record the search space and best params in the model card.
- **Calibration:** `CalibratedClassifierCV(method="isotonic", cv=TimeSeriesSplit(3))` applied to the chosen tree model (fit inside train); report calibrated and uncalibrated rows (`xgboost`, `xgboost_cal`).
- **Anomaly detectors (problem-statement mandatory):** `IsolationForest(n_estimators=300, contamination="auto")` fitted on **ON_TIME train rows only**; `OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")` fitted on a 10,000-row ON_TIME train sample. `anomaly_score` = min-max-scaled (train) negative decision function, written into `ml_predictions.csv` for every row of every model (same value per record).
- `risk_score = p_critical + 0.5 * p_at_risk` (clipped to 0-1).

**Interpretability [MUST]** `feature_importance_matrix.csv` (columns `feature, shap_mean_abs_critical, shap_mean_abs_at_risk, perm_importance_macro_f1, xgb_gain`) from SHAP (TreeExplainer, 2,000 sampled **val** rows - not test) and permutation importance on val. This is the problem statement's "Feature Importance Matrix".
**[SHOULD]** monotonic constraints on dwell/customs/weather in a second XGBoost (`xgboost_monotone`); LightGBM comparison; a regression head predicting `realized_delay_hours` with quantile loss (q10/q50/q90) saved as `delay_hours_model.joblib` - planners like hours more than classes (extra file `predictions/ml_delay_hours.csv`, no contract needed).
**[STRETCH]** conformal prediction sets for the class; SHAP interaction values for the top two features.

**Champion:** `model_card.json` lists `candidates` with val macro-F1 and val expected cost; `champion_proposal` = best val expected cost (use `common.evalkit.expected_cost` read-only). Role 4 writes the final `reports/eval_champion.json`; Role 3 serves that choice.

**Hand-off:** predictions for **val and test** of all models, validated; HANDOFF `03_model_engineer.md` READY; mention training time, library versions, any `--fast` mode.

### Day 2 - Stage 02 DL (`Stage02_DL/03_dl_engineer.py`)

Input: `hourly_node_telemetry.csv[.gz]` + `common.dataio.windows.build_windows` + `load_adjacency` (Role 3). Output: `predictions/dl_forecasts.csv[.gz]`, `artifacts/{lstm_forecaster.pt, graph_forecaster.pt, model_card.json}`.
Setup: input window 72 h, horizons (1, 6, 24, 72, 168) h, targets dwell and throughput, scaling with Role 3's train-only scaler, origins for test predictions every 6 h.

**Baselines [MUST]** `persistence` (last value), `seasonal_naive` (value 168 h earlier for each target hour), `xgb_lag` (XGBoost direct per horizon on lag features).
**LSTM forecaster [MUST]** one model shared by all nodes: node-id embedding (dim 8) concatenated to the 8-channel input; LSTM 2 layers, hidden 64, dropout 0.2; heads give 5 horizons × 2 targets × 3 quantiles (q10, q50, q90) with **pinball loss**; AdamW lr 1e-3, batch 64, ≤ 30 epochs, early stopping patience 5 on val; gradient clip 1.0. `y_pred` = q50, `y_lo` = q10, `y_hi` = q90.
**Graph forecaster [MUST]** (the problem statement's "Spatial Topology GNN"): per-node LSTM encoder → `H` (n_nodes × 64) → two graph layers `H' = ReLU(Â · H · W)` with `Â = load_adjacency(symmetric=True, self_loops=True, normalise=True)` implemented as a dense matrix multiplication in pure PyTorch (no extra library needed) → concatenate `[h_i, graph_i]` → same heads. 
**Ablations [MUST]** (all as separate `model_name`s so Role 4 can show them): `graph_none` (Â = identity), `graph_shuffled` (edges randomly permuted, seed 42), `graph_full`. If `graph_full` does not beat `graph_none`, say so - that is a finding, not a failure.
**[SHOULD]** `transformer_encoder` (2 layers, 4 heads, d=64) as a third architecture; a small hyperparameter study (hidden 32/64/128, window 48/72/168) logged to `artifacts/hparam_log.csv`; MC-dropout or conformal interval calibration so that the 80% interval covers ≈ 80% on val.
**[STRETCH]** 1D-CNN baseline; training-time augmentation from Role 3's helpers; per-node-type specialised heads.

Budget: whole script < 20 min on CPU in `--fast` mode (hidden 32, 10 epochs) and the saved models from the full run committed.
Hand-off: val+test forecasts for all models, validated; model cards (architecture, parameter count, training curves saved as `artifacts/training_curve.csv`), READY.
*Role 2 will run attribution on your saved model on the same day* - save a `load_model(path)` helper and the exact input pre-processing in `common/modelkit` or in the script (`if __name__` guarded) so they can import it.

### Day 3 - Stage 03 NLP (`Stage03_NLP/03_nlp_engineer.py`)

Input: `incident_messages.csv[.gz]`. Fit only on `split == train` (and `val` for tuning). The headline test set is the **HUMAN_GOLD** rows of `split == test`.
Output: `predictions/nlp_predictions.csv`, `predictions/nlp_entities_pred.jsonl`, `artifacts/*`, `model_card.json`.

**Urgency + risk-factor classifier [MUST]**
1. `tfidf_lr` baseline: `FeatureUnion` of word TF-IDF (1-2 grams, `min_df=2`, sublinear tf) and character TF-IDF (3-5 grams, `char_wb`) → `LogisticRegression(class_weight="balanced", max_iter=2000)` with `C` tuned on val. **Negation handled in training and inference by the same function** (mark tokens within 3 tokens after "no/not/without/never" with `NEG_`).
2. `minilm_lr`: sentence embeddings (`all-MiniLM-L6-v2`, normalised) → logistic regression. Cheap, usually much better on paraphrases.
3. Outputs: urgency probabilities (`p_routine/p_watch/p_urgent`), risk-factor label; write `y_true_*` from the labels, `pred_node_ids` from the entity linker.
**[SHOULD]** fine-tune `distilbert-base-uncased` (max_len 128, 3 epochs, lr 5e-5, batch 16, weight decay 0.01, early stop on val) as `distilbert`; temperature-scale its logits on val.

**Named-entity recognition [MUST]** entity types `PORT`, `CARRIER`, `DISRUPTION_TYPE`, `MITIGATION_ACTION` (problem statement). Models, each writing to `nlp_entities_pred.jsonl` under its own `model_name` field:
1. `gazetteer` baseline: spaCy `EntityRuler` built from `network_nodes.csv` aliases, the carrier list, and lexicons you build from `jargon_glossary.csv` + the training corpus (disruption words: "strike", "closure", "customs hold", "blank sailing", "omitted call", "congestion", "typhoon"...; mitigation words: "reroute", "divert", "expedite", "transship", "extend free time", "prioritise").
2. `spacy_ner`: fine-tune `en_core_web_sm` NER with the custom labels on train (weak + gold-train labels), with `EntityRuler` before NER as a hybrid. Dropout 0.2, 30 epochs, early stopping on val F1, batch compounding 4→32.
**Entity linking [MUST]** `artifacts/alias_resolver.py`+`alias_resolver.json`: normalise (casefold, strip punctuation), exact alias match against `network_nodes.csv` aliases, then fuzzy match (`rapidfuzz` `token_set_ratio ≥ 90`), then the ambiguity table from `jargon_glossary.csv` (`maps_to` with several nodes ⇒ return all, `ambiguous=True`); context rule examples: "LAX" next to "flight/airport/terminal 4" is **not** the port. Output `{span, label, node_ids, ambiguous, method}`. Unit tests with at least 25 cases in `tests/ml/test_alias_resolver.py`.
**[SHOULD]** `distilbert_ner` token classifier (seqeval-compatible BIO) as the third model; trade-lane classifier (origin>destination pair from linked nodes).
**[STRETCH]** active-learning loop: pick the 50 least-confident train messages for Role 4/2 to relabel.

### Day 4 - Stage 04 SLM (`Stage04_SLM/03_slm_engineer.py`)

Input: `slm_pairs.csv`. Output: `predictions/slm_generations.csv`, `artifacts/lora_adapter/` (adapter weights only), `artifacts/merged_model/` **only if < 50 MB** (otherwise ship the adapter + a `scripts`-free `load_model()` that merges at load), `model_card.json`.

**Three systems, same test pairs [MUST]**
1. `template` - deterministic: builds the two sentences from `source_facts_json` (worst node, counts, max dwell, main cause). The fair baseline.
2. `base_prompt` - `google/flan-t5-small` (or base) with a strict instruction prompt, no training.
3. `lora_ft` - LoRA fine-tune: `peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q","v"], task_type="SEQ_2_SEQ_LM")` on FLAN-T5-small (upgrade to base if the CPU budget allows); AdamW lr 5e-4 (T5 LoRA tolerates 3e-4 to 1e-3), batch 16, ≤ 5 epochs, max source length 384 / target 96, early stopping on val loss; **train split only**. Generation: `num_beams=2`, `do_sample=False`, `max_new_tokens=80`, `no_repeat_ngram_size=3`.
**Measurements written per generation [MUST]:** `latency_ms` (batch size 1, model loaded, 3 warm-up runs discarded, `time.perf_counter`, `torch.set_num_threads(4)`), `tokens_out`, `peak_ram_mb` (psutil resident-set max during generation), `device`. Role 4 re-measures independently; yours is the quick look.
**[SHOULD]** `lora_ft_int8`: PyTorch dynamic quantisation (`torch.quantization.quantize_dynamic` on `nn.Linear`) of the merged model; report quality and latency change.
**[STRETCH]** a decoder SLM (`Qwen/Qwen2.5-0.5B-Instruct` or `HuggingFaceTB/SmolLM2-360M-Instruct`) with LoRA (needs a GPU session; ship adapter only), exported for CPU if time permits. Only add it if the `[MUST]` list is complete.

Prompt/serialisation: use the format Role 2 recommends (`reports/eda_serialisation_decision.md`); do not invent your own on Day 4.
Hand-off: generations for **val and test** pairs for all systems, validated; `model_card.json` with training curve, adapter size, generation config; READY. If a GPU session was used, say so in the card.

### Day 5 - Stage 05 GenAI (`Stage05_GenAI/03_genai_engineer.py`)

Input: `Stage05_GenAI/data/baseline_profiles.json`, `Stage02_DL/data/hourly_node_telemetry.csv` (nominal train period), `Stage05_GenAI/scenarios/scenario_catalog.json` (Role 2). Output: `artifacts/{cvae.pt, model_card.json}`, `predictions/scenario_telemetry.csv[.gz]` (contract `scenario_telemetry`), `predictions/generator_comparison.csv`.

**Conditional VAE generator [MUST]**
- Data: nominal 168-hour windows per node (8 channels), from the **train** period only, scaled with the train scaler.
- Architecture: 1D-conv encoder → latent z (dim 32) → 1D-conv-transpose decoder; condition vector = [node-type one-hot, node embedding (8), scenario-type one-hot, severity (0-1), hour-of-week phase]; loss = reconstruction (Huber) + β·KL with β annealed 0→0.5 over the first epochs ("KL warm-up"); AdamW 1e-3, ≤ 60 epochs, early stop on val ELBO.
- **Constraint-projection layer (the USP) [MUST]** applied after decoding, written as a pure function `project_constraints(window, node_params)` with unit tests: clip every channel to its valid range; enforce `throughput ≤ capacity × (1 − labor_deficit) × equipment_availability`; recompute `inventory` from flows (`I_{t+1} = I_t + A_t − D_t`, never negative); recompute `avg_dwell_time_hours` ≈ `inventory / throughput` (Little's law) within the tolerance in the SIM_CARD; utilisation consistent with inventory.
- **Propagation [MUST]**: scenario shock hits its start nodes; downstream nodes receive the shock after `typical_transit_days × 24` h with strength `w_ij` (default 0.15 decaying along the path, or the lead-lag coefficients Role 2 estimates - prefer those if available).
- Generate all 20 catalogue scenarios + `WILDCARD` + a matching `NOMINAL` control run: 27 nodes, 14 days each, `severity` and `is_disrupted_hour` ground truth filled from the scenario definition.
**Comparator [MUST]** `simulator` generator: Role 3's fluid-queue simulator run with the scenario's event injected (`generator = SIMULATOR`) - the same scenarios, so Role 4 can compare CVAE vs simulator on realism and on downstream impact. Write `generator_comparison.csv` with generator × scenario × summary stats you computed (means, peaks), no judgement.
**[SHOULD]** `gan_baseline` (a small 1D-conv GAN, spectral norm on the discriminator) just to show mode-collapse risk in a table; scenario-conditioned interpolation (blend two scenarios).
**[STRETCH]** a tiny diffusion model on single-node windows; LLM-written scenario narratives that Role 3 turns into advisories (with Role 2's prompts).

**Hand-off:** `scenario_telemetry.csv` validated, `model_card.json` (latent dim, β, parameter count, reconstruction error on val, constraint-violation rate **before and after projection** - a number Role 4 will repeat), READY.

## 8. Hand-off checklist (every day)

- [ ] Predictions file(s) pass `validate_contract.py`; output pasted in HANDOFF
- [ ] Baselines present; model rows present; `model_version` set (`v1`, `v2`, ...)
- [ ] `model_card.json`: features, train window, seed, hyperparameters, training time, library versions, validation metrics (code-computed)
- [ ] `assert_features_allowed` ran; lag features unit-tested
- [ ] Artefact sizes < 50 MB; no checkpoints committed
- [ ] Script runs from the stage folder in one command; `--fast` documented
- [ ] Every logical step has a plain-English comment

## 9. Debate duties

Without notes you must explain:
- **Stage 1:** what XGBoost does (many small trees, each fixing the previous mistakes), why class weights, what calibration is and why probabilities must be calibrated before a cost-based threshold, how Isolation Forest isolates odd points, what One-Class SVM learns, how SHAP assigns credit.
- **Stage 2:** why a chronological split; what an LSTM remembers; what a graph layer does in one sentence (each node mixes in its neighbours' hidden state); why quantile loss gives intervals; what "skill vs persistence" means; what the ablation (no graph / shuffled graph / full graph) proved.
- **Stage 3:** TF-IDF vs embeddings; why the gold set exists; what NER is and how linking turns text spans into node IDs; the ambiguity example "LA/LB".
- **Stage 4:** what LoRA changes (small low-rank matrices added to attention weights, base frozen), why that makes CPU fine-tuning feasible, what beam search is, what quantisation trades.
- **Stage 5:** what a VAE learns (a smooth latent space), why KL warm-up, why a constraint layer is needed (a neural net can output physically impossible numbers), Little's law as the sanity check, why not a GAN first.
- The **one thing you would improve with more time** for each stage.

## 10. Cover-for-me checklist

Training commands live at the top of each `03_*` script; `model_card.json` records exact parameters; random seed is fixed. Another teammate can re-run any stage with `python Stage0N/03_*_engineer.py --fast` and compare to the card.

## 11. Common mistakes to avoid

- Reporting val metrics as test metrics (or the other way around).
- Fitting the scaler/imputer on all data.
- Rolling features that include the current row's label or later rows.
- Comparing a tuned model with an untuned baseline (tune baselines with the same budget).
- Saving only `y_pred` without probabilities (Role 4 needs them for calibration and thresholds).
- Committing a 200 MB checkpoint (hook will block; use adapters/quantisation).
- Changing the feature list between `v1` and `v2` without writing it in HANDOFF.
- Chasing a 0.99 score: if you reach it, assume leakage and tell Role 3 and Role 4 immediately.
