# ROLE 4 - Evaluation Engineer (the team's honesty engine)

> **For the AI assistant reading this file:** you are working for the teammate who holds Role 4. This file is your
> job description. Rules in `AGENTS.md` come first. Priority tags: **[MUST]** deliver or the day fails,
> **[SHOULD]** if MUSTs are done, **[STRETCH]** only with spare time.

## 0. At a glance

| | |
|---|---|
| **Programme role** | Evaluation Engineer: Day 1 metrics and errors, Day 2 accuracy and loss, Day 3 precision/recall/F1, Day 4 quality/latency/perplexity, Day 5 output quality of generated data |
| **Commit type** | `eval` (plus `docs` for `docs/KNOWN_LIMITATIONS.md` and `docs/debate_log.md`) |
| **Your scripts** | `StageNN/04_evaluation_engineer.py` (all five stages) |
| **Your folders** | `StageNN/reports/eval_*` (prefix `eval_` is your marker), `common/evalkit/`, `claims/`, `tests/eval/`, `notebooks/eval/` |
| **What you hand over** | `eval_metrics.json` + report + charts per stage, `eval_champion.json` (Stage 1-3: which model and which threshold the API must use), and updated `claims/CLAIMS_LEDGER.md` |
| **Your golden habit** | Report the baseline next to every model, put a confidence interval on every number, and publish what is wrong with our own work **before** the other team does |
| **Why you win debates** | You own the USP: leak-free evaluation, cost-based thresholds, honest limitations, a ledger of reproducible claims |

## 1. Mission

Turn "it seems to work" into **evidence**: independent, leak-free, uncertainty-aware, cost-aware measurement of every
model and of the data generator. You are also the team's quality gate (contracts, CI, clean-clone test) and its debate
timekeeper and fact-checker.

## 2. Reading order (Day 0)

`AGENTS.md` → `docs/00_PROJECT_BRIEF.md` → `docs/01_ARCHITECTURE.md` → `docs/02_USP_CASCADE.md` → `docs/05_DEBATE_PLAYBOOK.md` →
`docs/09_HANDOFF_PROTOCOL.md` → `docs/04_GITHUB_WORKFLOW.md` → `docs/08_COMMENTING_STANDARD.md` →
`contracts/` especially `label_policy.json`, `cost_policy.json`, `feature_policy.json` and the prediction schemas (`ml_predictions`, `dl_forecasts`, `nlp_predictions`, `slm_generations`, `stress_results`).

## 3. Your lane

| You OWN | You READ | You must NEVER touch |
|---|---|---|
| `StageNN/04_evaluation_engineer.py` | `StageNN/predictions/*`, `StageNN/artifacts/*` (read-only, e.g. to load a model for perplexity) | `StageNN/03_*`, `artifacts/`, `predictions/` (Role 1) - even to "fix" a bug; report it |
| `StageNN/reports/eval_*` | `StageNN/data/*`, `data/raw/sim/*`, `contracts/**` | `StageNN/data/**`, `contracts/**` (Role 3) |
| `StageNN/HANDOFF/04_evaluation_engineer.md` | `StageNN/reports/eda_*` | `StageNN/02_*`, `reports/eda_*` (Role 2) |
| `common/evalkit/**`, `claims/**`, `tests/eval/**`, `notebooks/eval/**` | `app.py`, `pages/**` (to verify what is shown) | `StageNN/01_*`, `05_*`, `app.py`, `pages/**` (Role 3) |
| `docs/KNOWN_LIMITATIONS.md`, `docs/debate_log.md` (type `docs`) | | |

## 4. Restrictions (hard)

1. **Test split is opened once per model version**, after the model and the threshold are frozen. The threshold is chosen on **val** only. Re-evaluating the test split after changing the model is a new version (`v2`) and must be recorded; never silently iterate against the test numbers.
2. **Baseline in every table; confidence interval on every headline number** (block bootstrap by day, 1,000 resamples; seed 42).
3. **Primary metric is never plain accuracy** on imbalanced classes. Use macro-F1, PR-AUC for the critical class, expected cost.
4. **Every number you report comes from code and is written to a JSON/CSV first.** Reports (`.md`) are rendered from those files. No manual typing of metrics.
5. **Every claim in the ledger has a reproduce-command** that regenerates it from a clean checkout.
6. **You do not edit others' files.** Problems found go to HANDOFF "Open questions" under the owner's role name and to the chat with evidence.
7. **Never soften a negative result.** If the graph model does not beat the baseline, the report says so in the first sentence.
8. **Keep human evaluation honest**: raters blind to which system wrote a summary; record who rated; small samples are labelled small.
9. **No fabricated comparisons.** If a cloud LLM was not run, the table says "not measured", not a number from memory.
10. **Plain-English comments on every step** (`docs/08_COMMENTING_STANDARD.md`).

## 5. How you work in parallel

| You wait for | Meanwhile |
|---|---|
| Role 1's predictions (≈ 45 min after data v1 for baselines) | Build evaluation functions and the report template on the **fixture** prediction files (`contracts/fixtures/*.csv`). When baseline predictions appear, run the pipeline immediately: this is the dress rehearsal |
| Role 3's data v1 | Prepare leakage tests (they need data only); write the simulator acceptance tests on Day 0 |
| Role 2's recommendations | Not needed; you consume their boundary values/guidelines on Day 3 and 5 |

Who waits on you: Role 3 (`eval_champion.json` and threshold for the API), the whole team (ledger numbers before each debate), Role 1 (feedback on calibration/errors), Role 2 (error slices).

## 6. Day 0 - Foundations (3-4 hours)

- [ ] **[MUST]** Environment as in `docs/04_GITHUB_WORKFLOW.md` §2 plus `scipy`, `scikit-learn`, `seqeval`, `rouge-score`, `psutil`.
- [ ] **[MUST]** `common/evalkit/` with unit tests in `tests/eval/`:
  - `expected_cost(y_true, y_pred, cost_matrix)`; `bayes_decision(proba, cost_matrix)` (argmin of expected cost)
  - `choose_threshold(y_val, p_val, cost_matrix, positive="CRITICAL_DELAY")` → threshold + the whole cost-vs-threshold table
  - `ece(y, p, n_bins=10)`, `reliability_table(...)`, `brier(...)`
  - `block_bootstrap_ci(metric_fn, *arrays, blocks, n=1000, seed=42)`; `paired_bootstrap_diff(...)`
  - `skill_score(err_model, err_baseline)`; `coverage_and_width(y, lo, hi)`; `pinball_loss(...)`
  - `lead_time(alert_times, onset_times, max_lookback_h=168)` and `episodes_from_series(...)` per the definition in Day 2
  - `grounding_check(summary, facts)` returning numeric_precision, entity_precision, unsupported list, critical_node_mentioned, sentence_count
  - `ledger.upsert(claim_id, ...)` which rewrites one row of `claims/CLAIMS_LEDGER.md` between marker lines (so no hand-editing, no merge noise)
- [ ] **[MUST]** Create `claims/CLAIMS_LEDGER.md` (table with markers: `ID | Day | Claim wording | Value [95% CI] | Source file | Reproduce command | Verified on | Status`), `docs/KNOWN_LIMITATIONS.md` (start with the five limitations in `docs/02_USP_CASCADE.md`), `docs/debate_log.md`.
- [ ] **[MUST]** **Simulator acceptance tests** `tests/eval/test_sim_acceptance.py` for Role 3 (non-negative inventory; Little's-law relative error < 10% on average; class mix within 3 points of 65/22/13; label not computable from snapshot columns - a depth-limited tree on allowed features stays well below perfect; no future timestamps in lag features; hourly grid has no missing hours after cleaning).
- [ ] **[SHOULD]** Draft the debate scorecard (`docs/debate_log.md` template) and practise 30-second answers to the question bank in `docs/05_DEBATE_PLAYBOOK.md`.
- [ ] **[SHOULD]** Read about: macro-F1, PR-AUC, calibration/ECE, block bootstrap, Brier score, MASE, pinball loss, seqeval strict vs partial matching, TSTR, discriminator two-sample test.

## 7. Day-by-day

Rhythm each day: **G2 (debate-ready)** needs your first numbers + 5 verified ledger rows; **G3** needs the full report, the champion file, limitations updated, a clean-clone test.

### Day 1 - Stage 01 ML (`Stage01_ML/04_evaluation_engineer.py`)

Input: `predictions/ml_predictions.csv`, `data/prepared_dataset.csv[.gz]` (for slices), `contracts/cost_policy.json`. Outputs in `reports/` prefixed `eval_`.

**[MUST]**
1. **Headline table** (test split, all models incl. baselines): macro-F1, per-class precision/recall/F1, balanced accuracy, PR-AUC(CRITICAL), log-loss, Brier, expected cost per shipment, each with 95% block-bootstrap CI → `eval_metrics.json`, `eval_model_comparison.csv`, `eval_model_comparison.png`.
2. **Confusion matrices** (counts + row-normalised) for the champion and the best baseline → `eval_confusion.png`.
3. **Cost-based decision (the Team Huddle answer):** build the cost-vs-threshold curve on **val**; pick the threshold `t*` that minimises expected cost for alert = `p_critical ≥ t`; also report the Bayes-optimal 3-class decision. Freeze `t*`; report on **test**: precision, recall, false alarms per 100 alerts, expected cost, vs the soft budget in `cost_policy.json`. Sensitivity grid over `missed_critical_cost ∈ {3,5,10,20,40}` → table of `t*` per ratio → `eval_threshold_decision.md`, `eval_cost_curve.png`, `eval_threshold_sensitivity.csv`.
4. **Champion selection:** best val expected cost among models that beat the best baseline (paired bootstrap on test-free val difference; if none beats baselines, say so). Write `reports/eval_champion.json`: `{stage, model_name, model_version, threshold, selected_on: "val", cost_policy_version, date}`. Role 3's API reads it.
5. **Calibration:** reliability diagram + ECE for the champion calibrated vs uncalibrated (`xgboost` vs `xgboost_cal`) → `eval_calibration.png`.
6. **Leakage tests** → `eval_leakage_tests.json` (these are the heart of the USP):
   a. *Feature-policy check*: every feature in `model_card.json` is allowed.
   b. *Shuffle-label test*: re-train a fixed, small gradient-boosting probe on train with labels shuffled; test macro-F1 must be near chance (≈ 0.33 ± CI).
   c. *Single-feature ceiling*: max univariate AUC of any allowed feature vs critical (should be well below 0.95).
   d. *Time-order check*: all train timestamps < val < test (minus purge) and no `shipment_id` in two splits.
   e. *Too-good-to-be-true alarm*: if champion macro-F1 > 0.95, mark the stage "SUSPECT" in the report.
7. **Slice analysis:** metrics by node, carrier, priority tier, month, overload flag; worst-5 slices → `eval_slices.csv`, `eval_slices.png`. Compare with Role 2's `eda_error_slices.csv`.
8. **Anomaly detectors:** ROC-AUC, PR-AUC and precision@k (k = number of CRITICAL in test) of `anomaly_score` for CRITICAL vs rest; how many CRITICAL shipments the classifier missed but the detector flagged → `eval_anomaly.json`.
9. **API check:** independent load test of the running API (Role 3) - 1,000 calls, p50/p95/p99 → `eval_api_latency.json`; verify every response has `requires_operator_review: true` and `model_version`.
10. **Ledger:** upsert ≥ 8 rows (headline score vs baseline, threshold, expected cost, false alarms/100, leakage test results, latency); update `docs/KNOWN_LIMITATIONS.md`.

**[SHOULD]** Learning-curve (train size vs val F1); stability across 3 sub-periods of the test split; McNemar test champion vs best baseline.
**[STRETCH]** Decision-curve analysis (net benefit); fairness-style disparity across carriers.

### Day 2 - Stage 02 DL (`Stage02_DL/04_evaluation_engineer.py`)

Input: `predictions/dl_forecasts.csv[.gz]` (all models incl. persistence, seasonal_naive, xgb_lag, LSTM, graph variants).

**[MUST]**
1. **Forecast accuracy** per model × horizon × target: MAE, RMSE, sMAPE; **skill score vs persistence** `1 − MAE_model/MAE_persistence`, with block-bootstrap CI → `eval_forecast_metrics.csv`, `eval_skill_by_horizon.png`. Headline: at how many of the 5 horizons does each model have positive skill?
2. **Interval quality:** empirical coverage of the 80% interval (target ≈ 0.80), mean width, pinball loss → `eval_interval_quality.csv`.
3. **Lead-time metric** (problem statement: "measure how far in advance the model predicts bottlenecks"):
   - *Episode definition* (agreed with Role 2): a node-episode **onset** = first hour where dwell exceeds the node's alert threshold (95th percentile of the node's nominal **train** dwell) for ≥ 6 consecutive hours, separated from the previous episode by ≥ 24 h.
   - *Forecast alert at origin t*: any horizon's q50 ≥ threshold (also report a probability version using the interval).
   - *Lead time* = onset − first alert within the 168 h before onset (0 if the alert first fires at onset, "missed" if none). *False alarm* = an alert not followed by an onset within 168 h. Report per model: episode recall, median lead time [IQR] in hours, false alarms per node-week.
   → `eval_lead_time.csv`, `eval_lead_time.png`. This is the Stage 2 headline.
4. **Ablation table:** `graph_none` vs `graph_shuffled` vs `graph_full` (skill and lead time) - does the real graph help, and is it the *real* topology rather than any extra parameters? → `eval_graph_ablation.csv`.
5. **Visual Prediction Benchmark** (problem statement deliverable): side-by-side figure for 3 selected episodes (truth vs persistence vs LSTM vs graph model, with interval) and a table comparing the **baseline ML feature detection** (`xgb_lag` and, if available, Stage 1's aggregated shipment `risk_score` per node-hour) with the DL forecaster on recall / lead time / false alarms → `eval_visual_benchmark.png`, `eval_visual_benchmark.csv`, `eval_visual_benchmark.md`.
6. **Per-node and per-type slices** (ports vs chokepoints vs rail/DC); worst nodes.
7. Leakage/time checks: forecast origins in test only; no target hours inside the window; persistence baseline sanity (skill of persistence vs itself = 0).
8. **Ledger** ≥ 6 rows (skill at 24 h and 168 h, lead time median, coverage, ablation outcome, parameter counts); limitations updated; `eval_champion.json` for Stage 2 (best model by skill at 24 h and lead time).

**[SHOULD]** Rolling-origin backtest: ask Role 1 for `--fold k` retrains (3 expanding folds) and report mean ± spread of skill; Diebold-Mariano test LSTM vs persistence.
**[STRETCH]** Cost-weighted lead time (earlier warning worth more, false alarms cost 1).

### Day 3 - Stage 03 NLP (`Stage03_NLP/04_evaluation_engineer.py`)

Input: `predictions/nlp_predictions.csv`, `predictions/nlp_entities_pred.jsonl`, `data/incident_messages.csv[.gz]`. **Headline test set = `label_origin == HUMAN_GOLD` rows of `split == test`.** Everything else is secondary and labelled so.

**[MUST]**
1. **Urgency:** macro-F1, per-class P/R/F1, confusion matrix, ECE, expected cost with the NLP cost matrix `{missed URGENT: 10, URGENT over-called: 1, WATCH↔ROUTINE: 1}` (written into `reports/eval_nlp_cost_matrix.json`, ratified by the team) - for each model vs `tfidf_lr` baseline, on gold and (separately) on synthetic test → the **domain gap** = synthetic − gold macro-F1. → `eval_urgency_metrics.json/csv`, `eval_urgency_confusion.png`.
2. **Risk factor** macro-F1 and confusion (which causes are confused).
3. **Confidence gate:** coverage–accuracy curve; at confidence < 0.70 how many messages go to review and what share of errors that catches → `eval_review_gate.csv/png`.
4. **NER** (`seqeval`/own matcher): entity-level **strict** and **partial-overlap** precision/recall/F1 **per type** (PORT, CARRIER, DISRUPTION_TYPE, MITIGATION_ACTION) for `gazetteer`, `spacy_ner` (and `distilbert_ner`) vs gold; error types (boundary, type confusion, missed, spurious) → `eval_ner_metrics.csv`, `eval_ner_errors.csv`.
5. **Node-linking accuracy** (problem statement: map jargon to node IDs): exact node-set match, plus recall of ambiguous cases (must return all candidate nodes) → `eval_linking.json`.
6. **Terminology Misinterpretation Audit** (deliverable): for each of Role 2's ≥ 25 ambiguous terms write 2-4 test sentences with expected resolution (`tests/eval/terminology_cases.csv`), run the pipeline, and produce a table: term | context sentence | expected | produced | correct? | root cause | proposed fix → `eval_terminology_audit.csv` and `eval_terminology_audit.md` (summary: x of N resolved correctly; the unresolved ones listed in "known limitations").
7. **Behavioural tests (CheckList style)** in `tests/eval/behaviour_cases_nlp.csv` (≥ 60 cases): *minimum-functionality* (clear URGENT/ROUTINE), *invariance* (typos, case change, swapping one port alias for another must not change urgency), *directional* (inserting "closed"/"strike" must not lower urgency; inserting negation "no delay expected" must not stay URGENT) → pass rate per test type → `eval_behaviour_tests.csv`.
8. **Annotation quality:** Cohen's kappa for urgency and entity-level agreement between the two annotators (numbers come from Role 3's merge file) → ledger.
9. Leakage check: template overlap between train and test = 0; near-duplicate similarity between test and train below 0.92; gold set not used for training.
10. **Ledger** ≥ 8 rows; limitations updated (e.g. weak spots in negation, domain gap).

**[SHOULD]** Advisory-to-telemetry association: do URGENT advisory counts precede dwell increases at the linked node? (lagged correlation with CI) → `eval_advisory_lead.csv`.
**[STRETCH]** Calibration of NER confidence; active-learning gain estimate.

### Day 4 - Stage 04 SLM (`Stage04_SLM/04_evaluation_engineer.py`)

Input: `predictions/slm_generations.csv`, `data/slm_pairs.csv[.gz]` (with `source_facts_json`), model artefacts (read-only, for perplexity and independent latency).

**[MUST]**
1. **Faithfulness / hallucination** with `grounding_check` for systems `template`, `base_prompt`, `lora_ft` (and `lora_ft_int8` if present) on the **test** pairs, split into `HUMAN` and non-human reference origins: numeric precision (every number in the brief appears in the facts), entity precision (every node/port named exists in the facts), **unsupported-claim rate** = share of briefs with ≥ 1 unsupported number/node, **critical-context recall** = worst/critical node mentioned (problem statement: "zero loss of critical context"), count accuracy → `eval_faithfulness.csv`, `eval_faithfulness.png`.
2. **Format compliance:** exactly two sentences, ≤ 60 words, no repeated n-grams, no empty output → `eval_format.csv`.
3. **Reference-based quality (secondary):** ROUGE-L and BERTScore (if installed) against references; **perplexity** of the reference summaries under the fine-tuned model on test → `eval_text_quality.csv`.
4. **Latency and memory suite** (problem statement: "Latency Benchmark Suite"): independent script run on **your own laptop CPU**, recording CPU model, RAM, threads; for each system: cold start (process start → first brief), then 100 warm briefs, batch size 1 → p50/p95/p99 ms, tokens/s, peak resident memory MB, model size on disk MB → `eval_latency.csv`, `eval_latency.png`, `eval_hardware.json`. **Target p95 < 5000 ms.** If a hosted LLM is available through Role 3's `.env`, add `cloud_ref` rows; otherwise leave them out and write "cloud comparison not run" in the report.
5. **Operator reading-time study** (problem statement team huddle): 6-10 trials with teammates: read a raw 100-alert table and answer "which hub is worst and why?" vs read the 2-sentence brief; record time and correctness; report mean ± range, state n → `eval_reading_study.csv`, `.md`.
6. **Human rating (small)**: 30 test briefs × 3 raters, blind, 1-5 for faithful / useful / readable → `eval_human_ratings.csv`; inter-rater spread.
7. **Offline proof:** run the offline test from Role 3 (`tests/integ/test_offline_demo.py`) and log pass/fail; repeat once manually with Wi-Fi off → `eval_offline_proof.md`.
8. Error taxonomy of the 20 worst fine-tuned briefs (omission / wrong number / invented cause / bad format) → `eval_error_taxonomy.csv`.
9. **Ledger** ≥ 8 rows; limitations updated; `eval_champion.json` (which system ships in the offline app).

**[SHOULD]** Significance (paired bootstrap) of `lora_ft` vs `base_prompt` on unsupported-claim rate; effect of trace length on hallucination; int8 quality loss.
**[STRETCH]** LLM-as-judge cross-check on 30 briefs (report agreement with humans, never as the primary metric).

### Day 5 - Stage 05 GenAI and the whole system (`Stage05_GenAI/04_evaluation_engineer.py`)

Input: `predictions/scenario_telemetry.csv[.gz]` (generator `CVAE`, `SIMULATOR`, optional `GAN`), `data/baseline_profiles.json`, all earlier artefacts (read-only).

**[MUST] A. Realism audit of the synthetic streams** (problem statement: "verify capacity-conservation constraints")
1. **Validity:** range violations (%), capacity violations (`throughput > capacity × (1 − labor_deficit) × equipment`), negative inventory, **Little's-law error** (median |dwell − inventory/throughput| / dwell), each **before and after** the projection layer (Role 1 supplies the "before" run) → `eval_realism_constraints.csv`.
2. **Fidelity vs held-out real-nominal windows (simulation year):** per-channel Kolmogorov-Smirnov statistic and normalised Wasserstein distance, correlation-matrix distance (Frobenius), ACF distance → `eval_realism_fidelity.csv`.
3. **Discriminative score:** train a classifier to separate real vs synthetic nominal windows (5-fold CV) - AUC closer to 0.5 is better → report honestly (high AUC means detectable synthetic).
4. **TSTR:** train a small forecaster on synthetic nominal windows, test on real; compare with train-on-real (MAE ratio).
5. **Coverage/diversity and memorisation:** nearest-neighbour distance of generated windows to training windows (must not be ≈ 0) and diversity (mean pairwise distance vs real).
→ `eval_realism_report.md` with a one-line verdict per generator (CVAE vs SIMULATOR vs GAN).

**[MUST] B. The 20-scenario stress test** (deliverable "Stress-Test Evaluation Report")
For each of S01-S20, `WILDCARD`, and the `NOMINAL` control:
- **Stage 1:** sample shipment snapshots from the scenario node states with Role 3's `common/dataio/shipment_sampler.py`, run the champion at its frozen threshold → detection rate (recall of CRITICAL), false alarms per 100 alerts.
- **Stage 2:** run the forecaster over the scenario stream → episode recall vs `is_disrupted_hour` onsets, median lead time, false alarms per node-week.
- **[SHOULD] Stage 3:** classify the advisories generated for the scenario (Role 2's prompts, Role 3 tool) → urgency macro-F1 and node-link accuracy.
- **[SHOULD] Stage 4:** briefs from the scenario alert traces → unsupported-claim rate.
- Degradation = control − scenario for every metric; **first stage to degrade** per scenario; bootstrap CIs.
→ `eval_stress_results.csv` (contract `stress_results`, validate it), `eval_robustness_heatmap.png`, `eval_stress_report.md` (scenario table, top-3 failure modes, which scenarios beat us, recommendations, limitations).
**[MUST] C. Wildcard** (canal blockage + labour strike + demand spike): one-page `eval_wildcard.md` + 2 figures: which stage degrades first, by how much, what the operator would have seen, recommended mitigation. Numbers must be reproducible live in Role 3's Scenario Lab; rehearse with Role 3.
**[MUST] D. System-level wrap-up**
- Final ledger review: every claim re-verified from a **clean clone** (`git clone` to a temp folder → install → `python run_pipeline.py` → the reproduce-commands). Mark `Verified on <date>` or fix.
- Final `docs/KNOWN_LIMITATIONS.md`, final cross-stage summary table `claims/SUMMARY.md` (one row per stage: headline, baseline, CI, caveat).
- Run the whole test suite and the CI; confirm no AI attribution in `git log` (`git log --format=%B | grep -iE "co-authored|generated with"` returns nothing).
**[SHOULD]** Sensitivity of the wildcard result to two parameters (grid from Role 2); fused triage score vs best single input on the test window (does fusion help? report either way).
**[STRETCH]** Compare synthetic-augmented training for Stage 1 on the Critical class (does adding CVAE scenarios improve recall on the real test split?).

## 8. Hand-off checklist (every day)

- [ ] `eval_metrics.json` + report + charts regenerate with one command from the stage folder
- [ ] Each headline number has baseline, CI and sample size
- [ ] Test split used once for the frozen model version; threshold from val
- [ ] `eval_champion.json` written (Stages 1-4) and Role 3 informed
- [ ] Ledger rows upserted and `Verified on` set; `docs/KNOWN_LIMITATIONS.md` updated
- [ ] Clean-clone test run at end of day; CI green
- [ ] Debate notes appended to `docs/debate_log.md`

## 9. Debate duties (you give the evidence and keep the clock)

Without notes you must explain:
- Why macro-F1/PR-AUC, not accuracy; what ECE and calibration are; what a confidence interval means for our numbers.
- The cost matrix, how the threshold was chosen on val, what changes when the miss cost changes (sensitivity grid), and what a false alarm costs in the field.
- Each leakage test and its result; why a chronological split with a purge gap.
- Skill score vs persistence, the lead-time definition and the headline value.
- Strict vs partial NER match; what the terminology audit found (including failures).
- How hallucination is measured (grounding check) and the measured rates for template / base / fine-tuned.
- How latency was measured (hardware, warm/cold, batch size, percentile) and the result.
- What we do **not** claim - read `docs/KNOWN_LIMITATIONS.md` aloud if needed.
- Timekeeping: you signal the speaker 30 seconds before their slot ends.

## 10. Cover-for-me checklist

All `04_*` scripts regenerate their outputs with one command and call `ledger.upsert`; `common/evalkit` has tests. Another teammate can re-run, read `eval_metrics.json`, and quote only ledger rows with status VERIFIED.

## 11. Common mistakes to avoid

- Reporting a test number that was seen before choosing the model (that is leakage by you).
- Averaging over a time-correlated sample and calling the CI tight (use block bootstrap).
- Using accuracy on an imbalanced target.
- Comparing the model with a baseline that was never tuned or never run on the same rows.
- Calling a latency number "CPU" without saying which CPU, how many threads, warm or cold.
- Hand-typing a metric into a slide.
- Hiding a bad slice: report the worst five slices every time.
- Treating synthetic-test performance as real-world performance (always give the gold/real subset number first).
