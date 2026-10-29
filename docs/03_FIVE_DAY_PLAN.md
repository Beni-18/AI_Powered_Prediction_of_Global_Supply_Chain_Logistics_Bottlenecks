# 03 - Five-day plan (plus Day 0)

Sessions per day (from the CAT Phase II plan): **FN (forenoon)** = project implementation guide + build;
**AN (afternoon)** = **debate** + build. We do not know exact clock times, so milestones are relative.
Every role file has the day-by-day detail; this page is the shared timetable and the deliverable matrix.

## Day structure (same every day)

| Block | What happens | Gate at the end |
|---|---|---|
| **FN-1** Guide session | Faculty guide for the day's domain. Everyone listens for rubric hints; Role 2 notes terms for the glossary | - |
| **FN-2** Build, hour 1 | Role 3 publishes **data v1** to `main` (target: within 60 min of FN-2 start). Others start against fixtures | **G1 Data-v1 gate** |
| **FN-3** Build, hours 2-3 | Role 1 trains baselines + first real model; Role 2 delivers first findings; Role 4 produces first evaluation run on baseline predictions; Role 3 builds API/UI against stub | **G2 Debate-ready gate**: one end-to-end number per role, claims ledger has ≥ 5 verified rows, 15-minute rehearsal |
| **AN-1** Debate (Evaluation N) | Team vs team. Everyone speaks (rubric is per individual) | - |
| **AN-2** Build, second half | Fix what the debate exposed; finish `[MUST]` then `[SHOULD]`; hand-offs to READY; PRs merged | **G3 Day-complete gate**: five HANDOFF files READY (or honestly explained), `python run_pipeline.py --stage N` runs, tag `dayN-complete`, ledger updated |
| **Evening (15 min)** | Retro: what the opponent said, what we must fix tomorrow; update `docs/KNOWN_LIMITATIONS.md` | - |

If time is shorter than expected: **keep the gates, shrink the scope** (drop `[STRETCH]`, then `[SHOULD]`). Never skip the contract check.

## Day 0 - Foundations (before Day 1; ~3-4 hours each, in parallel, no debate)

| Role | Day 0 tasks (details in role file) |
|---|---|
| **1 ML** | Environment + all stage requirements installed; **pre-download** all models (FLAN-T5, MiniLM/DistilBERT, spaCy model); build `common/modelkit`; verify a tiny LSTM and LoRA run on this laptop (hardware decision: CPU-only or Colab/Kaggle GPU for Day 4-5) |
| **2 EDA** | Environment; build `common/edakit` (one plot style, `psi`, noise-floor helper); study the real-event timelines (Ever Given/Suez 2021, Panama drought 2023, Red Sea 2024, LA/LB backlog 2021) and write `docs/decisions/` note proposing scenario magnitudes; read the reference EDA scripts |
| **3 Data/Integ** | Repo admin tasks (collaborators, branch rule, CI on); **data foundation**: fetch real public data, fit calibration numbers, build the calibrated simulator, publish `data/raw/sim/shipment_telemetry_raw.csv` + hourly node data; implement `common/dataio` (`windows.py`, `graph.py`) and `common/integration` stubs |
| **4 Eval** | Environment; build `common/evalkit` (cost matrix, threshold chooser, ECE, bootstrap CI, skill score, lead time, grounding check) with unit tests; create `claims/CLAIMS_LEDGER.md` and `docs/KNOWN_LIMITATIONS.md`; draft the debate scorecard; hand-label practice |
| **All** | Clone, hooks on, test commit on a throwaway branch, run `pytest`, read all of `docs/` once, **dress rehearsal**: each role opens a PR with a one-line change and merges it |

## Deliverable matrix (who is Accountable; others Contribute)

A = accountable owner, C = contributes.

| Problem-statement deliverable | Day | Role 1 | Role 2 | Role 3 | Role 4 |
|---|---|---|---|---|---|
| Bottleneck scoring engine (3 classes) | 1 | **A** | C (features) | C (data) | C (metrics) |
| Feature importance matrix | 1 | **A** | C | | C |
| Logistics Ops Field Guide (1 page) | 1 | C | **A** (content) | C (layout/PDF) | C (verifies numbers) |
| Real-time prediction API | 1 | C | | **A** | C (latency test) |
| False-alarm threshold decision | 1 | C | | | **A** |
| 7-day delay/throughput forecaster | 2 | **A** | C | C | C |
| Spatial topology GNN | 2 | **A** | C (lead-lag evidence) | C (adjacency) | C (ablation) |
| Visual prediction benchmark | 2 | C | C | | **A** |
| Disruption severity classifier | 3 | **A** | C | C | C |
| Named-entity recognizer (4 types, node-linked) | 3 | **A** | C (guidelines) | C (gold set tooling) | C |
| Terminology misinterpretation audit | 3 | C | C | | **A** |
| Search + tagging UI | 3 | | | **A** | C |
| Fine-tuned local SLM | 4 | **A** | C (prompt/tokens) | C (pairs) | C |
| Latency + accuracy benchmark suite | 4 | C | | C | **A** |
| Offline hub demo | 4 | C | | **A** | C (offline test) |
| Synthetic logistics generator | 5 | **A** | C (boundaries) | C (baselines) | C |
| 20-scenario stress-test report | 5 | C | C | C | **A** |
| Wildcard scenario + live defense | 5 | C | **A** (design) | C (live UI) | C (metrics) |
| Unified repo + LLD docs | all | C | C | **A** | C |

## Dependency map across days (what carries forward)

```
Day 1 prepared telemetry ──► Day 2 hourly aggregation ──► Day 5 baseline profiles
Day 1 node risk ───────────┐
Day 2 forecasts ───────────┼──► Day 4 summary pairs (facts come from Stage 1-3 outputs) ──► Day 5 stress test (Stages 1-4 on synthetic data)
Day 3 advisory signals ────┘
Day 1-3 artefacts must stay loadable all week: do not rename, move or re-train without a v2 note.
```

## Risk register (and the pre-agreed fallback)

| Risk | Likelihood | Fallback (decided now, so nobody debates it at 2 pm) |
|---|---|---|
| Real data fetch fails / licence problem | Medium | Calibrate on published summary statistics typed into `data/calibration/manual_anchors.csv` with citations; state it on slides |
| GNN does not beat LSTM | Medium | Report it: "graph adds X% / nothing on this data; here is the ablation". An honest negative result still scores |
| No GPU for LoRA on larger model | Medium | FLAN-T5-small/base LoRA on CPU is the `[MUST]`; decoder SLM is `[STRETCH]` |
| Hand-labelling gold set slips | High | Minimum gold set = 100 messages (60 real + 40 synthetic), labelled by two people; evaluate on it only |
| Debate exposes a leak | Low-Medium | Because Role 4 runs leakage tests daily, we already know; we show the test and the fix |
| Merge/conflict chaos | Low | Ownership lanes + CI; a `Cross-role:` line for the rare exception |
| One teammate absent | Medium | Roles are documented to be picked up: each role file lists a "cover for me" checklist; the other three never stop |
| Time overrun | High | Gates are fixed, scope shrinks: `[STRETCH]` → `[SHOULD]` → never `[MUST]` |

## Tags and releases

`day0-foundations`, `day1-complete` ... `day5-complete` are tags on `main` made by Role 3 at G3 (`git tag day1-complete && git push origin day1-complete`).
Tags let us roll back to a known-good state before a debate.
