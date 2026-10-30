# CASCADE - AI-powered prediction of global supply-chain logistics bottlenecks

Team project for the Data Science & AI CAT Phase II (problem statement 07). A four-person team builds, over five days,
an early-warning and decision-support system that **detects** shipment delay risk (ML), **forecasts** node congestion
for 7 days across a trade network (DL), **reads** carrier and customs advisories (NLP), **briefs** operators offline
(small language model), and **stress-tests** itself against 20 synthetic global disruptions (GenAI).

> Decision support only. Nothing in this repository executes a reroute. All shipment data is calibrated simulation,
> labelled as such in every file (`data_origin`). See `docs/02_USP_CASCADE.md` for what we do and do not claim.

## What makes this project different (the USP in six lines)

1. **Source-anchored data**: calibrated on public real-world data (IMF PortWatch, NY Fed GSCPI, Open-Meteo), with provenance on every row.
2. **Honest labels**: the label is the *realised outcome*, never a formula of the inputs; chronological split with purge gap; leakage tests every day.
3. **Beat-the-baseline discipline**: every model is shown next to a simple baseline with confidence intervals.
4. **Cost-justified alarms**: the false-alarm threshold comes from a written cost matrix, not a guess.
5. **Cascade-aware**: a real graph model with ablation, text entities linked to network node IDs, a conservation-respecting scenario generator.
6. **Verified and defensible**: LoRA-tuned offline briefing model with measured latency and hallucination rate; a claims ledger where every number we say aloud has a file and a command behind it.

## Team and roles

**Live app (Streamlit):** https://aipoweredpredictionofglobalsupplychainlogisticsbottlenecks-fut.streamlit.app/

| # | Member | Role | Scripts (all five stages) | Commit type | Role file |
|---|---|---|---|---|---|
| 1 | [Santhosh K](https://github.com/SANTHOSH-K06) | ML Engineer (model role) | `03_*` | `ml` | [ROLE_1_ML_ENGINEER](docs/roles/ROLE_1_ML_ENGINEER.md) |
| 2 | [Isha Miruthika](https://github.com/ishamiruthika23) | EDA Engineer (also prompt/workflow) | `02_*` | `eda` | [ROLE_2_EDA_ENGINEER](docs/roles/ROLE_2_EDA_ENGINEER.md) |
| 3 | [Pravin Lenin Naidu](https://github.com/Pravin292) | Data & Integration Engineer | `01_*`, `05_*`, app, repo | `data`, `integ` | [ROLE_3_DATA_INTEGRATION_ENGINEER](docs/roles/ROLE_3_DATA_INTEGRATION_ENGINEER.md) |
| 4 | [Bennish Christopher](https://github.com/Beni-18) | Evaluation Engineer | `04_*`, claims ledger | `eval` | [ROLE_4_EVALUATION_ENGINEER](docs/roles/ROLE_4_EVALUATION_ENGINEER.md) |

## Five-day plan

| Day | Stage | Folder | Headline deliverables |
|---|---|---|---|
| 0 | Foundations | `data/`, `common/` | Real-data calibration, calibrated simulator, shared helpers, environments, CI |
| 1 | Machine Learning | `Stage01_ML/` | Bottleneck scoring API, feature importance matrix, ops field guide, cost-based alert threshold |
| 2 | Deep Learning | `Stage02_DL/` | 7-day forecaster with intervals, graph neural model + ablation, lead-time metric, visual benchmark |
| 3 | NLP | `Stage03_NLP/` | Urgency/risk classifier, NER linked to node IDs, terminology audit, search + tagging UI |
| 4 | Small Language Model | `Stage04_SLM/` | LoRA-tuned 2-sentence briefs, latency/memory/hallucination suite, offline hub demo |
| 5 | Generative AI | `Stage05_GenAI/` | Constraint-aware generator, 20-scenario stress test, wildcard live run, fused triage view |

Details: [docs/03_FIVE_DAY_PLAN.md](docs/03_FIVE_DAY_PLAN.md). Debate preparation: [docs/05_DEBATE_PLAYBOOK.md](docs/05_DEBATE_PLAYBOOK.md).

## Start here (every teammate, once)

```bash
git clone https://github.com/Beni-18/AI_Powered_Prediction_of_Global_Supply_Chain_Logistics_Bottlenecks.git
cd AI_Powered_Prediction_of_Global_Supply_Chain_Logistics_Bottlenecks
git config user.name  "Your Real Name"
git config user.email "you@example.com"
git config core.hooksPath .githooks             # safety hooks: strips AI attribution, blocks secrets, checks commit format
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests -q
python scripts/board.py
```

Then:
1. Put the reference project and the problem-statement PDF in `_reference/` (local only, see [_reference/README.md](_reference/README.md)).
2. Read your role file and [docs/04_GITHUB_WORKFLOW.md](docs/04_GITHUB_WORKFLOW.md).
3. Open your IDE assistant and paste your role's prompt from [docs/07_KICKOFF_PROMPTS.md](docs/07_KICKOFF_PROMPTS.md).

## Documentation map

| File | What it answers |
|---|---|
| [docs/00_PROJECT_BRIEF.md](docs/00_PROJECT_BRIEF.md) | What the problem statement and the CAT plan require; how we beat the reference project |
| [docs/01_ARCHITECTURE.md](docs/01_ARCHITECTURE.md) | System design, folder layout, ownership, contracts, shared code interfaces |
| [docs/02_USP_CASCADE.md](docs/02_USP_CASCADE.md) | Our unique selling point and the evidence for each claim |
| [docs/03_FIVE_DAY_PLAN.md](docs/03_FIVE_DAY_PLAN.md) | Day structure, gates, deliverable matrix, risks and fallbacks |
| [docs/04_GITHUB_WORKFLOW.md](docs/04_GITHUB_WORKFLOW.md) | Branches, commits, PRs, no-AI-attribution rule, recovery recipes |
| [docs/05_DEBATE_PLAYBOOK.md](docs/05_DEBATE_PLAYBOOK.md) | Debate format, per-day claims, question bank, rehearsal checklist |
| [docs/06_DATA_SOURCES.md](docs/06_DATA_SOURCES.md) | Real sources, licences, how the simulator stays honest |
| [docs/07_KICKOFF_PROMPTS.md](docs/07_KICKOFF_PROMPTS.md) | Copy-paste first message for each role's IDE assistant |
| [docs/08_COMMENTING_STANDARD.md](docs/08_COMMENTING_STANDARD.md) | Plain-English commenting rules with examples |
| [docs/09_HANDOFF_PROTOCOL.md](docs/09_HANDOFF_PROTOCOL.md) | How four devices stay in sync without conflicts |
| [AGENTS.md](AGENTS.md) | The rules every AI assistant must follow (also mirrored for Claude, Gemini, Copilot, Cursor) |

## Everyday commands

```bash
python scripts/board.py                                     # who has handed over what
python scripts/validate_contract.py --list                  # contract names
python scripts/validate_contract.py telemetry Stage01_ML/data/prepared_dataset.csv
python scripts/check_ownership.py --range origin/main..HEAD # are my commits in my lane?
python run_pipeline.py --stage 1                            # run scripts 01-04 of one stage
python -m pytest tests -q
streamlit run app.py                                        # control tower (built up daily by Role 3)
```

## Status

Live status per role and stage: `python scripts/board.py`.

| Stage | State |
|---|---|
| 1 ML | **Complete.** Data, EDA, models, evaluation, API and dashboard are handed over; 15 claims verified from a fresh clone (`claims/CLAIMS_LEDGER.md`) |
| 2 DL | Models handed over (Role 1); data, EDA, evaluation and integration not started |
| 3 NLP, 4 SLM, 5 GenAI | Not started: scripts are stubs that exit with code 3 ("not implemented") |

**Stage 1 result** (test period, `Stage01_ML/reports/eval_metrics.json`): logistic regression at an alert threshold of 0.39 catches 77.3% of critical delays with 40.1 false alarms per 100 alerts (soft budget 40), at a cost of 0.337 per shipment. The dashboard can switch between the five trained models.

**About the data:** the simulator is driven by real weather, port traffic, port size and supply-chain pressure data (`data/raw/sim/SIM_CARD.md`, decision 0010), but shipment outcomes are still simulated, so every row is labelled `SYNTHETIC_CALIBRATED`.

**Still open for Stage 1:** votes on decisions 0008 and 0009; Role 2's field-guide rewrite and `eda_error_slices.csv`; pinning library versions in `requirements.txt` (exact numbers depend on them; see `docs/KNOWN_LIMITATIONS.md`).
