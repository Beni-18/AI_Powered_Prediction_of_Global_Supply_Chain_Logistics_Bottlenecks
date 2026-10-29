# 00 - Project brief (what we are building and how we are judged)

## 1. The problem (from `03 - AI_Powered_Prediction_of_Global_Supply_Chain_Logistics_Bottlenecks.pdf`)

> It is 3 AM at a major transpacific port. A customs backlog collides with a labour strike at an inland rail
> hub while a typhoon closes two alternate lanes. Dwell times spike. Without immediate rerouting, cascading
> delays reach 200 downstream warehouses within 48 hours. Build a multi-agent logistics intelligence engine that
> turns shipment telemetry, carrier logs and customs data into automated bottleneck triage.

Our system is **decision support**: it detects, forecasts, reads, briefs and stress-tests. A human approves any
action. (Stage 06 "Agentic AI" and automated rerouting are outside our five-day scope.)

## 2. The five stages we build (the problem statement's 12-day schedule is replaced by 5 days)

| Day | Stage | Mission | Problem-statement deliverables we must hit |
|---|---|---|---|
| 1 | **ML** | Spot the pattern: On-Time / At-Risk / Critical Delay | Real-time bottleneck prediction API; feature importance matrix; 1-page Logistics Ops Field Guide |
| 2 | **DL** | Develop instincts: temporal + network dynamics | Sequence forecaster (LSTM/Transformer, **7-day** horizon dwell & throughput); spatial topology **GNN**; visual prediction benchmark vs baseline ML |
| 3 | **NLP** | Decode the delay: advisories, customs notices, tickets | Urgency / lane / risk-factor classifier; **NER** (Port, Carrier, Disruption Type, Mitigation Action) mapped to **network node IDs**; terminology misinterpretation audit; search + tagging UI |
| 4 | **SLM** | Brief the ops desk offline in 5 seconds | **Fine-tuned** (PEFT/LoRA) local SLM giving 2-sentence briefs; latency + memory benchmark vs cloud LLMs; fully offline demo app |
| 5 | **GenAI** | Stress-test reality | Network-informed synthetic generator (GAN/VAE/diffusion); **20-scenario** stress-test report; live "Wildcard" scenario (canal blockage + labour strike + demand spike) |

Final expectation from the problem statement: unified code repository + low-level design documentation.
We keep `docs/` current every day so that LLD is a by-product, not a last-night scramble.

## 3. How we are judged (CAT Phase II session plan)

- **Team-based, role-driven application development**; evaluation = functionality, collaboration, contribution, demo.
- **Daily Team Battle of Debate** (Evaluation 1-5, one per day, after the forenoon build): team vs team on the same project topic; faculty may question anyone.
- **Score per individual out of 25**: Content Knowledge 5 + Communication 5 + Faculty Q&A 5 + Team Coordination 5 + Time Management 5. **Minimum 20/25 per individual** for "Won"; team winner = highest cumulative score.
- **Fixed roles across all five AI sections, no rotation.** The programme defines five roles (Data, EDA/Prompt/Workflow, Model/AI, Evaluation, Integration); with four people we merge **Data + Integration** into one person.

| Programme role | Our Role # | Person is responsible for |
|---|---|---|
| Model / AI Engineer | **1 - ML Engineer** | scripts `03_*` in all five stages (ML, DL, NLP, SLM, GenAI models) |
| EDA / Prompt / Workflow Engineer | **2 - EDA Engineer** | scripts `02_eda_engineer.py` in all five stages |
| Data Engineer + Integration Engineer | **3 - Data & Integration Engineer** | scripts `01_*` and `05_*`, shared data foundation, repo, app, API |
| Evaluation Engineer | **4 - Evaluation Engineer** | scripts `04_*` in all five stages, claims ledger, debate evidence |

Because every individual is scored, **every role file contains a "Debate duties" section**: what you must be able to
explain and defend alone, on the spot.

## 4. What the reference project teaches us - and where we beat it

The folder `Supply_Chain_Bottleneck_Industry_Oriented_Project` (on every device in `_reference/`) is a finished
prototype. We copy its **structure** (five stages, five role scripts per stage, `data/`, `run_pipeline.py`,
`app.py`, `tests/`) and nothing else. Its own slide decks list its honest gaps; each gap is one of our advantages.

| Reference weakness (documented in its own README / decks) | What rivals will therefore also show | Our answer |
|---|---|---|
| Labels are a rule of the features (`delay_status` follows dwell/utilisation), so ML scores are "near perfect" | 99% accuracy that means nothing | **Outcome-based labels** (`realized_delay_hours`), forbidden-feature policy, time-ordered split, leakage tests, baselines to beat |
| NLP urgency labels are copied from `delay_status` (`Routine/Urgent/Critical`), messages are templates of the numbers | Circular NLP score | Text generated from **events**, hand-labelled **gold test set**, group-aware split, CheckList-style behaviour tests |
| No baseline anywhere ("promising, not proven") | Models compared with nothing | Persistence / seasonal / rule / logistic baselines in every stage; **skill score** vs baseline |
| Forecast is next-hour only; "graph features" are neighbour averages, not a GNN | Short horizon, fake GNN | **7-day multi-horizon** forecast + a real graph neural layer + ablation with/without graph |
| NER = regular expressions for numbers; no carriers, disruption types, mitigation actions; no jargon audit | Regex "NER" | Trained NER with 4 required entity types, **linked to node IDs**, ambiguity-aware alias resolver, terminology audit |
| SLM is FLAN-T5 **as-is**, no fine-tuning, no latency/memory numbers, no offline app | Template fallback labelled as "SLM" | **LoRA fine-tune**, measured latency p50/p95 + RAM, hallucination rate, offline demo with network-off test |
| GenAI stage only writes incident briefs; no telemetry generator, no 20 scenarios, no wildcard | Briefs instead of the required generator | **Conditional VAE + constraint projection + simulator**, 20-scenario robustness matrix, live wildcard |
| Missing modules in the zip, incomplete README run steps | Code that does not run on another laptop | Contracts, hooks, CI, one-command pipeline, clean-clone test |
| 100% synthetic, no real-world anchor | Demo data only | **Real-anchored calibration** (public data, see `docs/06_DATA_SOURCES.md`) with provenance columns |

## 5. Definition of "done" for the whole project

1. `git clone` + `pip install -r requirements.txt` + `python run_pipeline.py` works on a fresh device.
2. `streamlit run app.py` shows all five stage pages and the fused triage view.
3. Every claim we make in a debate is in `claims/CLAIMS_LEDGER.md` with a file and a command that reproduces it.
4. `docs/` contains the architecture (LLD), the decisions log and the known-limitations list.
5. No AI attribution anywhere in git history.
