# 02 - Our unique selling point: CASCADE

**CASCADE** is our project name: delays *cascade* through a network, and our answer is an early-warning system that is
cascade-aware, source-anchored, claim-audited and defensible. (The name is a handle for the debate; the substance is
the six commitments below.)

> **One sentence for the debate:** *"Other teams will show you a demo; we will show you the evidence - leak-free
> labels, baselines we have to beat, a cost-justified alarm threshold, a real graph model, a measured offline
> model, and a stress test of 20 disruptions - and every number is reproducible from one command."*

## The six commitments (and the evidence that proves each one)

| # | Commitment | What it means in practice | Proof we can show live | Owner(s) |
|---|---|---|---|---|
| 1 | **Source-anchored data** | Scenario magnitudes and baseline dynamics are calibrated on public real-world data (chokepoint transits, supply-chain pressure index, weather); every row carries `data_origin` | `docs/06_DATA_SOURCES.md`, `data/calibration/`, `real_event_anchors.csv` plotted against our scenarios | 3, 2 |
| 2 | **Honest labels, no leakage** | Label = what *happened* (`realized_delay_hours`), not a formula of the inputs; time-ordered split with purge gap; forbidden-feature policy; shuffle-label and single-feature-AUC tests | `eval_leakage_tests.json` (shuffle-label accuracy ≈ chance; no feature with univariate AUC > 0.95) | 3, 2, 4 |
| 3 | **Beat-the-baseline discipline** | Every model is reported next to a dumb baseline (rule, logistic, persistence, seasonal-naive, TF-IDF, template, base prompt) as a *skill score* | One slide per stage: model vs baseline with bootstrap confidence intervals | 1, 4 |
| 4 | **Cost-justified alarms** | The Team Huddle question (acceptable false-alarm rate) is answered with a written cost matrix, an operating-point curve and a sensitivity grid, not a gut feeling | `eval_threshold_decision.md`, cost-vs-threshold chart, "at 10:1 miss-to-false-alarm cost we alert at p ≥ X" | 4 |
| 5 | **Cascade-aware network intelligence** | A real graph layer (not neighbour-averaging), text entities linked to node IDs, scenario generator that propagates shocks along lanes with delays learnt from data | GNN ablation (with/without graph), lead-lag heatmap between nodes, wildcard replayed on the map | 1, 2 |
| 6 | **Verified, offline, auditable** | LoRA-tuned SLM with measured latency/RAM/hallucination rate and a network-off test; claims ledger; known-limitations list we disclose *before* the other team finds them | `claims/CLAIMS_LEDGER.md`, `eval_latency.csv`, offline demo with Wi-Fi off | 1, 3, 4 |

## Why this wins a debate (and not only a demo)

The scoring rubric is Content Knowledge, Communication, Faculty Q&A, Team Coordination, Time Management (5 each).

- **Content knowledge:** each of us can answer *why* (e.g. why a time split, why LoRA, why a constraint layer) because the role files require it.
- **Faculty Q&A:** the questions an examiner asks first - "how do I know this isn't leakage?", "what does it beat?", "what does a false alarm cost?", "is the data real?", "does it hallucinate?" - each have a prepared answer **with a file behind it** (`docs/05_DEBATE_PLAYBOOK.md`).
- **Team coordination:** visible in the repo: board, hand-off files, contracts, CI. We can show it, not just say it.
- **Time management:** every day has a debate-ready gate before the debate and a day-complete gate after it.
- **Communication:** one-sentence claims, backed by one chart each, with the limitation stated up front.

## Where each stage carries the USP

| Stage | The "better than the other team" feature (the one thing to say in 20 seconds) |
|---|---|
| 1 ML | "Our 0.8x score is real: outcome labels, time split, cost-optimal threshold. Show us a 99% model and we will show you the leak." |
| 2 DL | "7-day forecast with intervals, a real GNN that we ablate, and a lead-time metric: how many hours before the jam we warn." |
| 3 NLP | "Our NER resolves jargon to node IDs ('LA/LB' → two ports) and we publish the misinterpretation audit, including what we still get wrong." |
| 4 SLM | "Fine-tuned, 2 sentences, under 5 seconds on a laptop CPU, Wi-Fi off, and a measured hallucination rate - not a template in disguise." |
| 5 GenAI | "Generator with conservation constraints, 20 scenarios scored through our own Stages 1-4, and a wildcard where we tell you which stage breaks first." |

## What we deliberately do NOT claim

Naming limits first is part of the USP (it kills the opponent's best attack):

- The data is calibrated simulation, not live port feeds. We say so on every slide.
- Absolute accuracy on synthetic data is not production accuracy; the *comparisons* (model vs baseline, with vs without graph) are the findings.
- The system recommends; humans decide. We do not automate rerouting.
- Stage 06 (Agentic AI) is outside the five-day plan.

## The three artefacts that make the USP checkable

1. `claims/CLAIMS_LEDGER.md` - every number we will say aloud: claim, value, source file, reproduce-command, verified-by, date.
2. `docs/KNOWN_LIMITATIONS.md` (created by Role 4 on Day 1, extended daily) - the honest list.
3. `python run_pipeline.py` + CI badge - proof the whole thing runs on a clean machine.
