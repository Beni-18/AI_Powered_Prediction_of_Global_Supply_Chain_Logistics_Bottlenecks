# Known limitations (we say these first, before the other team does)

Owner: Role 4. Updated at the end of every day. Each entry: what, why it matters, what we did about it, where the evidence is.

| # | Limitation | Why it matters | What we did / will do | Evidence |
|---|---|---|---|---|
| 1 | Shipment telemetry is **calibrated simulation**, not live port feeds | Absolute accuracy is not production accuracy | Anchored to public real data; comparisons (model vs baseline, with vs without graph) are the findings; every row labelled `data_origin` | `docs/06_DATA_SOURCES.md`, `data/raw/sim/SIM_CARD.md` |
| 2 | Labels depend on the simulator's outcome model | A model could learn the simulator, not logistics | Hidden-factor noise; outcome labels; leakage tests; report skill vs baselines | `Stage01_ML/reports/eval_leakage_tests.json` (Day 1) |
| 3 | The system recommends, humans decide | No automated rerouting | `requires_operator_review: true` in every response; no execution code path | API tests |
| 4 | Stage 06 (agentic rerouting) is outside the five-day scope | Problem statement lists it | Documented as future work | `docs/00_PROJECT_BRIEF.md` |
| 5 | Small human-labelled sets (gold NER/urgency set, 100 hand-written summaries, small rater panel) | Wide confidence intervals | Report n and CIs; never over-claim | Stage 3-4 eval reports |

(Role 4 adds one row per issue discovered each day.)
