# Known limitations (we say these first, before the other team does)

Owner: Role 4. Updated at the end of every day. Each entry: what, why it matters, what we did about it, where the evidence is.

| # | Limitation | Why it matters | What we did / will do | Evidence |
|---|---|---|---|---|
| 1 | Shipment telemetry is **calibrated simulation**, not live port feeds | Absolute accuracy is not production accuracy | Anchored to public real data; comparisons (model vs baseline, with vs without graph) are the findings; every row labelled `data_origin` | `docs/06_DATA_SOURCES.md`, `data/raw/sim/SIM_CARD.md` |
| 2 | Labels depend on the simulator's outcome model | A model could learn the simulator, not logistics | Hidden-factor noise; outcome labels; leakage tests; report skill vs baselines | `Stage01_ML/reports/eval_leakage_tests.json` (Day 1) |
| 3 | The system recommends, humans decide | No automated rerouting | `requires_operator_review: true` in every response; no execution code path | API tests |
| 4 | Stage 06 (agentic rerouting) is outside the five-day scope | Problem statement lists it | Documented as future work | `docs/00_PROJECT_BRIEF.md` |
| 5 | Small human-labelled sets (gold NER/urgency set, 100 hand-written summaries, small rater panel) | Wide confidence intervals | Report n and CIs; never over-claim | Stage 3-4 eval reports |

| 6 | The simulator is **strongly seasonal**: critical share by month is 2-3% in Jun-Jul and Nov-Dec, 17% in Aug, 33% in Sep, 13-17% in Jan-Feb. A chronological split puts validation in the busy season and test in the calm one (rehearsal: train 11%, val 17%, test 2.7% critical) | A threshold chosen on validation may not suit test; baseline-vs-model rankings on validation can flip on test; test has few CRITICAL rows (about 200) so intervals are wide | Report per-split class mix with every result (`eval_metrics.json`), quote test numbers with this caveat, raise at the Team Huddle (rolling-origin backtest or season-aware split, by vote) | `Stage01_ML/reports/eval_metrics.json` (`split_class_mix`), `eval_threshold_decision.md` |
| 7 | The Little's-law / flow-conservation acceptance check is not applicable: the merged simulator is the allowed fallback design (AR(1) congestion, no queueing model) | We cannot claim the data obeys conservation laws | Say "calibrated congestion simulation" not "physics-based"; compare models and baselines, not absolute accuracy | `data/raw/sim/SIM_CARD.md`, `tests/eval/test_sim_acceptance.py` |
| 8 | `dwell_time_hours` and `capacity_utilization_pct` alone reach AUC 0.89 / 0.88 for CRITICAL | A simple dwell rule is a strong baseline; model gains over it may look small | Always report the dwell-rule baseline next to every model; report skill versus it | `eval_leakage_tests.json` (single-feature ceiling) |

(Role 4 adds one row per issue discovered each day.)
