# 0004 - Cost policy and the Stage 1 alert threshold
Date: 2026-10-30   Deciders: Role 4 (proposer), Roles 1, 2, 3   Status: PROPOSED (needs the Team Huddle vote below)

## Context
The problem statement's Stage 1 Team Huddle asks: what false-alarm level is acceptable? Answering with a gut
feeling is not defensible in a debate, so we answer with costs. `contracts/cost_policy.json` says a missed critical
delay costs 10 units and an unnecessary emergency escalation costs 1. The threshold that minimises the expected cost
is chosen on the validation split only and then frozen.

All numbers below are copied from `Stage01_ML/reports/eval_threshold_sensitivity.csv` and `eval_metrics.json`
(test period, model `logreg`), reproduced from a fresh clone on 2026-10-30.

## Options
| A missed critical costs | Alert threshold on P(critical) | Critical delays caught | False alarms per 100 alerts |
|---|---|---|---|
| 3x a false alarm | 0.51 | 67% | 25 |
| 5x | 0.46 | 72% | 29 |
| **10x (proposed)** | **0.37** | **79%** | **38** |
| 20x | 0.16 | 93% | 61 |
| 40x | 0.07 | 96% | 72 |

- **Lower ratio (3x-5x):** fewer false alarms, but about 1 in 3 critical delays is missed.
- **10x (proposed):** catches about 4 in 5 critical delays and stays inside the soft budget of 40 false alarms per 100 alerts.
- **Higher ratio (20x-40x):** catches almost every critical delay, but 6-7 of every 10 alerts are false, so planners will start ignoring them.

## Decision (proposed)
- Keep `cost_policy.json` v1.0 as it is: miss = 10, false alarm = 1, soft budget 40 false alarms per 100 alerts.
- Alert when P(critical) >= **0.37** with the champion `logreg` (from `Stage01_ML/reports/eval_champion.json`).
- Re-check the threshold whenever the champion model changes (it is re-chosen on validation automatically).

## Consequences
- Role 3's API and dashboard read the threshold from `eval_champion.json`; nothing is hard-coded.
- Planners get about 6 real alerts for every 4 false ones; the field guide explains why that trade is worth it.
- What we give up: about 1 in 5 critical delays is still missed, and the chokepoints (Suez, Malacca, Panama) are weakest.

## Vote (fill in at the Team Huddle, then set Status to ACCEPTED)
| Role | Name | Agree? |
|---|---|---|
| 1 ML | Santhosh K | |
| 2 EDA | T.R. Isha Miruthika | |
| 3 Data & Integration | Pravin | |
| 4 Evaluation | Bennish Christopher | yes |
