# 0004 - Cost policy and the Stage 1 alert threshold
Date: 2026-10-30   Deciders: Roles 1, 2, 3, 4 (Team Huddle)   Status: ACCEPTED (Team Huddle, 2026-10-30)

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

## Decision
- Keep `cost_policy.json` v1.0 as it is: miss = 10, false alarm = 1, soft budget 40 false alarms per 100 alerts.
- Alert when P(critical) >= **0.37** with the champion `logreg` (from `Stage01_ML/reports/eval_champion.json`).
- Re-check the threshold whenever the champion model changes (it is re-chosen on validation automatically).

## Consequences
- Role 3's API and dashboard read the threshold from `eval_champion.json`; nothing is hard-coded.
- Planners get about 6 real alerts for every 4 false ones; the field guide explains why that trade is worth it.
- What we give up: about 1 in 5 critical delays is still missed, and the chokepoints (Suez, Malacca, Panama) are weakest.

## Vote (Team Huddle, 2026-10-30)
| Role | Name | Agree? |
|---|---|---|
| 1 ML | Santhosh K | yes |
| 2 EDA | T.R. Isha Miruthika | yes |
| 3 Data & Integration | Pravin | yes |
| 4 Evaluation | Bennish Christopher | yes |

## Amendment 1 (2026-10-30): new champion, threshold re-chosen by the same rule
Role 1's tuning (PR 14) produced `xgboost_monotone`, which beats the best baseline on validation cost by 0.024 per
shipment (95% range 0.007 to 0.040), so it replaced `logreg`. As this decision requires, the threshold was re-chosen on
validation: **alert when P(critical) >= 0.32**. On the test period it catches 82% of critical delays with 38 false alarms
per 100 alerts (inside the budget of 40). The cost policy and the alert budget are unchanged.
Numbers: `Stage01_ML/reports/eval_champion.json`, `eval_metrics.json`, `eval_threshold_sensitivity.csv`.

## Amendment 2 (2026-10-30): data and feature changes (decision 0009), threshold re-chosen by the same rule
After the data corrections and new inputs of decision 0009, logistic regression has the lowest validation cost and
is champion again. Re-chosen on validation: **alert when P(critical) >= 0.34**. On the test period it catches 82.8% of
critical delays with 39.6 false alarms per 100 alerts (inside the budget of 40). The cost policy is unchanged.

## Amendment 3 (2026-10-30): data driven by real sources (decision 0010)
On the real-driven data, logistic regression stays champion and the threshold, re-chosen on validation, is
**0.39**. On the test period it catches 77.3% of critical delays with 40.1 false alarms per 100 alerts, 0.1 above
the soft budget of 40; the budget is a discussion line, not a hard stop. The cost policy is unchanged.

