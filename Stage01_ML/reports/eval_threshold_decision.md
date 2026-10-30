# Alert threshold decision (Stage 01)

Date: 2026-10-30 - cost policy version 1.0 - status: PROPOSED until the Team Huddle ratifies it.

## The question
How many false alarms is acceptable? We answer it with costs, not opinion: a missed critical delay costs 10 units, an unnecessary emergency escalation costs 1 (see `contracts/cost_policy.json`).

## The answer

- Model: **xgboost_monotone**. Raise an alert when P(critical) >= **0.32**. This number was chosen on the validation split and frozen before looking at test.
- On test, that alert catches **82%** of critical delays, **62%** of alerts are real, so there are **38 false alarms per 100 alerts**.
- The soft budget is 40 false alarms per 100 alerts: this model is **within** the budget.
- Expected cost per shipment at this threshold: **0.408** [0.373, 0.441].

## Honest comparison with the baseline
- candidate beats the best baseline on validation cost (95% paired block bootstrap).
- Beats the best baseline (logreg): **True**.

## Sensitivity: what if a miss is worth more or less than 10?

| Miss costs | Threshold | Alerts | Recall of critical | False alarms /100 alerts | Test cost |
|---|---|---|---|---|---|
| 3 | 0.45 | 1400 | 73% | 26 | 0.279 |
| 5 | 0.42 | 1485 | 75% | 29 | 0.321 |
| 10 | 0.32 | 1869 | 82% | 38 | 0.408 |
| 20 | 0.14 | 3611 | 94% | 63 | 0.526 |
| 40 | 0.14 | 3611 | 94% | 63 | 0.710 |

The higher a miss costs, the lower the threshold and the more false alarms we accept. Row 10 is the proposed policy.

## Three-class decision
Choosing the cheapest of the three actions per shipment (Bayes decision) gives expected cost **0.337** and flags **3136** shipments as CRITICAL on test. With this cost matrix AT_RISK is a cheap hedge, so the three-class rule predicts it often; the single-threshold alert above is what the operator console should use.
