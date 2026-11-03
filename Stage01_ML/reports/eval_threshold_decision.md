# Alert threshold decision (Stage 01)

Date: 2026-11-03 - cost policy version 1.0 - status: PROPOSED until the Team Huddle ratifies it.

## The question
How many false alarms is acceptable? We answer it with costs, not opinion: a missed critical delay costs 10 units, an unnecessary emergency escalation costs 1 (see `contracts/cost_policy.json`).

## The answer

- Model: **xgboost**. Raise an alert when P(critical) >= **0.39**. This number was chosen on the validation split and frozen before looking at test.
- On test, that alert catches **80%** of critical delays, **63%** of alerts are real, so there are **37 false alarms per 100 alerts**.
- The soft budget is 40 false alarms per 100 alerts: this model is **within** the budget.
- Expected cost per shipment at this threshold: **0.322** [0.291, 0.356].

## Honest comparison with the baseline
- lowest validation cost of all candidates (decision 0012); its saving over logreg on validation is NOT statistically clear (the 95% interval includes 0).
- Beats the best baseline (logreg): **False**.

## Sensitivity: what if a miss is worth more or less than 10?

| Miss costs | Threshold | Alerts | Recall of critical | False alarms /100 alerts | Test cost |
|---|---|---|---|---|---|
| 3 | 0.63 | 847 | 65% | 23 | 0.216 |
| 5 | 0.50 | 1063 | 74% | 30 | 0.249 |
| 10 | 0.39 | 1284 | 80% | 37 | 0.322 |
| 20 | 0.23 | 1776 | 86% | 51 | 0.452 |
| 40 | 0.19 | 1987 | 88% | 56 | 0.672 |

The higher a miss costs, the lower the threshold and the more false alarms we accept. Row 10 is the proposed policy.

## Three-class decision
Choosing the cheapest of the three actions per shipment (Bayes decision) gives expected cost **0.341** and flags **2196** shipments as CRITICAL on test. With this cost matrix AT_RISK is a cheap hedge, so the three-class rule predicts it often; the single-threshold alert above is what the operator console should use.
