# Alert threshold decision (Stage 01)

Date: 2026-10-30 - cost policy version 1.0 - status: PROPOSED until the Team Huddle ratifies it.

## The question
How many false alarms is acceptable? We answer it with costs, not opinion: a missed critical delay costs 10 units, an unnecessary emergency escalation costs 1 (see `contracts/cost_policy.json`).

## The answer

- Model: **logreg**. Raise an alert when P(critical) >= **0.37**. This number was chosen on the validation split and frozen before looking at test.
- On test, that alert catches **79%** of critical delays, **62%** of alerts are real, so there are **38 false alarms per 100 alerts**.
- The soft budget is 40 false alarms per 100 alerts: this model is **within** the budget.
- Expected cost per shipment at this threshold: **0.485** [0.439, 0.530].

## Honest comparison with the baseline
- no candidate clearly beats the best baseline on validation; reporting the baseline honestly.
- Beats the best baseline (logreg): **False**.

## Sensitivity: what if a miss is worth more or less than 10?

| Miss costs | Threshold | Alerts | Recall of critical | False alarms /100 alerts | Test cost |
|---|---|---|---|---|---|
| 3 | 0.51 | 1273 | 67% | 25 | 0.328 |
| 5 | 0.46 | 1446 | 72% | 29 | 0.378 |
| 10 | 0.37 | 1802 | 79% | 38 | 0.485 |
| 20 | 0.16 | 3362 | 93% | 61 | 0.567 |
| 40 | 0.07 | 4809 | 96% | 72 | 0.673 |

The higher a miss costs, the lower the threshold and the more false alarms we accept. Row 10 is the proposed policy.

## Three-class decision
Choosing the cheapest of the three actions per shipment (Bayes decision) gives expected cost **0.341** and flags **3298** shipments as CRITICAL on test. With this cost matrix AT_RISK is a cheap hedge, so the three-class rule predicts it often; the single-threshold alert above is what the operator console should use.
