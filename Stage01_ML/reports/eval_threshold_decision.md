# Alert threshold decision (Stage 01)

Date: 2026-10-30 - cost policy version 1.0 - status: PROPOSED until the Team Huddle ratifies it.

## The question
How many false alarms is acceptable? We answer it with costs, not opinion: a missed critical delay costs 10 units, an unnecessary emergency escalation costs 1 (see `contracts/cost_policy.json`).

## The answer

- Model: **logreg**. Raise an alert when P(critical) >= **0.34**. This number was chosen on the validation split and frozen before looking at test.
- On test, that alert catches **83%** of critical delays, **60%** of alerts are real, so there are **40 false alarms per 100 alerts**.
- The soft budget is 40 false alarms per 100 alerts: this model is **within** the budget.
- Expected cost per shipment at this threshold: **0.369** [0.339, 0.399].

## Honest comparison with the baseline
- no candidate clearly beats the best baseline on validation; reporting the baseline honestly.
- Beats the best baseline (logreg): **False**.

## Sensitivity: what if a miss is worth more or less than 10?

| Miss costs | Threshold | Alerts | Recall of critical | False alarms /100 alerts | Test cost |
|---|---|---|---|---|---|
| 3 | 0.46 | 1497 | 76% | 28 | 0.251 |
| 5 | 0.46 | 1497 | 76% | 28 | 0.289 |
| 10 | 0.34 | 1939 | 83% | 40 | 0.369 |
| 20 | 0.24 | 2492 | 89% | 49 | 0.482 |
| 40 | 0.11 | 4001 | 95% | 66 | 0.651 |

The higher a miss costs, the lower the threshold and the more false alarms we accept. Row 10 is the proposed policy.

## Three-class decision
Choosing the cheapest of the three actions per shipment (Bayes decision) gives expected cost **0.336** and flags **3251** shipments as CRITICAL on test. With this cost matrix AT_RISK is a cheap hedge, so the three-class rule predicts it often; the single-threshold alert above is what the operator console should use.
