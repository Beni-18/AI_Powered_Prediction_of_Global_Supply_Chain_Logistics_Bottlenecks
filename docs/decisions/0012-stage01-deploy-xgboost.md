# 0012 - Stage 1: deploy the cheapest model on validation (XGBoost), not the baseline
Date: 2026-11-03   Deciders: team lead (Role 4, Bennish)   Status: ACCEPTED

## Context
The Stage 1 champion rule (Role 4's `04_evaluation_engineer.py`) picked the candidate with the lowest validation cost,
but only kept it if it beat the best baseline (logistic regression) with a 95% paired block-bootstrap interval above 0.
On the real-driven data, XGBoost has the lowest validation cost of every model (0.3160 per shipment vs 0.3222 for
logistic regression), but the saving, 0.0062 [-0.0059, 0.0188], is not statistically clear. So the rule fell back to
logistic regression, and the dashboard and API served a baseline in the machine-learning stage.

## Options
1. Keep logistic regression (the strict rule). Safe, but the deployed ML model is the baseline.
2. Deploy the candidate with the lowest validation cost (XGBoost) and report plainly that its gain over the baseline is
   not proven on validation. The choice still uses validation only; no test number is used to choose.
3. Pick a model by test results. Not allowed (AGENTS.md R5).

## Decision
Option 2. The champion is the non-baseline model with the lowest validation cost: `xgboost`, alert threshold 0.39
(its own threshold, chosen on validation by minimum cost). `eval_champion.json` keeps `beats_baseline: false` and the
validation saving with its interval, and the evaluation report says so in plain words.

## Consequences
- Role 4: `CHAMPION_RULE = "lowest_val_cost_candidate"` in `Stage01_ML/04_evaluation_engineer.py`; C-ML claims are
  recomputed for XGBoost and re-verified from a fresh clone. C-ML-12 still states that the champion does not clearly
  beat the baseline on validation.
- Role 3: the API and the dashboard read `eval_champion.json`, so they serve XGBoost with no code change; the
  operations field guide is regenerated.
- What we give up: the statistical guarantee that the deployed model beats the simplest one. We say this first if asked.
