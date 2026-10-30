# 0008 - Missing feature values: median fill or nullable columns
Date: 2026-10-30   Deciders: Roles 1, 2, 3, 4   Status: PROPOSED (needs a second approver; Role 3 proposes)

## Context
The raw shipment telemetry has gaps in 8 feature columns (for example `dwell_time_hours`, `customs_hold_hours`).
The Role 3 file says "keep missing values, the model imputes", but `contracts/schemas/telemetry.schema.json`
marks those 8 columns as non-nullable. Contracts are law (AGENTS.md R3), so Stage 1 data v1 followed the contract:
each gap was filled with the **train-split median** of its column, and a `was_missing_<col>=True` flag marks every
filled cell. Counts and medians per column are in `Stage01_ML/data/data_quality_report.json`
(`missing_filled_with_train_median`).

A second, related gap: `contracts/feature_policy.json` does not list `was_missing_*` or `overload_flag`, so no
model may use them yet, although both are known at snapshot time.

## Options
1. **Keep the median fill (contract unchanged).** Nothing already verified changes; Stage 1 results stay valid.
   Con: a filled value looks like a real one unless the model also sees the flag.
2. **Make the 8 columns nullable (contract change).** Real blanks reach the model, which imputes them itself.
   Con: every Stage 1 dataset, model and claim must be rebuilt and re-verified.
3. **Option 1 now, plus allow the flags as features from Stage 2 on (feature-policy change).** Models can learn
   "this value was missing" without touching Stage 1.

## Decision
Proposed: **option 3.** Keep the median fill for Stage 1. From Stage 2 on, add `was_missing_` (prefix) and
`overload_flag` to `contracts/feature_policy.json`, as a separate PR approved by Role 1 (who uses features)
and Role 4 (who checks leakage).

## Consequences
- Role 3: no data rebuild; writes the feature-policy PR only after this is accepted.
- Role 1: may use the flags from Stage 2 on; Stage 1 models stay as verified.
- Role 4: leakage check for the new features (they are set from the same row, at snapshot time).
- What we give up: Stage 1 models never see the missing-value flags.

## Vote
| Role | Name | Agree? |
|---|---|---|
| 1 ML | Santhosh K | |
| 2 EDA | T.R. Isha Miruthika | |
| 3 Data & Integration | Pravin | yes (proposer) |
| 4 Evaluation | Bennish Christopher | |
