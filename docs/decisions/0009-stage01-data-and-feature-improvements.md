# 0009 - Stage 1 data corrections and feature changes
Date: 2026-10-30   Deciders: Roles 1, 2, 3, 4   Status: PROPOSED (needs the team vote below)

## Context
After the first Stage 1 results, Role 4 re-checked the data cleaning, the EDA findings and the model inputs, and
ran a controlled experiment: Role 1's champion recipe with ONE change at a time, judged on validation cost only
(95% paired day-block bootstrap). Four issues stood out:
- Two dwell reporting faults found by Role 2's EDA (ZIM's offset from 1 July; x3 sensor spikes) were never corrected.
- `cal_month`: training covers January to September, so November and December were read as the September peak.
- The model did not know what is normal at each node (Role 2's noise-floor finding: busy gateways run long dwell).
- The hourly node table (queue of waiting vessels, inventory, utilisation trend) was not used at all.

## Options
1. **Keep the model as it is.** No risk, no gain; `cal_month` keeps distorting the test months.
2. **Fix the data and the inputs (proposed).** Experiment, validation cost saving vs the old recipe:
   drop `cal_month` +0.033 [+0.023, +0.041]; node-relative inputs +0.012 [+0.000, +0.024];
   node history +0.004 (not clear alone, better ranking); data fixes +0.002 (not clear alone, but correct).
   Combined: +0.041 [+0.026, +0.056].
3. **Add a separate alert line for chokepoints.** Not tested; chokepoints stay weak (few critical cases).

## Decision (proposed)
- `01_data_engineer.py` corrects the two dwell faults, estimating each correction from the data and flagging every
  changed row (`dwell_offset_corrected`, `dwell_spike_corrected`); logged in `data_quality_report.json`.
- The hourly node table becomes a model input source through `common/dataio/node_context.py`, using only the last
  COMPLETE hour before each shipment (`lag_` rule of `contracts/feature_policy.json`).
- New inputs `eng_dwell_vs_node`, `eng_util_vs_node` (value / node's train median) and 8 `lag_*` node-history inputs;
  `cal_month` removed.
- XGBoost chooses its number of trees on the last 15% of train, not on validation.

## Consequences
- Retrained result (the evaluation decides, not this file): logistic regression is champion again; test cost per
  shipment 0.369 (was 0.408), 82.8% of critical delays caught, 39.6 false alarms per 100 alerts, PR-AUC 0.829.
- The API needs the hourly node table to score (it reads `data/raw/sim/node_hourly_raw.csv.gz`). For a shipment
  after the data ends, the node history is missing and the model uses the training medians instead.
- Chokepoints remain the weakest slice (option 3 is still open).

## Vote (fill in, then set Status to ACCEPTED)
| Role | Name | Agree? |
|---|---|---|
| 1 ML | Santhosh K | |
| 2 EDA | T.R. Isha Miruthika | |
| 3 Data & Integration | Pravin | |
| 4 Evaluation | Bennish Christopher | yes |
