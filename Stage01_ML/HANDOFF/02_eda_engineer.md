# Stage 01 - Role 2 - EDA Engineer -> hand-off note (02_eda_engineer.py)

status: READY
updated_utc: 2026-10-29T14:12:00Z
git_commit: eda/stage01-eda
data_version: raw (shipment_telemetry_raw.csv.gz, 60,100 rows)

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

## Files produced (path | contract | rows | notes)
- `Stage01_ML/reports/eda_seasonality.csv` | report | 264 | Hourly, daily, monthly dwell/throughput/utilization profiles by node type
- `Stage01_ML/reports/eda_seasonality.png` | report | - | 3-panel chart illustrating diurnal cycles and Q3 peak surge
- `Stage01_ML/reports/eda_noise_floor.csv` | report | 228 | Per-node monthly robust noise floor (MAD × 1.4826)
- `Stage01_ML/reports/eda_noise_floor.png` | report | - | Heat-map of per-node noise floor variations across seasonal cycles
- `Stage01_ML/reports/eda_drift_report.json` | report | - | PSI & KS drift metrics between early and late annual thirds
- `Stage01_ML/reports/eda_drift.png` | report | - | Visual summary of PSI distribution shifts across columns
- `Stage01_ML/reports/eda_outliers.csv` | report | 8 | IQR/robust-z counts and record of 324 legitimate overloads preserved
- `Stage01_ML/reports/eda_class_balance.csv` | report | 123 | Class distribution overall, per node, per carrier, and per month
- `Stage01_ML/reports/eda_leakage_audit.json` | report | - | Univariate ROC-AUC and depth-3 tree leakage audit (Status: PASS)
- `Stage01_ML/reports/eda_correlations.csv` | report | 8 | Spearman rank correlation matrix across allowed numeric features
- `Stage01_ML/reports/eda_mutual_info.csv` | report | 8 | Mutual information scores with target on train split only
- `Stage01_ML/reports/eda_vif.csv` | report | 8 | Multicollinearity analysis (all VIFs < 6.0)
- `Stage01_ML/reports/eda_findings.md` | report | - | Comprehensive EDA deliverable with recommendations for Roles 1, 3, 4
- `Stage01_ML/reports/eda_field_guide_text.md` | report | 120 words | Exactly 120 words plain-English operational bottleneck summary

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
EDA outputs are report artifacts (eda_*) consumed as guidance by teammates.
Validated internally via unit tests in tests/eda/test_edakit.py (9 passed)
and full repo test suite (73 passed).
Contracts self-test passed:
PASS  dl_forecasts.csv  (4 rows, schema 'dl_forecasts')
PASS  hourly_node_telemetry.csv  (6 rows, schema 'hourly_node_telemetry')
PASS  incident_messages.csv  (4 rows, schema 'incident_messages')
PASS  ml_predictions.csv  (6 rows, schema 'ml_predictions')
PASS  nlp_predictions.csv  (3 rows, schema 'nlp_predictions')
PASS  scenario_telemetry.csv  (3 rows, schema 'scenario_telemetry')
PASS  slm_generations.csv  (4 rows, schema 'slm_generations')
PASS  slm_pairs.csv  (3 rows, schema 'slm_pairs')
PASS  stress_results.csv  (3 rows, schema 'stress_results')
PASS  telemetry.csv  (6 rows, schema 'telemetry')
```

## Decisions and assumptions (what you decided and why, in plain English)
- **Robust Noise Estimation:** Used rolling-median residual and MAD × 1.4826 rather than standard deviation so that rare extreme disruption shocks (typhoons, strikes) do not falsely inflate the baseline noise floor.
- **Seasonality vs Leakage:** Evaluated distribution shift with PSI; confirmed that dwell (PSI=0.215) and utilisation (PSI=0.243) shifts represent genuine physical operations changes during late summer / autumn peak shipping.
- **Preservation of Overloads:** Identified 324 records where `capacity_utilization_pct` exceeds 100% (reaching up to 131.4%); decided to retain all overloads as valid physical saturation events rather than clipping them.

## Things the next role must know (warnings, known issues, changed meanings)
- **For Role 1 (ML Engineer):**
  - Use `np.log1p(dwell_time_hours)` and `RobustScaler` due to heavy right skew.
  - Engineer interaction features (`eng_dwell_to_throughput_ratio`, `eng_customs_burden`).
  - Add calendar features (`cal_month`, `cal_day_of_week`) to account for autumn peak drift.
- **For Role 3 (Data & Integration Engineer):**
  - Drop the 100 exact duplicate rows identified in raw data during `Stage01_ML/01_data_engineer.py` cleaning.
  - Flag >100% utilisation as legitimate overloads in `data_quality_report.json`.
- **For Role 4 (Evaluation Engineer):**
  - Evaluate models on macro-F1 and cost-weighted loss to handle the 64.8% / 24.1% / 11.1% class imbalance.
  - Segment performance by operational regime (peak season vs nominal, chokepoints vs gateway ports).

## Open questions for other roles (name the role and the question)
- **Role 1 (ML Engineer):** Will you use node embeddings or target-encoding for the high-cardinality port identifiers?
- **Role 3 (Data Engineer):** When `prepared_dataset.csv` is published, please confirm that the 168-hour purge gap between train, val, and test splits is preserved.
