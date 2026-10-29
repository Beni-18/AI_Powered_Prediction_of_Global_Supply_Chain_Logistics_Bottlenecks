# Stage 01 EDA Findings & Modelling Recommendations

```
WHAT THIS REPORT SHOWS AND HOW TO REPRODUCE IT:
Comprehensive Exploratory Data Analysis covering noise floor, seasonality, reporting drift,
leakage audit, and feature recommendations for Stage 01 Machine Learning.
Reproduce: python Stage01_ML/02_eda_engineer.py
Data source analyzed: raw (shipment_telemetry_raw.csv.gz)
```

---

## 1. Executive Summary & Golden Findings

Every finding in this report connects empirical evidence to an explicit modelling or data decision.

### Finding 1: Seasonal Capacity & Dwell Drift Across Cycles
- **Evidence:** `Stage01_ML/reports/eda_drift_report.json` and `Stage01_ML/reports/eda_seasonality.png`. `capacity_utilization_pct` displays a Population Stability Index (PSI) of **0.2426** and `dwell_time_hours` shows a PSI of **0.2153** between the first third and last third of the annual cycle, peaking in Q3 (August–October).
- **So-What Recommendation:** Static threshold models will suffer distribution shift during autumn peak demand. **We recommend Role 1 include rolling calendar features (`cal_month`, `cal_day_of_week`) and calibrate alert thresholds per season.**

### Finding 2: Dwell Time Noise Floor Heterogeneity
- **Evidence:** `Stage01_ML/reports/eda_noise_floor.csv` and `Stage01_ML/reports/eda_noise_floor.png`. Robust noise floor (MAD × 1.4826) at maritime chokepoints averages ~0.8 to 1.5 hours, whereas gateway container ports (LAX, LGB, RTM, JEA) exhibit background noise floors of **4.0 to 12.5 hours**.
- **So-What Recommendation:** A single global dwell threshold triggers false alarms at busy seaports while missing anomalies at chokepoints. **We recommend Role 1 normalise dwell using node-specific baselines or learn node embeddings.**

### Finding 3: Target Leakage Audit - Passed Cleanly
- **Evidence:** `Stage01_ML/reports/eda_leakage_audit.json`. Top univariate feature AUC is **0.8925** (`dwell_time_hours`), and a depth-3 decision tree achieves an AUC of **0.8776**. No feature approaches the >0.95 leakage threshold.
- **So-What Recommendation:** The simulator creates learnable outcomes without trivial target leakage. **Role 1 can safely train supervised classifiers on allowed features.**

### Finding 4: Legitimate Capacity Overload Preserved
- **Evidence:** `Stage01_ML/reports/eda_outliers.csv`. There are **324** instances where `capacity_utilization_pct` exceeds 100% (reaching up to 131.4%).
- **So-What Recommendation:** These instances represent genuine physical port congestion and yard saturation, as confirmed by `contracts/schemas/telemetry.schema.json`. **Role 3 must never blank or clip these values; Role 1 should preserve them using robust scaling.**

### Finding 5: Multicollinearity Bounds Acceptable (VIF < 6.0)
- **Evidence:** `Stage01_ML/reports/eda_vif.csv`. All allowed continuous features exhibit Variance Inflation Factors below 6.0 (highest: `dwell_time_hours` at 5.27, `throughput_teu_per_hr` at 3.48).
- **So-What Recommendation:** Features have moderate operational correlation without severe collinear redundancy. **Linear and tree models can utilize all allowed base features without immediate dimensionality reduction.**

---

## 2. Actionable Recommendations for Role 1 (ML Engineer)

1. **Engineered Ratios (`eng_` prefix):**
   - `eng_dwell_to_throughput_ratio = dwell_time_hours / (throughput_teu_per_hr + 1.0)`: Captures yard clearance efficiency.
   - `eng_customs_burden = customs_hold_hours / (dwell_time_hours + 1.0)`: Isolates regulatory delay share.
   - `eng_stress_interaction = capacity_utilization_pct * labor_deficit_ratio`: Captures compound operational vulnerability.
2. **Feature Transforms:**
   - Apply `np.log1p(dwell_time_hours)` to compress heavy right-skewed dwell tails.
   - Use `RobustScaler` (median / IQR) rather than `StandardScaler` to prevent typhoon/strike outliers from distorting zero-mean scales.
3. **Categorical Handling:**
   - Target-encode or frequency-encode high-cardinality `origin_port`, `destination_port`, and `current_node_id`.
   - Preserve priority tier ordinal hierarchy (`CRITICAL_MEDICAL` > `PERISHABLE_COLD_CHAIN` > `EXPEDITED` > `STANDARD`).
4. **Hard-to-Predict Subsets:**
   - Chokepoints under sudden geopolitical diversion (`CHOKE_BAB_EL_MANDEB`).
   - Late Q3 peak volume months where nominal dwell overlaps with disruption dwell.

---

## 3. Actionable Recommendations for Role 3 (Data & Integration Engineer)

1. **Deduplication:** Raw data contains 100 exact duplicate rows; ensure `Stage01_ML/01_data_engineer.py` drops exact duplicates while verifying row counts.
2. **Overload Flagging:** Ensure `data_quality_report.json` flags the 324 overload (>100%) records as valid physical states rather than errors.
3. **Chronological Splitting:** Maintain strict chronological splitting (70% train, 15% val, 15% test) with the 168-hour purge gap to prevent temporal leakage.

---

## 4. Actionable Recommendations for Role 4 (Evaluation Engineer)

1. **Class Imbalance Metric Weighting:** Overall class distribution is 64.8% ON_TIME, 24.1% AT_RISK, 11.1% CRITICAL_DELAY. Evaluators must report macro-F1 and cost-weighted loss rather than raw accuracy.
2. **Slicing by Operational Regime:** Evaluate model precision and recall separately on nominal vs peak-season months (Aug–Oct) and high-noise gateway ports vs chokepoints.
