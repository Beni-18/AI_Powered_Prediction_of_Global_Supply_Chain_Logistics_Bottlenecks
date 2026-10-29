# Stage 01 - Role 3 - Data & Integration Engineer -> hand-off note (01_data_engineer.py)

status: READY
updated_utc: 2026-10-29
git_commit: see PR "data(stage01): prepared dataset, split and data helpers"
data_version: v1 (built from simulator v0.2, seed 42)

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

## Files produced (path | contract | rows | notes)
- `Stage01_ML/data/prepared_dataset.csv.gz` | telemetry | 60,000 | labels + split set; 11 extra helper columns (flags)
- `Stage01_ML/data/split_boundaries.json` | - | - | train_end 2025-09-14, val 2025-09-21 to 2025-11-07, test from 2025-11-14
- `Stage01_ML/data/data_quality_report.json` | - | - | every change with counts, class mix per split

Rebuild: `python scripts/build_sim_data.py --seed 42` then `cd Stage01_ML && python 01_data_engineer.py`.

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
PASS  Stage01_ML/data/prepared_dataset.csv.gz  (60000 rows, schema 'telemetry')
      note: extra columns not in the contract: ['dup_flag', 'in_purge_gap', 'overload_flag', 'was_missing_dwell_time_hours', 'was_missing_throughput_teu_per_hr', 'was_missing_capacity_utilization_pct', 'was_missing_customs_hold_hours', 'was_missing_weather_severity_index', 'was_missing_labor_deficit_ratio', 'was_missing_equipment_availability_pct', 'was_missing_historical_delay_rate']
```

## Decisions and assumptions (what you decided and why, in plain English)
- **Label:** `delay_status` comes only from `realized_delay_hours` (thresholds 6 h and 24 h from `label_policy.json`).
- **Split (chronological 70/15/15, 168 h purge):** a row in the 168 h after a boundary keeps the earlier split and gets `in_purge_gap=True`, which applies to 2,281 rows. No row is dropped. Each row is its own shipment, so no shipment can appear in two splits.

  | Split | Rows | ON_TIME / AT_RISK / CRITICAL |
  |---|---|---|
  | train | 43,180 | 67 / 20 / 12 % |
  | val | 8,921 | 51 / 31 / 17 % |
  | test | 7,899 | 55 / 27 / 18 % |

- **Duplicates:** 100 exact duplicate rows removed (a feed sent twice). Near-duplicates are only flagged, and there were none.
- **Overload:** 175 rows have utilisation above 100%. They are kept and marked `overload_flag=True`.
- **Missing values (CONFLICT, see open questions):** the contract forbids blanks, so each missing feature value is filled with the **train-split median** of its column. The column `was_missing_<col>=True` marks every filled cell. About 400-480 cells per column were filled, and the medians are listed in `data_quality_report.json`.

## Things the next role must know (warnings, known issues, changed meanings)
- **Role 1:** use only the features in `contracts/feature_policy.json`. You *may* also use the `was_missing_*` and `overload_flag` columns, because they are knowable at snapshot time. Never use `dup_flag`, `in_purge_gap` or `split` as features. Consider leaving `in_purge_gap=True` rows out of training.
- Val and test are later in the year and have more delays than train. That is a real time drift, so do not reshuffle the split.
- The raw data still contains planted quality issues, such as a reporting offset and outliers, for Role 2 to find. They are not fixed here on purpose.
- `common/dataio/windows.py` (`build_windows`) and `common/dataio/graph.py` (`load_adjacency`) are ready for Day 2 and covered by `tests/data/`.

## Open questions for other roles (name the role and the question)
- **All roles:** the Role 3 file says "keep missing values, the model imputes", but the `telemetry` contract marks those columns non-nullable. I followed the contract (rule R3) and filled the gaps with train medians plus flags. If the team prefers real blanks, the contract needs `"nullable": true` on those 8 columns. That is a contract change, so it needs a second approver.
