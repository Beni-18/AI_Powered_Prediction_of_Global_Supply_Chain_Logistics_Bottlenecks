# Stage 02 - Role 3 - Data & Integration Engineer -> hand-off note (01_data_engineer.py)

status: IN_PROGRESS
updated_utc: 2026-10-31
git_commit: not committed yet (branch to be opened by Bennish)
data_version: v1 (sim track) + real track v1 (IMF PortWatch to 2026-07-24)

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

**Owner: Pravin (Role 3).** First version written by Role 4 (Bennish) at the team lead's request so Stage 2 could run end to end.
**Pravin: review, re-run from a clean clone, then set READY** (see "Your tasks" at the bottom). Design: `docs/decisions/0011-stage02-forecasting-design.md`.

## Files produced (path | contract | rows | notes)
- `Stage02_DL/data/hourly_node_telemetry.csv.gz` | `hourly_node_telemetry` | 236,520 | 27 nodes x 8,760 h of 2025. 2,379 missing hours put back and flagged `is_imputed=True`; all were short gaps, filled by forward-fill (max 6 h); the train hour-of-week median fallback was not needed (0 rows)
- `Stage02_DL/data/splits.json` | - | - | the Stage 1 dates (train_end 2025-09-13, val 2025-09-20..11-07, test from 2025-11-14, 168 h purge) + the Stage 2 window rule
- `Stage02_DL/data/scaler.json` | - | - | robust median/IQR per node and channel, TRAIN hours only (imputed hours excluded); inverse = `scale_array(..., inverse=True)`
- `Stage02_DL/data/adjacency_matrix.csv` | - | 27 x 27 | symmetric 0/1 lanes, registry order (50 lanes, no isolated node)
- `Stage02_DL/data/adjacency_directed_transit_days.csv` | - | 27 x 27 | directed, value = `typical_transit_days` (62 directed links)
- `Stage02_DL/data/real_daily_activity.csv.gz` | - (no contract) | 63,526 | **REAL** track: IMF PortWatch daily container port calls / chokepoint transits, 23 nodes x 2,762 days (2019-01-01..2026-07-24), `data_origin=REAL`, 0 imputed days
- `Stage02_DL/data/real_splits.json`, `real_scaler.json` | - | - | calendar-year split (train <= 2023, val 2024, test >= 2025-01-08, 7-day purge); robust scaler on log(1+count), train days only
- `Stage02_DL/data/data_quality_report.json` | - | - | every count above, per split period
- New shared code: `common/dataio/hourly_grid.py` (grid, filling, scaler), `common/dataio/real_activity.py` (PortWatch loader, reuses `scripts/fetch_real_data.py` id tables)

## Contract check (paste the output of `python scripts/validate_contract.py <schema> <file>`)
```
PASS  Stage02_DL/data/hourly_node_telemetry.csv.gz  (236520 rows, schema 'hourly_node_telemetry')
```

## Decisions and assumptions (what you decided and why, in plain English)
- Filling never looks into the future: forward-fill from the past (<= 6 h), else the node's TRAIN hour-of-week median. Nothing is clipped (overloads > 100% are real congestion, SIM_CARD).
- Same split dates as Stage 1 (one split for the project). Long-context features may look further back than a split's start: that is history a forecaster really has, never a training target.
- Real track: LA and Long Beach are one PortWatch port, so only `PORT_US_LAX` is kept (a copy would be a fake duplicate series). Rail hub and DCs are not in PortWatch. The grid stops at the last day EVERY node has data (ports end 2026-07-24, chokepoints 2026-10-25), so no months are padded.
- 1,030 node-days with zero calls are real zeros (e.g. Chennai averages 1.6 calls/day), kept as they are.

## Things the next role must know (warnings, known issues, changed meanings)
- The real track needs `data/raw/real/portwatch_*.csv` (git-ignored). On a clean clone run `python scripts/fetch_real_data.py` once (~10 min, ~650 MB download, ~2 MB kept). Without it the script skips the real track and says so.
- **Caution:** `fetch_real_data.py` also rewrites `data/calibration/*`. If the PortWatch history has been revised, the simulator inputs change. Check `git diff data/calibration` before committing.
- Chokepoint "throughput" in the sim track is still TEU-equivalent with no documented conversion (SIM_CARD limitation 4). The schema asks for it to be documented here.

## Open questions for other roles (name the role and the question)
- Role 3 (you): keep the real-track split (calendar years) or align it to something else? Written up in decision 0011, status PROPOSED.

## Your tasks (Pravin)
1. [MUST] Review `01_data_engineer.py`, `common/dataio/hourly_grid.py`, `common/dataio/real_activity.py` and `tests/data/test_hourly_grid.py`; fix anything you disagree with.
2. [MUST] Clean clone: `python scripts/fetch_real_data.py`, then `cd Stage02_DL && python 01_data_engineer.py`. Paste the contract check above and set READY.
3. [SHOULD] Add the chokepoint throughput conversion note (schema requirement) to this file.
4. [SHOULD] Augmentation helpers (jitter / scaling / window-warp) in `common/dataio`, train windows only, unit-tested. Role 1 may then use them.
