# Stage 02 - Role 3 - Data & Integration Engineer -> hand-off note (01_data_engineer.py)

status: READY
updated_utc: 2026-10-31
git_commit: see PR from branch data/stage02-review
data_version: v1 (sim track) + real track v1 (IMF PortWatch to 2026-07-24)

> Allowed status values: NOT_STARTED, IN_PROGRESS, READY, BLOCKED.
> Only the role named in the title edits this file. Everyone else only reads it.
> Set READY only AFTER the contract check below passes and your work is pushed.

**Owner: Pravin (Role 3).** First version written by Role 4 (Bennish) at the team lead's request so Stage 2 could run end to end;
reviewed, re-run from a clean clone and taken over by Role 3 on 2026-10-31. Design: `docs/decisions/0011-stage02-forecasting-design.md`.

## Files produced (path | contract | rows | notes)
- `Stage02_DL/data/hourly_node_telemetry.csv.gz` | `hourly_node_telemetry` | 236,520 | 27 nodes x 8,760 h of 2025. 2,379 missing hours put back and flagged `is_imputed=True`; all were short gaps, filled by forward-fill (max 6 h); the train hour-of-week median fallback was not needed (0 rows)
- `Stage02_DL/data/splits.json` | - | - | the Stage 1 dates (train_end 2025-09-13, val 2025-09-20..11-07, test from 2025-11-14, 168 h purge) + the Stage 2 window rule
- `Stage02_DL/data/scaler.json` | - | - | robust median/IQR per node and channel, TRAIN hours only (imputed hours excluded); inverse = `scale_array(..., inverse=True)`
- `Stage02_DL/data/adjacency_matrix.csv` | - | 27 x 27 | symmetric 0/1 lanes, registry order (50 lanes, no isolated node)
- `Stage02_DL/data/adjacency_directed_transit_days.csv` | - | 27 x 27 | directed, value = `typical_transit_days` (62 directed links)
- `Stage02_DL/data/real_daily_activity.csv.gz` | - (no contract) | 63,526 | **REAL** track: IMF PortWatch daily container port calls / chokepoint transits, 23 nodes x 2,762 days (2019-01-01..2026-07-24), `data_origin=REAL`, 0 imputed days
- `Stage02_DL/data/real_splits.json`, `real_scaler.json` | - | - | calendar-year split (train <= 2023, val 2024, test >= 2025-01-08, 7-day purge); robust scaler on log(1+count), train days only
- `Stage02_DL/data/data_quality_report.json` | - | - | every count above, per split period
- New shared code: `common/dataio/augment.py` (train-only augmentation, see below), `common/dataio/hourly_grid.py` (grid, filling, scaler), `common/dataio/real_activity.py` (PortWatch loader, reuses `scripts/fetch_real_data.py` id tables)

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
- **Chokepoint throughput conversion (schema note for `throughput_teu_per_hr`).** Chokepoints have no boxes of their own,
  so the simulator gives them a TEU-equivalent flow: `900 TEU/h x relative_size`, where `relative_size` = the chokepoint's
  real 2025 container-ship transits per day (IMF PortWatch `n_container`) divided by the average node's real traffic
  (`data/calibration/activity_profile.csv`, clipped to 0.2-5). It then moves with utilisation and equipment like any node
  (`common/dataio/simulator.py`, step 7). Measured on the 2025 grid, this works out to about **1,000 TEU-equivalent per
  real container-ship transit** at every chokepoint (Suez 960, Panama 968, Malacca 941, Hormuz 1,005, Bab el-Mandeb 1,016,
  Cape 1,055). Read it as a scale unit, not a real box count: a real container ship carries many more TEU than that.
  Comparisons WITHIN a chokepoint over time are meaningful; chokepoint vs port volumes are not.
- **Augmentation helpers** (`common/dataio/augment.py`): `jitter` (sensor noise, inputs only), `scaling` (one factor per
  node and channel on inputs, anchor AND targets, so the past-to-future relation stays true) and `window_warp` (stretches a
  slice of the older half of the window; the origin hour never moves). They take `ForecastData.batch()` dictionaries and
  raise an error if any window is not "train". Not used by default; Role 1 decides. 17 tests in `tests/data/test_augment.py`.

## Open questions for other roles (name the role and the question)
- **All roles (decision 0011):** Role 3's answer on the real-track split is to **keep calendar years** (train 2019-2023, val 2024,
  test 2025-26, 7-day purge). Whole years keep every season in every split and make the story simple. The Stage 1 dates
  cannot be reused because they only cover 2025. Please confirm in 0011.

## Review by Role 3 (2026-10-31) - what changed
- **Leakage fix (real track):** gaps longer than 3 days were filled with the node's median over ALL years, test years
  included. They now use the TRAIN-years median (2019-2023), like the sim track. No effect on today's file (0 such days),
  but a future PortWatch revision could have created some.
- Checked and kept: filling never looks ahead, the scaler uses real TRAIN hours only, the split dates equal Stage 1's,
  duplicates are counted, nothing is clipped, and the matrices follow registry order.
- Added the augmentation helpers and the chokepoint conversion note (the two [SHOULD] tasks).

## Tasks (Pravin) - status
1. [MUST] Review the data code and tests - **done** (one leakage fix, see above).
2. [MUST] Clean-clone run - **done 2026-10-31**: fresh `git clone`, `python scripts/fetch_real_data.py` (16.5 min, PortWatch
   bulk download) then `python 01_data_engineer.py` (5 s). Same numbers as above (236,520 rows, 2,379 imputed; real track
   23 nodes x 2,762 days), contract PASS, and the unzipped CSVs are byte-identical to the committed ones (md5 match).
   `data/calibration/*` did not change. Output `.gz` files are now written with a fixed gzip timestamp, so a re-run no
   longer shows as a git change.
3. [SHOULD] Chokepoint throughput conversion note - **done** (above).
4. [SHOULD] Augmentation helpers - **done** (`common/dataio/augment.py`, 17 tests).
