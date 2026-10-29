# SIM_CARD - calibrated logistics simulator (v0.1, fallback design)

**Owner:** Role 3 (Data & Integration Engineer) · **Seed:** 42 · **Command:** `python scripts/build_sim_data.py --seed 42` (about 10 seconds)
**Code:** `common/dataio/simulator.py` (engine), `scripts/build_sim_data.py` (noise, planted issues, file writing)

## What it is
Every row is **simulated** (`data_origin = SYNTHETIC_CALIBRATED`, `is_synthetic = True`). We do not claim live or real shipment data.

We use the **fallback design** allowed by the Role 3 file (section 6.3), not the full fluid-queue network:
- Each of the 27 nodes has a hidden **congestion level** `c` that moves hour by hour towards a target (AR(1)-style mean reversion, 5% per hour, plus small shocks).
- The target rises with weather, labour shortage, customs holds, equipment loss, peak season (Aug-Oct), a slow global-stress wave, **events**, and **congestion arriving from upstream nodes**. Upstream congestion arrives after the lane's real `typical_transit_days` from `contracts/network_edges.csv`, so cascades travel along the network.
- Observable channels come from `c`:
  - utilisation = 45 + 40c, which can reach up to 150; anything above 100 is a real overload.
  - dwell = base × (0.4 + 2c).
  - throughput drops under gridlock.
  - inventory = yard × utilisation.

## How the label is made (no leakage by construction)
`realized_delay_hours` is the **outcome**, made up of three parts:
- extra waiting at the current node over the **next 24 h**, weighted 1.5;
- extra waiting at every later node **at the hour the shipment will arrive there**, weighted 0.15. Both parts come from future simulated states, which no column contains.
- **hidden factors** that are in no column: a base wobble (exponential, mean 1 h), missed vessel connections (3% of shipments, exponential mean 20 h) and random inspections (3%, mean 10 h).

The first two parts are added together, then passed through a curve `0.7s + 0.04s²` and multiplied by hidden carrier and priority factors. The hidden factors are added on top.

Planted issues and measurement noise affect only the feature columns. The label itself is not noised.

## Tuned once (never re-tune to flatter a model)
The weights 1.5 and 0.15, the curve 0.7 and 0.04, and the hidden-noise sizes were tuned once on 29 Oct 2026 to hit the target class mix:

| | Target | Achieved |
|---|---|---|
| ON_TIME (<6 h) | 65% | 64.8% |
| AT_RISK (6-24 h) | 22% | 24.1% |
| CRITICAL_DELAY (≥24 h) | 13% | 11.1% |

**Sanity check:** quick untuned models were trained on Jan to early Sep and tested on late Oct to Dec. A depth-4 tree scored a macro-F1 of 0.55 and a default gradient-boosting model 0.50. The label is hard but learnable, and nowhere near the 0.95 that would signal a leak. The role file's expected range is 0.6-0.85, so Role 1's feature engineering has room to add value.

## Outputs (`data/raw/sim/`)
| File | Rows | Notes |
|---|---|---|
| `shipment_telemetry_raw.csv.gz` | 60,100 | 60,000 shipments plus 100 planted duplicate rows. `delay_status` and `split` are empty because Stage01 `01_data_engineer.py` sets them |
| `node_hourly_raw.csv.gz` | 234,142 | 27 nodes × 8,760 h, with about 1% of hours deliberately missing for Stage 2 to rebuild |
| `events_ground_truth.csv` | 125 | `event_id, type, node_id, start_ts, end_ts, severity, cause` (includes the 60-day run-out after 2025) |
| `sim_summary.json` | - | Seed, counts and class mix |
| `INJECTED_ISSUES.json` | - | **Private.** Commit it only after Role 2 reports Day 1 findings |

## Known simplifications (say these openly in the debate)
1. **No real-data calibration yet.**
   - Weather is a synthetic seasonal pattern plus a typhoon bump, not Open-Meteo.
   - The global-stress wave stands in for the NY Fed GSCPI.
   - Base dwell, throughput and yard size per node *type* are rough public orders of magnitude, not per-port values.
   - Next step: `scripts/fetch_real_data.py` and `data/calibration/*`.
2. The event catalogue frequencies are our placeholder values until Role 2's `docs/decisions/0001-event-catalogue.md` is transcribed.
3. Shipment routes are shortest paths, with no rerouting during disruptions.
4. Chokepoint "throughput" is TEU-equivalent and has no real vessel-transit conversion yet.
5. `historical_delay_rate` uses outcomes from 7-37 days before each snapshot, so only outcomes that would already be known count. The first month falls back to the overall early rate.
