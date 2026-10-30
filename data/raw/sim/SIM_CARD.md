# SIM_CARD - calibrated logistics simulator (v0.3, real drivers, fallback design)

**Owner:** Role 3 (Data & Integration Engineer) · **Seed:** 42 · **Command:** `python scripts/build_sim_data.py --seed 42` (about 10 seconds)
**Code:** `common/dataio/simulator.py` (engine), `common/dataio/calibration.py` (real drivers), `scripts/build_sim_data.py` (noise, planted issues, file writing)

## v0.3: driven by real data (30 Oct 2026)
The simulator's drivers now come from **real public data** (`data/calibration/`, built by `scripts/fetch_real_data.py`):

| Driver | Before (v0.2) | Now (v0.3) | Real source |
|---|---|---|---|
| Weather at each node, every hour | synthetic seasonal curve + typhoon bump + random storm events | **real hourly weather** (wind gusts + rain → 0-10 severity), 1 Jan 2025 - 1 Mar 2026; synthetic storm events removed | Open-Meteo (ERA5), CC BY 4.0 |
| Busy and quiet months | one synthetic Aug-Oct peak for every node | **each node's real monthly traffic** vs its yearly mean | IMF PortWatch daily port calls / transits |
| Port size | one size per node type | **each node's real traffic** vs the average node scales throughput and yard | IMF PortWatch |
| Global supply-chain pressure | sine wave | **real monthly GSCPI** | NY Fed GSCPI |

Rail hub and distribution hubs are not in PortWatch, so they get neutral factors. The **shipment outcomes are still simulated** from these drivers, so rows stay `data_origin = SYNTHETIC_CALIBRATED`. Old drivers: `python scripts/build_sim_data.py --synthetic-drivers`.

## What it is
Every row is **simulated** (`data_origin = SYNTHETIC_CALIBRATED`, `is_synthetic = True`). We do not claim live or real shipment data.

We use the **fallback design** allowed by the Role 3 file (section 6.3), not the full fluid-queue network:
- Each of the 27 nodes has a hidden **congestion level** `c` that moves hour by hour towards a target (AR(1)-style mean reversion, 5% per hour, plus small shocks).
- The target rises with real weather, labour shortage, customs holds, equipment loss, each node's real busy months, the real GSCPI, **events**, and **congestion arriving from upstream nodes**. Upstream congestion arrives after the lane's real `typical_transit_days` from `contracts/network_edges.csv`, so cascades travel along the network.
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

The first two parts are added together, then passed through a curve `0.7s + 0.14s²` and multiplied by hidden carrier and priority factors. The hidden factors are added on top.

Planted issues and measurement noise affect only the feature columns. The label itself is not noised.

## Tuned once (never re-tune to flatter a model)
The weights 1.5 and 0.15, the curve 0.7 and 0.14, and the hidden-noise sizes were tuned on 29 Oct 2026 to hit the target class mix:

| | Target | Achieved |
|---|---|---|
| ON_TIME (<6 h) | 65% | 65.1% (v0.3, real drivers, no re-tuning) |
| AT_RISK (6-24 h) | 22% | 21.8% |
| CRITICAL_DELAY (≥24 h) | 13% | 13.1% |

**Sanity check:** quick untuned models were trained on Jan to early Sep and tested on late Oct to Dec. A depth-4 tree scored a macro-F1 of 0.59 and a default gradient-boosting model 0.67. The label is hard but learnable, inside the role file's expected 0.6-0.85, and far from the 0.95 that would signal a leak.

**v0.2 change (seasonality):** in v0.1 the seasonal swings were so large that September was 85% delayed and Nov-Dec 90% on time, which would have made the Stage 1 test split (Nov-Dec) almost all ON_TIME. v0.2 softens winter weather (amplitude 1.0 → 0.4), the typhoon season (2.0 → 0.8), peak season (0.25 → 0.08) and the global-stress wave (0.1 → 0.03, now one cycle per year), then re-tunes the curve once. The monthly ON_TIME share now stays between 57% and 80%.

## Outputs (`data/raw/sim/`)
| File | Rows | Notes |
|---|---|---|
| `shipment_telemetry_raw.csv.gz` | 60,100 | 60,000 shipments plus 100 planted duplicate rows. `delay_status` and `split` are empty because Stage01 `01_data_engineer.py` sets them |
| `node_hourly_raw.csv.gz` | 234,142 | 27 nodes × 8,760 h, with about 1% of hours deliberately missing for Stage 2 to rebuild |
| `events_ground_truth.csv` | 67 | `event_id, type, node_id, start_ts, end_ts, severity, cause` (includes the 60-day run-out after 2025; no storms in v0.3, real weather has them) |
| `sim_summary.json` | - | Seed, counts and class mix |
| `INJECTED_ISSUES.json` | - | **Private.** Commit it only after Role 2 reports Day 1 findings |

## Known simplifications (say these openly in the debate)
1. **Real drivers, simulated outcomes.** Weather, busy months, port size and GSCPI are real (v0.3). Still simulated:
   - the shipment outcomes (delays), because shipment-level delay records are not public;
   - labour, customs and equipment levels, and the disruption events (no public hourly source);
   - base dwell per node type (rough public orders of magnitude).
   - The 2025 Red Sea diversion is not modelled as a regime; PortWatch shows it, so it is a candidate for the next calibration step.
2. The event catalogue frequencies are our placeholder values until Role 2's `docs/decisions/0001-event-catalogue.md` is transcribed.
3. Shipment routes are shortest paths, with no rerouting during disruptions.
4. Chokepoint "throughput" is TEU-equivalent and has no real vessel-transit conversion yet.
5. `historical_delay_rate` uses outcomes from 7-37 days before each snapshot, so only outcomes that would already be known count. The first month falls back to the overall early rate.
