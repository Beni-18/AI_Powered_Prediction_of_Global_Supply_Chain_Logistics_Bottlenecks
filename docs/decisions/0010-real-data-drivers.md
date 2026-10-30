# 0010 - Drive the simulator with real data
Date: 2026-10-30   Deciders: Team lead (Bennish), Roles 1, 2, 3   Status: ACCEPTED (team lead's instruction, 2026-10-30; Pravin to review the code)

## Context
The team lead asked to move Phase 1 to real data. Shipment-level delay records (one row per shipment with its real
delay) are not publicly available, so two routes were possible:
1. a new, fully real "port-day" task built on IMF PortWatch traffic (new pipeline, API and dashboard), or
2. keep the shipment-level pipeline and drive the simulator with real data.
The team lead chose the faster route (2).

## Decision
`common/dataio/calibration.py` feeds four real drivers into `common/dataio/simulator.py` (SIM_CARD v0.3):
- **Weather:** real hourly wind gusts and rain at all 27 nodes (Open-Meteo / ERA5, 1 Jan 2025 - 1 Mar 2026),
  turned into the 0-10 severity index; the synthetic storm events are dropped because real weather has real storms.
- **Busy months:** each node's real monthly traffic relative to its yearly mean (IMF PortWatch).
- **Port size:** each node's real traffic relative to the average node scales throughput and yard (IMF PortWatch).
- **Global pressure:** the real monthly NY Fed GSCPI.
Each real driver is scaled to the same size of effect as the synthetic stand-in it replaces, so nothing was re-tuned
to flatter a model; the class mix came out at 65.1 / 21.8 / 13.1 against the 65 / 22 / 13 target.

## Consequences
- All Stage 1 data, models, evaluation, claims and the API were rebuilt; a fresh clone reproduces them exactly.
- Shipment outcomes are still simulated: rows stay `data_origin = SYNTHETIC_CALIBRATED` and we never call it "real data".
- Still synthetic: labour, customs and equipment levels, disruption events, base dwell per node type.
- Next calibration candidates: the 2025 Red Sea diversion regime (visible in PortWatch), and a fully real port-day
  task (route 1) if the team wants real outcomes.
