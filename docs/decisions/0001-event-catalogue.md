# 0001 - Event Catalogue Proposal for Logistics Simulator
Date: 2026-10-29   Deciders: Role 2 (EDA), Role 3 (Data)   Status: PROPOSED

## Context
The CASCADE supply-chain logistics simulator (`common/dataio/simulator.py`) generates synthetic calibrated telemetry, including disruption shocks and congestion waves that cascade across 27 global nodes. To make our stress-testing credible and evidence-based, every simulated shock must correspond to real-world logistics physics rather than arbitrary synthetic spikes. Role 3 requires a grounded catalogue of event types, frequencies, durations, parameter impacts, and historical precedents to calibrate event generation.

## Options
1. **Ad-hoc random perturbations:** Generate random Gaussian/Poisson shocks to dwell, throughput, and utilization without real-world anchoring.
   - *Pros:* Trivial to implement.
   - *Cons:* Fails external review; lacks credibility during debate; ignores correlated cross-node cascades and real duration distributions.
2. **Empirically anchored event catalogue (Recommended):** Define 8 canonical event categories with parameter effect envelopes and historical precedents from maritime, port, and customs records between 2017 and 2024.
   - *Pros:* Grounded in verifiable logistics history; supports realistic multi-node propagation; provides rigorous evidence for debate defense.
   - *Cons:* Requires maintaining parameter bounds and source citations.

## Decision
We adopt the empirically anchored event catalogue below across all stages. Role 3 incorporates these parameter ranges and frequencies into `common/dataio/simulator.py` and `scripts/build_sim_data.py`.

### Event Catalogue Specifications

| Event Type | Frequency (per yr) | Duration Range | Affected Node Types | Operational Parameter Effects | Real-World Precedent & Verified Source |
|---|---|---|---|---|---|
| **1. Customs Surge / Inspection Blitz** | 2 - 4 | 3 - 10 days | Seaports (import gateways: LAX, LGB, RTM, ANR, NSA) | `customs_hold_hours` × 3.0 to 6.0; `dwell_time_hours` × 1.5 to 2.5; gate throughput -25% to -40% | US CBP Uyghur Forced Labor Prevention Act (UFLPA) enforcement sweeps (June 2022–2023), detaining thousands of solar/electronics shipments for 14-30+ days (Source: US CBP Trade Statistics, CRS Report R47196). |
| **2. Labour Strike / Slowdown** | 1 - 3 | 4 - 14 days | Seaports, Rail Hubs (HAM, ANR, RTM, CHI_BNSF) | `labor_deficit_ratio` -> 0.70 to 0.95; throughput -60% to -90%; dwell time × 3.0 to 5.0; utilization reaches 120-140% | Port of Felixstowe 8-day dockworker strike by Unite union (August 21–29, 2022, BBC / Lloyd's List); German seaports 48-hr warning strike by ver.di (July 2022, Reuters). |
| **3. Severe Weather / Typhoon / Storm** | 3 - 6 | 2 - 5 days | Coastal Ports (SHA, NGB, YTN, PUS, SAV, NYC) | `weather_severity_index` -> 7.0 to 10.0; port berth throughput -> 0 (closure); anchorage queue spikes; transit delay +48-96h | Typhoon In-Fa (July 2021) shutting Shanghai and Ningbo-Zhoushan ports for 3 days with >150 anchored vessels (Source: S&P Global Platts, Lloyd's List). |
| **4. Equipment Failure / Crane Outage** | 4 - 8 | 1 - 3 days (24 - 72h) | Individual Terminals / Rail Ramps (LAX, CHI_BNSF, SIN) | `equipment_availability_pct` drops to 35% - 55%; crane moves/hr cut by half; truck gate queues surge | Chicago intermodal rail ramp crane freezes and chassis deficits during extreme polar vortex cold snaps (Jan 2019, Dec 2022, US Surface Transportation Board rail service reports). |
| **5. Chokepoint Blockage** | 0.2 - 0.5 | 6 - 20 days | Maritime Chokepoints (SUEZ, PANAMA, MALACCA, BAB_EL_MANDEB) | Utilization reaches 150% (gridlock); transit throughput drops -80% to -100%; upstream queues cascade by 10-20 days | *Ever Given* grounding blocking the Suez Canal (March 23–29, 2021), stranding ~400 ships and ~$9.6B daily cargo (Suez Canal Authority, Lloyd's List Intelligence). Panama Canal drought draft restrictions (2023–2024, ACP Advisories). |
| **6. Demand Surge / Peak Front-Loading** | 1 - 2 | 30 - 90 days | Import Gateways & Inland DCs (LAX, LGB, RTM, JOLIET) | Inbound volume +30% to +50%; sustained utilization 95% - 110%; yard dwell increases from baseline 3-4 days to 7-10 days | 2021–2022 San Pedro Bay container surge peaking at 109 container vessels at anchor off LA/Long Beach in January 2022 (Source: Marine Exchange of Southern California, Port of LA Monthly Briefings). |
| **7. Cyber Outage / TOS Shutdown** | 0.5 - 1 | 2 - 7 days | Ports & Carriers (Terminal Operating Systems, EDI) | Automated gate/EDI offline; throughput drops -70%; missing/delayed telemetry; manual paperwork backup | DP World Australia cyber attack (Nov 10–13, 2023), halting Sydney, Melbourne, Brisbane, Fremantle terminals with ~30,000 containers stranded (Source: DP World, Australian Cyber Security Centre). NotPetya ransomware disabling APM Terminals globally (June 2017, Maersk Annual Report). |
| **8. Route Diversion / Geopolitical Conflict** | 0.5 - 1 | 60 - 180 days | Chokepoints (BAB_EL_MANDEB, SUEZ) diverted via CAPE_GOOD_HOPE | Transit time +10 to 14 days (+3,500 nm); European port arrival bunching (RTM, ANR, HAM); transit cost and vessel capacity absorption | Red Sea crisis (December 2023–2024), where maritime attacks diverted >80% of Asia-Europe container vessels around the Cape of Good Hope (Source: UNCTAD Policy Brief 106, Clarksons Research). |

## Consequences
- **For Role 3 (Data & Integration Engineer):** `common/dataio/simulator.py` and `events_ground_truth.csv` adhere to these 8 categories, their duration intervals, and their parameter bounds.
- **For Role 2 (EDA Engineer):** EDA seasonality, anomaly detection, and change-point analysis cross-reference against these 8 event signatures.
- **For Role 1 (ML Engineer):** Feature engineering targets non-linear interactions characteristic of these compound events (e.g., weather + customs hold).
- **For Role 4 (Evaluation Engineer):** Scenario stress testing evaluates model robustness specifically against these anchored conditions.
