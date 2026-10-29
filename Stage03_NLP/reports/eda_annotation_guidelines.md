# EDA Annotation Guidelines: Freight Incident Classification & Named Entity Recognition

**Role:** Role 2 (EDA Engineer)  
**Target:** Stage 03 NLP (`incident_messages.csv` gold set curation & evaluator agreement)  
**Status:** DRAFT (Day 0) · For joint adjudication with Role 4 (Evaluation Engineer)  

---

## 1. Urgency Classification Scheme

Every logistics incident message must be assigned exactly one urgency level based on temporal horizon and operational impact:

| Urgency Level | Criteria & Temporal Window | Typical Triggers & Context |
|---|---|---|
| **URGENT** | Disruption **currently in effect** or **imminent within 24 hours**, resulting in direct cargo stoppage, missed connections, port closure, or severe operational delays (>24h). | Active dockworker strike; vessel grounding blocking a fairway; crane failure halting discharge; terminal gates closed; immediate blank sailing with rolled cargo. |
| **WATCH** | Potential, partial, or regional disruption **announced or anticipated within 24 to 72 hours**. Cargo movement ongoing but subject to impending bottlenecks. | Approaching tropical storm or typhoon warning; strike ballot passed with 48h notice; draft restriction tightening next week; planned customs inspection surge announced. |
| **ROUTINE** | Administrative, informational, or scheduled advisory with **no immediate or significant cargo delays**. Also includes resolved incidents and standard operational notices. | Normal gate hours posted; tariff schedule update; demurrage/detention free time policy reminder; notice that operations have fully resumed post-fog. |

---

## 2. Risk Factor Taxonomy

Messages may be tagged with one or more applicable risk factors (multi-label):

1. **WEATHER**: Meteorological phenomena impairing operations (typhoons, hurricanes, sea fog, freezing rain, river low-water levels / drought).
2. **LABOUR**: Industrial actions, work stoppages, slow-walks, union negotiation deadlines, picketing, or acute stevedore/chassis-driver shortages.
3. **CUSTOMS**: Regulatory inspections, non-intrusive VACIS/gamma scans, intensive physical exams, documentation discrepancies, sanction audits, tariff classification holds.
4. **CONGESTION**: Terminal saturation, anchorage queues, berth delays, chassis shortages, container dwell spikes, railhead backups, or truck turn-time surges.
5. **GEOPOLITICAL**: Armed conflict, maritime drone/missile strikes, chokepoint blockades, piracy, canal passage embargoes, sanctions enforcement.

---

## 3. Named Entity Recognition (NER) Span Rules

Annotate exact character spans without including leading/trailing punctuation or articles:

| Entity Type | Definition | Annotation Examples | Exclusions / Notes |
|---|---|---|---|
| `PORT` | Seaports, intermodal rail ramps, distribution centers, or recognised regional clusters/aliases. | `[PORT: Rotterdam]`, `[PORT: PORT_US_LAX]`, `[PORT: San Pedro Bay]`, `[PORT: BNSF Chicago]` | Do not annotate water bodies unless denoting the port complex (e.g., "San Pedro Bay" is accepted as LA/LB). |
| `CARRIER` | Ocean container carriers, rail operators, barge lines, or freight consortia. | `[CARRIER: Maersk]`, `[CARRIER: MSC]`, `[CARRIER: CMA CGM]`, `[CARRIER: BNSF]`, `[CARRIER: ONE]` | Do not annotate forwarders or freight brokers unless operating vessel/rail assets. |
| `DISRUPTION_TYPE` | Specific operational disruption mechanism causing the delay. | `[DISRUPTION_TYPE: crane power failure]`, `[DISRUPTION_TYPE: dockworker walkout]`, `[DISRUPTION_TYPE: dense sea fog]` | Must capture the root event noun phrase, not general adjectives. |
| `MITIGATION_ACTION` | Recommended or enacted corrective routing, operational bypass, or rescheduling. | `[MITIGATION_ACTION: divert via Cape of Good Hope]`, `[MITIGATION_ACTION: omit Hamburg call]`, `[MITIGATION_ACTION: rail bypass to Joliet]` | Action verbs and destinations indicating adaptive management. |

---

## 4. Worked Examples (10 Cases Including Edge Cases & Negations)

### Example 1: Active Terminal Gridlock (URGENT)
> **Text:** `"APM Terminals Maasvlakte II at Rotterdam reports complete yard paralysis following crane power failure; all gate-in appointments suspended until tomorrow morning."`  
> - **Urgency:** `URGENT` (Active disruption, immediate cargo stoppage)  
> - **Risk Factors:** `CONGESTION`  
> - **Entities:** `[PORT: Rotterdam]`, `[DISRUPTION_TYPE: crane power failure]`, `[MITIGATION_ACTION: gate-in appointments suspended]`

### Example 2: Impending Storm Advisory (WATCH)
> **Text:** `"Typhoon Gaemi is forecast to make landfall near Ningbo within 48 hours; terminal operators advise vessels to expedite departure prior to anticipated port closure."`  
> - **Urgency:** `WATCH` (Anticipated within 24-72 hours)  
> - **Risk Factors:** `WEATHER`  
> - **Entities:** `[PORT: Ningbo]`, `[DISRUPTION_TYPE: Typhoon Gaemi]`, `[MITIGATION_ACTION: expedite departure]`

### Example 3: Resolved Weather Event / Negation (ROUTINE)
> **Text:** `"Dense fog conditions at Shanghai Yangshan have lifted as of 06:00 UTC; normal vessel pilotage and berth discharge operations have fully resumed without remaining queue."`  
> - **Urgency:** `ROUTINE` (Event resolved, operations normal; negation of active delay)  
> - **Risk Factors:** `WEATHER`  
> - **Entities:** `[PORT: Shanghai Yangshan]`

### Example 4: Carrier Omission & Rolled Cargo (URGENT)
> **Text:** `"MSC announces vessel MSC Oscar will omit Hamburg due to persistent berth waiting times; export containers will be discharged at Antwerp and rolled to feeder service."`  
> - **Urgency:** `URGENT` (Vessel call omitted, immediate routing change)  
> - **Risk Factors:** `CONGESTION`  
> - **Entities:** `[CARRIER: MSC]`, `[PORT: Hamburg]`, `[PORT: Antwerp]`, `[DISRUPTION_TYPE: persistent berth waiting times]`, `[MITIGATION_ACTION: omit Hamburg]`, `[MITIGATION_ACTION: rolled to feeder service]`

### Example 5: Scheduled Fee Update / Administrative (ROUTINE)
> **Text:** `"Effective November 1, Hapag-Lloyd will implement a revised demurrage and detention tariff across North American ports, extending standard dry container free time to 5 calendar days."`  
> - **Urgency:** `ROUTINE` (Commercial tariff adjustment, no physical cargo stoppage)  
> - **Risk Factors:** *(None)*  
> - **Entities:** `[CARRIER: Hapag-Lloyd]`

### Example 6: Impending Union Strike Vote (WATCH)
> **Text:** `"ILA dockworkers at Savannah have voted to authorize a strike beginning in 72 hours if master contract negotiations fail; cargo owners should prepare inland contingency routings."`  
> - **Urgency:** `WATCH` (Imminent strike window 72 hours away)  
> - **Risk Factors:** `LABOUR`  
> - **Entities:** `[PORT: Savannah]`, `[DISRUPTION_TYPE: strike]`, `[MITIGATION_ACTION: prepare inland contingency routings]`

### Example 7: Regulatory Customs Intensive Exam (URGENT)
> **Text:** `"US CBP has issued VACIS hold notices and mandated intensive examination orders for 40 containers of automotive components arriving at LA/LB on CMA CGM voyage 441N."`  
> - **Urgency:** `URGENT` (Specific shipment holds causing immediate 5-10 day detention)  
> - **Risk Factors:** `CUSTOMS`  
> - **Entities:** `[PORT: LA/LB]`, `[CARRIER: CMA CGM]`, `[DISRUPTION_TYPE: VACIS hold notices]`, `[DISRUPTION_TYPE: intensive examination orders]`

### Example 8: Strategic Geopolitical Diversion (URGENT)
> **Text:** `"Following repeated drone incidents near Bab el-Mandeb, Maersk instructs all eastbound container vessels to divert via Cape of Good Hope, adding approximately 12 days to Asia-Europe transit."`  
> - **Urgency:** `URGENT` (Major transit delay in effect across global trade route)  
> - **Risk Factors:** `GEOPOLITICAL`  
> - **Entities:** `[PORT: Bab el-Mandeb]`, `[CARRIER: Maersk]`, `[PORT: Cape of Good Hope]`, `[DISRUPTION_TYPE: drone incidents]`, `[MITIGATION_ACTION: divert via Cape of Good Hope]`

### Example 9: Inland Railhead Chassis Shortage (WATCH)
> **Text:** `"BNSF Chicago terminal reports critical wheeled chassis availability under 15%; inbound dwell is projected to climb over the next 3 days unless drayage carriers return bare chassis."`  
> - **Urgency:** `WATCH` (Developing inland bottleneck projected over 3 days)  
> - **Risk Factors:** `CONGESTION`  
> - **Entities:** `[PORT: BNSF Chicago]`, `[CARRIER: BNSF]`, `[DISRUPTION_TYPE: critical wheeled chassis availability under 15%]`

### Example 10: False Alarm / False Positive Negation (ROUTINE)
> **Text:** `"Reports of a port-wide customs clearance blackout at Nhava Sheva (JNPT) were erroneous; terminal gate automation and EDI messaging remain fully functional with zero backlog."`  
> - **Urgency:** `ROUTINE` (Refutation of outage rumors; operations completely nominal)  
> - **Risk Factors:** `CUSTOMS`  
> - **Entities:** `[PORT: Nhava Sheva]`, `[PORT: JNPT]`
