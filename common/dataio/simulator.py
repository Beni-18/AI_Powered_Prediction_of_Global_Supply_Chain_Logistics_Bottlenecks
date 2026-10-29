"""ROLE: DATA ENGINEER  |  DAY 0 - FOUNDATIONS
Calibrated logistics simulator: the engine behind every dataset in this project.

This is the FALLBACK design allowed by the role file (section 6.3): every node has a hidden
"congestion level" that follows a mean-reverting AR(1) process, is pushed up by events (storms,
strikes, customs surges ...) and by congestion arriving from upstream nodes along the trade lanes
(with the lane's real transit time as the delay). Shipment delays are then read from the FUTURE
node states along each shipment's remaining route, plus hidden random factors that appear in no
column. That is why no model can predict the label perfectly.

READS : contracts/network_nodes.csv, contracts/network_edges.csv
WRITES: nothing itself - scripts/build_sim_data.py calls these functions and saves the files.
OWNER : Role 3 - Data & Integration Engineer
"""

from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

from common.paths import EDGES_CSV, NODES_CSV

# The simulation year is 2025 (role file 6.3). We simulate 60 extra days after it so that
# shipments snapshotted in late December still have a "future" to travel through.
SIM_START = pd.Timestamp("2025-01-01T00:00:00Z")
SIM_HOURS = 365 * 24
EXTRA_HOURS = 60 * 24

# Ten carriers from the telemetry contract, with a hidden reliability factor each
# (1.0 = average, above 1 = more delay). Made up for the simulation, documented in SIM_CARD.md.
CARRIER_FACTOR = {
    "MSC": 1.05, "MSK": 0.90, "CMA": 1.00, "COSCO": 1.05, "HAPAG": 0.90,
    "ONE": 0.95, "EVERGREEN": 1.00, "HMM": 1.10, "YANGMING": 1.10, "ZIM": 1.20,
}
CARRIER_SHARE = [0.20, 0.17, 0.13, 0.12, 0.08, 0.07, 0.08, 0.05, 0.05, 0.05]

# Priority tiers with realistic proportions, and how much faster each tier is handled.
PRIORITY_SHARE = {"STANDARD": 0.72, "EXPEDITED": 0.18, "PERISHABLE_COLD_CHAIN": 0.07, "CRITICAL_MEDICAL": 0.03}
PRIORITY_FACTOR = {"STANDARD": 1.0, "EXPEDITED": 0.8, "PERISHABLE_COLD_CHAIN": 0.85, "CRITICAL_MEDICAL": 0.7}

# Normal (calm-day) behaviour per node type: dwell in hours, throughput in TEU per hour,
# yard capacity in TEU, typical customs hold in hours. Rough public orders of magnitude.
NODE_TYPE_BASE = {
    "PORT":       {"dwell": 30.0, "throughput": 400.0, "yard": 60000.0, "customs": 4.0},
    "RAIL_HUB":   {"dwell": 20.0, "throughput": 250.0, "yard": 30000.0, "customs": 0.5},
    "DC":         {"dwell": 36.0, "throughput": 120.0, "yard": 15000.0, "customs": 0.0},
    "CHOKEPOINT": {"dwell": 10.0, "throughput": 900.0, "yard": 40000.0, "customs": 0.0},
}

# Event catalogue (fallback values until Role 2's decision note 0001 is transcribed).
# rate = expected events per node per year; dur = duration range in hours; types = node types hit.
EVENT_CATALOGUE = {
    "STORM":               {"rate": 2.0, "dur": (24, 96),   "types": ["PORT", "CHOKEPOINT"], "cause": "WEATHER"},
    "LABOR_STRIKE":        {"rate": 0.4, "dur": (48, 240),  "types": ["PORT", "RAIL_HUB"],   "cause": "LABOR"},
    "CUSTOMS_SURGE":       {"rate": 1.0, "dur": (48, 168),  "types": ["PORT"],               "cause": "CUSTOMS"},
    "EQUIPMENT_FAILURE":   {"rate": 1.2, "dur": (12, 72),   "types": ["PORT", "RAIL_HUB", "DC"], "cause": "EQUIPMENT"},
    "CHOKEPOINT_BLOCKAGE": {"rate": 0.3, "dur": (72, 168),  "types": ["CHOKEPOINT"],         "cause": "CONGESTION"},
    "DEMAND_SURGE":        {"rate": 0.5, "dur": (168, 504), "types": ["PORT", "DC"],         "cause": "CONGESTION"},
    "CYBER_OUTAGE":        {"rate": 0.15, "dur": (12, 72),  "types": ["PORT", "RAIL_HUB"],   "cause": "SECURITY"},
    "ROUTE_DIVERSION":     {"rate": 0.2, "dur": (336, 1008), "types": ["CHOKEPOINT"],        "cause": "SECURITY"},
}


def load_network() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official node registry and trade-lane edges from contracts/."""
    nodes = pd.read_csv(NODES_CSV)
    edges = pd.read_csv(EDGES_CSV)
    return nodes, edges


def _event_shape(length: int) -> np.ndarray:
    """Return an event's strength over its duration: ramp up 12 h, hold, ramp down 12 h (values 0-1)."""
    ramp = min(12, max(1, length // 3))
    shape = np.ones(length)
    shape[:ramp] = np.linspace(0.2, 1.0, ramp)
    shape[-ramp:] = np.linspace(1.0, 0.2, ramp)
    return shape


def sample_events(nodes: pd.DataFrame, total_hours: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw random disruption events for every node from the event catalogue."""
    rows = []
    years = total_hours / 8760
    for etype, spec in EVENT_CATALOGUE.items():
        for _, node in nodes[nodes["node_type"].isin(spec["types"])].iterrows():
            # Poisson count: on average `rate` events per node per year.
            for _ in range(rng.poisson(spec["rate"] * years)):
                start = int(rng.integers(0, total_hours - spec["dur"][0]))
                length = int(rng.integers(spec["dur"][0], spec["dur"][1] + 1))
                rows.append({
                    "event_id": f"EV{len(rows) + 1:05d}",
                    "type": etype,
                    "node_id": node["node_id"],
                    "start_h": start,
                    "end_h": min(start + length, total_hours),
                    # Severity 0.4-1.0 scales how strongly the event pushes the drivers.
                    "severity": round(float(rng.uniform(0.4, 1.0)), 2),
                    "cause": spec["cause"],
                })
    return pd.DataFrame(rows)


def _ar1(n_hours: int, n_nodes: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Mean-zero AR(1) noise: each hour keeps `phi` of the last hour's value plus a small shock."""
    out = np.zeros((n_hours, n_nodes))
    shocks = rng.normal(0, sigma, size=(n_hours, n_nodes))
    for t in range(1, n_hours):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


def simulate_nodes(nodes: pd.DataFrame, edges: pd.DataFrame, events: pd.DataFrame,
                   rng: np.random.Generator) -> dict:
    """Simulate every node for every hour. Returns a dict of (hours x nodes) arrays."""
    H = SIM_HOURS + EXTRA_HOURS
    N = len(nodes)
    idx = {nid: i for i, nid in enumerate(nodes["node_id"])}
    hours = SIM_START + pd.to_timedelta(np.arange(H), unit="h")
    day_of_year = hours.dayofyear.to_numpy()

    # --- 1. Weather: seasonal pattern + slow random wobble, worse for East Asia in typhoon season.
    season = 1.5 + 0.4 * np.cos(2 * np.pi * (day_of_year - 15) / 365)[:, None]  # winter storms
    typhoon = (nodes["region"].eq("East Asia").to_numpy()[None, :]
               * 0.8 * np.exp(-((day_of_year - 245) / 35.0) ** 2)[:, None])     # peaks early Sept
    weather = season + typhoon + _ar1(H, N, 0.98, 0.15, rng)

    # --- 2. Slow drivers around their calm-day level (AR(1) keeps them realistic and smooth).
    labor = 0.03 + _ar1(H, N, 0.995, 0.004, rng)
    customs_base = nodes["node_type"].map(lambda t: NODE_TYPE_BASE[t]["customs"]).to_numpy()
    customs = customs_base[None, :] * (1 + _ar1(H, N, 0.99, 0.05, rng))
    equipment = 93.0 + _ar1(H, N, 0.99, 0.4, rng)
    push = np.zeros((H, N))  # extra congestion pushed in directly by some events

    # --- 3. Add every sampled event on top of the drivers it affects.
    for ev in events.itertuples():
        j = idx[ev.node_id]
        span = slice(ev.start_h, ev.end_h)
        s = _event_shape(ev.end_h - ev.start_h) * ev.severity
        if ev.type == "STORM":
            weather[span, j] += 6.0 * s
        elif ev.type == "LABOR_STRIKE":
            labor[span, j] += 0.55 * s
        elif ev.type == "CUSTOMS_SURGE":
            customs[span, j] += 30.0 * s
        elif ev.type == "EQUIPMENT_FAILURE":
            equipment[span, j] -= 35.0 * s
        elif ev.type == "CYBER_OUTAGE":
            equipment[span, j] -= 55.0 * s
        elif ev.type == "CHOKEPOINT_BLOCKAGE":
            push[span, j] += 2.5 * s
        elif ev.type == "DEMAND_SURGE":
            push[span, j] += 0.8 * s
        elif ev.type == "ROUTE_DIVERSION":
            # Ships avoid this chokepoint and pile up on the long way round the Cape.
            push[span, j] += 1.0 * s
            if "CHOKE_CAPE_GOOD_HOPE" in idx:
                push[span, idx["CHOKE_CAPE_GOOD_HOPE"]] += 0.6 * s

    # Keep every driver inside its physical range (a ratio cannot go below 0, and so on).
    weather = np.clip(weather, 0, 10)
    labor = np.clip(labor, 0, 1)
    customs = np.clip(customs, 0, 240)
    equipment = np.clip(equipment, 0, 100)

    # --- 4. Peak season (Aug-Oct) and a slow global stress index (stand-in for NY Fed GSCPI
    # until scripts/fetch_real_data.py provides the real monthly values).
    peak = 0.08 * np.exp(-((day_of_year - 260) / 40.0) ** 2)
    global_stress = 0.03 * np.sin(2 * np.pi * day_of_year / 365 + 1.0)

    # --- 5. Upstream lanes: for each node, which nodes feed it and after how many hours.
    incoming = {j: [] for j in range(N)}
    for e in edges.itertuples():
        incoming[idx[e.dst]].append((idx[e.src], max(1, int(round(e.typical_transit_days * 24)))))

    # --- 6. Hidden congestion level c (0 = calm, 1 = busy, 2+ = gridlock), hour by hour.
    c = np.zeros((H, N))
    c[0] = 0.3
    shocks = rng.normal(0, 0.02, size=(H, N))
    for t in range(1, H):
        # Congestion that left upstream nodes `lag` hours ago now arrives here.
        upstream = np.zeros(N)
        for j, feeds in incoming.items():
            if feeds:
                vals = [max(0.0, c[t - lag, s] - 0.5) if t - lag >= 0 else 0.0 for s, lag in feeds]
                upstream[j] = np.mean(vals)
        # Where congestion "wants" to be, given today's drivers.
        target = (0.25 + 0.06 * weather[t] + 1.6 * labor[t] + 0.02 * customs[t]
                  + 0.025 * (93.0 - equipment[t]) + push[t] + 0.5 * upstream
                  + peak[t] + global_stress[t])
        # Move 5% of the way towards the target each hour (mean reversion) plus a small shock.
        c[t] = np.maximum(0.0, c[t - 1] + 0.05 * (target - c[t - 1]) + shocks[t])

    # --- 7. Turn hidden congestion into the observable channels the contracts ask for.
    base = {k: nodes["node_type"].map(lambda t: NODE_TYPE_BASE[t][k]).to_numpy()[None, :]
            for k in ("dwell", "throughput", "yard")}
    util = np.clip(45.0 + 40.0 * c, 0, 150)            # utilisation can exceed 100 = real overload
    dwell = np.clip(base["dwell"] * (0.4 + 2.0 * c), 0, 400)
    # Throughput rises with demand until about 100% utilisation, then collapses under gridlock.
    throughput = np.clip(base["throughput"] * np.minimum(util, 100) / 100
                         * (equipment / 100) * (1 - 0.3 * np.clip(c - 1.5, 0, 1)), 0, 2500)
    inventory = np.clip(base["yard"] * util / 100, 0, 500000)
    vessels = np.clip(np.round(2 + 6 * np.maximum(c - 0.5, 0)), 0, 500)

    return {
        "hours": hours, "node_ids": list(nodes["node_id"]), "congestion": c,
        "dwell": dwell, "throughput": throughput, "util": util, "customs": customs,
        "weather": weather, "labor": labor, "equipment": equipment,
        "inventory": inventory, "vessels": vessels,
        # "Normal" dwell = each node's typical (median) dwell over the year; only time above it is delay.
        "nominal_dwell": np.median(dwell[:SIM_HOURS], axis=0),
    }


def shortest_paths(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """Dijkstra over the lanes (weight = transit days). Returns {(src, dst): [node, node, ...]}."""
    graph = {n: [] for n in nodes["node_id"]}
    for e in edges.itertuples():
        graph[e.src].append((e.dst, float(e.typical_transit_days)))
    paths = {}
    for src in graph:
        dist, prev, heap = {src: 0.0}, {}, [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for v, w in graph[u]:
                if d + w < dist.get(v, np.inf):
                    dist[v], prev[v] = d + w, u
                    heapq.heappush(heap, (d + w, v))
        for dst in dist:
            if dst != src:
                path, node = [dst], dst
                while node != src:
                    node = prev[node]
                    path.append(node)
                paths[(src, dst)] = path[::-1]
    return paths


def simulate_shipments(state: dict, nodes: pd.DataFrame, edges: pd.DataFrame, n: int,
                       rng: np.random.Generator, hidden_scale: float = 1.0) -> pd.DataFrame:
    """Sample `n` shipment snapshots and compute each one's realised delay from the future."""
    idx = {nid: i for i, nid in enumerate(state["node_ids"])}
    transit = {(e.src, e.dst): float(e.typical_transit_days) * 24 for e in edges.itertuples()}
    paths = shortest_paths(nodes, edges)

    # Origins are Asian / Middle-East ports; destinations are US/EU ports and inland DCs.
    origins = ["PORT_CN_SHA", "PORT_CN_NGB", "PORT_CN_YTN", "PORT_KR_PUS", "PORT_SG_SIN",
               "PORT_MY_PKG", "PORT_IN_NSA", "PORT_IN_MAA", "PORT_AE_JEA"]
    dests = ["PORT_US_LAX", "PORT_US_LGB", "PORT_US_OAK", "PORT_US_NYC", "PORT_US_SAV",
             "PORT_NL_RTM", "PORT_BE_ANR", "PORT_DE_HAM", "DC_US_MW_JOLIET", "DC_EU_VENLO"]
    lanes = [(o, d) for o in origins for d in dests if (o, d) in paths]

    # Random choices for every shipment at once (fast and reproducible with the seed).
    lane_pick = rng.integers(0, len(lanes), n)
    carriers = rng.choice(list(CARRIER_FACTOR), n, p=CARRIER_SHARE)
    priorities = rng.choice(list(PRIORITY_SHARE), n, p=list(PRIORITY_SHARE.values()))
    snap_hour = rng.integers(0, SIM_HOURS, n)
    pos_u = rng.random(n)

    # Hidden factors (appear in NO column): missed vessel connection, random inspection,
    # yard-gang shortage. Heavy-tailed, so a few shipments get very large surprise delays.
    missed = rng.random(n) < 0.03
    hidden = (rng.exponential(1.0, n) + missed * rng.exponential(20.0, n)
              + (rng.random(n) < 0.03) * rng.exponential(10.0, n)) * hidden_scale

    rows = []
    for i in range(n):
        origin, dest = lanes[lane_pick[i]]
        path = paths[(origin, dest)]
        # The snapshot is taken at a random node on the route that is not the final one.
        k = int(pos_u[i] * (len(path) - 1))
        here, t = path[k], int(snap_hour[i])
        j = idx[here]

        # Realised delay = waiting beyond normal at this node and every node still to come,
        # read from the FUTURE simulated states (the model never sees these).
        here_wait, ahead_wait, clock = 0.0, 0.0, t
        for step, node in enumerate(path[k:]):
            jj = idx[node]
            h = min(clock, len(state["hours"]) - 1)
            # Waiting beyond normal at the current node over the NEXT 24 h (congestion lasts),
            # and at each later node at the hour the shipment will actually arrive there.
            if step == 0:
                here_wait = max(0.0, state["dwell"][h:h + 24, jj].mean() - state["nominal_dwell"][jj])
            else:
                ahead_wait += max(0.0, state["dwell"][h, jj] - state["nominal_dwell"][jj])
            if step + k + 1 < len(path):
                clock += int(transit[(node, path[k + step + 1])]) + int(state["nominal_dwell"][jj])
        # The current node matters most (we are stuck in it); later nodes add a smaller share.
        # The weights (1.5, 0.15) and the curve (0.7, 0.14) were tuned ONCE for the class mix (see SIM_CARD.md).
        # The squared term makes heavy congestion hurt much more than light congestion, like real queues.
        systematic = 1.5 * here_wait + 0.15 * ahead_wait
        delay = (0.7 * systematic + 0.14 * systematic ** 2) * CARRIER_FACTOR[carriers[i]] * PRIORITY_FACTOR[priorities[i]] + hidden[i]

        rows.append({
            "timestamp": state["hours"][t] + pd.Timedelta(minutes=int(rng.integers(0, 60))),
            "origin_port": origin, "destination_port": dest, "current_node_id": here,
            "carrier_code": carriers[i], "priority_tier": priorities[i],
            "dwell_time_hours": state["dwell"][t, j], "throughput_teu_per_hr": state["throughput"][t, j],
            "capacity_utilization_pct": state["util"][t, j], "customs_hold_hours": state["customs"][t, j],
            "weather_severity_index": state["weather"][t, j], "labor_deficit_ratio": state["labor"][t, j],
            "equipment_availability_pct": state["equipment"][t, j],
            "realized_delay_hours": min(delay, 2000.0),
        })
    return pd.DataFrame(rows)


def add_historical_delay_rate(df: pd.DataFrame) -> pd.Series:
    """Trailing delay rate per node: share of shipments seen 7-37 days earlier that ended >= 6 h late.

    The 7-day gap means we only use outcomes that would already be known at snapshot time.
    """
    out = pd.Series(np.nan, index=df.index)
    late = (df["realized_delay_hours"] >= 6).astype(float)
    for _, g in df.groupby("current_node_id"):
        g = g.sort_values("timestamp")
        # Work in whole hours since 1970 so the time arithmetic is plain integer maths.
        ts = (g["timestamp"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(hours=1)
        ts = ts.to_numpy()
        cum = np.concatenate([[0.0], np.cumsum(late.loc[g.index].to_numpy())])
        lo = np.searchsorted(ts, ts - 37 * 24)
        hi = np.searchsorted(ts, ts - 7 * 24)
        count = hi - lo
        rate = np.where(count > 0, (cum[hi] - cum[lo]) / np.maximum(count, 1), np.nan)
        out.loc[g.index] = rate
    # Early-January rows have no history yet: use the overall early rate (documented in SIM_CARD).
    return out.fillna(late.iloc[: max(1, len(df) // 20)].mean()).clip(0, 1)


def node_hourly_table(state: dict) -> pd.DataFrame:
    """Flatten the simulated arrays into one row per node per hour (simulation year only)."""
    H, N = SIM_HOURS, len(state["node_ids"])
    return pd.DataFrame({
        "hour_ts": np.repeat(state["hours"][:H], N),
        "node_id": np.tile(state["node_ids"], H),
        "avg_dwell_time_hours": state["dwell"][:H].ravel(),
        "throughput_teu_per_hr": state["throughput"][:H].ravel(),
        "capacity_utilization_pct": state["util"][:H].ravel(),
        "customs_hold_hours": state["customs"][:H].ravel(),
        "weather_severity_index": state["weather"][:H].ravel(),
        "labor_deficit_ratio": state["labor"][:H].ravel(),
        "equipment_availability_pct": state["equipment"][:H].ravel(),
        "inventory_level_teu": state["inventory"][:H].ravel(),
        "vessels_waiting": state["vessels"][:H].ravel().astype(int),
    })
