"""ROLE: DATA ENGINEER  |  DAY 0 - FOUNDATIONS
Downloads real public data and turns it into small calibration files for the simulator.

    python scripts/fetch_real_data.py            # uses cached downloads in data/raw/real/ when present
    python scripts/fetch_real_data.py --refresh  # download everything again

SOURCES (verified 29 Oct 2026, see docs/06_DATA_SOURCES.md):
    IMF PortWatch   - daily container port calls (Daily_Ports_Data) and chokepoint transits (Daily_Chokepoints_Data)
    NY Fed GSCPI    - monthly Global Supply Chain Pressure Index (Excel file)
    Open-Meteo      - hourly ERA5 wind gusts and precipitation at every node's coordinates (CC BY 4.0)

WRITES (raw downloads, git-ignored): data/raw/real/*
WRITES (committed, small):          data/calibration/activity_profile.csv
                                    data/calibration/real_event_anchors.csv
                                    data/calibration/gscpi_monthly.csv
                                    data/calibration/weather_hourly.csv.gz
OWNER : Role 3 - Data & Integration Engineer
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Make "import common" work when this file is run from the repo root or from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import NODES_CSV, REPO_ROOT  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw" / "real"
CAL_DIR = REPO_ROOT / "data" / "calibration"

# PortWatch bulk CSV download (the same link as the "Download" button on portwatch.imf.org).
PORTWATCH_DOWNLOAD = "https://portwatch.imf.org/api/download/v1/items/{item_id}/csv?layers=0"
PORTS_ITEM = "83b1bbc7b3354c5fb1f40673bb8f852e"    # Daily_Ports_Data
CHOKES_ITEM = "3da2b9ca97684916b75c4013f95d18ab"   # Daily_Chokepoints_Data
GSCPI_URL = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"

# Our node id -> PortWatch port id (looked up in PortWatch_ports_database on 29 Oct 2026).
# Los Angeles and Long Beach are ONE port in PortWatch ("Los Angeles-Long Beach"), so both share it.
PORT_IDS = {
    "PORT_CN_SHA": "port1188", "PORT_CN_NGB": "port824", "PORT_CN_YTN": "port1414", "PORT_KR_PUS": "port1065",
    "PORT_SG_SIN": "port1201", "PORT_MY_PKG": "port960", "PORT_LK_CMB": "port254", "PORT_IN_NSA": "port776",
    "PORT_IN_MAA": "port235", "PORT_AE_JEA": "port744", "PORT_NL_RTM": "port1114", "PORT_BE_ANR": "port57",
    "PORT_DE_HAM": "port446", "PORT_US_LAX": "port664", "PORT_US_LGB": "port664", "PORT_US_OAK": "port839",
    "PORT_US_NYC": "port815", "PORT_US_SAV": "port1170",
}
# Our chokepoint node id -> PortWatch chokepoint id (from Daily_Chokepoints_Data).
CHOKE_IDS = {
    "CHOKE_SUEZ": "chokepoint1", "CHOKE_PANAMA": "chokepoint2", "CHOKE_BAB_EL_MANDEB": "chokepoint4",
    "CHOKE_MALACCA": "chokepoint5", "CHOKE_HORMUZ": "chokepoint6", "CHOKE_CAPE_GOOD_HOPE": "chokepoint7",
}

# Well-documented real disruptions. The date windows are our choice (from public reporting);
# the % change is MEASURED from PortWatch against the 8 weeks before the start.
EVENTS = [
    ("SUEZ_EVER_GIVEN_2021", "CHOKE_SUEZ", "2021-03-23", "2021-03-29"),
    ("LA_LB_BACKLOG_2021", "PORT_US_LAX", "2021-09-01", "2021-11-30"),
    ("SHANGHAI_LOCKDOWN_2022", "PORT_CN_SHA", "2022-04-01", "2022-05-31"),
    ("PANAMA_DROUGHT_2023", "CHOKE_PANAMA", "2023-08-01", "2024-03-31"),
    ("RED_SEA_DIVERSION_2024", "CHOKE_BAB_EL_MANDEB", "2024-01-01", "2024-06-30"),
    ("RED_SEA_DIVERSION_2024", "CHOKE_SUEZ", "2024-01-01", "2024-06-30"),
    ("RED_SEA_DIVERSION_2024", "CHOKE_CAPE_GOOD_HOPE", "2024-01-01", "2024-06-30"),
]


def get_json(url: str, params: dict, tries: int = 10) -> dict:
    """GET a URL and parse JSON, retrying with growing pauses if the server is briefly unavailable."""
    full = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(full, timeout=120) as resp:
                data = json.load(resp)
            if "error" not in data:
                return data
        except (OSError, ValueError):
            pass
        time.sleep(3 * (attempt + 1))  # wait 3, 6, 9 ... seconds before trying again
    raise RuntimeError(f"Gave up after {tries} tries: {full[:150]}")


def download_portwatch_csv(item_id: str, ids: set, keep_cols: list, out: Path, refresh: bool) -> pd.DataFrame:
    """Download a PortWatch dataset as one bulk CSV, keep only our ids, and save that small file.

    Why bulk: the PortWatch query service often answers with empty or partial pages without an error,
    which silently lost whole ports in our first attempt. The bulk CSV (the site's own download button)
    is complete. The full ports file is about 650 MB, so it is filtered in pieces and then deleted.
    """
    if refresh or not out.exists():
        big = out.with_suffix(".full.csv")
        # A finished big file from an earlier run is reused (unless --refresh), so a crash after the
        # 10-minute download does not mean downloading again.
        if refresh or not big.exists():
            print(f"  PortWatch bulk download {item_id} (can take ~10 min) ...")
            urllib.request.urlretrieve(PORTWATCH_DOWNLOAD.format(item_id=item_id), big)
        # Read 500,000 rows at a time so the laptop's memory never holds the whole file.
        pieces = [chunk[chunk["portid"].isin(ids)] for chunk in
                  pd.read_csv(big, usecols=keep_cols, encoding="utf-8-sig", chunksize=500_000)]
        pd.concat(pieces, ignore_index=True).to_csv(out, index=False)
        big.unlink()
    df = pd.read_csv(out)
    # Check every id really arrived, so a broken download can never pass silently.
    missing = ids - set(df["portid"])
    if missing:
        raise RuntimeError(f"PortWatch download is missing {sorted(missing)}; re-run with --refresh")
    return df


def fetch_portwatch(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily container port calls for our ports and daily transits for our chokepoints (2019 onwards)."""
    ports = download_portwatch_csv(PORTS_ITEM, set(PORT_IDS.values()),
                                   ["date", "portid", "portname", "portcalls_container"],
                                   RAW_DIR / "portwatch_ports_daily.csv", refresh)
    chokes = download_portwatch_csv(CHOKES_ITEM, set(CHOKE_IDS.values()),
                                    ["date", "portid", "portname", "n_container", "n_total"],
                                    RAW_DIR / "portwatch_chokepoints_daily.csv", refresh)
    return ports, chokes


def node_daily_activity(ports: pd.DataFrame, chokes: pd.DataFrame) -> pd.DataFrame:
    """One tidy table: node_id, date, activity (container calls for ports, container transits for chokepoints)."""
    parts = []
    for node, pid in PORT_IDS.items():
        g = ports[ports["portid"] == pid][["date", "portcalls_container"]].rename(columns={"portcalls_container": "activity"})
        parts.append(g.assign(node_id=node, source_id=pid))
    for node, cid in CHOKE_IDS.items():
        g = chokes[chokes["portid"] == cid][["date", "n_container"]].rename(columns={"n_container": "activity"})
        parts.append(g.assign(node_id=node, source_id=cid))
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates(["node_id", "date"]).sort_values(["node_id", "date"])


def activity_profile(daily: pd.DataFrame) -> pd.DataFrame:
    """Per node: average 2025 activity, size relative to its node type, 2025-vs-2024 trend, weekday and month shape."""
    rows = []
    for node, g in daily.groupby("node_id"):
        y25 = g[g["date"].dt.year == 2025]
        y24 = g[g["date"].dt.year == 2024]
        mean25 = y25["activity"].mean()
        row = {"node_id": node, "source_id": g["source_id"].iloc[0], "days_2025": len(y25),
               "mean_daily_2025": round(mean25, 3),
               # Above 1 = busier in 2025 than in 2024.
               "trend_2025_vs_2024": round(mean25 / y24["activity"].mean(), 4) if len(y24) else np.nan}
        # Weekday shape: average of each weekday divided by the overall average (1.0 = a normal day).
        for wd, v in (y25.groupby(y25["date"].dt.day_of_week)["activity"].mean() / mean25).items():
            row[f"weekday_{wd}"] = round(v, 4)
        # Month shape: same idea per calendar month (real seasonality, used to check the simulator's).
        for m, v in (y25.groupby(y25["date"].dt.month)["activity"].mean() / mean25).items():
            row[f"month_{m:02d}"] = round(v, 4)
        rows.append(row)
    prof = pd.DataFrame(rows)
    # Relative size inside each node type (ports vs chokepoints are counted differently).
    is_port = prof["node_id"].str.startswith("PORT_")
    for mask in (is_port, ~is_port):
        prof.loc[mask, "relative_size"] = (prof.loc[mask, "mean_daily_2025"] / prof.loc[mask, "mean_daily_2025"].median()).round(4)
    return prof


def event_anchors(daily: pd.DataFrame) -> pd.DataFrame:
    """Measure each real event: average activity during it vs the 8 weeks just before it."""
    rows = []
    for event, node, start, end in EVENTS:
        g = daily[daily["node_id"] == node].set_index("date")["activity"]
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        before = g[(g.index >= s - pd.Timedelta(weeks=8)) & (g.index < s)]
        during = g[(g.index >= s) & (g.index <= e)]
        rows.append({
            "event": event, "node_id": node, "start": start, "end": end,
            "baseline_mean_8w": round(before.mean(), 3), "event_mean": round(during.mean(), 3),
            "pct_change": round(100 * (during.mean() / before.mean() - 1), 1) if before.mean() > 0 else np.nan,
            "days_baseline": len(before), "days_event": len(during),
            "source": "IMF PortWatch (container calls / transits)",
        })
    return pd.DataFrame(rows)


def fetch_gscpi(refresh: bool) -> pd.DataFrame:
    """Monthly GSCPI values (standard deviations from the long-run average; 0 = normal pressure)."""
    raw = RAW_DIR / "gscpi_data.xls"
    if refresh or not raw.exists():
        print("  NY Fed GSCPI ...")
        urllib.request.urlretrieve(GSCPI_URL, raw)
    df = pd.read_excel(raw, sheet_name="GSCPI Monthly Data", usecols=[0, 1]).dropna()
    df.columns = ["month_end", "gscpi"]
    df["month"] = pd.to_datetime(df["month_end"], format="%d-%b-%Y").dt.strftime("%Y-%m")
    return df[["month", "gscpi"]].round(4)


def fetch_weather(nodes: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    """Hourly 2025 wind gusts and rain at every node, plus the weather_severity_index from the role file."""
    folder = RAW_DIR / "open_meteo"
    folder.mkdir(exist_ok=True)
    parts = []
    for node in nodes.itertuples():
        path = folder / f"{node.node_id}.json"
        if refresh or not path.exists():
            print(f"  Open-Meteo {node.node_id} ...")
            data = get_json(OPEN_METEO, {"latitude": node.lat, "longitude": node.lon, "start_date": "2025-01-01",
                                         "end_date": "2025-12-31", "hourly": "wind_gusts_10m,precipitation",
                                         "timezone": "UTC"})
            path.write_text(json.dumps(data))
            time.sleep(1)  # be polite to the free API
        h = json.loads(path.read_text())["hourly"]
        parts.append(pd.DataFrame({"node_id": node.node_id, "hour_ts": h["time"],
                                   "wind_gust_kmh": h["wind_gusts_10m"], "precip_mm_h": h["precipitation"]}))
    w = pd.concat(parts, ignore_index=True)
    w["hour_ts"] = pd.to_datetime(w["hour_ts"]).dt.strftime("%Y-%m-%dT%H:00:00Z")
    # Role file formula: 10 * min(1, 0.6*gust/90 + 0.4*rain/20). A 90 km/h gust alone gives 6 out of 10.
    gust, rain = w["wind_gust_kmh"].fillna(0), w["precip_mm_h"].fillna(0)
    w["weather_severity_index"] = (10 * np.minimum(1, 0.6 * gust / 90 + 0.4 * rain / 20)).round(3)
    return w


def main() -> int:
    """Download all sources (or use the cache) and write the calibration files."""
    parser = argparse.ArgumentParser(description="Fetch real data and build calibration files.")
    parser.add_argument("--refresh", action="store_true", help="download again even if cached")
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(NODES_CSV)

    # Step 1: port and chokepoint activity from IMF PortWatch.
    ports, chokes = fetch_portwatch(args.refresh)
    daily = node_daily_activity(ports, chokes)
    activity_profile(daily).to_csv(CAL_DIR / "activity_profile.csv", index=False)
    event_anchors(daily).to_csv(CAL_DIR / "real_event_anchors.csv", index=False)

    # Step 2: monthly global supply-chain pressure.
    fetch_gscpi(args.refresh).to_csv(CAL_DIR / "gscpi_monthly.csv", index=False)

    # Step 3: hourly real weather at all 27 nodes (mtime=0 keeps the .gz identical between runs).
    fetch_weather(nodes, args.refresh).to_csv(CAL_DIR / "weather_hourly.csv.gz", index=False,
                                              compression={"method": "gzip", "mtime": 0})

    for f in sorted(CAL_DIR.glob("*.*")):
        print(f"wrote {f.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
