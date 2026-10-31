"""ROLE: INTEGRATION ENGINEER  |  STAGE 02 - DEEP LEARNING
Owner: Role 3 - Data & Integration Engineer (integration hat).
Built by Role 4 at the team lead's request so Stage 2 is complete end to end; Role 3 to review, run and deploy.
Full instructions: docs/roles/ROLE_3_DATA_INTEGRATION_ENGINEER.md (Day 2).

WHAT THIS FILE DOES
    1. A FastAPI service that forecasts a node's dwell and throughput 1 h to 7 days ahead with 80% intervals.
       Start it from the stage folder:   uvicorn 05_integration_engineer:app --port 8002
         GET  /health, /nodes, /network, /models
         POST /forecast                body = {"node_id": ..., "records": [72 consecutive hourly readings]}
         GET  /forecast/{node_id}?as_of=2025-12-01T00:00:00Z   replay from the stored hourly grid (demo / dashboard)
       Graph models also need every other node's recent history; the API takes it from the stored grid for the
       same hour and swaps in the posted node's 72 hours. Outside the stored period, use ?model=lstm.
    2. Run as a script (python 05_integration_engineer.py) it measures the API's own latency with 200 in-process
       calls per model and writes Stage02_DL/reports/integ_latency.json.

READS : Stage02_DL/artifacts/*.pt                                (Role 1: weights + interval margins)
        Stage02_DL/reports/eval_champion.json, eval_alert_thresholds.csv   (Role 4: default model, alert thresholds)
        Stage02_DL/data/hourly_node_telemetry.csv.gz, scaler.json  (Role 3: network history and scaling)
        contracts/network_nodes.csv, network_edges.csv, schemas/hourly_node_telemetry.schema.json
WRITES: Stage02_DL/reports/integ_latency.json (script mode only)

SAFETY: decision support only. Nothing here books or reroutes anything; every response carries
requires_operator_review: true, the model version and the evidence used.
"""

import os

# PyTorch and XGBoost-based code may share this process; one OpenMP thread avoids the macOS deadlock (see 03).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List, Optional  # noqa: E402

# Make "import common" work when this file is run from inside the stage folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field, create_model, field_validator  # noqa: E402

from common.contracts import load_schema  # noqa: E402
from common.dataio.hourly_grid import CHANNELS  # noqa: E402
from common.paths import EDGES_CSV, NODES_CSV, stage_path  # noqa: E402
from common.seed import set_global_seed  # noqa: E402

STAGE = 2
INPUT_HOURS = 72                  # the models read exactly the last 72 hours (role file, Day 2)
ALERT_HORIZON_H = 24              # alert_prob_24h = chance dwell is above the node's threshold 24 h ahead
LATENCY_CALLS = 200
MODELS = {"lstm": "lstm_forecaster", "transformer": "transformer_forecaster", "graph_full": "graph_forecaster",
          "graph_shuffled": "graph_shuffled", "graph_none": "graph_none"}


# ------------------------------------------------------------------ request body built from the contract

def _record_model() -> type[BaseModel]:
    """One hourly reading; ranges come from contracts/schemas/hourly_node_telemetry.schema.json."""
    cols = {c["name"]: c for c in load_schema("hourly_node_telemetry")["columns"]}
    fields = {name: (float, Field(..., ge=cols[name].get("min"), le=cols[name].get("max"))) for name in CHANNELS}
    fields["hour_ts"] = (str, Field(..., description="top of the hour, ISO-8601 UTC"))
    fields["is_imputed"] = (bool, False)
    return create_model("HourlyRecord", **fields)


HourlyRecord = _record_model()


class ForecastRequest(BaseModel):
    """The last 72 hours of ONE node, oldest first."""

    node_id: str
    records: List[HourlyRecord]

    @field_validator("node_id")
    @classmethod
    def _known_node(cls, value: str) -> str:
        if value not in set(pd.read_csv(NODES_CSV)["node_id"]):
            raise ValueError(f"unknown node id '{value}' (see contracts/network_nodes.csv)")
        return value

    @field_validator("records")
    @classmethod
    def _exactly_72_consecutive_hours(cls, records: list) -> list:
        # A gap or a wrong order would silently glue two different days together, so refuse it with a clear message.
        if len(records) != INPUT_HOURS:
            raise ValueError(f"send exactly {INPUT_HOURS} hourly records (got {len(records)})")
        ts = pd.to_datetime([r.hour_ts for r in records], utc=True)
        if (ts != ts.floor("h")).any():
            raise ValueError("every hour_ts must be the top of an hour")
        # Compare real time differences (not raw integers: pandas may store seconds or nanoseconds).
        if not ((ts[1:] - ts[:-1]) == pd.Timedelta(hours=1)).all():
            raise ValueError("records must be consecutive hours, oldest first, with no gaps or repeats")
        return records


# ------------------------------------------------------------------ the forecasting service

class Service:
    """Loads the models, the stored grid, the champion choice and the alert thresholds once per process."""

    def __init__(self):
        from common.modelkit.dl_infer import Stage2Forecaster  # PyTorch is only needed once the service starts

        self.node_ids = pd.read_csv(NODES_CSV)["node_id"].tolist()
        self.forecasters = {}
        for name, artifact in MODELS.items():
            if stage_path(STAGE, "artifacts", f"{artifact}.pt").exists():
                self.forecasters[name] = Stage2Forecaster.from_artifact(artifact)
        if not self.forecasters:
            raise RuntimeError("No Stage 2 model artefacts found; run Stage02_DL/03_dl_engineer.py first.")
        champ_file = stage_path(STAGE, "reports", "eval_champion.json")
        champ = json.loads(champ_file.read_text()) if champ_file.exists() else {}
        # Role 4's validation-chosen champion is the default; before evaluation exists, fall back to the LSTM.
        self.champion = champ.get("model") if champ.get("model") in self.forecasters else next(iter(self.forecasters))
        self.champion_file = champ
        thr_file = stage_path(STAGE, "reports", "eval_alert_thresholds.csv")
        self.thresholds = pd.read_csv(thr_file, index_col=0).iloc[:, 0].to_dict() if thr_file.exists() else {}
        self.grid = pd.read_csv(stage_path(STAGE, "data", "hourly_node_telemetry.csv.gz"))
        self.grid["hour_ts"] = pd.to_datetime(self.grid["hour_ts"], utc=True)
        self.grid_hours = (self.grid["hour_ts"].min(), self.grid["hour_ts"].max())

    def pick(self, model: Optional[str]):
        """Return (name, forecaster); an unknown model is a clear error, not a crash."""
        name = model or self.champion
        if name not in self.forecasters:
            raise ValueError(f"unknown model '{name}'; available: {sorted(self.forecasters)}")
        return name, self.forecasters[name]

    def run(self, node_id: str, model: Optional[str], records: Optional[pd.DataFrame] = None, as_of=None) -> dict:
        """Forecast one node from posted records (POST) or from the stored grid at `as_of` (GET replay)."""
        from common.modelkit.dl_infer import sim_window, single_node_window

        name, f = self.pick(model)
        last_hour = pd.Timestamp(records["hour_ts"].iloc[-1]) if records is not None else pd.Timestamp(as_of)
        last_hour = last_hour.tz_localize("UTC") if last_hour.tzinfo is None else last_hour.tz_convert("UTC")
        if records is not None and not f.needs_network:
            data, origin = single_node_window(records, node_id, self.node_ids, f.scaler)
            source = "posted 72 hours of this node only"
        else:
            if not (self.grid_hours[0] + pd.Timedelta(hours=INPUT_HOURS - 1) <= last_hour <= self.grid_hours[1]):
                raise ValueError(f"'{name}' needs the network's history at {last_hour:%Y-%m-%dT%H:%MZ}, which is outside the "
                                 f"stored grid ({self.grid_hours[0]:%Y-%m-%d} .. {self.grid_hours[1]:%Y-%m-%d}); use ?model=lstm")
            data, origin = sim_window(self.grid, self.node_ids, f.scaler, last_hour, override=records, override_node=node_id)
            source = ("posted 72 hours of this node + stored history of the other nodes" if records is not None
                      else "stored hourly grid (replay)")
        out = f.forecast(data, origin)
        i = self.node_ids.index(node_id)
        q = out["q"][i]                                                       # [H, T, 3]
        horizons = []
        for hi, h in enumerate(out["horizons_h"]):
            entry = {"horizon_h": int(h), "target_ts": (last_hour + pd.Timedelta(hours=int(h))).strftime("%Y-%m-%dT%H:%M:%SZ")}
            for ti, t in enumerate(out["targets"]):
                entry[t] = {"q10": round(float(q[hi, ti, 0]), 3), "q50": round(float(q[hi, ti, 1]), 3), "q90": round(float(q[hi, ti, 2]), 3)}
            horizons.append(entry)
        thr = self.thresholds.get(node_id)
        alert_prob = None
        if thr is not None and ALERT_HORIZON_H in out["horizons_h"]:
            from common.evalkit.forecast import exceedance_probability

            d = q[out["horizons_h"].index(ALERT_HORIZON_H), out["targets"].index("dwell")]
            alert_prob = round(float(exceedance_probability(d[0], d[1], d[2], thr)), 4)
        return {"node_id": node_id, "origin_ts": last_hour.strftime("%Y-%m-%dT%H:%M:%SZ"), "model_name": name,
                "model_version": f.meta.get("model_version"), "horizons": horizons,
                "interval": "80% (10th-90th percentile, split-conformal corrected on validation)",
                "alert_threshold_dwell_h": None if thr is None else round(float(thr), 3), "alert_prob_24h": alert_prob,
                "evidence": {"inputs": source, "threshold_source": "Stage02_DL/reports/eval_alert_thresholds.csv (node p95 of train dwell)",
                             "champion_chosen_on": self.champion_file.get("chosen_on")},
                "data_origin": "SYNTHETIC_CALIBRATED", "requires_operator_review": True}


@lru_cache(maxsize=1)
def service() -> Service:
    """Load everything once per process."""
    return Service()


app = FastAPI(title="CASCADE Stage 2 - 7-day node forecasts",
              description="Decision support only: every answer requires operator review.")


@app.get("/health")
def health() -> dict:
    """Liveness check and which models are loaded."""
    s = service()
    return {"status": "ok", "models": sorted(s.forecasters), "champion": s.champion, "requires_operator_review": True}


@app.get("/nodes")
def nodes() -> dict:
    """The node registry (contracts/network_nodes.csv)."""
    return {"nodes": pd.read_csv(NODES_CSV).fillna("").to_dict(orient="records")}


@app.get("/network")
def network() -> dict:
    """Nodes and lanes, for drawing the map."""
    return {"nodes": pd.read_csv(NODES_CSV).fillna("").to_dict(orient="records"),
            "edges": pd.read_csv(EDGES_CSV).to_dict(orient="records")}


@app.get("/models")
def models() -> dict:
    """Every servable model, its version and whether it needs the network history (graph models do)."""
    s = service()
    return {"champion": s.champion, "models": [{"model_name": n, "model_version": f.meta.get("model_version"),
                                                "needs_network_history": f.needs_network, "is_champion": n == s.champion}
                                               for n, f in s.forecasters.items()]}


@app.post("/forecast")
def forecast(body: ForecastRequest, model: Optional[str] = None) -> dict:
    """Forecast one node from its last 72 hourly readings. Optional ?model=lstm|transformer|graph_full|..."""
    records = pd.DataFrame([r.model_dump() for r in body.records])
    try:
        return service().run(body.node_id, model, records=records)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from None


@app.get("/forecast/{node_id}")
def forecast_replay(node_id: str, as_of: str, model: Optional[str] = None) -> dict:
    """Replay: forecast a node as if it were `as_of`, using the stored hourly grid (for demos and the dashboard)."""
    if node_id not in service().node_ids:
        raise HTTPException(status_code=422, detail=f"unknown node id '{node_id}'")
    try:
        return service().run(node_id, model, as_of=as_of)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from None


# ------------------------------------------------------------------ script mode: latency

def sample_bodies(n: int) -> list:
    """Real test-period windows from the stored grid, shaped as POST /forecast bodies."""
    s = service()
    rng = np.random.default_rng(42)
    test_start = pd.Timestamp(json.loads(stage_path(STAGE, "data", "splits.json").read_text())["test_start"])
    hours = s.grid.loc[s.grid["hour_ts"] >= test_start + pd.Timedelta(hours=INPUT_HOURS), "hour_ts"].unique()
    bodies = []
    for _ in range(n):
        node = s.node_ids[rng.integers(len(s.node_ids))]
        end = pd.Timestamp(hours[rng.integers(len(hours))])
        rows = s.grid[(s.grid["node_id"] == node) & s.grid["hour_ts"].between(end - pd.Timedelta(hours=INPUT_HOURS - 1), end)]
        recs = rows[["hour_ts", *CHANNELS, "is_imputed"]].copy()
        recs["hour_ts"] = recs["hour_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        recs["is_imputed"] = recs["is_imputed"].astype(str).str.lower().eq("true")
        bodies.append({"node_id": node, "records": recs.to_dict(orient="records")})
    return bodies


def measure_service_latency(bodies: list) -> dict:
    """Fallback when FastAPI's test client is not installed: time the forecasting service itself (no HTTP layer)."""
    result = {"calls_per_model": len(bodies), "date": date.today().isoformat(), "models": {},
              "method": "direct Service.run calls (no HTTP: starlette's test client needs httpx2, not installed here)"}
    for name in service().forecasters:
        times = []
        for body in bodies:
            records = pd.DataFrame(body["records"])
            start = time.perf_counter()
            answer = service().run(body["node_id"], name, records=records)
            times.append((time.perf_counter() - start) * 1000)
            assert answer["requires_operator_review"] is True
        ms = np.array(times)
        result["models"][name] = {"p50_ms": round(float(np.percentile(ms, 50)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2)}
    return result


def measure_latency(n: int = LATENCY_CALLS) -> dict:
    """Time n POST /forecast calls per model through the full HTTP stack (in-process test client)."""
    bodies = sample_bodies(n)
    try:
        from fastapi.testclient import TestClient
    except RuntimeError:
        return measure_service_latency(bodies)

    client = TestClient(app)
    result = {"calls_per_model": n, "method": "FastAPI TestClient, in-process, POST /forecast with 72 test-period hours",
              "date": date.today().isoformat(), "models": {}}
    for name in service().forecasters:
        client.post(f"/forecast?model={name}", json=bodies[0])  # warm-up call, not counted
        times = []
        for body in bodies:
            start = time.perf_counter()
            response = client.post(f"/forecast?model={name}", json=body)
            times.append((time.perf_counter() - start) * 1000)
            assert response.status_code == 200 and response.json()["requires_operator_review"] is True, response.text
        ms = np.array(times)
        result["models"][name] = {"p50_ms": round(float(np.percentile(ms, 50)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2)}
    return result


def main() -> int:
    """Script mode: measure latency and save it for the hand-off."""
    set_global_seed()
    import torch

    torch.set_num_threads(4)
    result = measure_latency()
    out = stage_path(STAGE, "reports", "integ_latency.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
