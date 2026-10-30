"""ROLE: INTEGRATION ENGINEER  |  STAGE 01 - MACHINE LEARNING
Owner: Role 3 - Data & Integration Engineer (integration hat).
Built by Role 4 at the team lead's request to finish Stage 1; Role 3 to review.
Full instructions: docs/roles/ROLE_3_DATA_INTEGRATION_ENGINEER.md (Day 1).

WHAT THIS FILE DOES
    1. A FastAPI service that scores shipments with the champion model chosen by Role 4.
       Start it from the stage folder:   uvicorn 05_integration_engineer:app --port 8001
    2. Run as a script (python 05_integration_engineer.py) it measures the API's own latency with 1,000
       in-process calls and writes the one-page Ops Field Guide for planners.

READS : Stage01_ML/artifacts/stage01_models.joblib, artifacts/model_card.json     (Role 1)
        Stage01_ML/reports/eval_champion.json, eval_metrics.json                   (Role 4: model + alert threshold)
        Stage01_ML/predictions/feature_importance_matrix.csv                       (Role 1: top drivers)
        Stage01_ML/data/prepared_dataset.csv.gz                                    (Role 3: recent node history)
WRITES: Stage01_ML/reports/integ_latency.json, integ_ops_field_guide.md (+ printable .html)

SAFETY: decision support only. Nothing here books, reroutes or changes a shipment, and every response
carries requires_operator_review: true.
"""

import importlib.util
import json
from contextlib import asynccontextmanager
import sys
import time
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

# Make "import common" work when this file is run from inside the stage folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field, create_model, field_validator  # noqa: E402

from common.contracts import load_node_ids, load_schema  # noqa: E402
from common.paths import CONTRACTS_DIR, stage_path  # noqa: E402
from common.seed import set_global_seed  # noqa: E402

STAGE = 1
CLASSES = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
MAX_BATCH = 1000            # from the Role 3 file: batch endpoint takes up to 1,000 rows
ROLLING_WINDOW = 24         # roll_node_dwell_mean_24h = mean of the previous 24 records at the node (Role 1's feature)
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"   # every timestamp is turned into this exact form before Role 1's code parses it
LATENCY_CALLS = 1000        # from the Role 3 file: measure latency with 1,000 calls
NUMERIC_FIELDS = ["dwell_time_hours", "throughput_teu_per_hr", "capacity_utilization_pct", "customs_hold_hours",
                  "weather_severity_index", "labor_deficit_ratio", "equipment_availability_pct", "historical_delay_rate"]


# ------------------------------------------------------------------ request model built from the contract

def _shipment_model() -> type[BaseModel]:
    """Build the request body from contracts/schemas/telemetry.schema.json so the API enforces the same ranges."""
    columns = {c["name"]: c for c in load_schema("telemetry")["columns"]}
    fields = {}
    for name in NUMERIC_FIELDS:
        col = columns[name]
        # Out-of-range values give a clear 422 error instead of a silent wrong score.
        fields[name] = (float, Field(..., ge=col.get("min"), le=col.get("max")))
    for name in ("priority_tier", "carrier_code"):
        fields[name] = (Literal[tuple(columns[name]["allowed"])], ...)
    for name in ("origin_port", "destination_port", "current_node_id"):
        fields[name] = (str, ...)
    fields["timestamp"] = (Optional[str], Field(None, description="ISO time of the snapshot; defaults to now (UTC)"))
    fields["record_id"] = (Optional[str], None)
    return create_model("Shipment", __base__=_NodeCheck, **fields)


class _NodeCheck(BaseModel):
    """Shared validator: node ids must exist in contracts/network_nodes.csv."""

    @field_validator("origin_port", "destination_port", "current_node_id", check_fields=False)
    @classmethod
    def _known_node(cls, value: str) -> str:
        if value not in _node_ids():
            raise ValueError(f"unknown node id '{value}' (see contracts/network_nodes.csv)")
        return value

    @field_validator("timestamp", check_fields=False)
    @classmethod
    def _iso_utc(cls, value: Optional[str]) -> Optional[str]:
        # One fixed format for every row, so a batch that mixes styles (or omits some times) still parses.
        if value is None:
            return None
        try:
            ts = pd.Timestamp(value)
        except (ValueError, TypeError):
            raise ValueError(f"timestamp '{value}' is not an ISO date-time, e.g. 2025-12-01T10:00:00Z") from None
        # A time without a zone is read as UTC, the zone every contract file uses.
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.strftime(TS_FORMAT)


@lru_cache(maxsize=1)
def _node_ids() -> set:
    """Official node ids, read once."""
    return load_node_ids()


Shipment = _shipment_model()


class Batch(BaseModel):
    """Up to 1,000 shipments in one call."""

    shipments: list[Shipment] = Field(..., min_length=1, max_length=MAX_BATCH)


# ------------------------------------------------------------------ loading the model once

def _load_role1_module():
    """Import Role 1's training script so its saved pipeline object (class Stage01Pipeline) can be unpickled."""
    spec = importlib.util.spec_from_file_location("stage01_ml_engineer", Path(__file__).with_name("03_ml_engineer.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The artefact was saved while 03_ml_engineer.py ran as __main__, so pickle looks for __main__.Stage01Pipeline.
    setattr(sys.modules["__main__"], "Stage01Pipeline", module.Stage01Pipeline)
    return module


class Service:
    """Everything the endpoints need: the fitted pipeline, the champion choice, and recent node history."""

    def __init__(self):
        champion_file = stage_path(STAGE, "reports", "eval_champion.json")
        artefact = stage_path(STAGE, "artifacts", "stage01_models.joblib")
        self.card = _read_json(stage_path(STAGE, "artifacts", "model_card.json"))
        self.champion = _read_json(champion_file)
        self.stub = not (artefact.exists() and self.champion)
        if self.stub:
            # Role 3 file: if the artefact is missing, serve a stub and say so loudly.
            self.model_name, self.model_version, self.threshold = "STUB", "STUB", 0.5
            return
        import joblib

        self.role1 = _load_role1_module()
        self.pipeline = joblib.load(artefact)
        self.model_name = self.champion["model_name"]
        self.model_version = str(self.champion.get("model_version", "unknown"))
        self.threshold = float(self.champion["threshold"])
        self.model = self.pipeline.models[self.model_name]
        # Every trained model can be served, each at its OWN alert threshold (chosen on validation by Role 4's
        # evaluation, eval_model_comparison.csv). The champion stays the default.
        self.thresholds = self._model_thresholds()
        self.available = [m for m in self.thresholds
                          if m in self.pipeline.models and hasattr(self.pipeline.models[m], "predict_proba")]
        self.detector = self.pipeline.models.get("isolation_forest")
        self.num_cols, self.cat_cols = self._feature_columns()
        self.history = self._node_dwell_history()
        self.global_drivers = _global_drivers()

    def _model_thresholds(self) -> dict:
        """Alert threshold per model from Role 4's evaluation (the champion's comes from eval_champion.json)."""
        path = stage_path(STAGE, "reports", "eval_model_comparison.csv")
        table = pd.read_csv(path).set_index("model_name")["threshold"].to_dict() if path.exists() else {}
        table[self.model_name] = self.threshold
        return {name: float(t) for name, t in table.items()}

    def resolve(self, model_name: str | None) -> str:
        """The model to use: the one asked for if it can be served, otherwise an error; None means the champion."""
        if model_name in (None, "", self.model_name):
            return self.model_name
        if self.stub or model_name not in self.available:
            raise ValueError(f"unknown model '{model_name}'; available: {', '.join(self.available)}")
        return model_name

    def _feature_columns(self) -> tuple:
        """Numeric and text feature lists exactly as Role 1's pipeline was fitted."""
        num_cols = list(self.pipeline.numeric_imputer_medians)
        cat_cols = list(self.pipeline.encoder.feature_names_in_)
        return num_cols, cat_cols

    def _node_dwell_history(self) -> dict:
        """Every node's dwell readings in time order, plus where each known record sits, for the rolling-dwell feature.

        Role 1 built roll_node_dwell_mean_24h as the mean of the 24 records at the same node BEFORE the shipment
        (file order, which is time order). To give the same score as the evaluated predictions, the API rebuilds it
        the same way instead of using one fixed "latest" value for every shipment.
        """
        path = stage_path(STAGE, "data", "prepared_dataset.csv.gz")
        if not path.exists():
            return {}
        df = pd.read_csv(path, usecols=["record_id", "timestamp", "current_node_id", "dwell_time_hours"])
        history = {"by_node": {}, "position": {}, "median": float(df["dwell_time_hours"].median())}
        for node, g in df.groupby("current_node_id", sort=False):
            # Times as plain UTC datetime64 (no zone object), so they can be binary-searched.
            times = pd.to_datetime(g["timestamp"], utc=True).dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
            history["by_node"][node] = (times, g["dwell_time_hours"].to_numpy())
            history["position"].update({rid: (node, i) for i, rid in enumerate(g["record_id"])})
        return history

    def _rolling_dwell(self, record_id, node: str, timestamp: str) -> float:
        """Mean dwell of the previous 24 records at this node, as Role 1 computed it (median when there are none)."""
        if not self.history or node not in self.history["by_node"]:
            return self.history.get("median", float("nan")) if self.history else float("nan")
        times, dwell = self.history["by_node"][node]
        known = self.history["position"].get(record_id)
        if known and known[0] == node:
            end = known[1]  # a shipment from the dataset: exactly the records before it
        else:
            # A new shipment: every record at the node that happened before its timestamp.
            moment = pd.Timestamp(timestamp).tz_convert("UTC").tz_localize(None).to_datetime64().astype("datetime64[ns]")
            end = int(np.searchsorted(times, moment, side="left"))
        window = dwell[max(0, end - ROLLING_WINDOW):end]
        return float(window.mean()) if len(window) else self.history["median"]

    def score(self, rows: list, model_name: str | None = None) -> list:
        """Score shipments with one model (default: the champion) and return contract-shaped answers."""
        if self.stub:
            return [_stub_answer(r) for r in rows]
        name = self.resolve(model_name)
        model, threshold = self.pipeline.models[name], self.thresholds[name]
        frame = pd.DataFrame([r.model_dump() for r in rows])
        frame["timestamp"] = frame["timestamp"].fillna(pd.Timestamp.now(tz="UTC").strftime(TS_FORMAT))
        # Same features as training: node-relative inputs use the train medians stored in the fitted pipeline.
        features, _ = self.role1.engineer_features(frame, node_medians=getattr(self.pipeline, "node_medians", None))
        # A request carries no history of its own, so the rolling dwell is rebuilt from the records before it.
        features["roll_node_dwell_mean_24h"] = [
            self._rolling_dwell(rid, n, ts) for rid, n, ts in zip(frame["record_id"], frame["current_node_id"], frame["timestamp"])]
        X = self.pipeline.transform(features, self.num_cols, self.cat_cols)
        proba = model.predict_proba(X)
        anomalous = self.detector.predict(X) == -1 if self.detector is not None else np.zeros(len(X), dtype=bool)
        drivers = self._drivers(X, model)
        out = []
        for i, row in enumerate(rows):
            p = proba[i]
            out.append({
                "record_id": row.record_id,
                "label": CLASSES[int(np.argmax(p))],
                "probabilities": {c: round(float(v), 4) for c, v in zip(CLASSES, p)},
                "risk_score": round(float(min(1.0, p[2] + 0.5 * p[1])), 4),
                # The alert uses Role 4's cost-based threshold, not "most likely class".
                "alert": bool(p[2] >= threshold),
                "alert_threshold": threshold,
                "anomaly_flag": bool(anomalous[i]),
                "top_drivers": drivers[i],
                "model_name": name,
                "is_champion": name == self.model_name,
                "model_version": self.model_version,
                "requires_operator_review": True,
            })
        return out

    def _drivers(self, X: np.ndarray, model) -> list:
        """Top 3 reasons per shipment: per-row contributions for a linear model, else the global SHAP ranking."""
        names = self.pipeline.feature_names_out
        if hasattr(model, "coef_"):
            # Contribution to the CRITICAL score = weight x standardised value; keep the three biggest pushes up.
            contrib = X * model.coef_[2]
            return [[names[j] for j in np.argsort(-row)[:3]] for row in contrib]
        return [self.global_drivers[:3]] * len(X)


def _stub_answer(row) -> dict:
    """Answer used only while no trained model exists."""
    return {"record_id": row.record_id, "label": "ON_TIME", "probabilities": dict(zip(CLASSES, [1.0, 0.0, 0.0])),
            "risk_score": 0.0, "alert": False, "alert_threshold": 0.5, "anomaly_flag": False, "top_drivers": [],
            "model_name": "STUB", "model_version": "STUB", "requires_operator_review": True}


def _read_json(path: Path) -> dict:
    """Read a JSON file, or return an empty dict if it does not exist yet."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _global_drivers() -> list:
    """Feature names ranked by mean |SHAP| for CRITICAL from Role 1's importance matrix."""
    path = stage_path(STAGE, "predictions", "feature_importance_matrix.csv")
    if not path.exists():
        return []
    return list(pd.read_csv(path).sort_values("shap_mean_abs_critical", ascending=False)["feature"])


@lru_cache(maxsize=1)
def service() -> Service:
    """Load the model once per process."""
    return Service()


# ------------------------------------------------------------------ the API

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model once at start-up, on the main thread.

    Loading it lazily inside a request worker thread crashed on macOS (XGBoost's OpenMP runtime), so we load first.
    """
    service()
    yield


app = FastAPI(title="CASCADE Stage 1 - shipment bottleneck scoring",
              description="Decision support only: every answer requires operator review.", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness check; also says whether a real model or the stub is being served."""
    return {"status": "ok", "model_version": service().model_version, "stub": service().stub}


@app.get("/model-info")
def model_info() -> dict:
    """Which model is served, at which alert threshold, and who chose it."""
    s = service()
    return {"model_name": s.model_name, "model_version": s.model_version, "alert_threshold": s.threshold,
            "champion_file": s.champion, "model_card": {k: s.card.get(k) for k in ("model_name", "created_utc", "feature_count",
                                                                                  "champion_proposal", "candidates")},
            "requires_operator_review": True}


@app.get("/models")
def models() -> dict:
    """Every model that can be served, with its own alert threshold; the champion is the default."""
    s = service()
    return {"champion": s.model_name, "models": [{"model_name": m, "alert_threshold": s.thresholds[m], "is_champion": m == s.model_name}
                                                for m in getattr(s, "available", [])]}


def _score_or_422(rows: list, model: str | None) -> list:
    """Score with the requested model; an unknown model name is a clear 422, not a crash."""
    try:
        return service().score(rows, model_name=model)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from None


@app.post("/predict")
def predict(shipment: Shipment, model: str | None = None) -> dict:
    """Score one shipment. Optional ?model=xgboost picks another trained model (see /models)."""
    return _score_or_422([shipment], model)[0]


@app.post("/predict/batch")
def predict_batch(batch: Batch, model: str | None = None) -> dict:
    """Score up to 1,000 shipments in one call. Optional ?model= as for /predict."""
    if len(batch.shipments) > MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"at most {MAX_BATCH} shipments per call")
    return {"results": _score_or_422(batch.shipments, model), "requires_operator_review": True}


# ------------------------------------------------------------------ script mode: latency + field guide

def sample_requests(n: int) -> list:
    """Real test-split shipments from the prepared dataset, shaped as API request bodies."""
    df = pd.read_csv(stage_path(STAGE, "data", "prepared_dataset.csv.gz"))
    test = df[df["split"] == "test"].sample(n=min(n, (df["split"] == "test").sum()), random_state=42)
    keep = NUMERIC_FIELDS + ["priority_tier", "carrier_code", "origin_port", "destination_port", "current_node_id", "timestamp", "record_id"]
    return test[keep].to_dict(orient="records")


def measure_latency(n: int = LATENCY_CALLS) -> dict:
    """Time n single-shipment calls through the full HTTP stack (in-process test client)."""
    from fastapi.testclient import TestClient

    service()  # load the model on the main thread before any request (see lifespan)
    bodies = sample_requests(n)
    client = TestClient(app)
    client.post("/predict", json=bodies[0])  # warm-up call, not counted
    times = []
    for body in bodies:
        start = time.perf_counter()
        response = client.post("/predict", json=body)
        times.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200 and response.json()["requires_operator_review"] is True
    ms = np.array(times)
    return {"calls": len(ms), "p50_ms": round(float(np.percentile(ms, 50)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2),
            "target_p95_ms": 50, "meets_target": bool(np.percentile(ms, 95) < 50), "model_version": service().model_version,
            "method": "FastAPI TestClient, in-process, single-shipment /predict calls on test-split rows", "date": date.today().isoformat()}


def write_field_guide(path: Path) -> None:
    """One page for planners. Every number is read from a team file, and the file is named in a footnote."""
    champ = _read_json(stage_path(STAGE, "reports", "eval_champion.json"))
    metrics = _read_json(stage_path(STAGE, "reports", "eval_metrics.json"))
    policy = _read_json(CONTRACTS_DIR / "cost_policy.json")
    row = next(m for m in metrics["models"] if m["model_name"] == champ["model_name"])
    drivers = [d for d in _global_drivers() if not d.startswith(("carrier_code_", "origin_port_", "destination_port_"))][:5]
    wording = stage_path(STAGE, "reports", "eda_field_guide_text.md")
    cm = policy["cost_matrix_rows_true_cols_pred"]
    lines = [
        "# Ops Field Guide - shipment delay alerts (Stage 1)", "",
        "**What this tool does:** for each shipment snapshot it estimates the chance of ON_TIME, AT_RISK or CRITICAL_DELAY "
        "and raises an alert when CRITICAL is likely enough to be worth acting on. **It never reroutes, books or changes "
        "anything: every alert needs a planner's decision.**", "",
        "## The three statuses",
        "| Status | Meaning |", "|---|---|",
        "| ON_TIME | Expected delay under 6 hours [1] |",
        "| AT_RISK | Expected delay 6 to 24 hours: watch it [1] |",
        "| CRITICAL_DELAY | Expected delay of 24 hours or more: act [1] |", "",
        "## When the tool raises an alert",
        f"- Alert when the chance of CRITICAL is **{champ['threshold']:.0%} or more** (model `{champ['model_name']}`) [2].",
        f"- On held-out data it caught **{row['at_threshold_critical_recall']:.0%}** of critical delays; "
        f"**{row['at_threshold_false_alarms_per_100_alerts']:.0f} of every 100 alerts** were false alarms [3].",
        f"- Why accept false alarms: a missed critical delay costs **{cm['CRITICAL_DELAY'][0]:.0f}x** an unnecessary escalation "
        f"({cm['ON_TIME'][2]:.0f} unit) [4].", "",
        "## Top five drivers of a CRITICAL prediction [5]",
        "(Ranked by Role 1's SHAP analysis of the XGBoost model; the served model may weigh them slightly differently, "
        "and each API answer lists its own top drivers.)", "",
    ]
    lines += [f"{i}. `{d}`" for i, d in enumerate(drivers, 1)]
    if wording.exists():
        lines += ["", "In plain words (Role 2):", "", wording.read_text(encoding="utf-8").split("\n", 2)[-1].strip()]
    lines += ["", "## What the tool will not do",
              "- It does not execute reroutes, bookings or any shipment change.",
              "- It is trained on calibrated simulation, not live port feeds; treat it as a second opinion.",
              "- It is weakest at maritime chokepoints (Suez, Malacca, Panama): check those by hand [6].", "",
              "---", "Sources: [1] `contracts/label_policy.json` [2] `Stage01_ML/reports/eval_champion.json` "
              "[3] `Stage01_ML/reports/eval_metrics.json` [4] `contracts/cost_policy.json` "
              "[5] `Stage01_ML/predictions/feature_importance_matrix.csv` [6] `Stage01_ML/reports/eval_slices.csv`.",
              f"Generated by `python Stage01_ML/05_integration_engineer.py` on {date.today().isoformat()}."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


PRINT_CSS = """
body { font: 11pt/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1b1f24; max-width: 46rem; margin: 2rem auto; padding: 0 1rem; }
h1 { font-size: 1.5rem; margin: 0 0 .75rem; } h2 { font-size: 1.1rem; margin: 1.25rem 0 .4rem; border-bottom: 1px solid #d0d7de; }
table { border-collapse: collapse; width: 100%; margin: .4rem 0; } th, td { border: 1px solid #d0d7de; padding: .3rem .5rem; text-align: left; }
th { background: #f6f8fa; } code { font: .9em ui-monospace, Menlo, monospace; background: #f6f8fa; padding: 0 .2rem; }
hr { border: 0; border-top: 1px solid #d0d7de; margin: 1.25rem 0; } .src { font-size: 9pt; color: #57606a; }
/* Printed copy must fit ONE A4 page (Role 3 file), so print uses smaller type and tighter spacing. */
@page { size: A4; margin: 12mm; }
@media print { body { margin: 0; max-width: none; font-size: 9pt; line-height: 1.3; } h1 { font-size: 1.25rem; margin-bottom: .4rem; }
  h2 { font-size: 1rem; margin: .7rem 0 .25rem; break-after: avoid; } p, ul, ol { margin: .3rem 0; } table { break-inside: avoid; }
  th, td { padding: .15rem .4rem; } hr { margin: .6rem 0; } .src { font-size: 7.5pt; } }
"""


def _inline(text: str) -> str:
    """Bold and code spans of one line of markdown, as HTML (text is escaped first)."""
    import html
    import re

    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


def markdown_to_html(markdown: str, title: str) -> str:
    """Tiny converter for the field guide's own markdown (headings, lists, tables, rules, paragraphs).

    No extra library is needed, so the printable copy can be rebuilt on any teammate's laptop.
    """
    out, in_list, table = [], None, []

    def flush_table():
        if table:
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in table if not set(r.replace("|", "").strip()) <= {"-", " "}]
            head, body = rows[0], rows[1:]
            out.append("<table><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr>"
                       + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body) + "</table>")
            table.clear()

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            table.append(stripped)
            continue
        flush_table()
        kind = "ul" if stripped.startswith("- ") else "ol" if stripped[:3].rstrip(".").isdigit() and ". " in stripped[:4] else None
        if in_list and kind != in_list:
            out.append(f"</{in_list}>")
            in_list = None
        if kind:
            if not in_list:
                out.append(f"<{kind}>")
                in_list = kind
            out.append(f"<li>{_inline(stripped.split(' ', 1)[1])}</li>")
        elif stripped.startswith("## "):
            out.append(f"<h2>{_inline(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            out.append(f"<h1>{_inline(stripped[2:])}</h1>")
        elif stripped == "---":
            out.append("<hr>")
        elif stripped:
            css = ' class="src"' if stripped.startswith(("Sources:", "Generated by")) else ""
            out.append(f"<p{css}>{_inline(stripped)}</p>")
    flush_table()
    if in_list:
        out.append(f"</{in_list}>")
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>{title}</title>"
            f"<style>{PRINT_CSS}</style></head><body>" + "\n".join(out) + "</body></html>\n")


def main() -> int:
    """Script mode: write the field guide (markdown + printable HTML) and the latency report."""
    if not stage_path(STAGE, "reports", "eval_champion.json").exists():
        print("WAITING: reports/eval_champion.json does not exist yet (Role 4). The API would serve the STUB model.")
        return 3
    # Load the model BEFORE seeding: set_global_seed() imports PyTorch when it is installed, and on macOS loading
    # PyTorch before XGBoost crashes the process (the two ship clashing copies of the OpenMP library).
    service()
    set_global_seed()  # same random numbers on every laptop
    guide = stage_path(STAGE, "reports", "integ_ops_field_guide.md")
    write_field_guide(guide)
    # Printable copy: open it in a browser and use Print -> Save as PDF.
    stage_path(STAGE, "reports", "integ_ops_field_guide.html").write_text(
        markdown_to_html(guide.read_text(encoding="utf-8"), "Ops Field Guide - Stage 1"), encoding="utf-8")
    latency = measure_latency()
    stage_path(STAGE, "reports", "integ_latency.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")
    print(f"field guide written; latency p50 {latency['p50_ms']} ms, p95 {latency['p95_ms']} ms (target < 50 ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
