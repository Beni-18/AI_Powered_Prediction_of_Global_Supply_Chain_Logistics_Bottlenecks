"""Builds a stand-in prepared dataset and stand-in predictions from the raw simulator files.

Why this file exists: Role 4 must not wait for Roles 1 and 3. This makes a quick, honest-enough copy of
what they will deliver (same columns, same split rules) so the evaluation script can be rehearsed end to
end. NOTHING made here is a team result: it is written to a temp folder and never committed.
Run by hand:  python tests/eval/rehearsal.py <output_folder>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.paths import REPO_ROOT as _ROOT  # noqa: E402,F401

CLASSES = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
NUMERIC = ["dwell_time_hours", "throughput_teu_per_hr", "capacity_utilization_pct", "customs_hold_hours",
           "weather_severity_index", "labor_deficit_ratio", "equipment_availability_pct", "historical_delay_rate"]


def build_prepared() -> pd.DataFrame:
    """The team's real prepared dataset when it exists; otherwise a quick stand-in built from the raw simulator files."""
    real = REPO_ROOT / "Stage01_ML" / "data" / "prepared_dataset.csv.gz"
    if real.exists():
        df = pd.read_csv(real)
        df = df[df["split"].isin(["train", "val", "test"]) & df["delay_status"].notna()].copy()
        df["priority_code"] = df["priority_tier"].astype("category").cat.codes
        return df
    raw = pd.read_csv(REPO_ROOT / "data" / "raw" / "sim" / "shipment_telemetry_raw.csv.gz")
    df = raw.drop_duplicates("record_id").dropna(subset=["realized_delay_hours"]).copy()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    # Label rule from contracts/label_policy.json: <6 h on time, 6-24 h at risk, >=24 h critical.
    df["delay_status"] = pd.cut(df["realized_delay_hours"], [-np.inf, 6, 24, np.inf], labels=CLASSES, right=False).astype(str)
    cut_a, cut_b = df["ts"].quantile(0.70), df["ts"].quantile(0.85)
    gap = pd.Timedelta(hours=168)
    df["split"] = np.select([df["ts"] < cut_a, (df["ts"] >= cut_a + gap) & (df["ts"] < cut_b), df["ts"] >= cut_b + gap],
                            ["train", "val", "test"], default="")
    df = df[df["split"] != ""].copy()
    df["priority_code"] = df["priority_tier"].astype("category").cat.codes
    return df.drop(columns="ts")


def build_predictions(prepared: pd.DataFrame) -> pd.DataFrame:
    """Three baselines/models in the ml_predictions contract format: a dwell rule, logistic, gradient boosting (+ calibrated)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = NUMERIC + ["priority_code"]
    code = {c: i for i, c in enumerate(CLASSES)}
    train = prepared[prepared["split"] == "train"]
    y_train = train["delay_status"].map(code).to_numpy()
    medians = train[feats].median()
    filled = prepared[feats].fillna(medians)
    iso = IsolationForest(n_estimators=50, random_state=42).fit(filled[prepared["split"] == "train"])
    anomaly = pd.Series(-iso.score_samples(filled), index=prepared.index).rank(pct=True)

    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)),
        "gbm": HistGradientBoostingClassifier(max_iter=80, max_depth=4, random_state=42),
    }
    models["gbm_cal"] = CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=80, max_depth=4, random_state=42), cv=3, method="isotonic")
    frames = []
    for name, model in models.items():
        model.fit(filled[prepared["split"] == "train"], y_train)
        frames.append(_frame(prepared, model.predict_proba(filled), name, anomaly))

    # Rule baseline: dwell of a day or more means critical (hard-ish probabilities).
    dwell = prepared["dwell_time_hours"].fillna(prepared["dwell_time_hours"].median()).to_numpy()
    rule = np.where((dwell >= 24)[:, None], [[0.1, 0.2, 0.7]], np.where((dwell >= 12)[:, None], [[0.2, 0.6, 0.2]], [[0.8, 0.15, 0.05]]))
    frames.append(_frame(prepared, rule, "rule_dwell24", anomaly))
    return pd.concat(frames, ignore_index=True)


def _frame(prepared: pd.DataFrame, proba: np.ndarray, name: str, anomaly: pd.Series) -> pd.DataFrame:
    """Wrap one model's probabilities in the contract columns."""
    proba = proba / proba.sum(axis=1, keepdims=True)
    return pd.DataFrame({
        "record_id": prepared["record_id"].to_numpy(), "split": prepared["split"].to_numpy(), "y_true": prepared["delay_status"].to_numpy(),
        "y_pred": np.array(CLASSES)[proba.argmax(axis=1)],
        "p_on_time": proba[:, 0].round(6), "p_at_risk": proba[:, 1].round(6), "p_critical": proba[:, 2].round(6),
        "risk_score": (proba[:, 2] + 0.5 * proba[:, 1]).clip(0, 1).round(6), "anomaly_score": anomaly.to_numpy().round(6),
        "model_name": name, "model_version": "rehearsal"})


def fix_rounding(pred: pd.DataFrame) -> pd.DataFrame:
    """Make the three probabilities sum to exactly 1 after rounding (the contract is strict about it)."""
    pred["p_on_time"] = (1 - pred["p_at_risk"] - pred["p_critical"]).round(6).clip(0, 1)
    return pred


def build(out_dir: Path) -> tuple:
    """Write prepared_dataset.csv and ml_predictions.csv into out_dir; return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = build_prepared()
    pred = fix_rounding(build_predictions(prepared))
    prepared_path, pred_path = out_dir / "prepared_dataset.csv", out_dir / "ml_predictions.csv"
    prepared.to_csv(prepared_path, index=False)
    pred.to_csv(pred_path, index=False)
    return pred_path, prepared_path


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("rehearsal_out")
    print(build(folder))
