"""Leakage tests: evidence that a high score comes from learning logistics, not from cheating.

Why this file exists: this is the heart of our "honest numbers" promise. Each function answers one
question and returns a small dict with `status` = PASS / FAIL / NOT_RUN and the numbers behind it,
so the results can be written straight to eval_leakage_tests.json and quoted in a debate.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common.evalkit.metrics import macro_f1, roc_auc
from common.paths import CONTRACTS_DIR

SINGLE_FEATURE_AUC_LIMIT = 0.95  # from Role 4's file: "well below 0.95"
SHUFFLED_F1_LIMIT = 0.40         # chance level for 3 classes is about 0.33; 0.40 leaves room for noise
SUSPECT_F1 = 0.95                # a champion above this is "too good to be true"
ID_EXCESS_AUC_LIMIT = 0.05       # an id may beat the clock by at most this much AUC before we call it a leak


def load_feature_policy() -> dict:
    """Read the team's frozen feature policy."""
    return json.loads((CONTRACTS_DIR / "feature_policy.json").read_text(encoding="utf-8"))


def feature_policy_check(features: list) -> dict:
    """Test (a): every feature a model used must be allowed by contracts/feature_policy.json."""
    policy = load_feature_policy()
    allowed = set(policy["allowed_base_features"])
    prefixes = tuple(policy["allowed_derived_prefixes"])
    forbidden = set(policy["forbidden_features"])
    bad = [f for f in features if f in forbidden or not (f in allowed or f.startswith(prefixes))]
    return {"status": "PASS" if not bad else "FAIL", "n_features": len(features), "illegal_features": bad}


def single_feature_ceiling(df: pd.DataFrame, features: list, is_critical: np.ndarray) -> dict:
    """Test (c): the best any ONE numeric feature can do alone at spotting CRITICAL (ROC-AUC, direction-free).

    Features that are not numeric columns of `df` (derived lag_/roll_/eng_ columns built inside the model
    script, or text columns) cannot be tested here. They are listed in `untested_features`, and the status is
    PARTIAL instead of PASS, so a green light never hides features nobody looked at.
    """
    aucs, untested = {}, []
    for f in features:
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
            col = df[f].fillna(df[f].median()).to_numpy()
            a = roc_auc(is_critical, col)
            aucs[f] = max(a, 1 - a)
        else:
            untested.append(f)
    worst = max(aucs, key=aucs.get) if aucs else None
    top = aucs[worst] if worst else float("nan")
    if top >= SINGLE_FEATURE_AUC_LIMIT:
        status = "FAIL"
    else:
        status = "PARTIAL" if untested else "PASS"
    return {"status": status, "max_auc": top, "max_feature": worst, "limit": SINGLE_FEATURE_AUC_LIMIT,
            "untested_features": untested, "auc_by_feature": aucs}


def bookkeeping_ceiling(df: pd.DataFrame, is_critical: np.ndarray) -> dict:
    """Test (c2): do the id columns say MORE about the outcome than the clock alone does? (they must not)

    Ids such as record_id / shipment_id are forbidden features. They are issued in time order, so they always
    carry whatever the clock carries (seasonality). The leak signal is an id that ranks outcomes better than
    the timestamp does: then the ids were generated in an order that encodes the label.
    """
    aucs = {}
    for col in ("record_id", "shipment_id", "container_id"):
        if col in df.columns:
            digits = pd.to_numeric(df[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
            if digits.notna().mean() > 0.9:
                a = roc_auc(is_critical, digits.fillna(digits.median()).to_numpy())
                aucs[col] = max(a, 1 - a)
    clock = float("nan")
    if "timestamp" in df.columns:
        hours = pd.to_datetime(df["timestamp"], utc=True).astype("int64").to_numpy()
        a = roc_auc(is_critical, hours)
        clock = max(a, 1 - a)
    worst_id = max(aucs.values()) if aucs else float("nan")
    # Compare with the clock; if there is no timestamp column, fall back to a coin-flip-plus-slack limit.
    baseline = clock if clock == clock else 0.5
    excess = worst_id - baseline
    return {"status": "FAIL" if excess > ID_EXCESS_AUC_LIMIT else "PASS", "max_id_auc": worst_id, "clock_auc": clock,
            "excess_over_clock": excess, "limit_excess": ID_EXCESS_AUC_LIMIT, "auc_by_column": aucs}


def time_order_check(df: pd.DataFrame, purge_hours: int) -> dict:
    """Test (d): train comes before val, val before test, with the purge gap, and no shipment crosses splits."""
    ts = pd.to_datetime(df["timestamp"], utc=True)
    span = {s: (ts[df["split"] == s].min(), ts[df["split"] == s].max()) for s in ("train", "val", "test")}
    gap = pd.Timedelta(hours=purge_hours)
    gap_tv = span["val"][0] - span["train"][1]
    gap_vt = span["test"][0] - span["val"][1]
    shared = int((df.groupby("shipment_id")["split"].nunique() > 1).sum())
    ok = gap_tv >= gap and gap_vt >= gap and shared == 0
    return {"status": "PASS" if ok else "FAIL", "purge_hours_required": purge_hours,
            "gap_train_to_val_h": gap_tv / pd.Timedelta(hours=1), "gap_val_to_test_h": gap_vt / pd.Timedelta(hours=1),
            "shipments_in_two_splits": shared}


def shuffle_label_probe(train: pd.DataFrame, test: pd.DataFrame, features: list, label_col: str, seed: int = 42) -> dict:
    """Test (b): a SANITY check of the scoring pipeline, not a leakage detector.

    Train a small model on SHUFFLED train labels, then score it on the real test labels. It must land near
    chance. This catches a scoring bug (for example real test labels leaking into the evaluation), but it can NOT
    detect a leaky feature, because shuffled labels teach the model nothing whatever the features contain.
    Leaky features are caught by tests (a), (c) and (c2). Do not quote this one as proof of "no leakage".
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    classes = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
    code = {c: i for i, c in enumerate(classes)}
    X_train, X_test = _numeric_matrix(train, features, reference=pd.concat([train, test])), _numeric_matrix(test, features, reference=pd.concat([train, test]))
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(train[label_col].map(code).to_numpy())
    # Small, shallow probe: it only needs to show the pipeline scores chance as chance.
    probe = HistGradientBoostingClassifier(max_iter=60, max_depth=3, learning_rate=0.1, random_state=seed)
    probe.fit(X_train, y_shuffled)
    score = macro_f1(test[label_col].map(code).to_numpy(), probe.predict(X_test))
    return {"status": "PASS" if score <= SHUFFLED_F1_LIMIT else "FAIL", "test_macro_f1_with_shuffled_labels": score,
            "limit": SHUFFLED_F1_LIMIT, "chance_level_about": 0.33,
            "scope": "pipeline sanity check only; does not detect leaky features"}


def too_good_alarm(champion_macro_f1: float) -> dict:
    """Test (e): a macro-F1 above 0.95 on this kind of data is more likely a leak than a triumph."""
    suspect = champion_macro_f1 > SUSPECT_F1
    return {"status": "FAIL" if suspect else "PASS", "champion_macro_f1": champion_macro_f1, "suspect_above": SUSPECT_F1,
            "verdict": "SUSPECT" if suspect else "plausible"}


def _numeric_matrix(df: pd.DataFrame, features: list, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Numbers as they are; text columns become integer codes so the probe can read them.

    `reference` fixes the code list (train + test together) so the same value gets the same code in both frames.
    """
    reference = df if reference is None else reference
    out = pd.DataFrame(index=df.index)
    for f in features:
        if pd.api.types.is_numeric_dtype(df[f]):
            out[f] = df[f]
        else:
            levels = pd.Index(sorted(reference[f].dropna().astype(str).unique()))
            out[f] = pd.Categorical(df[f].astype(str), categories=levels).codes
    return out
