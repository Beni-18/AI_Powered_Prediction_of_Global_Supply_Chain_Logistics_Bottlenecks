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
    """Test (c): the best any ONE numeric feature can do alone at spotting CRITICAL (ROC-AUC, direction-free)."""
    aucs = {}
    for f in features:
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
            col = df[f].fillna(df[f].median()).to_numpy()
            a = roc_auc(is_critical, col)
            aucs[f] = max(a, 1 - a)
    worst = max(aucs, key=aucs.get) if aucs else None
    top = aucs[worst] if worst else float("nan")
    return {"status": "PASS" if top < SINGLE_FEATURE_AUC_LIMIT else "FAIL", "max_auc": top,
            "max_feature": worst, "limit": SINGLE_FEATURE_AUC_LIMIT, "auc_by_feature": aucs}


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
    """Test (b): train a small model on SHUFFLED train labels; on the real test labels it must score near chance.

    If it scores well, the features or the split leak the answer by some route other than the label itself.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    classes = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
    code = {c: i for i, c in enumerate(classes)}
    X_train, X_test = _numeric_matrix(train, features), _numeric_matrix(test, features)
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(train[label_col].map(code).to_numpy())
    # Small, shallow probe: it only needs to be able to find a leak, not to be a good model.
    probe = HistGradientBoostingClassifier(max_iter=60, max_depth=3, learning_rate=0.1, random_state=seed)
    probe.fit(X_train, y_shuffled)
    score = macro_f1(test[label_col].map(code).to_numpy(), probe.predict(X_test))
    return {"status": "PASS" if score <= SHUFFLED_F1_LIMIT else "FAIL", "test_macro_f1_with_shuffled_labels": score,
            "limit": SHUFFLED_F1_LIMIT, "chance_level_about": 0.33}


def too_good_alarm(champion_macro_f1: float) -> dict:
    """Test (e): a macro-F1 above 0.95 on this kind of data is more likely a leak than a triumph."""
    suspect = champion_macro_f1 > SUSPECT_F1
    return {"status": "FAIL" if suspect else "PASS", "champion_macro_f1": champion_macro_f1, "suspect_above": SUSPECT_F1,
            "verdict": "SUSPECT" if suspect else "plausible"}


def _numeric_matrix(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Numbers as they are; text columns become integer codes so the probe can read them."""
    out = pd.DataFrame(index=df.index)
    for f in features:
        out[f] = df[f] if pd.api.types.is_numeric_dtype(df[f]) else df[f].astype("category").cat.codes
    return out
