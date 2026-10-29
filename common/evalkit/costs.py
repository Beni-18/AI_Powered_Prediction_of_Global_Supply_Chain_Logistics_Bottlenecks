"""Cost-based decisions: what a wrong prediction costs and where to put the alert threshold.

Why this file exists: the problem statement asks for an "acceptable false-alarm threshold". We answer it
with money-like cost units from contracts/cost_policy.json instead of a gut feeling. Missing a critical
delay costs far more than one needless alarm, so the best threshold is NOT 0.5.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common.paths import CONTRACTS_DIR

CLASSES = ["ON_TIME", "AT_RISK", "CRITICAL_DELAY"]
POSITIVE = "CRITICAL_DELAY"


def load_cost_policy() -> dict:
    """Read the team's frozen cost policy file."""
    return json.loads((CONTRACTS_DIR / "cost_policy.json").read_text(encoding="utf-8"))


def load_cost_matrix(policy: dict | None = None) -> np.ndarray:
    """Return the 3x3 cost matrix (rows = truth, columns = prediction) in CLASSES order."""
    policy = policy or load_cost_policy()
    block = policy["cost_matrix_rows_true_cols_pred"]
    assert block["classes"] == CLASSES, "cost_policy.json class order changed; update evalkit.costs.CLASSES"
    return np.array([block[c] for c in CLASSES], dtype=float)


def scaled_cost_matrix(missed_critical_cost: float, policy: dict | None = None) -> np.ndarray:
    """Copy of the cost matrix where a missed critical costs `missed_critical_cost`.

    The CRITICAL -> AT_RISK cell (a half-caught critical) keeps its policy ratio to the miss (3 / 10).
    All other cells stay as in the policy, so only the "how bad is a miss" question changes.
    """
    base = load_cost_matrix(policy)
    out = base.copy()
    ratio_half_caught = base[2, 1] / base[2, 0]
    out[2, 0] = missed_critical_cost
    out[2, 1] = missed_critical_cost * ratio_half_caught
    return out


def _to_index(labels) -> np.ndarray:
    """Turn class names into 0/1/2 so they can index the cost matrix."""
    lookup = {c: i for i, c in enumerate(CLASSES)}
    return np.array([lookup[v] for v in labels], dtype=int)


def expected_cost(y_true, y_pred, cost_matrix: np.ndarray) -> float:
    """Average cost per shipment of a set of predictions (lower is better)."""
    t, p = _to_index(y_true), _to_index(y_pred)
    return float(cost_matrix[t, p].mean())


def bayes_decision(proba: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """For each row pick the class with the lowest expected cost given its probabilities.

    proba has columns in CLASSES order. Expected cost of saying class j = sum_i P(truth=i) * cost[i, j].
    """
    expected = np.asarray(proba) @ cost_matrix
    return expected.argmin(axis=1)


def decide_with_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Alert rule: predict CRITICAL when p_critical >= threshold, else the likelier of the other two."""
    proba = np.asarray(proba)
    pred = np.where(proba[:, 0] >= proba[:, 1], 0, 1)
    pred = np.where(proba[:, 2] >= threshold, 2, pred)
    return pred


def cost_vs_threshold(y_true, proba: np.ndarray, cost_matrix: np.ndarray, grid: np.ndarray | None = None) -> pd.DataFrame:
    """Expected cost, precision, recall and alert counts for every candidate threshold."""
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2) if grid is None else grid
    t_idx = _to_index(y_true)
    is_crit = t_idx == 2
    rows = []
    for thr in grid:
        pred = decide_with_threshold(proba, thr)
        alert = pred == 2
        tp = int((alert & is_crit).sum())
        n_alert = int(alert.sum())
        rows.append({
            "threshold": float(thr),
            "expected_cost": float(cost_matrix[t_idx, pred].mean()),
            "alerts": n_alert,
            "precision": tp / n_alert if n_alert else np.nan,
            "recall": tp / int(is_crit.sum()) if is_crit.sum() else np.nan,
            "false_alarms_per_100_alerts": 100.0 * (n_alert - tp) / n_alert if n_alert else np.nan,
        })
    return pd.DataFrame(rows)


def choose_threshold(y_val, proba_val: np.ndarray, cost_matrix: np.ndarray, grid: np.ndarray | None = None) -> tuple[float, pd.DataFrame]:
    """Pick the threshold with the lowest expected cost on VALIDATION data.

    Returns (best_threshold, the whole cost-vs-threshold table). Ties go to the higher threshold,
    which means fewer alerts, so the quieter option wins when costs are equal.
    """
    table = cost_vs_threshold(y_val, proba_val, cost_matrix, grid)
    best_cost = table["expected_cost"].min()
    best = table[np.isclose(table["expected_cost"], best_cost)]["threshold"].max()
    return float(best), table
