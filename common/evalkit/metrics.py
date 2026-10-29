"""Plain classification metrics written in NumPy so they are fast inside a bootstrap loop.

Why this file exists: the bootstrap calls each metric 1,000 times, so each one takes simple arrays and
returns one number. Class labels are the integers 0/1/2 in the order ON_TIME, AT_RISK, CRITICAL_DELAY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_CLASSES = 3


def confusion_counts(t: np.ndarray, p: np.ndarray, k: int = N_CLASSES) -> np.ndarray:
    """k x k table; rows are the truth, columns are the prediction."""
    return np.bincount(t * k + p, minlength=k * k).reshape(k, k)


def per_class_prf(t: np.ndarray, p: np.ndarray, k: int = N_CLASSES) -> dict:
    """Precision, recall and F1 for each class (0 when a class is never predicted or never occurs)."""
    cm = confusion_counts(t, p, k)
    tp = np.diag(cm).astype(float)
    pred_n, true_n = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, pred_n, out=np.zeros(k), where=pred_n > 0)
    recall = np.divide(tp, true_n, out=np.zeros(k), where=true_n > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(k), where=denom > 0)
    return {"precision": precision, "recall": recall, "f1": f1}


def _present_classes(t: np.ndarray, p: np.ndarray, k: int = N_CLASSES) -> np.ndarray:
    """Classes that occur in the truth or in the predictions (same rule as scikit-learn's macro average).

    A class that never occurs anywhere (e.g. no CRITICAL shipment at one small node) must not drag the
    average down with a fake zero.
    """
    return (np.bincount(t, minlength=k) + np.bincount(p, minlength=k)) > 0


def macro_f1(t: np.ndarray, p: np.ndarray) -> float:
    """Average F1 over the classes that occur, so the rare CRITICAL class counts as much as ON_TIME."""
    return float(per_class_prf(t, p)["f1"][_present_classes(t, p)].mean())


def balanced_accuracy(t: np.ndarray, p: np.ndarray) -> float:
    """Average recall over the classes that occur in the truth (immune to class imbalance, unlike accuracy)."""
    has_truth = np.bincount(t, minlength=N_CLASSES) > 0
    return float(per_class_prf(t, p)["recall"][has_truth].mean())


def average_precision(is_positive: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC as average precision: how well the score ranks real positives above the rest.

    Rows with the same score are one step (one threshold), so the answer does not depend on row order.
    """
    n_pos = int(is_positive.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    s, hit = score[order], is_positive[order].astype(float)
    tp, n_seen = np.cumsum(hit), np.arange(1, len(hit) + 1)
    # Keep only the last row of each group of equal scores: that is where a real threshold can cut.
    last_of_group = np.append(s[1:] != s[:-1], True)
    precision, recall = tp[last_of_group] / n_seen[last_of_group], tp[last_of_group] / n_pos
    step = np.diff(np.concatenate([[0.0], recall]))
    return float((precision * step).sum())


def roc_auc(is_positive: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via ranks (probability a random positive scores above a random negative)."""
    n_pos = int(is_positive.sum())
    n_neg = len(is_positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Average ranks give ties half credit; pandas does this without needing scipy.
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[is_positive.astype(bool)].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def log_loss(t: np.ndarray, proba: np.ndarray, eps: float = 1e-12) -> float:
    """Multi-class log loss: punishes confident wrong answers heavily."""
    picked = np.clip(proba[np.arange(len(t)), t], eps, 1.0)
    return float(-np.log(picked).mean())


def multiclass_brier(t: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared gap between the probability vector and the one-hot truth."""
    onehot = np.eye(proba.shape[1])[t]
    return float(((proba - onehot) ** 2).sum(axis=1).mean())
