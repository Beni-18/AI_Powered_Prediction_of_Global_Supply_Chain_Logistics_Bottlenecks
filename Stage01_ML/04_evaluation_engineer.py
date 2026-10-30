"""ROLE: EVALUATION ENGINEER  |  STAGE 01 - MACHINE LEARNING
Owner: Role 4 - Evaluation Engineer. Full instructions: docs/roles/ROLE_4_EVALUATION_ENGINEER.md (Day 1).

WHAT THIS SCRIPT DOES
    1. Scores every model on the TEST split (macro-F1, PR-AUC, cost ...) with block-bootstrap confidence intervals.
    2. Chooses the alert threshold on the VALIDATION split by minimum expected cost, freezes it, reports on TEST.
    3. Picks the champion (must beat the best baseline) and writes eval_champion.json for the API.
    4. Draws confusion matrices, the cost curve, calibration, and slice results.
    5. Runs the leakage tests and the anomaly-detector check.
    6. Writes claims into claims/CLAIMS_LEDGER.md.

READS : Stage01_ML/predictions/ml_predictions.csv          (Role 1)
        Stage01_ML/data/prepared_dataset.csv[.gz]            (Role 3; optional, needed for slices and leakage tests)
        Stage01_ML/artifacts/model_card.json                 (Role 1; optional, needed for the feature-policy check)
        contracts/cost_policy.json, contracts/label_policy.json
WRITES: Stage01_ML/reports/eval_*.json|csv|md|png, claims/CLAIMS_LEDGER.md

HOW TO RUN
    python Stage01_ML/04_evaluation_engineer.py                       (real run)
    python Stage01_ML/04_evaluation_engineer.py --predictions X.csv --out-dir /tmp/eval_try     (dry run, no ledger)
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# Make "import common" work when this file is run from inside the stage folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # draw to files only, so it works on a machine with no screen
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import contracts  # noqa: E402
from common.evalkit import (  # noqa: E402
    CLASSES, average_precision, balanced_accuracy, bayes_decision, block_bootstrap_ci, choose_threshold,
    confusion_counts, decide_with_threshold, ece, ledger, load_cost_matrix, load_cost_policy,
    log_loss, macro_f1, multiclass_brier, paired_bootstrap_diff, per_class_prf, reliability_table, roc_auc,
    scaled_cost_matrix,
)
from common.evalkit import leakage  # noqa: E402
from common.paths import CONTRACTS_DIR, stage_path  # noqa: E402
from common.seed import SEED, set_global_seed  # noqa: E402

STAGE = 1
PROBA_COLS = ["p_on_time", "p_at_risk", "p_critical"]
# Models whose name starts with one of these are the "beat me first" baselines, not candidates for champion.
# Role 1's file names them majority, rule_dwell and logreg; the other spellings are kept for older files.
BASELINE_PREFIXES = ("majority", "rule", "logreg", "logistic", "baseline")
FALLBACK_BLOCK_ROWS = 100  # used only when no timestamps are available to form day-blocks
MIN_SLICE_ROWS = 50        # a slice with fewer test rows is too small to rank as "worst"


# ------------------------------------------------------------------ loading

def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, default=stage_path(STAGE, "predictions", "ml_predictions.csv"))
    ap.add_argument("--prepared", type=Path, default=None, help="prepared dataset (default: Stage01_ML/data/prepared_dataset.csv[.gz])")
    ap.add_argument("--model-card", type=Path, default=stage_path(STAGE, "artifacts", "model_card.json"))
    ap.add_argument("--out-dir", type=Path, default=None, help="where to write eval_* files (default: Stage01_ML/reports; a custom folder also disables the ledger)")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples (1000 for real runs)")
    ap.add_argument("--no-ledger", action="store_true", help="do not touch claims/CLAIMS_LEDGER.md")
    return ap.parse_args()


def find_prepared(explicit: Path | None) -> Path | None:
    """Return the prepared dataset path if it exists (either .csv.gz or .csv)."""
    candidates = [explicit] if explicit else [stage_path(STAGE, "data", "prepared_dataset.csv.gz"), stage_path(STAGE, "data", "prepared_dataset.csv")]
    return next((p for p in candidates if p and p.exists()), None)


def load_predictions(path: Path) -> pd.DataFrame:
    """Load Role 1's predictions after checking they follow the shared contract."""
    report = contracts.validate_csv("ml_predictions", path)
    if not report.ok:
        for violation in report.violations[:10]:
            print(f"  - {violation}")
        raise SystemExit("ml_predictions.csv breaks the contract; ask Role 1 to fix it (scripts/validate_contract.py shows details).")
    df = pd.read_csv(path)
    return df.sort_values(["model_name", "record_id"]).reset_index(drop=True)


def check_integrity(pred: pd.DataFrame, prepared: pd.DataFrame | None) -> dict:
    """Make sure the predictions file can be trusted before any number is computed from it.

    Hard errors (the run stops): models scored different rows, the split or label in the file disagrees with
    the frozen prepared dataset, or records are unknown to it. Warnings: a model skipped some rows, or the
    y_pred column is not the argmax of the probabilities.
    """
    errors, warnings = [], []

    # Every model must have scored exactly the same rows in each split, or comparisons are not like for like.
    for split, part in pred.groupby("split"):
        sets = {m: frozenset(g["record_id"]) for m, g in part.groupby("model_name")}
        reference = max(sets.values(), key=len)
        for model, ids in sets.items():
            if ids != reference:
                errors.append(f"model '{model}' scored a different set of {split} rows ({len(ids)} vs {len(reference)}).")

    if prepared is not None:
        truth = prepared.drop_duplicates("record_id").set_index("record_id")
        known = pred["record_id"].isin(truth.index)
        if not known.all():
            errors.append(f"{int((~known).sum())} prediction rows have a record_id that is not in the prepared dataset.")
        joined = pred[known].join(truth[["split", "delay_status"]].rename(columns={"split": "prep_split", "delay_status": "prep_label"}), on="record_id")
        bad_split = int((joined["split"] != joined["prep_split"]).sum())
        bad_label = int((joined["y_true"] != joined["prep_label"]).sum())
        if bad_split:
            errors.append(f"{bad_split} prediction rows sit in a different split than the frozen prepared dataset (possible re-split or leakage).")
        if bad_label:
            errors.append(f"{bad_label} prediction rows have a y_true that differs from delay_status in the prepared dataset.")
        # Coverage: a model that silently skips hard test rows would look better than it is.
        for split in ("val", "test"):
            expected = int((truth["split"] == split).sum())
            got = pred[pred["split"] == split].groupby("model_name")["record_id"].nunique()
            for model, n in got.items():
                if n < expected:
                    warnings.append(f"model '{model}' covers {n} of {expected} {split} rows ({n / expected:.1%}).")

    # y_pred should be the most likely class. We recompute it from the probabilities for every metric.
    recomputed = np.array(CLASSES)[pred[PROBA_COLS].to_numpy().argmax(axis=1)]
    mismatch = float((recomputed != pred["y_pred"].to_numpy()).mean())
    if mismatch > 0.001:
        warnings.append(f"y_pred differs from the argmax of the probabilities in {mismatch:.1%} of rows; metrics use the recomputed argmax.")
    return {"errors": errors, "warnings": warnings, "y_pred_mismatch_share": mismatch}


def attach_context(pred: pd.DataFrame, prepared: pd.DataFrame | None) -> pd.DataFrame:
    """Add timestamp and slice columns from the prepared dataset, and a 'block' column for the bootstrap."""
    if prepared is not None:
        keep = [c for c in ["record_id", "timestamp", "current_node_id", "carrier_code", "priority_tier", "capacity_utilization_pct"] if c in prepared.columns]
        pred = pred.merge(prepared[keep].drop_duplicates("record_id"), on="record_id", how="left")
    if "timestamp" in pred.columns and pred["timestamp"].notna().all():
        # One block per calendar day: shipments on the same day share weather and congestion.
        pred["block"] = pd.to_datetime(pred["timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    else:
        # No timestamps: use runs of consecutive record ids as blocks (rougher, and we say so in the report).
        pred["block"] = (pred.groupby("model_name").cumcount() // FALLBACK_BLOCK_ROWS).astype(str)
        print("WARNING: no timestamps found, bootstrap blocks are runs of 100 rows. Intervals may be too narrow.")
    return pred


# ------------------------------------------------------------------ per-model numbers

def class_index(labels: pd.Series) -> np.ndarray:
    """Class names -> 0/1/2."""
    return labels.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


def model_arrays(df: pd.DataFrame, model: str, split: str) -> dict:
    """Pull one model's rows for one split as plain arrays."""
    part = df[(df["model_name"] == model) & (df["split"] == split)]
    proba = part[PROBA_COLS].to_numpy()
    # The most likely class is recomputed from the probabilities, not trusted from the y_pred column.
    return {"t": class_index(part["y_true"]), "argmax": proba.argmax(axis=1), "proba": proba,
            "blocks": part["block"].to_numpy(), "frame": part}


def is_baseline(name: str) -> bool:
    """True for the simple comparison models."""
    return name.lower().startswith(BASELINE_PREFIXES)


def alert_stats(t: np.ndarray, pred: np.ndarray) -> dict:
    """Precision, recall and false alarms per 100 alerts for the CRITICAL alert."""
    alert, crit = pred == 2, t == 2
    tp, n_alert = int((alert & crit).sum()), int(alert.sum())
    return {"alerts": n_alert,
            "alert_precision": tp / n_alert if n_alert else float("nan"),
            "critical_recall": tp / int(crit.sum()) if crit.sum() else float("nan"),
            "false_alarms_per_100_alerts": 100.0 * (n_alert - tp) / n_alert if n_alert else float("nan")}


def evaluate_models(df: pd.DataFrame, cm: np.ndarray, n_boot: int) -> tuple:
    """Score every model: choose its threshold on val, freeze it, report on test. Returns (rows, thresholds, val tables)."""
    rows, thresholds, val_tables = [], {}, {}
    cost_fn = lambda t, p: float(cm[t, p].mean())  # noqa: E731  (cost of a set of predictions)
    for model in sorted(df["model_name"].unique()):
        val, test = model_arrays(df, model, "val"), model_arrays(df, model, "test")
        # Threshold chosen on VALIDATION only; the test split never influences it.
        thr, table = choose_threshold(val["frame"]["y_true"], val["proba"], cm)
        thresholds[model], val_tables[model] = thr, table
        pred_t = decide_with_threshold(test["proba"], thr)
        is_crit = (test["t"] == 2).astype(int)
        p_crit = test["proba"][:, 2]
        prf = per_class_prf(test["t"], test["argmax"])

        f1, f1_lo, f1_hi = block_bootstrap_ci(macro_f1, test["t"], test["argmax"], blocks=test["blocks"], n=n_boot)
        ap, ap_lo, ap_hi = block_bootstrap_ci(average_precision, is_crit, p_crit, blocks=test["blocks"], n=n_boot)
        co, co_lo, co_hi = block_bootstrap_ci(cost_fn, test["t"], pred_t, blocks=test["blocks"], n=n_boot)
        # Macro-F1 of the decision the operator actually gets (the frozen alert threshold), not just the argmax.
        ft, ft_lo, ft_hi = block_bootstrap_ci(macro_f1, test["t"], pred_t, blocks=test["blocks"], n=n_boot)

        row = {"model_name": model, "model_version": test["frame"]["model_version"].iloc[0], "is_baseline": is_baseline(model),
               "n_test": len(test["t"]), "n_test_critical": int(is_crit.sum()),
               "macro_f1": f1, "macro_f1_lo": f1_lo, "macro_f1_hi": f1_hi,
               "macro_f1_at_threshold": ft, "macro_f1_at_threshold_lo": ft_lo, "macro_f1_at_threshold_hi": ft_hi,
               "balanced_accuracy": balanced_accuracy(test["t"], test["argmax"]),
               "pr_auc_critical": ap, "pr_auc_critical_lo": ap_lo, "pr_auc_critical_hi": ap_hi,
               "log_loss": log_loss(test["t"], test["proba"]), "brier_multiclass": multiclass_brier(test["t"], test["proba"]),
               "expected_cost_argmax": cost_fn(test["t"], test["argmax"]),
               "expected_cost_at_threshold": co, "expected_cost_lo": co_lo, "expected_cost_hi": co_hi,
               "threshold": thr, "val_expected_cost": float(table["expected_cost"].min()),
               "f1_on_time": prf["f1"][0], "f1_at_risk": prf["f1"][1], "f1_critical": prf["f1"][2],
               "precision_critical": prf["precision"][2], "recall_critical": prf["recall"][2]}
        row.update({f"at_threshold_{k}": v for k, v in alert_stats(test["t"], pred_t).items()})
        rows.append(row)
        print(f"  {model:<22} macro-F1(argmax) {f1:.3f} [{f1_lo:.3f}, {f1_hi:.3f}]  macro-F1@t* {ft:.3f}  PR-AUC(crit) {ap:.3f}  cost@t* {co:.3f}  t*={thr:.2f}")
    return pd.DataFrame(rows), thresholds, val_tables


def pick_champion(df: pd.DataFrame, table: pd.DataFrame, thresholds: dict, cm: np.ndarray, n_boot: int) -> dict:
    """Best candidate by validation cost, but only if it beats the best baseline (paired bootstrap on val)."""
    baselines, candidates = table[table["is_baseline"]], table[~table["is_baseline"]]
    best_base = baselines.sort_values("val_expected_cost").iloc[0]["model_name"] if len(baselines) else None
    if len(candidates) == 0:
        return {"model_name": best_base, "beats_baseline": False, "best_baseline": best_base, "reason": "no non-baseline model in the predictions file"}
    best_cand = candidates.sort_values("val_expected_cost").iloc[0]["model_name"]
    if best_base is None:
        return {"model_name": best_cand, "beats_baseline": None, "best_baseline": None, "reason": "no baseline in the predictions file, cannot test"}

    cand, base = model_arrays(df, best_cand, "val"), model_arrays(df, best_base, "val")
    pred_c = decide_with_threshold(cand["proba"], thresholds[best_cand])
    pred_b = decide_with_threshold(base["proba"], thresholds[best_base])
    # metric = negative cost, so a positive difference means the candidate is cheaper.
    neg_cost = lambda t, p: -float(cm[t, p].mean())  # noqa: E731
    diff = paired_bootstrap_diff(neg_cost, (cand["t"], pred_c), (base["t"], pred_b), cand["blocks"], n=n_boot)
    beats = diff["lo"] > 0
    chosen = best_cand if beats else best_base
    return {"model_name": chosen, "beats_baseline": bool(beats), "best_baseline": best_base, "best_candidate": best_cand,
            "val_cost_saving_vs_baseline": diff["diff"], "val_cost_saving_ci": [diff["lo"], diff["hi"]],
            "reason": "candidate beats the best baseline on validation cost (95% paired block bootstrap)" if beats
            else "no candidate clearly beats the best baseline on validation; reporting the baseline honestly"}


# ------------------------------------------------------------------ figures

def plot_model_comparison(table: pd.DataFrame, out: Path) -> None:
    """Two panels: macro-F1 with CI (higher is better) and expected cost with CI (lower is better)."""
    t = table.sort_values("macro_f1")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.6 * len(t) + 2.2), sharey=True)
    y = np.arange(len(t))
    colour = np.where(t["is_baseline"], "#9aa5b1", "#1f6feb")
    for ax, col, lo, hi, title in [(axes[0], "macro_f1", "macro_f1_lo", "macro_f1_hi", "Macro-F1 on test (higher is better)"),
                                   (axes[1], "expected_cost_at_threshold", "expected_cost_lo", "expected_cost_hi", "Expected cost per shipment on test (lower is better)")]:
        ax.barh(y, t[col], color=colour, xerr=[(t[col] - t[lo]).clip(lower=0), (t[hi] - t[col]).clip(lower=0)], ecolor="#333", capsize=3)
        ax.set_title(title, fontsize=10)
        ax.set_yticks(y)
        ax.set_yticklabels(t["model_name"])
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Grey = baselines, blue = candidate models. Bars show 95% block-bootstrap intervals.", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_confusions(df: pd.DataFrame, models: list, thresholds: dict, out: Path) -> None:
    """Counts and row-percentages for each named model at its frozen threshold (test split)."""
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6), squeeze=False)
    for ax, model in zip(axes[0], models):
        test = model_arrays(df, model, "test")
        pred = decide_with_threshold(test["proba"], thresholds[model])
        cmx = confusion_counts(test["t"], pred)
        pct = cmx / cmx.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmx[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=9, color="white" if pct[i, j] > 0.5 else "black")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["On time", "At risk", "Critical"])
        ax.set_yticklabels(["On time", "At risk", "Critical"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("What really happened")
        ax.set_title(f"{model}  (threshold {thresholds[model]:.2f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cost_curve(val_tables: dict, models: list, thresholds: dict, out: Path) -> None:
    """Validation expected cost against the alert threshold; the dot marks the chosen threshold."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for model in models:
        tab = val_tables[model]
        line, = ax.plot(tab["threshold"], tab["expected_cost"], label=model)
        best = tab.loc[tab["threshold"] == thresholds[model]].iloc[0]
        ax.plot([best["threshold"]], [best["expected_cost"]], "o", color=line.get_color())
    ax.set_xlabel("Alert threshold on P(critical)")
    ax.set_ylabel("Expected cost per shipment (validation)")
    ax.set_title("Where to put the alert threshold: lowest point of each curve", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_calibration(df: pd.DataFrame, models: list, out: Path) -> dict:
    """Reliability diagram (is 'x% critical' right x% of the time?) for each model on test; returns ECE per model."""
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect")
    eces = {}
    for model in models:
        test = model_arrays(df, model, "test")
        y, p = (test["t"] == 2).astype(int), test["proba"][:, 2]
        tab = reliability_table(y, p).dropna()
        eces[model] = ece(y, p)
        ax.plot(tab["mean_predicted"], tab["observed_rate"], "o-", label=f"{model} (ECE {eces[model]:.3f})")
    ax.set_xlabel("Predicted probability of CRITICAL")
    ax.set_ylabel("How often it really was CRITICAL")
    ax.set_title("Calibration on test", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return eces


# ------------------------------------------------------------------ decision memo

def sensitivity_table(df: pd.DataFrame, champion: str, policy: dict) -> pd.DataFrame:
    """Best threshold on val, and the resulting test alert quality, for each 'how bad is a miss' value."""
    val, test = model_arrays(df, champion, "val"), model_arrays(df, champion, "test")
    rows = []
    for miss in policy["sensitivity_grid"]["missed_critical_cost"]:
        cm_m = scaled_cost_matrix(miss, policy)
        thr, _ = choose_threshold(val["frame"]["y_true"], val["proba"], cm_m)
        stats = alert_stats(test["t"], decide_with_threshold(test["proba"], thr))
        rows.append({"missed_critical_cost": miss, "false_alarm_cost": policy["sensitivity_grid"]["false_alarm_cost"][0],
                     "threshold": thr, "test_expected_cost": float(cm_m[test["t"], decide_with_threshold(test["proba"], thr)].mean()), **stats})
    return pd.DataFrame(rows)


def split_class_mix(df: pd.DataFrame) -> dict:
    """Share of each class inside train / val / test, plus a flag when the CRITICAL share differs a lot between splits.

    Why: a threshold chosen on val only transfers to test if the two periods look alike. Seasonal data can break that.
    """
    one_model = df[df["model_name"] == df["model_name"].iloc[0]]
    mix = one_model.groupby("split")["y_true"].value_counts(normalize=True).unstack().reindex(columns=CLASSES).fillna(0.0)
    crit = mix["CRITICAL_DELAY"]
    shifted = bool(crit.max() > 2 * max(crit.min(), 1e-9))  # more than 2x apart = clearly different periods
    return {"share_by_split": mix.round(4).to_dict(orient="index"), "critical_share_differs_2x_between_splits": shifted}


def write_decision_memo(path: Path, table: pd.DataFrame, champion: dict, sens: pd.DataFrame, policy: dict, bayes: dict, mix: dict) -> None:
    """Plain-English answer to the Team Huddle question: what false-alarm level is acceptable?"""
    row = table[table["model_name"] == champion["model_name"]].iloc[0]
    budget = policy["alert_budget"]["max_false_alarms_per_100_alerts"]
    within = row["at_threshold_false_alarms_per_100_alerts"] <= budget
    lines = [
        "# Alert threshold decision (Stage 01)", "",
        f"Date: {date.today().isoformat()} - cost policy version {policy['version']} - status: PROPOSED until the Team Huddle ratifies it.", "",
        "## The question", "How many false alarms is acceptable? We answer it with costs, not opinion: a missed critical delay costs 10 units, "
        "an unnecessary emergency escalation costs 1 (see `contracts/cost_policy.json`).", "",
        "## The answer", "",
        f"- Model: **{champion['model_name']}**. Raise an alert when P(critical) >= **{row['threshold']:.2f}**. This number was chosen on the validation split and frozen before looking at test.",
        f"- On test, that alert catches **{row['at_threshold_critical_recall']:.0%}** of critical delays, **{row['at_threshold_alert_precision']:.0%}** of alerts are real, "
        f"so there are **{row['at_threshold_false_alarms_per_100_alerts']:.0f} false alarms per 100 alerts**.",
        f"- The soft budget is {budget} false alarms per 100 alerts: this model is **{'within' if within else 'OVER'}** the budget.",
        f"- Expected cost per shipment at this threshold: **{row['expected_cost_at_threshold']:.3f}** [{row['expected_cost_lo']:.3f}, {row['expected_cost_hi']:.3f}].", "",
        "## Honest comparison with the baseline",
        f"- {champion['reason']}.",
        f"- Beats the best baseline ({champion.get('best_baseline')}): **{champion.get('beats_baseline')}**.", "",
        "## Sensitivity: what if a miss is worth more or less than 10?", "",
        "| Miss costs | Threshold | Alerts | Recall of critical | False alarms /100 alerts | Test cost |", "|---|---|---|---|---|---|",
    ]
    for _, r in sens.iterrows():
        lines.append(f"| {r['missed_critical_cost']:.0f} | {r['threshold']:.2f} | {int(r['alerts'])} | {r['critical_recall']:.0%} | "
                     f"{r['false_alarms_per_100_alerts']:.0f} | {r['test_expected_cost']:.3f} |")
    if mix["critical_share_differs_2x_between_splits"]:
        shares = ", ".join(f"{s}: {v['CRITICAL_DELAY']:.1%}" for s, v in mix["share_by_split"].items())
        lines += ["", "## WARNING: the periods do not look alike",
                  f"The CRITICAL share differs more than 2x between splits ({shares}). The data is seasonal, so a threshold chosen on validation "
                  "may not suit the test period, and baseline-vs-candidate rankings on validation can flip on test. Quote test numbers together with this caveat "
                  "and raise it at the Team Huddle (options: a rolling-origin backtest, or a season-aware split decided by vote)."]
    lines += ["", "The higher a miss costs, the lower the threshold and the more false alarms we accept. Row 10 is the proposed policy.", "",
              "## Three-class decision", f"Choosing the cheapest of the three actions per shipment (Bayes decision) gives expected cost **{bayes['cost']:.3f}** and "
              f"flags **{bayes['critical_alerts']}** shipments as CRITICAL on test. With this cost matrix AT_RISK is a cheap hedge, so the three-class rule predicts it often; "
              "the single-threshold alert above is what the operator console should use.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ slices, anomaly, leakage

def slice_analysis(df: pd.DataFrame, champion: str, thr: float, out_csv: Path, out_png: Path) -> pd.DataFrame | None:
    """Metrics per node / carrier / priority / month / overload for the champion on test; None if no context columns."""
    test = df[(df["model_name"] == champion) & (df["split"] == "test")].copy()
    if "current_node_id" not in test.columns or test["current_node_id"].isna().all():
        return None
    test["month"] = pd.to_datetime(test["timestamp"], utc=True).dt.strftime("%Y-%m")
    test["overload"] = np.where(test["capacity_utilization_pct"] > 100, "over_100pct", "within_capacity")
    rows = []
    for dim in ["current_node_id", "carrier_code", "priority_tier", "month", "overload"]:
        for value, part in test.groupby(dim):
            t = class_index(part["y_true"])
            pred = decide_with_threshold(part[PROBA_COLS].to_numpy(), thr)
            stats = alert_stats(t, pred)
            rows.append({"dimension": dim, "slice": value, "n": len(part), "n_critical": int((t == 2).sum()),
                         "macro_f1": macro_f1(t, pred), "critical_recall": stats["critical_recall"],
                         "alert_precision": stats["alert_precision"]})
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    worst = out[out["n"] >= MIN_SLICE_ROWS].sort_values("macro_f1").head(10)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(worst["dimension"] + ": " + worst["slice"].astype(str), worst["macro_f1"], color="#d1495b")
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 on test")
    ax.set_title(f"Ten weakest slices with at least {MIN_SLICE_ROWS} test rows ({champion})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out


def anomaly_check(df: pd.DataFrame, champion: str, thr: float) -> dict:
    """How well does the anomaly score on its own spot CRITICAL shipments, and does it catch what the classifier missed?"""
    test = df[(df["model_name"] == champion) & (df["split"] == "test")]
    if "anomaly_score" not in test.columns or test["anomaly_score"].isna().all():
        return {"status": "NOT_RUN", "reason": "no anomaly_score in the predictions file"}
    t, score = class_index(test["y_true"]), test["anomaly_score"].fillna(0).to_numpy()
    crit = (t == 2)
    k = int(crit.sum())
    top_k = np.argsort(-score)[:k]
    pred = decide_with_threshold(test[PROBA_COLS].to_numpy(), thr)
    missed_by_classifier = crit & (pred != 2)
    flagged = np.zeros(len(t), dtype=bool)
    flagged[top_k] = True
    return {"status": "OK", "roc_auc_critical": roc_auc(crit, score), "pr_auc_critical": average_precision(crit, score),
            f"precision_at_k_{k}": float(crit[top_k].mean()) if k else float("nan"),
            "critical_missed_by_classifier": int(missed_by_classifier.sum()),
            "of_those_flagged_by_detector": int((missed_by_classifier & flagged).sum())}


def run_leakage_tests(table: pd.DataFrame, champion: str, prepared: pd.DataFrame | None, model_card: Path) -> dict:
    """The leakage tests from the Role 4 file. A test that cannot run says NOT_RUN, and one that only covered part of the features says PARTIAL; neither is shown as a plain PASS."""
    # The purge gap comes from the frozen label policy (168 h).
    purge = json.loads((CONTRACTS_DIR / "label_policy.json").read_text(encoding="utf-8"))["split_policy"]["purge_gap_hours"]
    out = {}

    # (a) every feature the model used must be on the allowed list. The card's feature list key may vary by author.
    features = None
    if model_card.exists():
        card = json.loads(model_card.read_text(encoding="utf-8"))
        features = next((card[k] for k in ("features", "feature_columns", "feature_names") if isinstance(card.get(k), list)), None)
    if features is not None:
        out["a_feature_policy"] = leakage.feature_policy_check(features)
    else:
        out["a_feature_policy"] = {"status": "NOT_RUN", "reason": "model_card.json missing or has no 'features' list"}

    if prepared is None:
        for key in ("b_label_shuffle_sanity", "c_single_feature_ceiling", "c2_bookkeeping_ceiling", "d_time_order"):
            out[key] = {"status": "NOT_RUN", "reason": "prepared dataset not found"}
    else:
        labelled = prepared[prepared["delay_status"].notna()]
        # Test the features the model really used when the card lists them, otherwise the allowed base features.
        tested = features or leakage.load_feature_policy()["allowed_base_features"]
        usable = [f for f in tested if f in labelled.columns]
        train, test = labelled[labelled["split"] == "train"], labelled[labelled["split"] == "test"]
        is_crit_train = (train["delay_status"] == "CRITICAL_DELAY").to_numpy()
        out["b_label_shuffle_sanity"] = leakage.shuffle_label_probe(train, test, usable, "delay_status", seed=SEED)
        out["c_single_feature_ceiling"] = leakage.single_feature_ceiling(train, tested, is_crit_train)
        out["c2_bookkeeping_ceiling"] = leakage.bookkeeping_ceiling(train, is_crit_train)
        out["d_time_order"] = leakage.time_order_check(labelled, purge)

    # (e) a macro-F1 above 0.95 (by either rule) is more likely a leak than a triumph.
    row = table.loc[table["model_name"] == champion].iloc[0]
    out["e_too_good_to_be_true"] = leakage.too_good_alarm(float(max(row["macro_f1"], row["macro_f1_at_threshold"])))
    statuses = [v.get("status") for v in out.values() if isinstance(v, dict)]
    out["overall"] = "FAIL" if "FAIL" in statuses else ("PARTIAL" if ("PARTIAL" in statuses or "NOT_RUN" in statuses) else "PASS")
    return out


# ------------------------------------------------------------------ ledger

def write_ledger(table: pd.DataFrame, champion: dict, leak: dict, eces: dict) -> None:
    """One ledger row per number we may quote in a debate. Rows start as PENDING until a clean-clone run verifies them."""
    c = table[table["model_name"] == champion["model_name"]].iloc[0]
    base_name = champion.get("best_baseline")
    b = table[table["model_name"] == base_name].iloc[0] if base_name else None
    cmd, src = "python Stage01_ML/04_evaluation_engineer.py", "Stage01_ML/reports/eval_metrics.json"
    fmt = lambda v, lo, hi: f"{v:.3f} [{lo:.3f}, {hi:.3f}]"  # noqa: E731
    name = champion["model_name"]
    rows = [
        ("C-ML-01", f"{name} macro-F1 on the held-out test period (most-likely-class rule)", fmt(c.macro_f1, c.macro_f1_lo, c.macro_f1_hi), src),
        ("C-ML-02", f"{base_name} (best baseline) macro-F1 on test (most-likely-class rule)", fmt(b.macro_f1, b.macro_f1_lo, b.macro_f1_hi) if b is not None else "n/a", src),
        ("C-ML-03", f"{name} PR-AUC for CRITICAL_DELAY on test", fmt(c.pr_auc_critical, c.pr_auc_critical_lo, c.pr_auc_critical_hi), src),
        ("C-ML-04", "Alert threshold on P(critical), chosen on validation by minimum cost", f"{c.threshold:.2f}", "Stage01_ML/reports/eval_champion.json"),
        ("C-ML-05", f"{name} expected cost per shipment at that threshold (test)", fmt(c.expected_cost_at_threshold, c.expected_cost_lo, c.expected_cost_hi), src),
        ("C-ML-06", "Share of CRITICAL delays caught at the frozen threshold (test)", f"{c.at_threshold_critical_recall:.3f}", src),
        ("C-ML-07", "False alarms per 100 alerts at the frozen threshold (test)", f"{c.at_threshold_false_alarms_per_100_alerts:.1f}", src),
        ("C-ML-08", "Scoring sanity check: model trained on shuffled labels scores near chance (does not prove absence of leakage)", _fmt_leak(leak["b_label_shuffle_sanity"], "test_macro_f1_with_shuffled_labels"), "Stage01_ML/reports/eval_leakage_tests.json"),
        ("C-ML-09", "Strongest single feature's AUC for CRITICAL (must stay under 0.95)", _fmt_leak(leak["c_single_feature_ceiling"], "max_auc"), "Stage01_ML/reports/eval_leakage_tests.json"),
        ("C-ML-10", "Time-order and purge-gap check between train, val and test", leak["d_time_order"]["status"], "Stage01_ML/reports/eval_leakage_tests.json"),
        ("C-ML-11", f"{name} calibration error (ECE) for P(critical) on test", f"{eces.get(name, float('nan')):.3f}", "Stage01_ML/reports/eval_calibration.png"),
        ("C-ML-12", "Does the champion beat the best baseline on validation cost?", str(champion.get("beats_baseline")), "Stage01_ML/reports/eval_champion.json"),
        ("C-ML-13", f"{name} macro-F1 at the frozen alert threshold (the decision the operator gets)", fmt(c.macro_f1_at_threshold, c.macro_f1_at_threshold_lo, c.macro_f1_at_threshold_hi), src),
        ("C-ML-14", "Id-column check: how much more than the clock an id's numeric order says about CRITICAL, in AUC (must stay under 0.05)", _fmt_leak(leak["c2_bookkeeping_ceiling"], "excess_over_clock"), "Stage01_ML/reports/eval_leakage_tests.json"),
    ]
    for cid, claim, value, source in rows:
        ledger.upsert(cid, 1, claim, value, source, cmd, status="PENDING")


def _fmt_leak(result: dict, key: str) -> str:
    """Format a leakage number, or say that the test could not run."""
    return f"{result[key]:.3f}" if key in result else f"{result['status']}"


# ------------------------------------------------------------------ main

def main() -> int:
    """Run the whole Day 1 evaluation."""
    args = parse_args()
    set_global_seed()  # same random numbers on every laptop
    out_dir = args.out_dir or stage_path(STAGE, "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    policy, cm = load_cost_policy(), load_cost_matrix()

    if not args.predictions.exists():
        print(f"WAITING: {args.predictions} does not exist yet. Ask Role 1 for baseline predictions (contracts/fixtures/ml_predictions.csv shows the format).")
        return 3  # same code the stubs use, so run_pipeline.py shows SKIPPED instead of FAILED

    # Step 1: load predictions, plus the prepared dataset when it exists (for timestamps, slices, leakage tests).
    prepared_path = find_prepared(args.prepared)
    prepared = pd.read_csv(prepared_path) if prepared_path else None
    print(f"predictions: {args.predictions}\nprepared dataset: {prepared_path or 'NOT FOUND (slices and leakage tests will say NOT_RUN)'}")
    raw_pred = load_predictions(args.predictions)

    # Step 1b: trust check. A broken predictions file stops the run (exit 5) before any number is computed.
    integrity = check_integrity(raw_pred, prepared)
    for note in integrity["warnings"]:
        print(f"WARNING: {note}")
    if integrity["errors"]:
        for note in integrity["errors"]:
            print(f"ERROR: {note}")
        print("STOPPED: fix the predictions file (ask Role 1) and run again. No evaluation files were written.")
        return 5
    df = attach_context(raw_pred, prepared)
    if not ({"val", "test"} <= set(df["split"])):
        raise SystemExit("predictions need both 'val' and 'test' rows (val picks the threshold, test reports it).")

    # Step 2: headline table with a frozen, validation-chosen threshold per model.
    print("scoring models on test:")
    table, thresholds, val_tables = evaluate_models(df, cm, args.n_boot)
    table.to_csv(out_dir / "eval_model_comparison.csv", index=False)
    plot_model_comparison(table, out_dir / "eval_model_comparison.png")

    # Step 3: champion = best candidate, only if it really beats the best baseline.
    champion = pick_champion(df, table, thresholds, cm, args.n_boot)
    name = champion["model_name"]
    print(f"champion: {name}  (beats baseline: {champion['beats_baseline']})")
    champ_row = table[table["model_name"] == name].iloc[0]

    # Step 3b: leakage tests come BEFORE anything is published. A FAIL (or a SUSPECT score) blocks the champion file
    # and the ledger, so a leaky model can never be served by the API or quoted in a debate by accident.
    leak = run_leakage_tests(table, name, prepared, args.model_card)
    (out_dir / "eval_leakage_tests.json").write_text(json.dumps(leak, indent=2, default=float), encoding="utf-8")
    if leak["overall"] == "FAIL":
        failed = [k for k, v in leak.items() if isinstance(v, dict) and v.get("status") == "FAIL"]
        print(f"BLOCKED: leakage tests failed: {failed}. No champion file and no ledger rows were written. See eval_leakage_tests.json.")
        return 4
    (out_dir / "eval_champion.json").write_text(json.dumps({
        "stage": STAGE, "model_name": name, "model_version": str(champ_row["model_version"]), "threshold": thresholds[name],
        "selected_on": "val", "cost_policy_version": policy["version"], "date": date.today().isoformat(),
        "beats_baseline": champion["beats_baseline"]}, indent=2), encoding="utf-8")

    # Step 4: figures and the threshold memo.
    shown = [name] + ([champion["best_baseline"]] if champion.get("best_baseline") and champion["best_baseline"] != name else [])
    plot_confusions(df, shown, thresholds, out_dir / "eval_confusion.png")
    plot_cost_curve(val_tables, shown, thresholds, out_dir / "eval_cost_curve.png")
    # Show the champion next to its calibrated or uncalibrated twin (xgboost vs xgboost_cal), whichever it is.
    twin = name[:-4] if name.endswith("_cal") else f"{name}_cal"
    calib_models = [name] + [m for m in df["model_name"].unique() if m == twin]
    eces = plot_calibration(df, calib_models, out_dir / "eval_calibration.png")
    sens = sensitivity_table(df, name, policy)
    sens.to_csv(out_dir / "eval_threshold_sensitivity.csv", index=False)
    test_c = model_arrays(df, name, "test")
    bayes_pred = bayes_decision(test_c["proba"], cm)
    bayes = {"cost": float(cm[test_c["t"], bayes_pred].mean()), "critical_alerts": int((bayes_pred == 2).sum()), "macro_f1": macro_f1(test_c["t"], bayes_pred)}
    mix = split_class_mix(df)
    write_decision_memo(out_dir / "eval_threshold_decision.md", table, champion, sens, policy, bayes, mix)

    # Step 5: slices and the anomaly detector.
    slices = slice_analysis(df, name, thresholds[name], out_dir / "eval_slices.csv", out_dir / "eval_slices.png")
    anomaly = anomaly_check(df, name, thresholds[name])
    (out_dir / "eval_anomaly.json").write_text(json.dumps(anomaly, indent=2), encoding="utf-8")

    # Step 6: one JSON with everything, for the dashboard and the report.
    (out_dir / "eval_metrics.json").write_text(json.dumps({
        "stage": STAGE, "date": date.today().isoformat(), "seed": SEED, "bootstrap_resamples": args.n_boot,
        "integrity": integrity, "champion": champion, "bayes_three_class": bayes, "split_class_mix": mix, "slices_available": slices is not None,
        "models": table.to_dict(orient="records")}, indent=2, default=float), encoding="utf-8")

    # Step 7: claims ledger, only for real runs (a custom --out-dir means a dry run).
    if not args.no_ledger and args.out_dir is None:
        write_ledger(table, champion, leak, eces)
        print("ledger: 14 rows written to claims/CLAIMS_LEDGER.md (status PENDING until a clean-clone run)")
    print(f"leakage tests overall: {leak['overall']}\nwritten to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
