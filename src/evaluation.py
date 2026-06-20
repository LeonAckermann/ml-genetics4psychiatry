"""Metric computation and reporting for nested CV."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R², Pearson r/r², Spearman ρ/ρ² with p-values."""
    from sklearn.metrics import r2_score

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    r, pearson_p = pearsonr(y_true, y_pred)
    if np.isnan(r):
        r, pearson_p = 0.0, 1.0

    rho, spearman_p = spearmanr(y_true, y_pred)
    if np.isnan(rho):
        rho, spearman_p = 0.0, 1.0

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(r),
        "pearson_r2": float(r ** 2),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(rho),
        "spearman_rho2": float(rho ** 2),
        "spearman_p": float(spearman_p),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """Accuracy, precision, recall, F1, balanced accuracy, AUC-ROC, Pearson r², confusion matrix.

    threshold: decision boundary applied to y_pred to produce binary predictions.
    If None, defaults to 0.5 when predictions are in [0, 1], else 0.0.
    Pass threshold=1.0 for MDN outputs (expected Z-score in sign-flipped space).
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true_bin = np.asarray(y_true, dtype=int).ravel()
    preds = np.asarray(y_pred, dtype=float).ravel()
    if threshold is None:
        threshold = 0.5 if np.all((preds >= 0.0) & (preds <= 1.0)) else 0.0
    y_pred_bin = (preds >= threshold).astype(int)

    pearson_r2, pearson_p = 0.0, 1.0
    if np.unique(y_true_bin).size > 1 and np.unique(preds).size > 1:
        r, p = pearsonr(y_true_bin, preds)
        if not np.isnan(r):
            pearson_r2, pearson_p = float(r ** 2), float(p)

    # AUC-ROC requires both classes to be present in y_true.
    try:
        roc_auc = float(roc_auc_score(y_true_bin, preds))
    except ValueError:
        roc_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_bin, y_pred_bin)),
        "roc_auc": roc_auc,
        "pearson_r2": pearson_r2,
        "pearson_p": pearson_p,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> dict:
    """Dispatch to classification or regression metrics based on task_type."""
    if task_type == "binary_classification":
        return classification_metrics(y_true, y_pred)
    return regression_metrics(y_true, y_pred)


def label_distribution(y, is_binary: bool = True) -> dict:
    """Per-class counts and fractions.

    For binary tasks counts class 0/1.
    For regression tasks counts negative/non-negative values.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    if is_binary:
        y_int = y_arr.astype(int)
        counts = np.bincount(y_int, minlength=2)
        total = int(counts.sum())
        return {
            "counts": {"negative": int(counts[0]), "positive": int(counts[1])},
            "fractions": {
                "negative": float(counts[0] / total) if total else 0.0,
                "positive": float(counts[1] / total) if total else 0.0,
            },
            "total": total,
        }
    neg = int(np.sum(y_arr < 0))
    pos = int(np.sum(y_arr >= 0))
    total = neg + pos
    return {
        "counts": {"negative": neg, "positive": pos},
        "fractions": {
            "negative": float(neg / total) if total else 0.0,
            "positive": float(pos / total) if total else 0.0,
        },
        "total": total,
    }


def report_fold_metrics(
    metrics: dict,
    fold: int,
    outer_cv: int,
    experiment_name: str,
    task_type: str,
) -> None:
    tag = f"[{experiment_name}] " if experiment_name else ""
    print(f"  {tag}Fold {fold}/{outer_cv}")
    if task_type == "binary_classification":
        print(f"    Accuracy:     {metrics['accuracy']:.4f}")
        print(f"    Precision:    {metrics['precision']:.4f}")
        print(f"    Recall:       {metrics['recall']:.4f}")
        print(f"    F1:           {metrics['f1']:.4f}")
        print(f"    Balanced Acc: {metrics['balanced_accuracy']:.4f}")
        print(f"    AUC-ROC:      {metrics['roc_auc']:.4f}")
        print(f"    Pearson r²:   {metrics['pearson_r2']:.4f}  (p={metrics['pearson_p']:.4e})")
        print(f"    TP/FP/FN/TN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']}")
    else:
        print(f"    R²:           {metrics['r2']:.4f}")
        print(f"    Pearson r:    {metrics['pearson_r']:.4f}")
        print(f"    Pearson r²:   {metrics['pearson_r2']:.4f}  (p={metrics['pearson_p']:.4e})")
        print(f"    Spearman ρ²:  {metrics['spearman_rho2']:.4f}  (p={metrics['spearman_p']:.4e})")


MDN_CONFIDENCE_THRESHOLDS: list[float] = [round(t, 2) for t in np.arange(0.0, 0.95, 0.05)]


def mdn_confidence_metrics(
    pi: np.ndarray,
    mu: np.ndarray,
    y_true: np.ndarray,
    thresholds: list[float] = MDN_CONFIDENCE_THRESHOLDS,
) -> list[dict]:
    """Per-threshold balanced accuracy for one MDN fold.

    Confidence = margin between the top-1 and top-2 mixing weights.
    A sample is predicted positive when its winning component's mu >= 0.
    """
    from sklearn.metrics import balanced_accuracy_score

    # Z >= 1 threshold in sign-flipped space: positive class = high |Z| cluster
    y_true_bin = (np.asarray(y_true) >= 1).astype(int)
    best_component = np.argmax(pi, axis=1)
    best_mu = mu[np.arange(len(mu)), best_component]
    y_pred = (best_mu >= 1).astype(int)

    sorted_pi = np.sort(pi, axis=1)[:, ::-1]
    margins = sorted_pi[:, 0] - sorted_pi[:, 1]

    results = []
    for t in thresholds:
        confident = margins >= t
        n_confident = int(confident.sum())
        n_total = len(y_true)

        if n_confident == 0 or len(np.unique(y_true_bin[confident])) < 2:
            bal_acc = None
        else:
            bal_acc = float(balanced_accuracy_score(y_true_bin[confident], y_pred[confident]))

        results.append({
            "threshold": t,
            "n_confident": n_confident,
            "n_total": n_total,
            "coverage": float(n_confident / n_total),
            "balanced_accuracy": bal_acc,
        })
    return results


def aggregate_confidence_metrics(fold_conf: list[list[dict]]) -> list[dict]:
    """Average per-threshold results across folds."""
    if not fold_conf:
        return []
    thresholds = [r["threshold"] for r in fold_conf[0]]
    aggregated = []
    for t_idx, t in enumerate(thresholds):
        bal_accs = [
            f[t_idx]["balanced_accuracy"] for f in fold_conf
            if f[t_idx]["balanced_accuracy"] is not None
        ]
        n_confs = [f[t_idx]["n_confident"] for f in fold_conf]
        coverages = [f[t_idx]["coverage"] for f in fold_conf]
        aggregated.append({
            "threshold": t,
            "mean_balanced_accuracy": float(np.mean(bal_accs)) if bal_accs else None,
            "std_balanced_accuracy": float(np.std(bal_accs)) if len(bal_accs) > 1 else 0.0,
            "n_folds_with_confident_samples": len(bal_accs),
            "mean_n_confident": float(np.mean(n_confs)),
            "mean_coverage": float(np.mean(coverages)),
        })
    return aggregated


def aggregate_metrics(fold_metrics: list[dict], task_type: str, experiment_name: str) -> dict:
    """Mean ± std over folds with a printed summary."""
    tag = f"[{experiment_name}] " if experiment_name else ""
    keys = (
        ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "roc_auc", "pearson_r2", "pearson_p"]
        if task_type == "binary_classification"
        else ["r2", "pearson_r", "pearson_r2", "pearson_p", "spearman_rho", "spearman_rho2", "spearman_p"]
    )
    result: dict = {}
    print(f"\n{tag}=== Nested CV Summary ({task_type}) ===")
    for key in keys:
        vals = [m[key] for m in fold_metrics if key in m]
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))
        result[f"mean_{key}"] = mean
        result[f"std_{key}"] = std
        print(f"  {key:<22} {mean:.4f} ± {std:.4f}")
    return result
