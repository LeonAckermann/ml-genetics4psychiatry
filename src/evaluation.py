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


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, precision, recall, F1, balanced accuracy, AUC-ROC, Pearson r², confusion matrix."""
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
