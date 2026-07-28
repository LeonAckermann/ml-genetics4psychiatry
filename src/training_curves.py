"""Training-curve plots for the iterative models (DNN, MDN, XGBoost).

Fed by the per-step records that ``src.training.train`` appends to its
``history`` list and that ``src.cv.nested_cv`` collects per outer fold. The
step is an epoch for the neural models and a boosting round for XGBoost;
``step_name`` in each record drives the axis label. Writes to
``<results_dir>/training/`` when ``cfg["training_curves"]["plots"]`` is on.

Two curves per fold — loss and the task score (accuracy for classification, R²
for regression) — each with the outer-fold train and test split drawn together,
plus one overview figure across folds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# One hue per split, held fixed across every figure so the identity of a line
# never changes between the per-fold and overview plots.
TRAIN_COLOR = "#2a78d6"
TEST_COLOR = "#e07b39"
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def _score_name(fold_curves: list[dict]) -> str:
    for f in fold_curves:
        for rec in f.get("epochs", []):
            if rec.get("score_name"):
                return rec["score_name"]
    return "score"


def _step_name(fold_curves: list[dict]) -> str:
    """x-axis unit: 'epoch' for the neural models, 'boosting round' for XGBoost."""
    for f in fold_curves:
        for rec in f.get("epochs", []):
            if rec.get("step_name"):
                return rec["step_name"]
    return "epoch"


def _series(epochs: list[dict], key: str) -> np.ndarray:
    return np.array([rec.get(key, np.nan) for rec in epochs], dtype=float)


def _style(ax) -> None:
    ax.grid(True, alpha=0.18, linewidth=0.8, color=MUTED)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)


def _plot_fold(ax_loss, ax_score, epochs: list[dict], score_name: str,
               step_name: str = "epoch") -> None:
    x = _series(epochs, "epoch")
    ax_loss.plot(x, _series(epochs, "train_loss"), lw=2, color=TRAIN_COLOR, label="train")
    ax_loss.plot(x, _series(epochs, "test_loss"), lw=2, color=TEST_COLOR, label="test")

    ax_score.plot(x, _series(epochs, f"train_{score_name}"), lw=2, color=TRAIN_COLOR, label="train")
    ax_score.plot(x, _series(epochs, f"test_{score_name}"), lw=2, color=TEST_COLOR, label="test")

    # Mark the epoch early stopping selected — the weights actually used.
    val = _series(epochs, "val_loss")
    if np.isfinite(val).any():
        best = int(np.nanargmin(val))
        for ax in (ax_loss, ax_score):
            ax.axvline(x[best], color=MUTED, lw=1, ls="--", alpha=0.7)
        ax_loss.annotate(f"best val ({step_name} {int(x[best])})", (x[best], ax_loss.get_ylim()[1]),
                         textcoords="offset points", xytext=(4, -12),
                         fontsize=8, color=MUTED)


def plot_training_curves(
    fold_curves: list[dict],
    results_dir: Path | str,
    experiment_name: str = "",
    dir_name: str = "training",
) -> Path | None:
    """Render one figure per fold plus an across-fold overview.

    ``fold_curves`` is ``[{"fold": k, "epochs": [ {...}, ... ]}, ...]``.
    Returns the output directory, or None if there was nothing to plot.
    """
    folds = [f for f in fold_curves if f.get("epochs")]
    if not folds:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(results_dir) / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    score_name = _score_name(folds)
    step_name = _step_name(folds)
    label = {"r2": "R²", "accuracy": "accuracy"}.get(score_name, score_name)

    # ── One figure per fold ──────────────────────────────────────────────────
    for f in folds:
        fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        for ax in (ax_loss, ax_score):
            ax.set_facecolor(SURFACE)
        _plot_fold(ax_loss, ax_score, f["epochs"], score_name, step_name)

        ax_loss.set_xlabel(step_name, fontsize=10, color=INK)
        ax_loss.set_ylabel("loss", fontsize=10, color=INK)
        ax_score.set_xlabel(step_name, fontsize=10, color=INK)
        ax_score.set_ylabel(label, fontsize=10, color=INK)
        for ax in (ax_loss, ax_score):
            _style(ax)
            ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
        fig.suptitle(f"Fold {f['fold']} — outer-fold train vs test per {step_name}"
                     + (f"\n{experiment_name}" if experiment_name else ""),
                     fontsize=11.5, color=INK, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out_dir / f"fold_{f['fold']}.png", facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)

    # ── Overview: every fold on shared axes, train vs test ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, ylabel) in zip(axes, [("loss", "loss"), (score_name, label)]):
        ax.set_facecolor(SURFACE)
        for f in folds:
            x = _series(f["epochs"], "epoch")
            tr = _series(f["epochs"], "train_loss" if key == "loss" else f"train_{score_name}")
            te = _series(f["epochs"], "test_loss" if key == "loss" else f"test_{score_name}")
            # Folds are not separate series — they are repeats of the same two,
            # so they share the split's color and are de-emphasised individually.
            ax.plot(x, tr, lw=1.2, color=TRAIN_COLOR, alpha=0.45)
            ax.plot(x, te, lw=1.2, color=TEST_COLOR, alpha=0.45)
        ax.plot([], [], lw=2, color=TRAIN_COLOR, label="train")
        ax.plot([], [], lw=2, color=TEST_COLOR, label="test")
        ax.set_xlabel(step_name, fontsize=10, color=INK)
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
        _style(ax)
        ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    fig.suptitle(f"All {len(folds)} outer folds"
                 + (f" — {experiment_name}" if experiment_name else ""),
                 fontsize=11.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "all_folds.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    return out_dir
