"""Experiment logging: Weights & Biases live logging, plus local
training-curve plots.

## Weights & Biases (init_run and friends)

One run per experiment (one src.cv.nested_cv call). Gated by
``cfg["wandb"]["enabled"]``; every function here is a no-op when that's off
(or when ``init_run`` was never called / returned ``None``), so callers never
need their own enabled-check before calling them.

Auth is never handled here -- wandb picks it up from the ``WANDB_API_KEY``
env var or a prior ``wandb login``, exactly as the plain wandb SDK does;
nothing in this repo reads, stores, or asks for a credential. See
documentation/4_wandb_logging.md.

All outer folds of one experiment share a single run, so a model with several
folds of different lengths (e.g. XGBoost with early stopping firing at a
different round each fold) still lines up correctly -- each fold's curve
keeps its own x-axis (epoch / boosting-round count) rather than being forced
onto one shared step counter.

## Training-curve plots (plot_training_curves)

Fed by the per-step records that ``src.training.train`` appends to its
``history`` list and that ``src.cv.nested_cv`` collects per outer fold. The
step is an epoch for the neural models and a boosting round for XGBoost;
``step_name`` in each record drives the axis label. Writes to
``<results_dir>/training/`` when ``cfg["training_curves"]["plots"]`` is on.

Two curves per fold -- loss and the task score (accuracy for classification,
R² for regression) -- each with the outer-fold train and test split drawn
together, plus one overview figure across folds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def init_run(cfg: dict, experiment_name: str, model_name: str, task_type: str):
    """Start one W&B run for this experiment, or return None if wandb.enabled is off."""
    wandb_cfg = cfg.get("wandb", {}) or {}
    if not bool(wandb_cfg.get("enabled", False)):
        return None

    import wandb

    return wandb.init(
        project=wandb_cfg.get("project", "ml-genetics4psychiatry"),
        entity=wandb_cfg.get("entity"),
        mode=wandb_cfg.get("mode", "online"),
        tags=wandb_cfg.get("tags"),
        name=experiment_name or None,
        config=cfg,
        reinit="finish_previous",
    )


def make_epoch_logger(run: Any, fold: int):
    """Return a callback that logs one training record live, the moment it's
    produced -- called from inside src/training.py's epoch/boosting-round loops
    (via `on_epoch=`). None if `run` is None, so callers can pass the result
    straight through without their own enabled-check.

    Uses `wandb.define_metric` so this fold's points plot against its own
    `fold_{k}/epoch` x-axis instead of the run's global step -- required
    because different folds (e.g. XGBoost stopping early at different rounds)
    reach different lengths, and a shared step would misalign them.
    """
    if run is None:
        return None
    import wandb

    prefix = f"fold_{fold + 1}"
    wandb.define_metric(f"{prefix}/epoch")
    wandb.define_metric(f"{prefix}/*", step_metric=f"{prefix}/epoch")

    def _on_epoch(record: dict) -> None:
        score_name = record.get("score_name", "score")
        payload = {f"{prefix}/epoch": record["epoch"]}
        for key in ("train_loss", "test_loss", f"train_{score_name}", f"test_{score_name}"):
            if key in record:
                payload[f"{prefix}/{key}"] = record[key]
        run.log(payload)

    return _on_epoch


def log_fold_hyperparameters(run: Any, fold: int, best_params: dict) -> None:
    """Record this fold's selected hyperparameters (the Optuna-search result,
    or the pinned config values when there was no search) in W&B the moment
    they're known -- called right after src/cv.py resolves them, before that
    fold's model is even built, not after training finishes. Written to the
    run summary under a per-fold key so it's visible immediately. See also
    log_hyperparameter_table(), which logs the same data again as one
    cross-fold comparison table once every fold is done."""
    if run is None:
        return
    run.summary.update({f"fold_{fold + 1}/hyperparameters": dict(best_params)})


def log_hyperparameter_table(run: Any, fold_best_params: list[dict]) -> None:
    """Log one wandb.Table row per fold (columns = every hyperparameter key
    seen across folds), so hyperparameters can be compared side by side in
    the W&B UI -- sortable, and plottable per column. Logged once, after
    every fold is done, since a Table needs all its rows to render as one
    table rather than growing param-by-param."""
    if run is None or not fold_best_params:
        return
    import wandb

    all_keys: list[str] = []
    for params in fold_best_params:
        for k in params:
            if k not in all_keys:
                all_keys.append(k)

    table = wandb.Table(columns=["fold", *all_keys])
    for i, params in enumerate(fold_best_params):
        table.add_data(i + 1, *[params.get(k) for k in all_keys])
    run.log({"fold_hyperparameters": table})


def log_summary(run: Any, aggregated: dict) -> None:
    """Log the final aggregated (mean/std across outer folds) metrics as the
    run's summary, so runs are comparable in the W&B project table."""
    if run is None:
        return
    run.summary.update(aggregated)


def log_performance_table(run: Any, fold_metrics: list[dict], aggregated: dict) -> None:
    """Log the same per-fold metrics + aggregated mean/std that
    aggregate_metrics() already prints to the console (e.g. "r2 -0.0101 ±
    0.0231") as one wandb.Table -- one row per fold, plus a `mean` and a `std`
    row at the bottom. Columns are numeric (not pre-formatted "mean ± std"
    strings) so they stay sortable/plottable in the W&B UI."""
    if run is None or not fold_metrics:
        return
    import wandb

    all_keys: list[str] = []
    for m in fold_metrics:
        for k in m:
            if k not in all_keys:
                all_keys.append(k)

    table = wandb.Table(columns=["fold", *all_keys])
    for i, m in enumerate(fold_metrics):
        table.add_data(f"fold {i + 1}", *[m.get(k) for k in all_keys])
    table.add_data("mean", *[aggregated.get(f"mean_{k}") for k in all_keys])
    table.add_data("std", *[aggregated.get(f"std_{k}") for k in all_keys])

    run.log({"performance_summary": table})


def finish_run(run: Any) -> None:
    if run is not None:
        run.finish()


# =============================================================================
# Local training-curve plots
# =============================================================================

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
