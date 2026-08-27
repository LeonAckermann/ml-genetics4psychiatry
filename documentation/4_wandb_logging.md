# 4. Performance report — Weights & Biases logging

Uploads training curves and final metrics to [Weights & Biases](https://wandb.ai), as a complement to the local JSON report (see the README's §4). Off by default (`wandb.enabled: false`). Implemented in [`src/log.py`](../src/log.py), called from [`src/cv.py::nested_cv`](../src/cv.py).

## Auth

Nothing in this repo reads, stores, or asks for a W&B API key — `wandb.init()` picks it up exactly the way the plain W&B SDK always does:

```bash
wandb login          # once, interactively — stores a key in ~/.netrc
# or
export WANDB_API_KEY=...   # e.g. in a batch script / cluster job
```

If neither is set, `wandb.init()` will prompt for a key on first use (interactively) or fail (in a non-interactive job) — not something this integration adds or changes.

## One run per experiment, logged live

```
nested_cv() starts
        │
        ▼
wandb.init(project, entity, mode, tags, name=experiment_name, config=cfg)
        │
        ▼
for each outer fold:
    define_metric(fold_{k}/epoch)  as that fold's own x-axis
        │
        ▼
    train  ──▶  every epoch / boosting round, the instant it's computed:
                  wandb.log({fold_{k}/epoch, .../train_loss, .../test_loss,
                              .../train_{score}, .../test_{score}})
                  ── visible in the W&B dashboard immediately, while the fold
                     is still training, not after it finishes
        │
        ▼
after all outer folds:
    log_hyperparameter_table()  →  one wandb.Table, one row per fold
    log_performance_table()       →  one wandb.Table, one row per fold + mean/std rows
    log_summary()                    →  final aggregated (mean/std) metrics as run summary
    finish_run()                       →  wandb.finish()
```

Every outer fold shares one run rather than getting its own — keeps the W&B project table at one row per experiment instead of `outer_cv` (default 5) rows. `reinit="finish_previous"` lets a single `main.py` process safely start a new run for the next experiment in a multi-experiment job (`model.names`, multiple phenotypes, …) without needing to restart the process.

## Live logging, per model family

Implemented via an `on_epoch` callback threaded through [`src/training.py`](../src/training.py), built per-fold by [`make_epoch_logger`](../src/log.py):

- **DNN / ResidualDNN / MDN**: called once per epoch, right where the training loop already builds that epoch's `history` record (`src/training.py::_train_dnn` / `train_mdn`) — genuinely live, since these are plain Python loops.
- **XGBoost**: `.fit()` is a single blocking call into XGBoost's C++ training loop, so there's no Python-level epoch loop to hook into directly. Instead, `_make_xgb_live_callback` builds an `xgboost.callback.TrainingCallback` whose `after_iteration` hook fires once per boosting round *during* `.fit()` — confirmed by test: for `n_estimators: 15` over 2 outer folds, exactly 30 live calls are logged (one per round per fold), not 1 batched call at the end. This works whether or not `early_stopping_rounds` is set — the curve isn't tied to early stopping, only to `training_curves.enabled` (on by default).
- **Linear / Lasso / Ridge / TabPFN**: closed-form or single-shot fits — no per-step curve exists to stream, so a run for one of these gets the config and final-metric summary only, no `fold_*/epoch` charts.

## Per-fold hyperparameters

Each outer fold's selected hyperparameters (the Optuna-search result under nested CV, or the pinned config values when there's no search) are logged the moment they're resolved — **before that fold's model is even built**, not after training finishes:

- `log_fold_hyperparameters()` writes them to the run summary under `fold_{k}/hyperparameters` immediately (confirmed by test to land before that fold's first `fold_{k}/epoch` point).
- `log_hyperparameter_table()` additionally logs a `wandb.Table` (one row per fold, one column per hyperparameter key seen across folds) once every fold is done, for a side-by-side comparison view in the W&B UI.

Populated for any model with hyperparameters to select (via HPO or pinned config) — not restricted to a specific model family, since `fold_best_params` already exists generically in `nested_cv()`.

## Performance table

The same per-fold metrics and aggregated mean/std that [`aggregate_metrics()`](../src/evaluation.py) already prints to the console (`r2  -0.1479 ± 0.0674`, …) and writes into the local result JSON, logged once as a `wandb.Table` (`log_performance_table()`) after every fold is done: one row per fold (`fold 1`, `fold 2`, …), plus a `mean` row and a `std` row at the bottom — the two together are exactly the "mean ± std" the console prints, just as two numeric rows instead of one formatted string, so the columns stay sortable/plottable in the W&B UI. Columns are whatever [`compute_metrics()`](../src/evaluation.py) produced for the task type (`r2`/`pearson_r`/… for regression, `accuracy`/`balanced_accuracy`/… for classification).

## Why each fold gets its own x-axis, not a shared step counter

XGBoost with early stopping can stop at a different boosting round in each fold; DNN-family models can early-stop at a different epoch too. Logging every fold's curve against one shared W&B "step" would misalign them (fold 1's round 20 isn't the same point as fold 2's round 20 if they stopped at different lengths). `wandb.define_metric(f"fold_{k}/epoch")` + `define_metric(f"fold_{k}/*", step_metric=f"fold_{k}/epoch")` gives each fold's live metrics their own x-axis, so a fold that stops early just produces a shorter live line, not a misaligned one.

## What gets logged

- **Config** (`wandb.init(config=cfg)`): the full per-iteration config for that experiment — model, data, hpo, etc. — so runs are comparable and reproducible from the W&B UI alone.
- **`fold_{k}/epoch`, `fold_{k}/train_loss`, `fold_{k}/test_loss`, `fold_{k}/train_{score}`, `fold_{k}/test_{score}`**: live, one point per epoch/round, per fold, as training happens.
- **`fold_{k}/hyperparameters`** (run summary) and the **`fold_hyperparameters`** table: see above.
- **`performance_summary`** table: per-fold + mean/std metrics, see above.
- **Run summary**: the same aggregated mean/std metrics that land in the local result JSON's top level (`mean_r2`, `std_r2`, …).

No end-of-run combined training-curve chart is logged — the live `fold_{k}/*` points during training are the only curve view. (The local result JSON still has the full `training_curves` array regardless of `wandb.enabled` — this only affects what's uploaded to W&B.)

Only populated for models that actually produce a curve — currently `xgboost`, `dnn`, `residual_dnn`, `mdn` (see [`epoch_history`](../src/cv.py) and [`_xgb_curve_records`](../src/training.py)).

## Config

```yaml
wandb:
  enabled: true
  project: ml-genetics4psychiatry   # default
  entity: your-team-or-username      # optional; null = your W&B account default
  mode: online                        # online (default) | offline | disabled
  tags: [xgboost, adhd]                # optional
```

| Key | Default | Notes |
|---|---|---|
| `enabled` | `false` | Master switch — everything else is inert when off. |
| `project` | `"ml-genetics4psychiatry"` | W&B project name. |
| `entity` | `null` | W&B team/username; unset uses your account's default. |
| `mode` | `"online"` | `"offline"` writes locally under `./wandb/` for a later `wandb sync` (useful on an offline cluster node); `"disabled"` no-ops the W&B SDK entirely while keeping the same code path. |
| `tags` | `null` | Freeform tags for filtering runs in the W&B UI. |

`./wandb/` (local run cache, used in offline mode or as W&B's working directory generally) is gitignored.
