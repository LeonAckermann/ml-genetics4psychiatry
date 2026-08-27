# Hyperparameter space of each model

The tunable hyperparameters and default Optuna search ranges for the primary supported models (see the README's "Available models" table), as registered in `_DEFAULT_SPACES` in [`src/hpo.py`](../src/hpo.py:39). A model with no entry here has nothing to search — it always runs under [plain cross-fold validation](3_cross_fold_validation.md) (`n_trials` auto-set to `0`), even if `hpo.run: true`.

Ranges are `[low, high]`; `hpo.search_space` in the config can override any of them (see [3_nested_cross_fold_validation.md](3_nested_cross_fold_validation.md#search-space-resolution-and-pinning)).

## Linear Regression (`linear`)

No hyperparameters — plain closed-form OLS. Always runs under [plain cross-fold validation](3_cross_fold_validation.md).

## Lasso (`lasso`)

| Hyperparameter | Range | Controls |
|---|---|---|
| `alpha` | `[1e-4, 10.0]` (log-scale) | L1 penalty strength — higher values push more coefficients to exactly zero. |

## Ridge (`ridge`)

| Hyperparameter | Range | Controls |
|---|---|---|
| `alpha` | `[1e-4, 10000.0]` (log-scale) | L2 penalty strength — higher values shrink coefficients toward zero without zeroing them out. |

## XGBoost (`xgboost`)

| Hyperparameter | Range | Controls |
|---|---|---|
| `n_estimators` | `[100, 1000]` | Number of boosting rounds (trees). |
| `max_depth` | `[3, 10]` | Maximum depth of each tree — deeper trees fit more complex interactions, and overfit more easily. |
| `learning_rate` | `[0.01, 0.3]` (log-scale) | Shrinkage applied to each tree's contribution. |
| `subsample` | `[0.6, 1.0]` | Fraction of rows sampled per tree. |
| `colsample_bytree` | `[0.6, 1.0]` | Fraction of features sampled per tree. |
| `reg_alpha` | `[0.0, 1.0]` | L1 regularization on leaf weights. |
| `reg_lambda` | `[0.0, 5.0]` | L2 regularization on leaf weights. |
| `min_child_weight` | `[1, 10]` | Minimum sum of instance weight needed in a child — higher values make splits more conservative. |
| `gamma` | `[0.0, 1.0]` | Minimum loss reduction required to make a further split. |
| `early_stopping_rounds` | `[10, 50]` | Regression only. Stops boosting once the validation metric hasn't improved for this many rounds; when present, `build_model()` switches to raw `xgb.XGBRegressor` with an early-stopping callback instead of the plain wrapper class. |

## Residual DNN (`residual_dnn`)

Same search space as the plain `dnn` (shares `_DEFAULT_SPACES[("dnn", task_type)]`) — the only difference between the two is that `residual_dnn`'s architecture adds residual connections between hidden layers; the hyperparameters mean the same thing for both.

| Hyperparameter | Range | Controls |
|---|---|---|
| `hidden_dim` | `[32, 128]` | Width of each hidden layer (used with `n_layers` to build `hidden_dims = [hidden_dim] * n_layers`; a config can instead set `hidden_dims` directly as an explicit list). |
| `n_layers` | `[1, 4]` | Number of hidden layers. |
| `dropout` | `[0.0, 0.5]` | Dropout probability applied between hidden layers. |
| `learning_rate` | `[1e-4, 1e-2]` (log-scale) | Adam optimizer learning rate. |
| `batch_size` | `[16, 64]` | Mini-batch size. |
| `epochs` | `[20, 60]` | Maximum training epochs. |
| `patience` | `[5, 30]` | Early-stopping patience (epochs without validation-loss improvement before stopping). |

Needs a held-out validation split for early stopping (`NEEDS_VAL_SPLIT`) and per-inner-fold `StandardScaler` (`NEEDS_SCALING`) — both handled automatically by `nested_cv()`.

## TabPFNv3 (`tabpfn`)

No default HPO search space — always runs under [plain cross-fold validation](3_cross_fold_validation.md) (`n_trials` auto-set to `0`). Two extra config-driven parameters aren't part of a search space but still reach `build_model()` (via the "pinned scalar" mechanism, see [3_nested_cross_fold_validation.md](3_nested_cross_fold_validation.md#search-space-resolution-and-pinning)):

| Parameter | Default | Controls |
|---|---|---|
| `finetune` | `false` | Switches from the frozen pretrained model to `FinetunedTabPFNModel`, which fine-tunes on the outer-fold training data. |
| `learning_rate` | `1e-5` | Fine-tuning learning rate — only used when `finetune: true`. |
| `epochs` | `30` | Fine-tuning epochs — only used when `finetune: true`. |

```yaml
model:
  name: tabpfn
  finetune: true
  learning_rate: 1.0e-5
  epochs: 30
```

When running several models in one job via `model.names`, scope these under `model.overrides.tabpfn` instead of the shared `model:` block — see the README's "Running multiple models in one job".
