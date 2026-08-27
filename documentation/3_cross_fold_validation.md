# 3.1 Cross-fold validation

Plain outer cross-validation: each fold trains once, using the hyperparameters straight from the config file (falling back to a curated default per model — see below) — no inner loop, no hyperparameter search, nothing that could bias the reported metrics toward a particular hyperparameter choice. Implemented as the `n_trials == 0` branch of [`nested_cv()`](../src/cv.py:226) in `src/cv.py`.

## When to use it

Start here for models with few or no real hyperparameters (`linear`, `tabpfn` without `finetune`), or for a first pass on a new model or dataset. It's fast — one train/evaluate per outer fold, nothing nested inside — and there's no hyperparameter-selection step to worry about overfitting.

## Diagram

```
                        for each outer fold (default 5, hpo.outer_cv):
                                     │
                                     ▼
                    split: this fold's train rows / test rows
                                     │
                                     ▼
              build_model(model_name, params, cfg)   params = {**defaults, **pinned}
                                     │                (no HPO trial ever runs)
                                     ▼
                    train on the fold's train rows
                                     │
                                     ▼
                  evaluate once on the fold's test rows
                                     │
                                     ▼
                          record this fold's metrics
                                     │
                                     ▼
              (repeat for the next outer fold)
                                     │
                                     ▼
        aggregate_metrics()  →  mean / std across all outer folds
```

## Config

```yaml
model:
  name: linear

hpo:
  run: false   # or omit the hpo: block entirely
```

`params` passed to `build_model()` is the same for every fold — identical hyperparameters each time, since nothing is being searched. It's built as `{**get_default_params(model_name, task_type), **pinned}` (`src/hpo.py`): a curated default hyperparameter dict for that model (one interior point of its own `_DEFAULT_SPACES` entry, or a standard estimator default for a model with no search space) with any pinned scalar already set under `model:` in the config overriding the matching key. A model with no entry (`linear_regression`, `tabpfn` without `finetune`) gets `{}` from `get_default_params` and falls back to whatever `build_model` itself hard-codes — unchanged, since those models have no real hyperparameter to default. This is also the mode used automatically when a model has no default search space at all (e.g. `linear`, `tabpfn`) even if `hpo.run: true` is set — see [3_nested_cross_fold_validation.md](3_nested_cross_fold_validation.md).

### Current defaults per model

| Model | Default hyperparameters |
| --- | --- |
| `xgboost` (regression) | `n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8` |
| `xgboost` (binary) | same, plus `reg_alpha=0.0, reg_lambda=1.0, min_child_weight=1.0, gamma=0.0` |
| `dnn` / `residual_dnn` | `hidden_dim=64, n_layers=2, dropout=0.2, learning_rate=1e-3, batch_size=32, epochs=100, patience=20` |
| `mdn` | `hidden_dim=64, n_layers=2, dropout=0.3, learning_rate=1e-4, weight_decay=0.0, batch_size=32, epochs=200, patience=20, number_of_components=2` |
| `lasso_regression` | `alpha=1.0` |
| `ridge_regression` | `alpha=1000.0` (deliberately high — this codebase's feature matrices are p ≫ n) |
| `elastic_regression` | `alpha=0.07, l1_ratio=0.1` |
| `logistic_regression` / `ridge_logistic_regression` / `lasso_logistic_regression` | `C=1.0, class_weight="balanced"` (label imbalance is the norm under a p-value binary threshold) |
| `elastic_logistic_regression` | `C=1.0, l1_ratio=0.5, class_weight="balanced"` |
| `bayesian_ridge_regression` | `alpha_1=alpha_2=lambda_1=lambda_2=1e-6, max_iter=300, tol=1e-3, fit_intercept=True` (near-flat hyperpriors — alpha/lambda themselves are fit by evidence maximization, not searched) |
| `linear_regression`, `tabpfn` (no `finetune`) | none — no real hyperparameter to default |

See `src/hpo.py::_DEFAULT_PARAMS` for the source of truth; edit there (or override per-run via `model:` in the config) rather than here.

## Loading pre-optimized parameters instead

A third, related mode: if `--load-best-params <path>` (CLI) or `load_best_params: true` (config, reading from `best_params/`) supplies a `fold_best_params` list, evaluation also runs with `n_trials = 0` — but using per-fold parameters previously found by a nested cross-validation run, rather than the config file's fixed values. Useful for re-evaluating a model (e.g. with SHAP enabled) without re-running the hyperparameter search.
