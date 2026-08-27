# 3. Adding a new model

Every model goes through one generic dispatch point: [`build_model(model_name, params, cfg)`](../src/hpo.py) in `src/hpo.py`. Nothing else in `src/cv.py` or `main.py` needs to know a new model exists beyond that function and a couple of registries next to it.

## Checklist

1. **Implement the model** in [`model/`](../model), as either:
   - a plain **sklearn-compatible object** (`.fit(X, y)` / `.predict(X)`) — used directly, e.g. `LassoRegressionModel`, `XGBoostTreeModel`; or
   - a **DNN-style config dict**, for anything trained by [`src/training.py::train`](../src/training.py) (epoch loop, early stopping, optional val split): `{"class": MyModel, **hyperparameter kwargs}`. `DNN`, `ResidualDNN`, `MDN` follow this pattern.

2. **Add a branch to `build_model()`** in [`src/hpo.py`](../src/hpo.py:160) that instantiates it from `params` (an Optuna-trial dict, or `cfg["model"]` directly on the no-HPO path) and `cfg` (for `task_type`, `seed`, `device`, etc.):

   ```python
   if model_name == "my_model":
       from model import MyModel
       return MyModel(
           some_param=float(params.get("some_param", 1.0)),
           random_state=seed,
       )
   ```

3. **Register scaling/val-split needs**, if applicable, in the two frozensets at the top of `src/hpo.py`:
   - `NEEDS_SCALING` — add your model name if it needs `StandardScaler` applied per inner fold (linear-family and DNN-family models do; tree models and TabPFN don't).
   - `NEEDS_VAL_SPLIT` — add it if training needs a held-out validation split for early stopping (DNN/MDN do; XGBoost only does when `early_stopping_rounds` is in `params`, checked dynamically).

4. **Add a default HPO search space**, if the model has tunable hyperparameters, to `_DEFAULT_SPACES` keyed by `(model_name, task_type)`:

   ```python
   _DEFAULT_SPACES[("my_model", "regression")] = {
       "some_param": [0.01, 10.0],   # [low, high] → numeric range (log-scale if in _LOG_PARAMS)
       "some_flag": [True, False],   # categorical (any non-2-numeric list)
   }
   ```

   No entry → `get_default_search_space` returns `{}` → `main.py` auto-sets `n_trials=0` for that model (plain evaluation with config-file params, no HPO). This is the correct choice for models with no real hyperparameters to tune (e.g. `linear_regression`) or ones whose "hyperparameters" are fitted internally (e.g. `bayesian_ridge_regression`'s evidence maximisation).

5. **Model-specific extra parameters** (not part of the HPO search space, but still config-driven — e.g. TabPFN's `finetune`/`learning_rate`, or a device flag) don't need any extra plumbing: any scalar (non-list, non-dict) key under `cfg["model"]` reaches `build_model` automatically via `params.get(...)` — see `_pinned_params()` in [`src/cv.py`](../src/cv.py:81). When looping over several models in one job via `model.names`, scope these under `model.overrides.<model_name>` so they don't leak into another model's pinned params — see README §3.

6. **Resolve the family → concrete name mapping**, only if your model's config `name` differs between `regression` and `binary_classification` (like `linear` → `linear_regression`/`logistic_regression`). Add the pair to `_MODEL_NAME_MAP` in [`main.py`](../main.py:65). Models whose name is identical for both task types (most of them) don't need an entry.

## Available models

| `model.name` | Task types | Key hyperparameters | HPO search space |
|---|---|---|---|
| `linear` → `linear_regression` / `logistic_regression` | both | — (logistic: `C`, `class_weight`) | regression: none; logistic: yes |
| `ridge` → `ridge_regression` / `ridge_logistic_regression` | both | `alpha` (or `C` for logistic) | yes |
| `lasso` → `lasso_regression` / `lasso_logistic_regression` | both | `alpha` (or `C`) | yes |
| `elastic` → `elastic_regression` / `elastic_logistic_regression` | both | `alpha`, `l1_ratio` | yes |
| `bayesian_ridge` → `bayesian_ridge_regression` | regression only | `alpha_1/2`, `lambda_1/2` (Gamma hyperpriors; precisions fitted by evidence maximisation) | none (`n_trials=0`) |
| `xgboost` | both | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha/lambda`, `min_child_weight`, `gamma`, `early_stopping_rounds` (regression only) | yes |
| `dnn` | both | `hidden_dim`/`n_layers` (or `hidden_dims`), `dropout`, `learning_rate`, `batch_size`, `epochs`, `patience` | yes |
| `residual_dnn` | both | same as `dnn` | yes (shares DNN's space) |
| `mdn` (mixture density network) | both | same as DNN + `number_of_components`, `weight_decay` | yes |
| `tabpfn` | both | `finetune`, `learning_rate`, `epochs` (only used when `finetune: true`) | none (`n_trials=0`) |
| `baseline` | — | — (predicts the training-fold mean) | none |

## Best practices — which evaluation mode to use

The pipeline always runs [`nested_cv()`](../src/cv.py:226); the choice is really *"HPO or not"*, controlled by `n_trials`:

- **Plain outer cross-validation (`n_trials=0`)** — start here for models with few or no real hyperparameters (`linear_regression`, `bayesian_ridge_regression`, `tabpfn` without `finetune`), or for a first pass on a new model/dataset. Trains once per outer fold with the config-file (or default) params, no inner loop — fast, and there's no hyperparameter-selection bias to worry about.
- **Nested cross-validation (`n_trials > 0`, via `hpo.run: true` / `hpo.n_trials`)** — use once a model has enough tunable hyperparameters that picking them by hand risks overfitting to the outer test fold (`xgboost`, `dnn`, `residual_dnn`, `mdn`, or any regularized linear model where the penalty strength matters). Each outer fold runs its own inner-fold Optuna search (`hpo.inner_cv`, default 3) before evaluating on that fold's held-out test set — so the reported metrics are never inflated by hyperparameters chosen using the same data they're scored on.

Rule of thumb: if you'd have to hand-pick a hyperparameter to get a reasonable result, it belongs in HPO; if the model works about the same across a wide range of values, plain CV is enough and much cheaper.
