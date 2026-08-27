# 3.1 Cross-fold validation

Plain outer cross-validation: each fold trains once, using the hyperparameters straight from the config file (or the model's built-in defaults) — no inner loop, no hyperparameter search, nothing that could bias the reported metrics toward a particular hyperparameter choice. Implemented as the `n_trials == 0` branch of [`nested_cv()`](../src/cv.py:226) in `src/cv.py`.

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
              build_model(model_name, params, cfg)   params = cfg["model"]
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

`params` passed to `build_model()` is just `dict(cfg["model"])` for every fold — identical hyperparameters each time, since nothing is being searched. This is also the mode used automatically when a model has no default search space at all (e.g. `linear`, `tabpfn`) even if `hpo.run: true` is set — see [3_nested_cross_fold_validation.md](3_nested_cross_fold_validation.md).

## Loading pre-optimized parameters instead

A third, related mode: if `--load-best-params <path>` (CLI) or `load_best_params: true` (config, reading from `best_params/`) supplies a `fold_best_params` list, evaluation also runs with `n_trials = 0` — but using per-fold parameters previously found by a nested cross-validation run, rather than the config file's fixed values. Useful for re-evaluating a model (e.g. with SHAP enabled) without re-running the hyperparameter search.
