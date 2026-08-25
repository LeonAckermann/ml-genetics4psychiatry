# 3.2 Nested cross-fold validation

Every outer fold runs its own hyperparameter search *before* being scored — so the hyperparameters a fold is evaluated with were never chosen using that fold's own test rows. Implemented as the `n_trials > 0` branch of [`nested_cv()`](../src/cv.py:226), which calls [`_run_hpo()`](../src/cv.py:610) once per outer fold.

## When to use it

Use this once a model has enough tunable hyperparameters that hand-picking them risks overfitting to the outer test fold — `xgboost`, `residual_dnn`, or any regularized linear model where the penalty strength matters. If you'd have to hand-pick a value to get a reasonable result, it belongs here; if the model works about the same across a wide range of values, [plain cross-fold validation](3_cross_fold_validation.md) is enough and much cheaper.

## Diagram

```
                    for each outer fold (default 5, hpo.outer_cv):
                                 │
                                 ▼
                split: this fold's train rows / test rows
                                 │
                                 ▼
        ┌── inner hyperparameter search (Optuna, this outer fold's train rows only) ──┐
        │                                                                              │
        │   for each of hpo.n_trials trials:                                          │
        │       sample one hyperparameter set (TPESampler)                            │
        │                    │                                                        │
        │                    ▼                                                        │
        │   for each of hpo.inner_cv inner folds of the train rows:                    │
        │       train with that set on the inner-train split                          │
        │       score on the inner-test split (r2 / balanced_accuracy)                │
        │                    │                                                        │
        │                    ▼                                                        │
        │        average the inner-fold scores → this trial's score                  │
        │                                                                              │
        │   keep the best-scoring trial's hyperparameter set                          │
        └──────────────────────────────────┬───────────────────────────────────────────┘
                                            ▼
              build_model(model_name, best_params, cfg)  (+ any pinned config values)
                                            ▼
                    train on the FULL outer-fold train rows
                                            ▼
                  evaluate once on the outer-fold's test rows
                                            ▼
                          record this fold's metrics + its best_params
                                            │
                                            ▼
              (repeat for the next outer fold — each gets its own search)
                                            │
                                            ▼
        aggregate_metrics()  →  mean / std across all outer folds
```

## Config

```yaml
model:
  name: xgboost

hpo:
  run: true
  n_trials: 100      # Optuna trials per outer fold
  inner_cv: 3          # inner folds per trial
  outer_cv: 5           # optional override; defaults to 5
  # search_space:        # optional — overrides the model's default search space
  #   max_depth: [3, 6]
```

A model with no default search space (`linear`, `tabpfn`, `bayesian_ridge`) auto-falls-back to [plain cross-fold validation](3_cross_fold_validation.md) even with `hpo.run: true` — `n_trials` is silently set to `0` for it, since there's nothing to search. See [3_model_hyperparameters.md](3_model_hyperparameters.md) for which models have a search space and what it covers.

## Search space resolution and pinning

The search space searched per trial is either the model's built-in default (`get_default_search_space(model_name, task_type)`, keyed by `(model_name, task_type)` in [`src/hpo.py`](../src/hpo.py)) or the explicit `hpo.search_space` override shown above. Any scalar value already set under `model:` in the config (e.g. a fixed `device`) is "pinned": removed from the search space so Optuna never samples it, then re-applied to the best-found params afterward — so a value you've deliberately fixed can never be silently overwritten by a sampled one. This is also how model-specific extras like TabPFN's `finetune`/`learning_rate` reach `build_model()` without needing their own search-space entry — see the README's "Running multiple models in one job" for scoping those with `model.overrides` when several models share a job.
