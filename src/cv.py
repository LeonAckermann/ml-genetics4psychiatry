"""Single generic nested cross-validation with optional Optuna HPO."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .evaluation import (
    aggregate_confidence_metrics,
    aggregate_metrics,
    classification_metrics,
    compute_metrics,
    label_distribution,
    mdn_confidence_metrics,
    report_fold_metrics,
)
from .hpo import NEEDS_SCALING, NEEDS_VAL_SPLIT, build_model, get_default_search_space, suggest_params
from .log import (
    finish_run,
    init_run,
    log_fold_hyperparameters,
    log_hyperparameter_table,
    log_performance_table,
    log_summary,
    make_epoch_logger,
    plot_training_curves,
)
from .shap import aggregate_fold_shap, explain_fold
from .training import train, train_mdn


def _apply_pca(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    setting: float | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on X_train, transform all splits.

    setting:
      float (0, 1)  → minimum components explaining that fraction of variance
      'effective'   → entropy-based effective rank of the training matrix
    """
    from sklearn.decomposition import PCA

    if setting == "effective":
        _, s, _ = np.linalg.svd(X_train, full_matrices=False)
        s_norm = s / (s.sum() + 1e-12)
        entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
        n_components = max(1, min(int(np.round(np.exp(entropy))),
                                  X_train.shape[0] - 1, X_train.shape[1]))
        pca = PCA(n_components=n_components, random_state=42)
    else:
        pca = PCA(n_components=float(setting), svd_solver="full", random_state=42)

    X_train_t = pca.fit_transform(X_train)
    print(f"    PCA ({setting}): {pca.n_components_} components → "
          f"{pca.explained_variance_ratio_.sum():.1%} variance (d={X_train.shape[1]})")
    return X_train_t, pca.transform(X_val), pca.transform(X_test)


def _pca_setting(cfg: dict) -> float | str | None:
    """Return cfg['data']['pca']: a variance float, 'effective', or None."""
    v = cfg.get("data", {}).get("pca")
    if v is None:
        return None
    if str(v).lower() == "effective":
        return "effective"
    return float(v)


def _add_noise(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add i.i.d. Gaussian noise N(0, sigma²) to X. No-op when sigma <= 0."""
    if sigma <= 0:
        return X
    return (X + rng.normal(0, sigma, X.shape)).astype(X.dtype)


def _noise_sigma(cfg: dict) -> float:
    """Return the noise sigma from cfg['noise']['sigma'], defaulting to 0."""
    n = cfg.get("noise", {}) or {}
    return float(n.get("sigma", 0.0) or 0.0)


def _pinned_params(cfg: dict) -> dict:
    """Scalar model-config values that must not be overridden by HPO."""
    return {
        k: v for k, v in cfg.get("model", {}).items()
        if not isinstance(v, (list, dict))
        and k not in ("name", "type", "p_value_binary")
    }


def _make_sampler(cfg: dict, seed: int):
    """Build the Optuna sampler chosen via hpo.sampler (default 'tpe', i.e.
    TPESampler -- unchanged from before this was configurable). See
    documentation/3_nested_cross_fold_validation.md#hyperparameter-optimization."""
    hpo_cfg = cfg.get("hpo")
    name = hpo_cfg.get("sampler", "tpe") if isinstance(hpo_cfg, dict) else "tpe"
    if name == "random":
        return RandomSampler(seed=seed)
    if name == "cmaes":
        return CmaEsSampler(seed=seed)
    return TPESampler(seed=seed)


def _needs_val_split(model_name: str, params: dict) -> bool:
    """True for models that require a held-out val set (early stopping)."""
    return (
        model_name in NEEDS_VAL_SPLIT
        or (model_name == "xgboost" and "early_stopping_rounds" in params)
    )


def _residual_target(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
    cfg: dict,
    experiment_name: str,
    feature_names: list | None = None,
    results_dir: Path | None = None,
    shap_cfg: dict | None = None,
    shap_enabled: bool = False,
    save_predictions: bool = False,
    row_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, dict, np.ndarray, list[dict]]:
    """Out-of-fold linear-regression residual of ``y_arr``, on the exact same
    outer folds the real model (e.g. TabPFN) is evaluated on below.

    For each fold, a plain (unregularised) LinearRegression is fit on that
    fold's training split only and used to predict the held-out test split --
    it never sees a sample's label before being scored against it, so the
    residual for every sample comes from a model that never trained on that
    sample. Stitching those held-out residuals back together in original row
    order gives one out-of-fold residual per sample, which becomes the new
    regression target for the real model configured via ``model_name``.

    Preprocessing (optional PCA, StandardScaler, optional noise) mirrors
    exactly what the main fold loop below does for ``linear_regression``, so
    this auxiliary model sees the same feature representation the real model
    will train on, not a differently-preprocessed proxy.

    When ``save_predictions`` is set, this model's own out-of-fold predictions
    (the linear fit, not the residual) are stitched into ``linear_preds`` for
    the caller to persist alongside the real model's predictions -- independent
    of SHAP. Separately, when ``shap_enabled`` and ``shap_cfg["store"]`` are
    both set, each fold's fitted LinearRegression is explained via
    ``explain_fold`` (writing under ``<results_dir>/residual_baseline/shap/``
    so it never collides with the real model's own SHAP output) and the
    per-fold summaries are returned for the caller to aggregate the same way
    as the real model's. ``row_ids`` (aligned to ``y_arr``, the dataset's own
    row identifiers rather than positional indices) is threaded through to
    ``explain_fold`` so its saved per-row values can be traced back to a
    sample.

    Returns ``(residual_y, baseline_metrics, linear_preds, baseline_shap_folds)``
    -- ``baseline_metrics`` is aggregated the same way as the real model's own
    results (mean/std per fold), for direct comparison against an ordinary
    ``linear_regression`` run on this phenotype. ``linear_preds`` is all-NaN
    and ``baseline_shap_folds`` is empty when their respective flags are off.
    """
    print(f"[{experiment_name}] data.residual=true -- building out-of-fold "
          f"linear-regression residual target ({len(fold_splits)} folds)")
    residual_y = np.full_like(y_arr, np.nan, dtype=np.float32)
    linear_preds = np.full_like(y_arr, np.nan, dtype=np.float32)
    fold_metrics: list[dict] = []
    baseline_shap_folds: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(fold_splits):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        pca = _pca_setting(cfg)
        if pca is not None:
            # No real val split needed (linear_regression does not use one);
            # X_train doubles as the unused "val" input, same convention the
            # main loop uses below when _needs_val_split is False.
            X_train, _, X_test = _apply_pca(X_train, X_train, X_test, pca)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        sigma = _noise_sigma(cfg)
        if sigma > 0:
            rng = np.random.default_rng(42 + fold)
            X_train = _add_noise(X_train, sigma, rng)
            X_test = _add_noise(X_test, sigma, rng)

        # Plain LinearRegression: build_model("linear_regression", ...) has no
        # hyperparameters, so an empty params dict (rather than the outer
        # model's, e.g. TabPFN's, pinned config values) is the correct and
        # only sensible input here.
        model = build_model("linear_regression", {}, cfg)
        explain_baseline = shap_enabled and bool((shap_cfg or {}).get("store", False))
        if explain_baseline:
            preds, fitted_model = train(
                model, X_train, y_train, X_train, y_train, X_test, y_test, cfg,
                return_model=True,
            )
        else:
            preds = train(model, X_train, y_train, X_train, y_train, X_test, y_test, cfg)

        residual_y[test_idx] = y_test - preds
        if save_predictions:
            linear_preds[test_idx] = preds
        metrics = compute_metrics(y_test, preds, "regression")
        fold_metrics.append(metrics)
        print(f"  [{experiment_name}] residual baseline fold {fold + 1}/{len(fold_splits)}: "
              f"pearson_r2={metrics.get('pearson_r2', float('nan')):.4f}")

        if explain_baseline:
            try:
                effective_names = (
                    [f"PC{i + 1}" for i in range(X_train.shape[1])]
                    if pca is not None else feature_names
                )
                base_dir = Path(results_dir) if results_dir else Path("./results") / experiment_name
                fold_summary = explain_fold(
                    fitted_model, "linear_regression", "regression",
                    X_background=X_train, X_test=X_test,
                    feature_names=effective_names, fold=fold,
                    experiment_name=f"{experiment_name} (residual baseline)",
                    shap_cfg=shap_cfg or {},
                    results_dir=base_dir / "residual_baseline",
                    y_background=y_train,
                    test_idx=test_idx,
                    row_ids=row_ids[test_idx] if row_ids is not None else None,
                )
                if fold_summary:
                    baseline_shap_folds.append(fold_summary)
            except Exception as e:
                print(f"  [{experiment_name}] residual baseline SHAP explanation "
                      f"failed for fold {fold + 1}: {e}")

    baseline = aggregate_metrics(fold_metrics, "regression",
                                 f"{experiment_name} residual baseline")
    baseline["fold_metrics"] = fold_metrics
    return residual_y, baseline, linear_preds, baseline_shap_folds


def nested_cv(
    X,
    y,
    model_name: str,
    cfg: dict,
    outer_cv: int = 5,
    inner_cv: int = 3,
    n_trials: int = 50,
    search_space: dict | None = None,
    best_params_list: list | None = None,
    val_size: float = 0.1,
    experiment_name: str = "",
    feature_names: list | None = None,
    results_dir: str | None = None,
    row_ids: np.ndarray | None = None,
) -> dict:
    """Generic nested cross-validation with optional Optuna HPO.

    Three operating modes controlled by ``best_params_list`` and ``n_trials``:

    1. ``best_params_list`` is provided → use those params per fold, skip HPO.
    2. ``n_trials > 0`` → run inner-fold HPO via Optuna, then evaluate on outer test.
    3. ``n_trials == 0`` → train with default / empty params (no HPO); useful for
       models without hyperparameters such as ``linear_regression``.

    Scaling (StandardScaler) is applied per inner fold for all models in
    ``NEEDS_SCALING`` (lasso/ridge/elastic/linear regression, DNN variants).
    The scaler is always fit on the inner training split only to avoid leakage.

    When ``cfg["shap"]["enabled"]`` is true, each outer fold's final model is
    explained on its test set via ``src.shap.explain_fold`` after that
    fold's metrics are reported (never during inner-fold HPO). ``feature_names``
    and ``results_dir`` control where plots are written and how features are
    labeled; see ``src/shap.py`` for the full config schema.

    When ``cfg["predictions"]["enabled"]`` is true, out-of-fold predictions
    (stitched across the outer folds into original row order) are saved to
    ``<results_dir>/predictions/predictions.npz``, including ``row_ids``.
    Under ``data.residual=true`` this also includes the auxiliary linear-
    regression baseline's own predictions (of the original target).

    Independently, when ``cfg["shap"]["enabled"]`` and ``cfg["shap"]["store"]``
    are both true, each outer fold's per-row SHAP/interaction values (already
    computed for the beeswarm plot) are additionally written to
    ``<results_dir>/shap/fold_<k>/row_interactions.npz`` -- and, under
    ``data.residual=true``, the linear-regression baseline is explained the
    same way, written under ``<results_dir>/residual_baseline/shap/``. See
    ``src/shap.py::save_row_interactions``. This does not require
    ``predictions.enabled``.

    ``row_ids`` are the dataset's own row identifiers (e.g. subject IDs),
    aligned to ``X``/``y`` in the same order -- used (instead of a plain
    positional index) wherever a saved array needs to be traced back to a
    sample. Defaults to the positional index when not given.
    """
    task_type = cfg.get("model", {}).get("type", "regression")
    is_classification = task_type == "binary_classification"
    needs_scaling = model_name in NEEDS_SCALING

    residual_enabled = bool(cfg.get("data", {}).get("residual", False))
    if residual_enabled and is_classification:
        raise ValueError(
            "data.residual=true is only defined for regression targets: it "
            "residualises a continuous y against an out-of-fold linear-"
            "regression fit, which has no equivalent for a binary label "
            "without a different (unimplemented) construction. Set "
            "data.residual=false for binary_classification runs."
        )

    shap_cfg = cfg.get("shap", {}) or {}
    shap_enabled = bool(shap_cfg.get("enabled", False))

    predictions_cfg = cfg.get("predictions", {}) or {}
    save_predictions = bool(predictions_cfg.get("enabled", False))

    # Per-epoch train/test curves on the outer fold. Recording defaults on (it
    # only costs anything for epoch-based models); rendering is opt-in.
    curves_cfg = cfg.get("training_curves", {}) or {}
    curves_enabled = bool(curves_cfg.get("enabled", True))
    curves_plots = bool(curves_cfg.get("plots", False))

    wandb_run = init_run(cfg, experiment_name, model_name, task_type)

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32).ravel()
    y_original = y_arr.copy()
    row_ids_arr = np.asarray(row_ids) if row_ids is not None else np.arange(len(y_arr))

    outer_kfold = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
    # Materialised once so the residual pass and the real model below are
    # scored on identical fold assignments -- not just "the same by
    # construction" via a second .split() call with a matching seed.
    fold_splits = list(outer_kfold.split(X_arr))

    residual_baseline: dict | None = None
    linear_baseline_preds: np.ndarray | None = None
    baseline_shap_folds: list[dict] = []
    if residual_enabled:
        y_arr, residual_baseline, linear_baseline_preds, baseline_shap_folds = _residual_target(
            X_arr, y_arr, fold_splits, cfg, experiment_name,
            feature_names=feature_names, results_dir=results_dir,
            shap_cfg=shap_cfg, shap_enabled=shap_enabled,
            save_predictions=save_predictions, row_ids=row_ids_arr,
        )

    fold_metrics: list[dict] = []
    fold_best_params: list[dict] = []
    fold_label_distributions: list[dict] = []
    fold_confidence_metrics: list[list[dict]] = []  # populated only for MDN
    fold_init_params: list[dict] = []               # populated only for MDN
    fold_training_curves: list[dict] = []           # populated only for epoch-based models
    fold_shap: list[dict] = []                      # populated only when shap.enabled
    main_preds = np.full(len(y_arr), np.nan, dtype=np.float32)  # populated only when save_predictions

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if best_params_list is not None:
        print(
            f"[{experiment_name}] Using {len(best_params_list)} pre-loaded fold params"
            " — skipping HPO"
        )

    for fold, (train_idx, test_idx) in enumerate(fold_splits):
        print(f"\n[{experiment_name}] --- Outer Fold {fold + 1}/{outer_cv} ---")
        X_train_outer = X_arr[train_idx]
        X_test_outer = X_arr[test_idx]
        y_train_outer = y_arr[train_idx]
        y_test_outer = y_arr[test_idx]

        # Per-epoch curves for this fold, filled in by the epoch-based trainers
        # (DNN/MDN). Left None when disabled so no extra eval passes run.
        epoch_history: list[dict] | None = [] if curves_enabled else None

        is_binary_labels = is_classification and model_name != "mdn"
        fold_label_distributions.append({
            "fold": fold + 1,
            "train": label_distribution(y_train_outer, is_binary=is_binary_labels),
            "test": label_distribution(y_test_outer, is_binary=is_binary_labels),
        })

        # ── Select hyperparameters ────────────────────────────────────────────
        pinned = _pinned_params(cfg)

        if best_params_list is not None:
            best_params = best_params_list[fold]
            print(f"  Using params: {best_params}")
        elif n_trials > 0:
            space = search_space or get_default_search_space(model_name, task_type)
            # Scalar values in cfg["model"] are pinned — remove from search space.
            if pinned:
                space = {k: v for k, v in space.items() if k not in pinned}
            best_params = _run_hpo(
                X_outer=X_train_outer,
                y_outer=y_train_outer,
                inner_cv=inner_cv,
                model_name=model_name,
                search_space=space,
                cfg=cfg,
                val_size=val_size,
                task_type=task_type,
                n_trials=n_trials,
                fold=fold,
                experiment_name=experiment_name,
                needs_scaling=needs_scaling,
            )
            # Re-apply pinned params so build_model sees them.
            if pinned:
                best_params = {**best_params, **pinned}
        else:
            # n_trials=0: seed from cfg["model"] so scalar config values
            # (e.g. finetune, device) reach build_model.
            best_params = pinned

        fold_best_params.append(best_params)
        log_fold_hyperparameters(wandb_run, fold, best_params)

        # ── Train best model on full outer train (val split only when needed) ──
        if _needs_val_split(model_name, best_params):
            X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
                X_train_outer, y_train_outer, test_size=val_size, random_state=42 + fold
            )
        else:
            X_train_final, y_train_final = X_train_outer, y_train_outer
            X_val_final, y_val_final = X_train_outer, y_train_outer  # unused by sklearn

        pca = _pca_setting(cfg)
        if pca is not None:
            X_train_final, X_val_final, X_test_outer = _apply_pca(
                X_train_final, X_val_final, X_test_outer, pca
            )

        if needs_scaling:
            scaler = StandardScaler()
            X_train_final = scaler.fit_transform(X_train_final)
            X_val_final = scaler.transform(X_val_final)
            X_test_outer = scaler.transform(X_test_outer)

        sigma = _noise_sigma(cfg)
        if sigma > 0:
            rng = np.random.default_rng(42 + fold)
            X_train_final = _add_noise(X_train_final, sigma, rng)
            X_val_final   = _add_noise(X_val_final,   sigma, rng)
            X_test_outer  = _add_noise(X_test_outer,  sigma, rng)

        model = build_model(model_name, best_params, cfg)
        on_epoch = make_epoch_logger(wandb_run, fold)
        if model_name == "mdn":
            preds, fold_pi, fold_mu, init_mu, init_sigma = train_mdn(
                model,
                X_train_final, y_train_final,
                X_val_final, y_val_final,
                X_test_outer,
                cfg,
                y_test=y_test_outer,
                history=epoch_history,
                on_epoch=on_epoch,
            )
            fold_init_params.append({
                "init_mu": init_mu,
                "init_sigma": init_sigma,
            })
            conf_metrics = mdn_confidence_metrics(fold_pi, fold_mu, y_test_outer)
            fold_confidence_metrics.append(conf_metrics)
            report_t = cfg.get("evaluation", {}).get("confidence_threshold", 0.5)
            print(
                f"  [{experiment_name}] Fold {fold + 1} MDN confidence @ {report_t}:"
                f" bal_acc={next((r['balanced_accuracy'] for r in conf_metrics if r['threshold'] == report_t), None)}"
                f"  n_conf={next((r['n_confident'] for r in conf_metrics if r['threshold'] == report_t), 0)}"
                f"/{len(y_test_outer)}"
            )
        else:
            fitted_model = None
            if shap_enabled:
                preds, fitted_model = train(
                    model,
                    X_train_final, y_train_final,
                    X_val_final, y_val_final,
                    X_test_outer, y_test_outer,
                    cfg,
                    return_model=True,
                    history=epoch_history,
                    on_epoch=on_epoch,
                )
            else:
                preds = train(
                    model,
                    X_train_final, y_train_final,
                    X_val_final, y_val_final,
                    X_test_outer, y_test_outer,
                    cfg,
                    history=epoch_history,
                    on_epoch=on_epoch,
                )

        if save_predictions:
            main_preds[test_idx] = np.asarray(preds, dtype=np.float32).ravel()

        if model_name == "mdn" and task_type == "binary_classification":
            metrics = classification_metrics(
                (np.asarray(y_test_outer) >= 1.0).astype(int),
                preds,
                threshold=1.0,
            )
        else:
            metrics = compute_metrics(y_test_outer, preds, task_type)
        report_fold_metrics(metrics, fold + 1, outer_cv, experiment_name, task_type)
        fold_metrics.append(metrics)

        if epoch_history:
            fold_training_curves.append({"fold": fold + 1, "epochs": epoch_history})
            last = epoch_history[-1]
            sn = last.get("score_name", "score")
            print(f"  [{experiment_name}] fold {fold + 1}: {len(epoch_history)} epochs recorded "
                  f"(final train/test loss {last['train_loss']:.4f}/{last['test_loss']:.4f}, "
                  f"{sn} {last[f'train_{sn}']:.4f}/{last[f'test_{sn}']:.4f})")

        if shap_enabled and model_name != "mdn":
            try:
                effective_names = (
                    [f"PC{i + 1}" for i in range(X_train_final.shape[1])]
                    if pca is not None else feature_names
                )
                fold_summary = explain_fold(
                    fitted_model, model_name, task_type,
                    X_background=X_train_final, X_test=X_test_outer,
                    feature_names=effective_names, fold=fold,
                    experiment_name=experiment_name, shap_cfg=shap_cfg,
                    results_dir=Path(results_dir) if results_dir else Path("./results") / experiment_name,
                    y_background=y_train_final,
                    test_idx=test_idx,
                    row_ids=row_ids_arr[test_idx],
                )
                if fold_summary:
                    fold_shap.append(fold_summary)
            except Exception as e:
                print(f"  [{experiment_name}] SHAP explanation failed for fold {fold + 1}: {e}")

        gc.collect()

    aggregated = aggregate_metrics(fold_metrics, task_type, experiment_name)
    result = {
        "fold_metrics": fold_metrics,
        "fold_best_params": fold_best_params,
        "fold_label_distributions": fold_label_distributions,
        **aggregated,
    }
    if residual_baseline is not None:
        # The out-of-fold linear-regression pass whose residual became this
        # run's y -- saved so the residual construction can be sanity-checked
        # against an ordinary linear_regression run on the same phenotype
        # without needing to re-derive it.
        result["residual_baseline"] = residual_baseline
        print(f"[{experiment_name}] residual_baseline included in result "
              f"(auxiliary linear-regression mean_pearson_r2="
              f"{residual_baseline.get('mean_pearson_r2', float('nan')):.4f})")
    if fold_confidence_metrics:
        result["fold_confidence_metrics"] = fold_confidence_metrics
        result["confidence_threshold_evaluation"] = aggregate_confidence_metrics(fold_confidence_metrics)
    if fold_init_params:
        result["fold_init_params"] = fold_init_params
    if fold_training_curves:
        result["training_curves"] = fold_training_curves
        if curves_plots:
            out_root = Path(results_dir) if results_dir else Path("./results") / experiment_name
            try:
                out = plot_training_curves(
                    fold_training_curves,
                    out_root,
                    experiment_name=experiment_name,
                    dir_name=curves_cfg.get("dir", "training"),
                )
                if out:
                    print(f"[{experiment_name}] Training curves written to {out}")
            except Exception as e:
                print(f"[{experiment_name}] Training-curve plots failed: {e}")
    if fold_shap:
        # Mean attribution (and interactions, in interaction mode) averaged over
        # the outer folds; also writes the averaged InteractionValues and its
        # plots under <results_dir>/shap/. Folds whose explanation raised are
        # simply absent, so this reflects the folds that produced values.
        result["shap_mean_values"] = aggregate_fold_shap(
            fold_shap,
            results_dir=Path(results_dir) if results_dir else Path("./results") / experiment_name,
            experiment_name=experiment_name,
            shap_cfg=shap_cfg,
        )
    if baseline_shap_folds:
        # Same aggregation as fold_shap above, but for the residual baseline's
        # LinearRegression -- written under residual_baseline/shap/ so its mean
        # InteractionValues JSON and plots never collide with the real model's.
        baseline_shap_mean = aggregate_fold_shap(
            baseline_shap_folds,
            results_dir=(Path(results_dir) if results_dir else Path("./results") / experiment_name)
                / "residual_baseline",
            experiment_name=f"{experiment_name} residual baseline",
            shap_cfg=shap_cfg,
        )
        if residual_baseline is not None:
            residual_baseline["shap_mean_values"] = baseline_shap_mean

    if save_predictions:
        out_root = Path(results_dir) if results_dir else Path("./results") / experiment_name
        pred_dir = out_root / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        fold_assignment = np.zeros(len(y_original), dtype=np.int32)
        for fold, (_, test_idx) in enumerate(fold_splits):
            fold_assignment[test_idx] = fold + 1

        save_kwargs = dict(
            row_ids=row_ids_arr,
            y_true=y_original,
            preds=main_preds,
            fold_assignment=fold_assignment,
        )
        if residual_enabled:
            # y_arr at this point is the residual target the real model was
            # trained on (see the reassignment near the top of this function).
            save_kwargs["residual_target"] = y_arr
            save_kwargs["linear_baseline_preds"] = linear_baseline_preds

        pred_path = pred_dir / "predictions.npz"
        np.savez(pred_path, **save_kwargs)
        print(f"[{experiment_name}] predictions saved to {pred_path}")

    if wandb_run is not None:
        log_hyperparameter_table(wandb_run, fold_best_params)
        log_performance_table(wandb_run, fold_metrics, aggregated)
        log_summary(wandb_run, aggregated)
        run_url = wandb_run.url
        finish_run(wandb_run)
        print(f"[{experiment_name}] logged to W&B: {run_url or '(offline run, not synced)'}")

    return result


# ---------------------------------------------------------------------------
# Private HPO helper
# ---------------------------------------------------------------------------

def _run_hpo(
    X_outer: np.ndarray,
    y_outer: np.ndarray,
    inner_cv: int,
    model_name: str,
    search_space: dict,
    cfg: dict,
    val_size: float,
    task_type: str,
    n_trials: int,
    fold: int,
    experiment_name: str,
    needs_scaling: bool,
) -> dict:
    """Run Optuna on the outer training fold and return the best params dict."""
    inner_kfold = KFold(n_splits=inner_cv, shuffle=True, random_state=42)
    primary_metric = "balanced_accuracy" if task_type == "binary_classification" else "r2"

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, search_space, model_name)
        inner_scores: list[float] = []

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            inner_kfold.split(X_outer)
        ):
            X_inner_train = X_outer[inner_train_idx]
            y_inner_train = y_outer[inner_train_idx]
            X_inner_test = X_outer[inner_val_idx]
            y_inner_test = y_outer[inner_val_idx]

            # Val split only for models that use it (DNN early stopping, XGBoost ES).
            if _needs_val_split(model_name, params):
                X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
                    X_inner_train, y_inner_train, test_size=val_size, random_state=42 + inner_fold
                )
            else:
                X_inner_val, y_inner_val = X_inner_train, y_inner_train

            n_pca = _pca_setting(cfg)
            if n_pca:
                X_inner_train, X_inner_val, X_inner_test = _apply_pca(
                    X_inner_train, X_inner_val, X_inner_test, n_pca
                )

            if needs_scaling:
                scaler = StandardScaler()
                X_inner_train = scaler.fit_transform(X_inner_train)
                X_inner_val = scaler.transform(X_inner_val)
                X_inner_test = scaler.transform(X_inner_test)

            sigma = _noise_sigma(cfg)
            if sigma > 0:
                rng = np.random.default_rng(fold * 10000 + trial.number * 100 + inner_fold)
                X_inner_train = _add_noise(X_inner_train, sigma, rng)
                X_inner_val   = _add_noise(X_inner_val,   sigma, rng)
                X_inner_test  = _add_noise(X_inner_test,  sigma, rng)

            model = build_model(model_name, params, cfg)
            preds = train(
                model,
                X_inner_train, y_inner_train,
                X_inner_val, y_inner_val,
                X_inner_test, y_inner_test,
                cfg,
            )
            if model_name == "mdn" and task_type == "binary_classification":
                score = classification_metrics(
                    (np.asarray(y_inner_test) >= 1.0).astype(int),
                    preds,
                    threshold=1.0,
                )[primary_metric]
            else:
                score = compute_metrics(y_inner_test, preds, task_type)[primary_metric]
            inner_scores.append(score)
            print(
                f"    [{experiment_name}] trial {trial.number + 1}"
                f" inner fold {inner_fold + 1}: {primary_metric}={score:.4f}"
            )

            del model, X_inner_train, y_inner_train, X_inner_val
            del y_inner_val, X_inner_test, y_inner_test, preds

        mean_score = float(np.mean(inner_scores))
        print(
            f"  [{experiment_name}] trial {trial.number + 1}:"
            f" mean {primary_metric}={mean_score:.4f}"
        )
        gc.collect()
        return mean_score

    sampler = _make_sampler(cfg, seed=42 + fold)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(
        f"  [{experiment_name}] Fold {fold + 1}"
        f" best inner {primary_metric}: {study.best_value:.4f}"
    )
    print(f"  [{experiment_name}] Best params: {study.best_params}")
    return study.best_params
