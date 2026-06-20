"""Single generic nested cross-validation with optional Optuna HPO."""
from __future__ import annotations

import gc

import numpy as np
import optuna
from optuna.samplers import TPESampler
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


def _needs_val_split(model_name: str, params: dict) -> bool:
    """True for models that require a held-out val set (early stopping)."""
    return (
        model_name in NEEDS_VAL_SPLIT
        or (model_name == "xgboost" and "early_stopping_rounds" in params)
    )


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
    """
    task_type = cfg.get("model", {}).get("type", "regression")
    is_classification = task_type == "binary_classification"
    needs_scaling = model_name in NEEDS_SCALING

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32).ravel()

    outer_kfold = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
    fold_metrics: list[dict] = []
    fold_best_params: list[dict] = []
    fold_label_distributions: list[dict] = []
    fold_confidence_metrics: list[list[dict]] = []  # populated only for MDN
    fold_init_params: list[dict] = []               # populated only for MDN

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if best_params_list is not None:
        print(
            f"[{experiment_name}] Using {len(best_params_list)} pre-loaded fold params"
            " — skipping HPO"
        )

    for fold, (train_idx, test_idx) in enumerate(outer_kfold.split(X_arr)):
        print(f"\n[{experiment_name}] --- Outer Fold {fold + 1}/{outer_cv} ---")
        X_train_outer = X_arr[train_idx]
        X_test_outer = X_arr[test_idx]
        y_train_outer = y_arr[train_idx]
        y_test_outer = y_arr[test_idx]

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
        if model_name == "mdn":
            preds, fold_pi, fold_mu, init_mu, init_sigma = train_mdn(
                model,
                X_train_final, y_train_final,
                X_val_final, y_val_final,
                X_test_outer,
                cfg,
                y_test=y_test_outer,
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
            preds = train(
                model,
                X_train_final, y_train_final,
                X_val_final, y_val_final,
                X_test_outer, y_test_outer,
                cfg,
            )

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
        gc.collect()

    aggregated = aggregate_metrics(fold_metrics, task_type, experiment_name)
    result = {
        "fold_metrics": fold_metrics,
        "fold_best_params": fold_best_params,
        "fold_label_distributions": fold_label_distributions,
        **aggregated,
    }
    if fold_confidence_metrics:
        result["fold_confidence_metrics"] = fold_confidence_metrics
        result["confidence_threshold_evaluation"] = aggregate_confidence_metrics(fold_confidence_metrics)
    if fold_init_params:
        result["fold_init_params"] = fold_init_params
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

    sampler = TPESampler(seed=42 + fold)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(
        f"  [{experiment_name}] Fold {fold + 1}"
        f" best inner {primary_metric}: {study.best_value:.4f}"
    )
    print(f"  [{experiment_name}] Best params: {study.best_params}")
    return study.best_params
