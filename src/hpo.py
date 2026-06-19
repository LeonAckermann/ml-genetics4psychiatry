"""Generic model factory and hyperparameter search utilities."""
from __future__ import annotations

# Models that require StandardScaler applied per inner fold.
NEEDS_SCALING: frozenset[str] = frozenset({
    "linear_regression",
    "lasso_regression",
    "ridge_regression",
    "elastic_regression",
    "logistic_regression",
    "ridge_logistic_regression",
    "lasso_logistic_regression",
    "elastic_logistic_regression",
    "dnn",
    "residual_dnn",
})

# Models that require a held-out validation split (for early stopping).
# XGBoost regression also needs one when early_stopping_rounds is in params,
# but that is checked dynamically in cv.py.
NEEDS_VAL_SPLIT: frozenset[str] = frozenset({
    "dnn",
    "residual_dnn",
})

_INT_PARAMS: frozenset[str] = frozenset({
    "n_estimators", "max_depth", "min_child_weight",
    "epochs", "patience", "batch_size", "n_layers", "hidden_dim",
    "early_stopping_rounds",
})
_LOG_PARAMS: frozenset[str] = frozenset({
    "learning_rate", "C", "alpha", "reg_alpha", "reg_lambda",
})

# Default search spaces keyed by (model_name, task_type).
_DEFAULT_SPACES: dict[tuple[str, str], dict] = {
    ("xgboost", "regression"): {
        "n_estimators": [100, 1000],
        "max_depth": [3, 10],
        "learning_rate": [0.01, 0.3],
        "subsample": [0.6, 1.0],
        "colsample_bytree": [0.6, 1.0],
        "reg_alpha": [0.0, 1.0],
        "reg_lambda": [0.0, 5.0],
        "min_child_weight": [1, 10],
        "gamma": [0.0, 1.0],
        "early_stopping_rounds": [10, 50],
    },
    ("xgboost", "binary_classification"): {
        "n_estimators": [100, 1000],
        "max_depth": [3, 10],
        "learning_rate": [0.01, 0.3],
        "subsample": [0.6, 1.0],
        "colsample_bytree": [0.6, 1.0],
        "reg_alpha": [0.0, 1.0],
        "reg_lambda": [0.0, 5.0],
        "min_child_weight": [1, 10],
        "gamma": [0.0, 1.0],
    },
    ("dnn", "regression"): {
        "hidden_dim": [32, 128],
        "n_layers": [1, 4],
        "dropout": [0.0, 0.5],
        "learning_rate": [1e-4, 1e-2],
        "batch_size": [16, 64],
        "epochs": [20, 60],
        "patience": [5, 30],
    },
    ("dnn", "binary_classification"): {
        "hidden_dim": [32, 128],
        "n_layers": [1, 4],
        "dropout": [0.0, 0.5],
        "learning_rate": [1e-4, 1e-2],
        "batch_size": [16, 64],
        "epochs": [20, 60],
        "patience": [5, 30],
    },
    ("lasso_regression", "regression"): {"alpha": [1e-4, 10.0]},
    ("ridge_regression", "regression"): {"alpha": [1e-4, 10000.0]},
    ("elastic_regression", "regression"): {
        "alpha": [1e-4, 10.0],
        "l1_ratio": [0.0, 1.0],
    },
    ("logistic_regression", "binary_classification"): {
        "C": [1e-4, 100.0],
        "class_weight": [None, "balanced"],
    },
    ("ridge_logistic_regression", "binary_classification"): {
        "C": [1e-4, 100.0],
        "class_weight": [None, "balanced"],
    },
    ("lasso_logistic_regression", "binary_classification"): {
        "C": [1e-4, 100.0],
        "class_weight": [None, "balanced"],
    },
    ("elastic_logistic_regression", "binary_classification"): {
        "C": [1e-4, 100.0],
        "l1_ratio": [0.0, 1.0],
        "class_weight": [None, "balanced"],
    },
    # tabpfn has no default HPO search space → n_trials is auto-set to 0 in
    # main.py, falling back to plain nested CV with config-file params.
}
# residual_dnn shares DNN spaces
_DEFAULT_SPACES[("residual_dnn", "regression")] = _DEFAULT_SPACES[("dnn", "regression")]
_DEFAULT_SPACES[("residual_dnn", "binary_classification")] = _DEFAULT_SPACES[("dnn", "binary_classification")]


def get_default_search_space(model_name: str, task_type: str) -> dict:
    """Return the default Optuna search space for the given model and task."""
    return _DEFAULT_SPACES.get((model_name, task_type), {})


def suggest_params(trial, search_space: dict, model_name: str) -> dict:
    """Sample one hyperparameter configuration from search_space via an Optuna trial."""
    def _is_numeric(v) -> bool:
        """True for int/float but not bool (bool is a subclass of int in Python)."""
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    params: dict = {}
    for name, bounds in search_space.items():
        # Treat as a [low, high] numeric range only when both elements are real
        # numbers.  [True, False], [None, "balanced"], etc. fall through to
        # suggest_categorical even though they also have length 2.
        if isinstance(bounds, list) and len(bounds) == 2 and all(_is_numeric(v) for v in bounds):
            low, high = bounds
            if name in _INT_PARAMS:
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                # log=True requires low > 0; fall back to linear scale if the
                # lower bound is zero (e.g. reg_alpha, reg_lambda, dropout).
                use_log = (name in _LOG_PARAMS) and (float(low) > 0)
                params[name] = trial.suggest_float(
                    name, float(low), float(high), log=use_log
                )
        elif isinstance(bounds, list):
            # Categorical
            params[name] = trial.suggest_categorical(name, bounds)
        else:
            params[name] = bounds
    return params


def build_model(model_name: str, params: dict, cfg: dict):
    """Instantiate a model from its name and a hyperparameter dict.

    Returns either a sklearn-compatible object (fit/predict interface) or a
    DNN config dict consumed by src.training.train.

    ``params`` can come from an Optuna trial (HPO path) or directly from
    ``cfg['model']`` (main experiment path).  The full ``cfg`` is passed for
    additional settings such as device, seed, and task type.
    """
    task_type = cfg.get("model", {}).get("type", "regression")
    seed = int(cfg.get("seed", 42))

    # ── XGBoost ──────────────────────────────────────────────────────────────
    if model_name == "xgboost":
        if task_type == "binary_classification":
            from model import XGBoostBinaryClassifierModel
            kw = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
            return XGBoostBinaryClassifierModel(random_state=seed, **kw)

        # Regression: use early stopping only when the param is explicitly present
        # (it is included in the default HPO search space but not in a plain cfg).
        if "early_stopping_rounds" in params:
            import torch
            import xgboost as xgb
            device = "cuda" if torch.cuda.is_available() else "cpu"
            p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
            p.update({
                "random_state": seed,
                "verbosity": 0,
                "n_jobs": 1,
                "tree_method": "hist",
                "objective": "reg:squarederror",
            })
            callbacks = [
                xgb.callback.EarlyStopping(
                    rounds=int(params["early_stopping_rounds"]),
                    metric_name="rmse",
                    data_name="validation_0",
                    save_best=True,
                )
            ]
            return xgb.XGBRegressor(**p, callbacks=callbacks, device=device)

        from model import XGBoostTreeModel
        return XGBoostTreeModel(
            random_state=seed,
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        )

    # ── DNN / ResidualDNN ────────────────────────────────────────────────────
    if model_name in ("dnn", "residual_dnn"):
        from model import DNN, ResidualDNN
        model_class = DNN if model_name == "dnn" else ResidualDNN

        # Support both HPO params (hidden_dim + n_layers) and cfg-style (hidden_dims list)
        if "hidden_dims" in params:
            hidden_dims = list(params["hidden_dims"])
        else:
            hidden_dim = int(params.get("hidden_dim", 64))
            n_layers = int(params.get("n_layers", 2))
            hidden_dims = [hidden_dim] * n_layers

        return {
            "class": model_class,
            "hidden_dims": hidden_dims,
            "output_dim": int(params.get("output_dim", 1)),
            "dropout": params.get("dropout"),
            "lr": float(params.get("learning_rate", params.get("lr", 1e-3))),
            "weight_decay": float(params.get("weight_decay", 0.0)),
            "epochs": int(params.get("epochs", 100)),
            "batch_size": int(params.get("batch_size", 32)),
            "patience": int(params.get("patience", 20)),
            "seed": seed,
        }

    # ── Regularised regression ───────────────────────────────────────────────
    if model_name == "lasso_regression":
        from sklearn.linear_model import Lasso
        return Lasso(
            alpha=float(params.get("alpha", params.get("best_alpha", 1.0))),
            max_iter=10000,
            random_state=seed,
        )

    if model_name == "ridge_regression":
        from sklearn.linear_model import Ridge
        return Ridge(
            alpha=float(params.get("alpha", params.get("best_alpha", 1000.0))),
            max_iter=10000,
            random_state=seed,
        )

    if model_name == "elastic_regression":
        from model import ElasticRegressionModel
        return ElasticRegressionModel(
            best_l1_ratio=float(params.get("l1_ratio", params.get("best_l1_ratio", 0.1))),
            best_alpha=float(params.get("alpha", params.get("best_alpha", 0.07))),
        )

    if model_name == "linear_regression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    # ── Logistic regression ──────────────────────────────────────────────────
    if model_name == "logistic_regression":
        from model import LogisticRegressionModel
        return LogisticRegressionModel(
            C=float(params.get("C", 1.0)),
            random_state=seed,
            class_weight=params.get("class_weight"),
        )

    if model_name == "ridge_logistic_regression":
        from model import RidgeLogisticRegressionModel
        return RidgeLogisticRegressionModel(
            C=float(params.get("C", 1.0)),
            random_state=seed,
            class_weight=params.get("class_weight"),
        )

    if model_name == "lasso_logistic_regression":
        from model import LassoLogisticRegressionModel
        return LassoLogisticRegressionModel(
            C=float(params.get("C", 1.0)),
            random_state=seed,
            class_weight=params.get("class_weight"),
        )

    if model_name == "elastic_logistic_regression":
        from model import ElasticLogisticRegressionModel
        return ElasticLogisticRegressionModel(
            C=float(params.get("C", 1.0)),
            l1_ratio=float(params.get("l1_ratio", 0.5)),
            random_state=seed,
            class_weight=params.get("class_weight"),
        )

    # ── TabPFN ───────────────────────────────────────────────────────────────
    if model_name == "tabpfn":
        device = cfg.get("model", {}).get("device", "cpu")
        if task_type == "binary_classification":
            if params.get("finetune", False):
                from model import FinetunedTabPFNBinaryClassifierModel
                return FinetunedTabPFNBinaryClassifierModel(
                    random_state=seed,
                    device=device,
                    epochs=int(params.get("epochs", 30)),
                    learning_rate=float(params.get("learning_rate", 1e-5)),
                )
            from model import TabPFNBinaryClassifierModel
            return TabPFNBinaryClassifierModel(random_state=seed)

        if params.get("finetune", False):
            from model import FinetunedTabPFNModel
            return FinetunedTabPFNModel(
                random_state=seed,
                device=device,
                epochs=int(params.get("epochs", 30)),
                learning_rate=float(params.get("learning_rate", 1e-5)),
            )
        from model import TabPFNModel
        return TabPFNModel(random_state=seed)

    # ── Baseline ─────────────────────────────────────────────────────────────
    if model_name == "baseline":
        from model import BaselineModel
        return BaselineModel(target_column=cfg["data"]["target"])

    raise ValueError(f"Unknown model: {model_name!r}")
