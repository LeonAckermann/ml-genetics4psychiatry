from __future__ import annotations

import xgboost as xgb


class XGBoostBinaryClassifierModel:
    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        scale_pos_weight: float | None = None,
    ):
        model_kwargs = {
            "random_state": random_state,
            "verbosity": 0,
            "n_jobs": 1,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
        }
        if scale_pos_weight is not None:
            model_kwargs["scale_pos_weight"] = scale_pos_weight

        self.model = xgb.XGBClassifier(**model_kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, verbose=False)

    def predict(self, X_test):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)[:, 1]
        return self.model.predict(X_test)
