from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class _BaseLogisticRegressionModel:
    def __init__(
        self,
        *,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        l1_ratio: float | None = None,
        max_iter: int = 1000,
        class_weight=None,
        random_state: int = 42,
    ):
        logreg_kwargs = {
            "penalty": penalty,
            "C": C,
            "solver": solver,
            "max_iter": max_iter,
            "class_weight": class_weight,
            "random_state": random_state,
        }
        if penalty == "elasticnet":
            logreg_kwargs["l1_ratio"] = 0.5 if l1_ratio is None else l1_ratio

        self.model = Pipeline(
            [
                #("scaler", StandardScaler()),
                ("logreg", LogisticRegression(**logreg_kwargs)),
            ]
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)[:, 1]
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X_test)
        return self.model.predict(X_test)


class LogisticRegressionModel(_BaseLogisticRegressionModel):
    def __init__(self, C: float = 1.0, random_state: int = 42, class_weight=None):
        super().__init__(penalty="l2", C=C, solver="lbfgs", random_state=random_state, class_weight=class_weight)


class RidgeLogisticRegressionModel(_BaseLogisticRegressionModel):
    def __init__(self, C: float = 1.0, random_state: int = 42, class_weight=None):
        super().__init__(penalty="l2", C=C, solver="lbfgs", random_state=random_state, class_weight=class_weight)


class LassoLogisticRegressionModel(_BaseLogisticRegressionModel):
    def __init__(self, C: float = 1.0, random_state: int = 42, class_weight=None):
        super().__init__(penalty="l1", C=C, solver="saga", max_iter=5000, random_state=random_state, class_weight=class_weight)


class ElasticLogisticRegressionModel(_BaseLogisticRegressionModel):
    def __init__(
        self,
        C: float = 1.0,
        l1_ratio: float = 0.5,
        random_state: int = 42,
        class_weight=None,
    ):
        super().__init__(
            penalty="elasticnet",
            C=C,
            solver="saga",
            l1_ratio=l1_ratio,
            max_iter=5000,
            random_state=random_state,
            class_weight=class_weight,
        )
