from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class _BaseLogisticRegressionModel:
    def __init__(
        self,
        *,
        l1_ratio: float = 0.0,  # 0 = L2 (ridge), 1 = L1 (lasso), (0,1) = elastic
        C: float = 1.0,
        max_iter: int = 5000,
        class_weight=None,
        random_state: int = 42,
    ):
        # lbfgs is faster for pure L2; saga is required for any L1 component.
        solver = "lbfgs" if l1_ratio == 0.0 else "saga"
        self.model = Pipeline([
            ("logreg", LogisticRegression(
                l1_ratio=l1_ratio,
                C=C,
                solver=solver,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
            )),
        ])

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)[:, 1]
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X_test)
        return self.model.predict(X_test)


class LogisticRegressionModel(_BaseLogisticRegressionModel):
    """L2-regularised logistic regression (l1_ratio=0)."""
    def __init__(self, C: float = 1.0, random_state: int = 42, class_weight=None):
        super().__init__(l1_ratio=0.0, C=C, random_state=random_state, class_weight=class_weight)


class RidgeLogisticRegressionModel(_BaseLogisticRegressionModel):
    """L2-regularised logistic regression (l1_ratio=0)."""
    def __init__(self, C: float = 1.0, random_state: int = 42, class_weight=None):
        super().__init__(l1_ratio=0.0, C=C, random_state=random_state, class_weight=class_weight)


class LassoLogisticRegressionModel(_BaseLogisticRegressionModel):
    """L1-regularised logistic regression (l1_ratio=1)."""
    def __init__(self, C: float = 1.0, random_state: int = 42, class_weight=None):
        super().__init__(l1_ratio=1.0, C=C, random_state=random_state, class_weight=class_weight)


class ElasticLogisticRegressionModel(_BaseLogisticRegressionModel):
    """Elastic-net logistic regression (l1_ratio in (0, 1))."""
    def __init__(
        self,
        C: float = 1.0,
        l1_ratio: float = 0.5,
        random_state: int = 42,
        class_weight=None,
    ):
        super().__init__(l1_ratio=l1_ratio, C=C, random_state=random_state, class_weight=class_weight)
