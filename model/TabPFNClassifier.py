from __future__ import annotations

try:
    from tabpfn import TabPFNClassifier
    from tabpfn.finetuning import FinetunedTabPFNClassifier
except ImportError:  # pragma: no cover - handled at runtime when dependency is missing
    TabPFNClassifier = None
    FinetunedTabPFNClassifier = None


class TabPFNBinaryClassifierModel:
    def __init__(self, random_state: int = 42):
        if TabPFNClassifier is None:
            raise ImportError("tabpfn is required for TabPFNBinaryClassifierModel")
        self.model = TabPFNClassifier(random_state=random_state)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)[:, 1]
        return self.model.predict(X_test)


class FinetunedTabPFNBinaryClassifierModel:
    def __init__(self, random_state: int = 42, device: str = "cuda", epochs: int = 30, learning_rate: float = 1e-5):
        if FinetunedTabPFNClassifier is None:
            raise ImportError("tabpfn is required for FinetunedTabPFNBinaryClassifierModel")
        self.model = FinetunedTabPFNClassifier(
            device=device,
            epochs=epochs,
            learning_rate=float(learning_rate),
            random_state=random_state,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)[:, 1]
        return self.model.predict(X_test)
